"""Contract envelopes for the three independent review tools, not a workflow."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
import uuid

import cv2
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from lagent.schema import ActionReturn, ActionStatusCode
from lagent.actions.parser import JsonParser, ParseError

ROOT = Path(__file__).resolve().parents[1]


class ReviewJsonParser(JsonParser):
    """Object-only JSON for these tools; keep legacy Interpreter parsing untouched."""
    def parse_inputs(self, inputs, name='run'):
        if name not in self._api2param:
            raise ParseError('Undeclared review-tool API')
        if isinstance(inputs, str):
            encoded = inputs.strip()
            if encoded.startswith('```json\n') and encoded.endswith('```'):
                encoded = encoded[8:-3].strip()
            try:
                inputs = json.loads(encoded)
            except (ValueError, TypeError) as error:
                raise ParseError('Review tools require a JSON object') from error
        if not isinstance(inputs, dict):
            raise ParseError('Review tools require a JSON object')
        return super().parse_inputs(inputs, name)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    # Every invocation owns a fresh directory, so existing records are not replaced.
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def description(tool, name, text):
    value = deepcopy(tool.__tool_description__)
    value['name'] = name
    value['description'] = text
    return value


def _arguments(image_before, image_after, output_dir):
    values = {'image_before': image_before, 'image_after': image_after, 'output_dir': output_dir}
    for name, value in values.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(name + ' must be a nonempty path string')
    before, after, output = [Path(values[name]).expanduser().resolve()
                             for name in ('image_before', 'image_after', 'output_dir')]
    for name, path in (('image_before', before), ('image_after', after)):
        if not path.is_file():
            raise ValueError(name + ' file does not exist: ' + str(path))
    if output.exists() and not output.is_dir():
        raise ValueError('output_dir must be a directory, not a file')
    return before, after, output


def _artifact(path, kind, reference=None, **fields):
    result = dict(kind=kind, path=str(path.resolve()), sha256=sha256(path), **fields)
    if reference is not None:
        result.update(reference_image=str(reference), size=[256, 256], coordinate_system='pixel_xy_top_left')
    return result


def _quality_artifacts(report, run_dir, before, after):
    artifacts = []
    log_values = [np.log1p(tile['sharpness']['laplacian_variance'])
                  for image in report['images'].values() for tile in image['tiles']]
    minimum, maximum = float(min(log_values)), float(max(log_values))
    for key, reference in (('before', before), ('after', after)):
        canvas = np.full((256, 256), 128, np.uint8)
        for tile in report['images'][key]['tiles']:
            x0, y0, x1, y1 = tile['bbox']
            value = np.log1p(tile['sharpness']['laplacian_variance'])
            encoded = int(round(255 * (value-minimum)/(maximum-minimum))) if maximum > minimum else 128
            canvas[y0:y1, x0:x1] = encoded
        path = run_dir / ('laplacian_tiles_' + key + '.png')
        Image.fromarray(canvas).save(path)
        artifacts.append(_artifact(path, 'measurement_visualization', reference,
             encoding='uint8 display of tile log1p(Laplacian variance), shared pair min/max; not a quality/NoData mask',
             display_range_log1p=[minimum, maximum], bbox_convention='[x0,y0,x1,y1), exclusive upper bounds'))
    metrics_path = run_dir / 'quality_metrics.json'
    write_json(metrics_path, report)
    artifacts.append(_artifact(metrics_path, 'quality_measurements'))
    diagnostics = {key: report[key] for key in ('assessment', 'threshold_profile', 'heuristic_flags', 'pair')}
    diagnostics['images'] = {}
    for key, value in report['images'].items():
        short = {name: value[name] for name in ('validation', 'brightness', 'clipping', 'sharpness',
                                               'texture_assessment', 'assessment', 'heuristic_flags')}
        short['local_review_regions'] = [dict(bbox=tile['bbox'], assessment=tile['assessment'],
             heuristic_flags=tile['heuristic_flags'], sharpness=tile['sharpness'])
             for tile in value['tiles'] if tile['heuristic_flags']]
        diagnostics['images'][key] = short
    return artifacts, diagnostics


def invoke(tool, operation, image_before, image_after, output_dir):
    try:
        before, after, output = _arguments(image_before, image_after, output_dir)
    except (OSError, ValueError, TypeError) as error:
        return ActionReturn(type=tool.name, state=ActionStatusCode.ARGS_ERROR, errmsg=str(error))
    started = time.perf_counter()
    try:
        hashes = {'image_before': sha256(before), 'image_after': sha256(after)}
        if operation == 'quality':
            report = tool.inspector.inspect(str(before), str(after))
        elif operation == 'alignment':
            report = tool.inspector.inspect(str(before), str(after))
        elif operation == 'registration':
            report = tool.registrar.register(str(before), str(after))
        else:
            raise RuntimeError('Unsupported review operation')
    except ValueError as error:
        return ActionReturn(type=tool.name, state=ActionStatusCode.ARGS_ERROR, errmsg=str(error))
    except Exception as error:
        return ActionReturn(type=tool.name, state=ActionStatusCode.API_ERROR, errmsg=str(error))
    try:
        if any(sha256(path) != hashes[name] for name, path in (('image_before', before), ('image_after', after))):
            raise RuntimeError('Input changed during execution; refusing inconsistent provenance')
        run_id = uuid.uuid4().hex
        run_dir = output / (tool.name + '_' + run_id)
        run_dir.mkdir(parents=True, exist_ok=False)
        artifacts, warnings = [], []
        if operation == 'quality':
            artifacts, diagnostics = _quality_artifacts(report, run_dir, before, after)
            warnings.extend(report['limitations'])
            summary = 'Quality measurements computed; heuristic review status=' + report['assessment'] + '; not a blur diagnosis'
            configuration = report['config']
            version = report['algorithm_version']
        else:
            from algorithms.registration.inference import VERSION
            version, configuration = VERSION, report['config']
            rgb, valid = report.pop('aligned_image', None), report.pop('valid_mask', None)
            diagnostics = report
            warnings.extend(report['warnings'])
            diagnostics['threshold_profile'] = 'uncalibrated_translation_v1'
            if report['transform'] is not None:
                transform = dict(report['transform'], reference_image=str(before), moving_image=str(after))
                path = run_dir / 'transform.json'
                write_json(path, transform)
                artifacts.append(_artifact(path, 'pixel_transform', before))
            if operation == 'registration' and report['status'] in ('registered', 'noop'):
                if rgb is None or valid is None or rgb.shape != (256, 256, 3) or rgb.dtype != np.uint8:
                    raise RuntimeError('Registrar returned an invalid RGB image')
                if valid.shape != (256, 256) or not np.isin(valid, (0, 1)).all():
                    raise RuntimeError('Registrar returned an invalid valid-mask')
                aligned_path, valid_path = run_dir / 'image_after_aligned.png', run_dir / 'valid_mask.png'
                fully_valid = bool(valid.all())
                metadata = PngInfo()
                if not fully_valid:
                    # This declares an explicit validity mask, not a magic RGB
                    # NoData value. Our full-valid input readers reject it.
                    metadata.add_text('NoData_mask',str(valid_path))
                Image.fromarray(rgb).save(aligned_path,pnginfo=metadata)
                Image.fromarray(valid.astype(np.uint8)).save(valid_path)
                artifacts.extend([_artifact(aligned_path, 'registered_image', before, moving_image=str(after), encoding='RGB uint8'),
                                  _artifact(valid_path, 'valid_mask', before, labels={'0':'invalid','1':'valid'})])
                diagnostics['downstream'] = dict(fully_valid_rgb256=fully_valid,
                    compatible_with_existing_full_valid_256_detectors=fully_valid,
                    requires_mask_aware_processing=not fully_valid,
                    explanation='Existing detector contracts require fully valid RGB256. Invalid borders require an explicit mask/grid handling strategy before detection; do not treat fill as change.')
                pair_path = run_dir / 'pair.json'
                write_json(pair_path, dict(image_before=str(before), image_after=str(aligned_path),
                    valid_mask=str(valid_path), reference_image=str(before), shape=[256, 256],
                    coordinate_system='pixel_xy_top_left', transform=report['transform'],
                    downstream=diagnostics['downstream']))
                artifacts.append(_artifact(pair_path, 'registered_pair'))
                if not fully_valid:
                    warnings.append('Invalid output edges exist. Direct invocation of the current full-valid RGB256 detectors is not supported; pair.json records this precondition.')
            elif operation == 'registration':
                if rgb is not None or valid is not None:
                    raise RuntimeError('Unknown/refused registration must not return corrected pixels')
                diagnostics['downstream'] = dict(fully_valid_rgb256=False,
                    compatible_with_existing_full_valid_256_detectors=False,
                    explanation='No corrected image produced; registration was not accepted.')
            summary = operation + ' status=' + report['status'] + '; reason=' + report['reason']
            warnings.append('Thresholds are conservative uncalibrated screening gates, not independently validated registration accuracy.')
        sources = ['tools/review_tool_utils.py', str(Path(tool.source_file).relative_to(ROOT)).replace('\\','/')]
        sources.append('algorithms/image_quality/inference.py' if operation == 'quality' else 'algorithms/registration/inference.py')
        result = dict(schema_version='0.1',run_id=run_id,summary=summary,artifacts=artifacts,
            diagnostics=diagnostics,warnings=warnings,elapsed_seconds=time.perf_counter()-started,
            provenance=dict(algorithm=tool.algorithm_name,algorithm_version=version,weights='not_applicable',
                capability=tool.capability,configuration=configuration,
                inputs={'image_before':str(before),'image_after':str(after)},input_sha256=hashes,
                source_sha256={name:sha256(ROOT/name) for name in sources},
                runtime={'numpy':np.__version__,'opencv':cv2.__version__},
                timing_scope='input hashes+reads+validation+algorithm+artifact writes; excludes final result.json write',
                original_inputs_modified=False,coordinate_system='raw pixel_xy_top_left; no georeferencing'))
        record_path = run_dir / 'result.json'
        result['artifacts'].append(dict(kind='tool_record',path=str(record_path)))
        write_json(record_path,result)
        return result
    except Exception as error:
        return ActionReturn(type=tool.name,state=ActionStatusCode.API_ERROR,errmsg=str(error))
