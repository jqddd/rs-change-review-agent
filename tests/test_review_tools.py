"""Real CPU ActionExecutor/ReAct contracts, not mock model accuracy tests."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image
from lagent.actions import ActionExecutor
from lagent.agents import ReAct
from lagent.schema import ActionReturn,ActionStatusCode
from tools.demo_review_tools_agent import ReviewActionExecutor,make_executor
from tools.image_quality_tool import ImageQualityCheckTool
from tools.alignment_check_tool import AlignmentCheckTool
from tools.registration_tool import RegistrationCorrectionTool
from tools.review_tool_utils import sha256


class ReviewToolTests(unittest.TestCase):
    def setUp(self):
        parent=Path(__file__).resolve().parents[1]/'tmp_dir'
        parent.mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=parent,prefix='review_tools_test_')
        self.root=Path(self.temp.name)
        self.a,self.b=self.root/'before.png',self.root/'after.png'
        rng=np.random.default_rng(87)
        noise=rng.integers(25,225,(256,256,3),dtype=np.uint8)
        self.rgb=cv2.GaussianBlur(noise,(3,3),.6)
        Image.fromarray(self.rgb).save(self.a)
        shifted=cv2.warpAffine(self.rgb,np.float32([[1,0,4],[0,1,-3]]),(256,256),borderMode=cv2.BORDER_REFLECT_101)
        Image.fromarray(shifted).save(self.b)
        self.args=dict(image_before=str(self.a),image_after=str(self.b),output_dir=str(self.root/'out'))

    def tearDown(self):self.temp.cleanup()

    def call(self,executor,name,values=None):
        action=executor(name,json.dumps(self.args if values is None else values))
        self.assertEqual(action.state,ActionStatusCode.SUCCESS,action.errmsg)
        return json.loads(action.result[0]['content'])

    def test_registered_names_and_all_three_real_calls(self):
        executor=make_executor()
        self.assertEqual(executor.action_names(),['image_quality_check','alignment_check','registration_correct'])
        for item in executor.get_actions_info():
            self.assertEqual(set(item['required']),{'image_before','image_after'})
        original=(sha256(self.a),sha256(self.b))
        for name in executor.action_names():
            result=self.call(executor,name)
            self.assertEqual(result['schema_version'],'0.1')
            self.assertEqual(result['provenance']['weights'],'not_applicable')
            self.assertFalse(result['provenance']['original_inputs_modified'])
            for artifact in result['artifacts']:
                path=Path(artifact['path'])
                self.assertTrue(path.is_absolute() and path.is_file())
                if 'sha256' in artifact:self.assertEqual(sha256(path),artifact['sha256'])
            self.assertTrue(Path(result['provenance']['inputs']['image_before']).is_absolute())
        self.assertEqual(original,(sha256(self.a),sha256(self.b)))

    def test_registration_output_support_and_reference_grid(self):
        result=self.call(make_executor(),'registration_correct')
        self.assertEqual(result['diagnostics']['status'],'registered')
        artifacts={a['kind']:a for a in result['artifacts']}
        valid=np.asarray(Image.open(artifacts['valid_mask']['path']))
        corrected=np.asarray(Image.open(artifacts['registered_image']['path']))
        self.assertEqual(valid.shape,(256,256));self.assertTrue(np.isin(valid,[0,1]).all())
        self.assertFalse(valid.all())
        # Independent known synthetic coordinates, not optimizer's fitted residual.
        self.assertLess(np.mean(np.abs(corrected[valid==1].astype(float)-self.rgb[valid==1])),2.0)
        self.assertFalse(result['diagnostics']['downstream']['compatible_with_existing_full_valid_256_detectors'])
        self.assertTrue(result['diagnostics']['downstream']['requires_mask_aware_processing'])
        pair=json.loads(Path(artifacts['registered_pair']['path']).read_text())
        self.assertEqual(pair['image_before'],str(self.a.resolve()))
        self.assertEqual(artifacts['valid_mask']['labels'],{'0':'invalid','1':'valid'})

    def test_noop_does_not_resample_pixels(self):
        values=dict(self.args,image_after=str(self.a))
        result=self.call(make_executor(),'registration_correct',values)
        self.assertEqual(result['diagnostics']['status'],'noop')
        artifact=next(a for a in result['artifacts'] if a['kind']=='registered_image')
        np.testing.assert_array_equal(np.asarray(Image.open(artifact['path'])),self.rgb)
        self.assertTrue(result['diagnostics']['downstream']['fully_valid_rgb256'])

    def test_corrected_invalid_edges_are_explicitly_declared_and_not_reused_as_full_valid(self):
        result=self.call(make_executor(),'registration_correct')
        path=next(a['path'] for a in result['artifacts'] if a['kind']=='registered_image')
        with Image.open(path) as image:
            image.load()
            self.assertIn('NoData_mask',image.info)
        values=dict(self.args,image_after=path)
        for name in make_executor().action_names():
            action=make_executor()(name,json.dumps(values))
            self.assertEqual(action.state,ActionStatusCode.ARGS_ERROR)

    def test_unknown_is_successful_diagnostic_without_corrected_image(self):
        Image.fromarray(np.full((256,256,3),100,np.uint8)).save(self.a)
        Image.fromarray(np.full((256,256,3),100,np.uint8)).save(self.b)
        executor=make_executor()
        inspection=self.call(executor,'alignment_check')
        result=self.call(executor,'registration_correct')
        self.assertEqual(inspection['diagnostics']['status'],'unknown')
        self.assertEqual(result['diagnostics']['status'],'unknown')
        self.assertNotIn('registered_image',[a['kind'] for a in result['artifacts']])
        self.assertFalse(result['diagnostics']['downstream']['compatible_with_existing_full_valid_256_detectors'])

    def test_quality_raw_tiles_and_uncalibrated_status_are_preserved(self):
        result=self.call(make_executor(),'image_quality_check')
        self.assertEqual(result['diagnostics']['threshold_profile'],'uncalibrated_v1')
        record=next(a for a in result['artifacts'] if a['kind']=='quality_measurements')
        full=json.loads(Path(record['path']).read_text())
        self.assertEqual(len(full['images']['before']['tiles']),16)
        self.assertNotIn('confidence',result['diagnostics'])
        maps=[a for a in result['artifacts'] if a['kind']=='measurement_visualization']
        self.assertEqual(len(maps),2)
        self.assertEqual(maps[0]['display_range_log1p'],maps[1]['display_range_log1p'])

    def test_json_scalars_lists_and_code_are_args_errors_not_executor_crashes(self):
        for raw in ('null','1','true','[]','[{}]','NaN','print(1)','{"image_before": 1}'):
            with self.subTest(raw=raw):
                action=make_executor()('image_quality_check',raw)
                self.assertEqual(action.state,ActionStatusCode.ARGS_ERROR)

    def test_types_unknown_arguments_and_invalid_image_are_args_errors(self):
        for name in make_executor().action_names():
            for key,value in [('image_before',True),('image_after',[]),('output_dir',17),('output_dir',''),('image_before','missing.png')]:
                with self.subTest(name=name,key=key,value=value):
                    action=make_executor()(name,json.dumps(dict(self.args,**{key:value})))
                    self.assertEqual(action.state,ActionStatusCode.ARGS_ERROR)
            action=make_executor()(name,json.dumps(dict(self.args,config={'max_displacement_px':99})))
            self.assertEqual(action.state,ActionStatusCode.ARGS_ERROR)
        Image.fromarray(np.zeros((256,256),np.uint8)).save(self.b)
        for name in make_executor().action_names():
            self.assertEqual(make_executor()(name,json.dumps(self.args)).state,ActionStatusCode.ARGS_ERROR)

    def test_output_path_to_file_and_io_failure_are_distinguished(self):
        action=make_executor()('image_quality_check',json.dumps(dict(self.args,output_dir=str(self.a))))
        self.assertEqual(action.state,ActionStatusCode.ARGS_ERROR)
        with patch('tools.review_tool_utils.write_json',side_effect=OSError('simulated write failure')):
            action=make_executor()('image_quality_check',json.dumps(self.args))
        self.assertEqual(action.state,ActionStatusCode.API_ERROR)

    def test_fenced_json_object_and_unique_artifacts(self):
        executor=make_executor()
        a=self.call(executor,'alignment_check')
        action=executor('alignment_check','```json\n'+json.dumps(self.args)+'\n```')
        self.assertEqual(action.state,ActionStatusCode.SUCCESS,action.errmsg)
        b=json.loads(action.result[0]['content'])
        self.assertNotEqual(a['run_id'],b['run_id'])
        self.assertNotEqual(a['artifacts'][-1]['path'],b['artifacts'][-1]['path'])

    def test_fixed_budgets_are_enforced_by_executor(self):
        executor=ReviewActionExecutor(max_calls=2,max_registration_calls=1)
        self.call(executor,'registration_correct')
        denied=executor('registration_correct',json.dumps(self.args))
        self.assertEqual(denied.state,ActionStatusCode.API_ERROR)
        self.call(executor,'alignment_check')
        self.assertEqual(executor('image_quality_check',json.dumps(self.args)).state,ActionStatusCode.API_ERROR)

    def test_dotted_api_aliases_cannot_bypass_call_budget_or_crash_executor(self):
        executor=ReviewActionExecutor(max_calls=1,max_registration_calls=1)
        for name in ('registration_correct.run','registration_correct.__init__',
                     'registration_correct.run.extra','image_quality_check.run',None,17):
            action=executor(name,json.dumps(self.args))
            self.assertEqual(action.state,ActionStatusCode.ARGS_ERROR)
        self.assertEqual(executor.calls,0)
        self.call(executor,'registration_correct')
        self.assertEqual(executor.calls,1)
        self.assertEqual(executor('registration_correct.run',json.dumps(self.args)).state,ActionStatusCode.ARGS_ERROR)
        self.assertEqual(executor('registration_correct',json.dumps(self.args)).state,ActionStatusCode.API_ERROR)

    def test_undeclared_method_is_rejected_by_local_parser_in_stock_executor(self):
        executor=ActionExecutor([ImageQualityCheckTool()])
        for name in ('image_quality_check.__init__','image_quality_check.inspector'):
            action=executor(name,json.dumps(self.args))
            self.assertEqual(action.state,ActionStatusCode.ARGS_ERROR)

    def test_real_react_loop_with_scripted_llm_calls_actual_tools(self):
        # This verifies protocol/execution plumbing; not real DeepSeek or intelligent choice.
        executor=make_executor()
        commands=['Thought: perform requested diagnostic\nAction: '+name+'\nAction Input: '+json.dumps(self.args)
                  for name in executor.action_names()]
        commands.append('Thought: completed\nFinal Answer: tools executed; see actual diagnostic records')
        class ScriptedLLM:
            def __init__(self):self.responses=iter(commands)
            def chat(self,prompt,**kwargs):return next(self.responses)
        result=ReAct(llm=ScriptedLLM(),action_executor=executor,max_turn=6).chat('Explicitly invoke all three tools')
        names=[a.type for a in result.actions if a.type in executor.action_names()]
        self.assertEqual(names,executor.action_names())
        self.assertTrue(all(a.state==ActionStatusCode.SUCCESS for a in result.actions))


if __name__=='__main__':unittest.main()
