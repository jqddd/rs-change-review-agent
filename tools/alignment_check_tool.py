"""Phase-based translation inspection; does not correct input files."""
from lagent.actions.base_action import BaseAction
from algorithms.registration import AlignmentInspector
from tools.review_tool_utils import ReviewJsonParser, description, invoke

CAPABILITY = dict(output_encoding='diagnostics_json+pixel_transform',target_object='global_translation',
                  trained_on='not_applicable',input_size=256,method_family='phase_correlation')


class AlignmentCheckTool(BaseAction):
    """Inspect global translation with phase correlation and conservative internal gates."""
    __tool_name__ = 'alignment_check'
    source_file = __file__
    algorithm_name = 'OpenCV phase translation inspection'
    capability = CAPABILITY

    def __init__(self, inspector=None):
        if inspector is not None and (not isinstance(inspector,AlignmentInspector) or inspector.config.method!='phase'):
            raise ValueError('inspector must be an AlignmentInspector with fixed phase profile')
        self.inspector = inspector if inspector is not None else AlignmentInspector()
        super().__init__(parser=ReviewJsonParser,description=description(self,'alignment_check',
            'Estimate T2 displacement relative to T1 on strict RGB8 256px inputs, without changing images. '
            'Returns aligned/misaligned/unknown/refused with internal evidence and an accepted candidate transform when available. '
            'Phase scores are not confidence or independent TRE; real temporal change may invalidate translation. CPU, no weights.'))

    def run(self,image_before:str,image_after:str,output_dir:str='output'):
        """Inspect possible whole-image translation, keeping original pixels unchanged.

        Args:
            image_before (str): Reference T1 RGB8 256x256 PNG/JPEG.
            image_after (str): Moving T2 RGB8 256x256 PNG/JPEG.
            output_dir (str): Directory for a unique diagnostic and optional transform record.

        Returns:
            dict: Candidate translation evidence or unknown/refused status; not ground-truth accuracy.
        """
        return invoke(self,'alignment',image_before,image_after,output_dir)
