"""Independent quality-measurement action with fixed application configuration."""
from lagent.actions.base_action import BaseAction
from algorithms.image_quality import ImageQualityInspector
from tools.review_tool_utils import ReviewJsonParser, description, invoke

CAPABILITY = dict(output_encoding='diagnostics_json',target_object='image_quality',trained_on='not_applicable',
                  input_size=256,method_family='classical_image_statistics')


class ImageQualityCheckTool(BaseAction):
    """Measure RGB image quality clues without diagnosing true changes or calibrated blur."""
    __tool_name__ = 'image_quality_check'
    source_file = __file__
    algorithm_name = 'OpenCV quality statistics'
    capability = CAPABILITY

    def __init__(self, inspector=None):
        if inspector is not None and not isinstance(inspector,ImageQualityInspector):
            raise ValueError('inspector must be an ImageQualityInspector')
        self.inspector = inspector if inspector is not None else ImageQualityInspector()
        super().__init__(parser=ReviewJsonParser,description=description(self,'image_quality_check',
            'Measure brightness, clipping, Laplacian/Sobel and local review regions for two fully valid RGB8 256px PNG/JPEG images. '
            'Statistics and uncalibrated flags are clues, not a blur diagnosis, NoData/cloud mask or confidence. '
            'Writes measurements and visualization only, never corrects or detects changes. CPU, no model weights.'))

    def run(self,image_before:str,image_after:str,output_dir:str='output'):
        """Measure global/local quality of the original fully valid image pair.

        Args:
            image_before (str): Original RGB8 256x256 PNG/JPEG at T1.
            image_after (str): Original RGB8 256x256 PNG/JPEG at T2.
            output_dir (str): Directory for a unique record and measurement visualizations.

        Returns:
            dict: Measurements and uncalibrated review clues, or an explicit action error.
        """
        return invoke(self,'quality',image_before,image_after,output_dir)
