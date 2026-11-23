"""Conservative phase-initialized translation ECC correction on the T1 grid."""
from lagent.actions.base_action import BaseAction
from algorithms.registration import PairRegistrar
from tools.review_tool_utils import ReviewJsonParser, description, invoke

CAPABILITY = dict(output_encoding='rgb_uint8+valid_mask_01+pixel_transform',target_object='global_translation',
                  trained_on='not_applicable',input_size=256,method_family='phase_initialized_translation_ecc')


class RegistrationCorrectionTool(BaseAction):
    """Correct a supported global translation, preserving original files and validity."""
    __tool_name__ = 'registration_correct'
    source_file = __file__
    algorithm_name = 'OpenCV phase-initialized translation ECC'
    capability = CAPABILITY

    def __init__(self, registrar=None):
        if registrar is not None and (not isinstance(registrar,PairRegistrar) or registrar.config.method!='ecc'):
            raise ValueError('registrar must be a PairRegistrar with fixed ecc profile')
        self.registrar = registrar if registrar is not None else PairRegistrar()
        super().__init__(parser=ReviewJsonParser,description=description(self,'registration_correct',
            'Put moving T2 on reference T1 pixel grid using phase-initialized translation ECC and one bilinear resampling. '
            'Returns registered/noop/unknown/refused; unknown/refused produces no corrected image. '
            'Always preserve originals; accepted output includes valid_mask and pair.json. Invalid borders cannot be passed directly '
            'to existing fully-valid RGB256 detectors. No rotation, nonrigid warping or independent accuracy claim. CPU, no weights.'))

    def run(self,image_before:str,image_after:str,output_dir:str='output'):
        """Produce a derived T2 image on T1 grid if conservative translation gates accept.

        Args:
            image_before (str): Reference T1 RGB8 256x256 PNG/JPEG.
            image_after (str): Moving T2 RGB8 256x256 PNG/JPEG.
            output_dir (str): Directory for a unique corrected image, valid-mask and metadata.

        Returns:
            dict: Accepted derived pair or explicit diagnostic rejection; no hidden detection workflow.
        """
        return invoke(self,'registration',image_before,image_after,output_dir)
