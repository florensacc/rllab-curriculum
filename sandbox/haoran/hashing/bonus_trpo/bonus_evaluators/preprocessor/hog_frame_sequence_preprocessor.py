from sandbox.haoran.hashing.bonus_trpo.bonus_evaluators.preprocessor.base import Preprocessor
import numpy as np

class HOGFrameSequencePreprocessor(Preprocessor):
    """
    Accepts an input composed of consecutive frames. Produce an output suitable for HOGFeatureExtractor.
    """
    def __init__(self,n_last_screens,img_height,img_width, option="cur_minus_prev"):
        self.n_last_screens = n_last_screens
        self.img_height = img_height
        self.img_width = img_width
        self._input_dim = (n_last_screens, img_height, img_width)

        if option == "cur_minus_prev":
            self._output_dim = np.prod([img_height, img_width])
        else:
            raise NotImplementedError
        self.option = option

    def get_input_dim(self):
        return self._input_dim

    def get_output_dim(self):
        return self._output_dim

    def process(self,items):
        # input image: ()
        frame_start_indices = [
            self.img_height * self.img_width * i
            for i in range(self.n_last_screens)
        ]
        frame_size = self.img_height * self.img_width

        if self.option == "cur_minus_prev":
            processed_items = np.asarray([
                item[frame_start_indices[-1]:] - \
                item[frame_start_indices[-2]:frame_start_indices[-1]]
                for item in items
            ])
        else:
            raise NotImplementedError
        return processed_items
