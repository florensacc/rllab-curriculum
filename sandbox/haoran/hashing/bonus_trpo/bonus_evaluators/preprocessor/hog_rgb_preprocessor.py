from sandbox.haoran.hashing.bonus_trpo.bonus_evaluators.preprocessor.base import Preprocessor
import numpy as np

class HOGRGBPreprocessor(Preprocessor):
    def __init__(self,img_height,img_width,img_channel):
        self.img_height = img_height
        self.img_width = img_width
        self.img_channel = img_channel
        self._input_dim = (img_height, img_width, img_channel)
        self._output_dim = np.prod([img_height, img_width, img_channel])

    def get_input_dim(self):
        return self._input_dim

    def get_output_dim(self):
        return self._output_dim

    def process(self,items):
        # input image: (height, width, channel)
        processed_items = np.asarray([
            np.transpose(item,(2,1,0))
            for item in items
        ])
        return processed_items
