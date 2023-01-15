import os
import sys
import onnxruntime as ort


class EffRecognizer:

    def __init__(self, model, num_cores = None, providers=None):
        
        sess_options = ort.SessionOptions()
        if num_cores is not None:
            sess_options.intra_op_num_threads = num_cores

        if providers is None:
            providers = ort.get_available_providers()

        self._eng_net = ort.InferenceSession(
                    model,
                    sess_options,
                    providers=providers,
                )
        
    def __call__(self, imgs):
        return self.run(imgs)
    
    def run(self, imgs):
        return self._eng_net.run(None, {'imgs': imgs})

