#
# Copyright 2018 Picovoice Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
from collections import namedtuple
from enum import Enum

import openwakeword
from scipy.signal import resample_poly
import numpy as np
import pvporcupine



class Engines(Enum):
    OPEN_WAKE_WORD = 'OpenWakeWord'
    PORCUPINE = 'Porcupine'



SensitivityInfo = namedtuple('SensitivityInfo', 'min, max, step')


class Engine(object):
    def process(self, pcm):
        raise NotImplementedError()

    def release(self):
        raise NotImplementedError()

    def __str__(self):
        raise NotImplementedError()

    #@staticmethod
    def frame_length(self):
        if isinstance(self, OpenWakeWordEngine):
            print("Using frame length 1280 for OpenWakeWord")
            return 1280
        elif isinstance(self, PorcupineEngine):
            return 512

        #return 1280

    @staticmethod
    def sensitivity_info(engine_type):
        if engine_type is Engines.OPEN_WAKE_WORD:
            return SensitivityInfo(0, 0.5, 0.1)
        elif engine_type is Engines.PORCUPINE:
            return SensitivityInfo(0, 1, 0.1)
        else:
            raise ValueError("no sensitivity range for '%s'", engine_type.value)

    @staticmethod
    def create(engine, keyword, sensitivity, **kwargs):
        if engine is Engines.OPEN_WAKE_WORD:
            return OpenWakeWordEngine(keyword, sensitivity, **kwargs)
        elif engine is Engines.PORCUPINE:
            return PorcupineEngine(keyword, sensitivity, **kwargs)
        else:
            raise ValueError(f"cannot create engine of type '{engine.value}'")


class PorcupineEngine(Engine):
    def __init__(self, keyword, sensitivity):
        self._porcupine = pvporcupine.create(
            access_key="BEQ0XFaUI4XX+OthpLQtH6B1r3+imqG7fq6ojbyxBC9rUB1TAMy05w==",
            keywords=[keyword.lower()],
            sensitivities=[sensitivity])

    def process(self, pcm):
        assert pcm.dtype == np.int16
        #print(f"PorcupineEngine processing pcm of shape: {pcm.shape}, dtype: {pcm.dtype}")
        #scores = self._porcupine.process(pcm)
        #print(f"Scores: {scores}")
        return self._porcupine.process(pcm) == 0

    def release(self):
        self._porcupine.delete()

    def __str__(self):
        return 'Porcupine'



class OpenWakeWordEngine(Engine):
    def __init__(self, keyword, sensitivity, model_path=None, sample_rate=16000, **kwargs):
        #if model_path is None:
        #    openwakeword.utils.download_models()
        #    self._model = openwakeword.Model(wakeword_models=[keyword.lower()])
        #else:
        self._model = openwakeword.Model(wakeword_models=["eval_models/Hey_Ford.onnx"], inference_framework="onnx")

        self._sensitivity = sensitivity
        print(f"OpenWakeWordEngine initialized with sensitivity {self._sensitivity}")
        self._keyword = keyword.lower()
        #print(f"Initialized OpenWakeWordEngine with keyword '{self._keyword}' and sensitivity {self._sensitivity}")
        self._sample_rate = sample_rate  # expected 16 kHz
    def process(self, pcm, input_rate=16000):
        """
        pcm: numpy array of mono audio samples.
        Automatically resamples and converts to 16-bit 16 kHz PCM if needed.
        """

        # --- ensure 1D mono numpy array ---
        pcm = np.asarray(pcm)#.flatten()
        scores = self._model.predict(pcm)
        
        # --- convert to float32 range [-1, 1] ---
        #pcm_f = pcm.astype(np.float32) / 32768.0

        # --- run inference ---
        score = 0.0
        for mdl in self._model.prediction_buffer.keys():
            scores = list(self._model.prediction_buffer[mdl])
            if not scores:
                continue
            score = scores[-1]
        '''
        score = list(scores.values())[-1]
        #print(f"Score for '{self._keyword}': {score}")
        #print(score >= self._sensitivity)
        '''
        return score >= self._sensitivity

    def release(self):
        pass

    def __str__(self):
        return f"OpenWakeWord({self._keyword})"
    