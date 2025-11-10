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
from pocketsphinx import (
    Config,
    Decoder,
    get_model_path
)

#from engines import snowboydetect


class Engines(Enum):
    POCKET_SPHINX = 'PocketSphinx'
    PORCUPINE = 'Porcupine'
    OPEN_WAKE_WORD = 'OpenWakeWord'
    EFFICIENTWORD = 'EfficientWordNet'

    #SNOWBOY = 'Snowboy'


SensitivityInfo = namedtuple('SensitivityInfo', 'min, max, step')


class Engine(object):
    def process(self, pcm):
        raise NotImplementedError()

    def release(self):
        raise NotImplementedError()

    def __str__(self):
        raise NotImplementedError()

    @staticmethod
    def frame_length(engine_type):
        if engine_type is Engines.OPEN_WAKE_WORD:
            return 12800
        elif engine_type is Engines.PORCUPINE:
            return 512
        elif engine_type is Engines.EFFICIENTWORD:
            return 24000
        elif engine_type is Engines.POCKET_SPHINX:
            return 512
        #elif engine_type is Engines.SNOWBOY:
        #    return 512
        return 512

    @staticmethod
    def sensitivity_info(engine_type):
        if engine_type is Engines.OPEN_WAKE_WORD:
            return SensitivityInfo(0, 1, 0.1)
        elif engine_type is Engines.PORCUPINE:
            return SensitivityInfo(0, 1, 0.1)
        elif engine_type is Engines.EFFICIENTWORD:
            return SensitivityInfo(0, 1, 0.1)
        if engine_type is Engines.POCKET_SPHINX:
            return SensitivityInfo(-21, 15, 3)
        #elif engine_type is Engines.SNOWBOY:
        #    return SensitivityInfo(0, 1, 0.05)
        else:
            raise ValueError("no sensitivity range for '%s'", engine_type.value)

    @staticmethod
    def create(engine, keyword, sensitivity, **kwargs):
        if engine is Engines.OPEN_WAKE_WORD:
            return OpenWakeWordEngine(keyword, sensitivity, **kwargs)
        elif engine is Engines.PORCUPINE:
            return PorcupineEngine(keyword, sensitivity, **kwargs)
        elif engine is Engines.EFFICIENTWORD:
            return EfficientWordNetEngine(keyword, sensitivity, **kwargs)
        elif engine is Engines.POCKET_SPHINX:
            return PocketSphinxEngine(keyword, sensitivity)
        else:
            raise ValueError(f"cannot create engine of type '{engine.value}'")


class PocketSphinxEngine(Engine):
    def __init__(self, keyword, sensitivity):
        config = Config()
        config.set_string('-logfn', '/dev/null')
        config.set_string('-hmm', os.path.join(get_model_path('en-us'), 'en-us'))
        config.set_string('-dict', os.path.join(get_model_path('en-us'), 'cmudict-en-us.dict'))
        config.set_string('-lm', None)
        config.set_string('-keyphrase', keyword if keyword != 'snowboy' else 'snow boy')
        config.set_float('-kws_threshold', 10 ** -sensitivity)

        self._decoder = Decoder(config)
        self._decoder.start_utt()

    def process(self, pcm):
        assert pcm.dtype == np.int16

        self._decoder.process_raw(pcm.tobytes(), False, False)

        detected = self._decoder.hyp()
        if detected:
            self._decoder.end_utt()
            self._decoder.start_utt()

        return detected

    def release(self):
        self._decoder.end_utt()

    def __str__(self):
        return 'PocketSphinx'


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


class SnowboyEngine(Engine):
    def __init__(self, keyword, sensitivity):
        keyword = keyword.lower()
        if keyword == 'alexa':
            model_relative_path = 'engines/snowboy/resources/alexa/alexa-avs-sample-app/alexa.umdl'
        else:
            model_relative_path = 'engines/snowboy/resources/models/%s.umdl' % keyword.replace(' ', '_')

        model_str = os.path.join(os.path.dirname(__file__), model_relative_path).encode()
        resource_filename = os.path.join(os.path.dirname(__file__), 'engines/snowboy/resources/common.res').encode()
        self._snowboy = snowboydetect.SnowboyDetect(resource_filename=resource_filename, model_str=model_str)

        # https://github.com/Kitt-AI/snowboy#pretrained-universal-models

        if keyword == 'jarvis':
            self._snowboy.SetSensitivity(('%f,%f' % (sensitivity, sensitivity)).encode())
        else:
            self._snowboy.SetSensitivity(str(sensitivity).encode())

        if keyword in {'alexa', 'computer', 'jarvis', 'view glass'}:
            self._snowboy.ApplyFrontend(True)
        else:
            self._snowboy.ApplyFrontend(False)

    def process(self, pcm):
        assert pcm.dtype == np.int16

        return self._snowboy.RunDetection(pcm.tobytes()) == 1

    def release(self):
        pass

    def __str__(self):
        return 'Snowboy'



class OpenWakeWordEngine(Engine):
    def __init__(self, keyword, sensitivity, model_path=None, sample_rate=16000, **kwargs):
        if model_path is None:
            openwakeword.utils.download_models()
            self._model = openwakeword.Model(wakeword_models=[keyword.lower()])
        else:
            self._model = openwakeword.Model(wakeword_models=[model_path])

        self._sensitivity = sensitivity
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
        
        # --- convert to float32 range [-1, 1] ---
        #pcm_f = pcm.astype(np.float32) / 32768.0

        # --- run inference ---
        scores = self._model.predict(pcm)
        score = list(scores.values())[0]
        #print(f"Score for '{self._keyword}': {score}")
        #print(score >= self._sensitivity)
        return score >= self._sensitivity

    def release(self):
        pass

    def __str__(self):
        return f"OpenWakeWord({self._keyword})"
    
import numpy as np
from eff_word_net.engine import HotwordDetector
from eff_word_net.streams import SimpleMicStream  # or for file processing
from eff_word_net import samples_loc
print(samples_loc)
# other imports as required
from eff_word_net.audio_processing import Resnet50_Arc_loss

class EfficientWordNetEngine(Engine):
    def __init__(self, keyword, sensitivity, reference_file=None, model_file=None):
        """
        keyword: e.g., 'alexa'
        sensitivity: numeric threshold for confidence (0.0-1.0)
        reference_file: path to _ref.json generated for the keyword
        model_file: optional custom model path
        """
        self.keyword = keyword.lower()
        self.sensitivity = sensitivity
        if reference_file is None:
            # default built-in sample ref
            from eff_word_net import samples_loc
            reference_file = os.path.join(samples_loc, f"{self.keyword}_ref.json")
        self.detector = HotwordDetector(
            model= Resnet50_Arc_loss(),
            hotword="Alexa",
            reference_file=reference_file,
            threshold=sensitivity,
            relaxation_time=0.8
        )

    @staticmethod
    def frame_length(engine_type):
        # EfficientWord-Net typically uses e.g. ~1 s windows? you’ll calibrate.
        return 24000  # e.g., 1 second @16kHz

    def process(self, pcm):
        """
        pcm: numpy array of int16 mono samples at correct sample rate
        Returns True if wake-word detected in this frame.
        """
        # Ensure dtype
        assert pcm.dtype == np.int16
        res = self.detector.scoreFrame(pcm)
        if res is None:
            return False
        match = res.get("match", False)
        if match and res.get("confidence", 0.0) >= self.sensitivity:
            return True
        return False

    def release(self):
        # any cleanup; likely none
        pass

    def __str__(self):
        return f"EfficientWordNet({self.keyword})"
