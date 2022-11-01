from typing import Dict, List
import os

import torch
from ts.torch_handler.base_handler import BaseHandler
import whisper
import uuid
from pathlib import Path
import logging


from whisper.decoding import DecodingOptions, DecodingResult
from torch.profiler import ProfilerActivity



ipex_enabled = False
if os.environ.get("TS_IPEX_ENABLE", "false") == "true":
    try:
        import intel_extension_for_pytorch as ipex

        ipex_enabled = True
    except ImportError as error:
        logging.warning(
            "IPEX is enabled but intel-extension-for-pytorch is not installed. Proceeding without IPEX."
        )

class WhisperHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.initialized = False

    def initialize(self, ctx):
        self.manifest = ctx.manifest
        # torch.set_num_threads(4)
        self.profiler_args = {
            "activities": [ProfilerActivity.CPU],
            "record_shapes": True,
        }

        self.model = whisper.load_model(
            self.manifest["model"]["modelType"], 
            download_root=self.manifest["model"]["modelDir"]
        )
        self.model.eval()
        if ipex_enabled:
            self.model = self.model.to(memory_format=torch.channels_last)
            self.model = ipex.optimize(self.model)

        self.device = self.model.device
        self.option = DecodingOptions(
            task="transcribe",
            language="en",
            temperature=0.0,
            prompt=[],
            without_timestamps=False,
            fp16=False,
        )

        self.initialized = True

    def preprocess(
        self, batch_audio: List[Dict[str, bytes]]
    ) -> List[torch.FloatTensor]:
        batch_mel_spec = []
        for audio_byte in batch_audio:
            tmp_file_path = Path(f"/tmp/{uuid.uuid4()}")
            with open(tmp_file_path, "wb") as f:
                f.write(audio_byte["data"])
            audio = whisper.load_audio(str(tmp_file_path))
            padded_audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(padded_audio).to(self.device)
            batch_mel_spec.append(mel)
            tmp_file_path.unlink()
        return batch_mel_spec

    def inference(self, batch_mel_spec: List[torch.FloatTensor]) -> List[torch.Tensor]:
        list_pred = []
        for tensor in batch_mel_spec:
            result = whisper.decode(self.model, tensor, self.option)
            list_pred.append(result)
        return list_pred

    def postprocess(self, list_infer_pred: List[DecodingResult]) -> List[str]:
        list_result = []
        for infer_pred in list_infer_pred:
            list_result.append(infer_pred.text)
        return list_result
