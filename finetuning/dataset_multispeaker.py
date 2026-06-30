# coding=utf-8
# Multi-speaker extension of Qwen3-TTS official finetuning/dataset.py.
# It keeps the official prefix layout and only adds speaker_names to the batch.

from typing import Any, List, Tuple, Union

import librosa
import numpy as np
import torch
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
from torch.utils.data import Dataset

AudioLike = Union[str, np.ndarray, Tuple[np.ndarray, int]]
MaybeList = Union[Any, List[Any]]


class TTSDataset(Dataset):
    def __init__(self, data_list, processor, config: Qwen3TTSConfig, lag_num=-1, speaker_key="speaker"):
        self.data_list = data_list
        self.processor = processor
        self.lag_num = lag_num
        self.config = config
        self.speaker_key = speaker_key

    def __len__(self):
        return len(self.data_list)

    def _load_audio_to_np(self, x: str) -> Tuple[np.ndarray, int]:
        audio, sr = librosa.load(x, sr=None, mono=True)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)
        return audio.astype(np.float32), int(sr)

    def _normalize_audio_inputs(self, audios: Union[AudioLike, List[AudioLike]]) -> List[Tuple[np.ndarray, int]]:
        if isinstance(audios, list):
            items = audios
        else:
            items = [audios]
        out: List[Tuple[np.ndarray, int]] = []
        for a in items:
            if isinstance(a, str):
                out.append(self._load_audio_to_np(a))
            elif isinstance(a, tuple) and len(a) == 2 and isinstance(a[0], np.ndarray):
                out.append((a[0].astype(np.float32), int(a[1])))
            elif isinstance(a, np.ndarray):
                raise ValueError("For numpy waveform input, pass a tuple (audio, sr).")
            else:
                raise TypeError(f"Unsupported audio input type: {type(a)}")
        return out

    def _build_assistant_text(self, text: str) -> str:
        return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"

    def _ensure_list(self, x: MaybeList) -> List[Any]:
        return x if isinstance(x, list) else [x]

    def _tokenize_texts(self, text) -> List[torch.Tensor]:
        inputs = self.processor(text=text, return_tensors="pt", padding=True)
        input_id = inputs["input_ids"]
        input_id = input_id.unsqueeze(0) if input_id.dim() == 1 else input_id
        return input_id

    @torch.inference_mode()
    def extract_mels(self, audio, sr):
        assert sr == 24000, "Only support 24kHz audio; resample wav/ref_audio to 24kHz."
        mels = mel_spectrogram(
            torch.from_numpy(audio).unsqueeze(0),
            n_fft=1024,
            num_mels=128,
            sampling_rate=24000,
            hop_size=256,
            win_size=1024,
            fmin=0,
            fmax=12000,
        ).transpose(1, 2)
        return mels

    def _get_speaker_name(self, item) -> str:
        for key in (self.speaker_key, "speaker", "channel"):
            if key in item and item[key]:
                return str(item[key]).strip()
        raise ValueError(f"Missing speaker field. Expected '{self.speaker_key}', 'speaker', or 'channel'. Item keys: {list(item.keys())}")

    def __getitem__(self, idx):
        item = self.data_list[idx]
        text = item["text"]
        audio_codes = item["audio_codes"]
        ref_audio_path = item["ref_audio"]
        speaker_name = self._get_speaker_name(item)

        text = self._build_assistant_text(text)
        text_ids = self._tokenize_texts(text)
        audio_codes = torch.tensor(audio_codes, dtype=torch.long)

        ref_audio_list = self._ensure_list(ref_audio_path)
        normalized = self._normalize_audio_inputs(ref_audio_list)
        wav, sr = normalized[0]
        ref_mel = self.extract_mels(audio=wav, sr=sr)

        return {
            "text_ids": text_ids[:, :-5],
            "audio_codes": audio_codes,
            "ref_mel": ref_mel,
            "speaker_name": speaker_name,
        }

    def collate_fn(self, batch):
        assert self.lag_num == -1
        item_length = [b["text_ids"].shape[1] + b["audio_codes"].shape[0] for b in batch]
        max_length = max(item_length) + 8
        bsz, t = len(batch), max_length

        input_ids = torch.zeros((bsz, t, 2), dtype=torch.long)
        codec_ids = torch.zeros((bsz, t, 16), dtype=torch.long)
        text_embedding_mask = torch.zeros((bsz, t), dtype=torch.bool)
        codec_embedding_mask = torch.zeros((bsz, t), dtype=torch.bool)
        codec_mask = torch.zeros((bsz, t), dtype=torch.bool)
        attention_mask = torch.zeros((bsz, t), dtype=torch.long)
        codec_0_labels = torch.full((bsz, t), -100, dtype=torch.long)

        for i, data in enumerate(batch):
            text_ids = data["text_ids"]
            audio_codec_0 = data["audio_codes"][:, 0]
            audio_codecs = data["audio_codes"]
            text_ids_len = text_ids.shape[1]
            codec_ids_len = audio_codec_0.shape[0]

            # text channel, same as official Qwen finetuning/dataset.py
            input_ids[i, :3, 0] = text_ids[0, :3]
            input_ids[i, 3:7, 0] = self.config.tts_pad_token_id
            input_ids[i, 7, 0] = self.config.tts_bos_token_id
            input_ids[i, 8:8 + text_ids_len - 3, 0] = text_ids[0, 3:]
            input_ids[i, 8 + text_ids_len - 3, 0] = self.config.tts_eos_token_id
            input_ids[i, 8 + text_ids_len - 2:8 + text_ids_len + codec_ids_len, 0] = self.config.tts_pad_token_id
            text_embedding_mask[i, :8 + text_ids_len + codec_ids_len] = True

            # codec channel, same as official Qwen finetuning/dataset.py
            input_ids[i, 3:8, 1] = torch.tensor(
                [
                    self.config.talker_config.codec_nothink_id,
                    self.config.talker_config.codec_think_bos_id,
                    self.config.talker_config.codec_think_eos_id,
                    0,  # speaker embedding slot at position 6
                    self.config.talker_config.codec_pad_id,
                ],
                dtype=torch.long,
            )
            input_ids[i, 8:8 + text_ids_len - 3, 1] = self.config.talker_config.codec_pad_id
            input_ids[i, 8 + text_ids_len - 3, 1] = self.config.talker_config.codec_pad_id
            input_ids[i, 8 + text_ids_len - 2, 1] = self.config.talker_config.codec_bos_id
            input_ids[i, 8 + text_ids_len - 1:8 + text_ids_len - 1 + codec_ids_len, 1] = audio_codec_0
            input_ids[i, 8 + text_ids_len - 1 + codec_ids_len, 1] = self.config.talker_config.codec_eos_token_id

            codec_0_labels[i, 8 + text_ids_len - 1:8 + text_ids_len - 1 + codec_ids_len] = audio_codec_0
            codec_0_labels[i, 8 + text_ids_len - 1 + codec_ids_len] = self.config.talker_config.codec_eos_token_id
            codec_ids[i, 8 + text_ids_len - 1:8 + text_ids_len - 1 + codec_ids_len, :] = audio_codecs

            codec_embedding_mask[i, 3:8 + text_ids_len + codec_ids_len] = True
            codec_embedding_mask[i, 6] = False  # speaker embedding slot
            codec_mask[i, 8 + text_ids_len - 1:8 + text_ids_len - 1 + codec_ids_len] = True
            attention_mask[i, :8 + text_ids_len + codec_ids_len] = True

        # Multi-speaker batches can contain different reference-audio lengths.
        # Each ref_mel is shaped [1, T_ref, 128]. torch.cat(..., dim=0) only
        # works when all T_ref are identical, which is usually false when each
        # speaker has its own fixed ref_audio. Pad along the time axis instead.
        ref_mel_list = [data["ref_mel"] for data in batch]
        max_ref_len = max(m.shape[1] if m.dim() == 3 else m.shape[0] for m in ref_mel_list)
        n_mels = ref_mel_list[0].shape[-1]
        ref_mels = ref_mel_list[0].new_zeros((len(ref_mel_list), max_ref_len, n_mels))

        for i, m in enumerate(ref_mel_list):
            if m.dim() == 2:
                # [T_ref, 128] -> [1, T_ref, 128]
                m = m.unsqueeze(0)
            if m.shape[0] != 1:
                raise ValueError(f"Expected ref_mel shape [1, T, 128], got {tuple(m.shape)}")
            ref_len = m.shape[1]
            ref_mels[i, :ref_len, :] = m[0]

        return {
            "input_ids": input_ids,
            "ref_mels": ref_mels,
            "attention_mask": attention_mask,
            "text_embedding_mask": text_embedding_mask.unsqueeze(-1),
            "codec_embedding_mask": codec_embedding_mask.unsqueeze(-1),
            "codec_0_labels": codec_0_labels,
            "codec_ids": codec_ids,
            "codec_mask": codec_mask,
            "speaker_names": [data["speaker_name"] for data in batch],
        }
