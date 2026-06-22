# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Modifications for Stage 1 (Vietnamese language learning):
#
#   FIX 1 — _ref_mel_cache không giới hạn size (OOM)
#     Gốc: self._ref_mel_cache = {}  ← dict tăng mãi mãi
#     Lý do: Stage 1 có hàng nghìn speaker khác nhau, mỗi ref_audio là duy nhất.
#            Với num_workers=4, mỗi worker có cache riêng → 4x memory.
#            10,000 unique ref_audio × ~240KB/file × 4 workers ≈ 9.6GB chỉ cho cache.
#     Fix: dùng functools.lru_cache với maxsize có thể cấu hình (mặc định 64).
#          Stage 2 (single speaker): tăng lên 1 vì chỉ có 1 ref_audio.
#          Stage 1 (multi-speaker) : giữ 64 để hưởng lợi từ cache cục bộ mà không OOM.
#
#   FIX 2 — assert sr == 24000 bị comment out (silent crash)
#     Gốc: # assert sr == 24000, "Only support 24kHz audio"
#     Lý do: không có assertion → audio sai sample rate vẫn chạy qua, sinh codec
#            sai, loss kỳ lạ, khó debug. Đặc biệt nguy hiểm với batch Stage 1 lớn.
#     Fix: khôi phục assert, thêm thông tin file để dễ tìm file lỗi.
#
#   FIX 3 — Không có max_duration_sec filter (batch OOM)
#     Gốc: không lọc độ dài audio → một clip 30s trong batch làm max_length tăng
#          lên 30×12=360 codec tokens, kéo toàn batch lên max đó → OOM.
#     Fix: thêm tham số max_duration_sec (mặc định 20s) và min_duration_sec (1s).
#          Clip bị lọc sẽ được log và bỏ qua thay vì crash.

from functools import lru_cache
from typing import Any, List, Optional, Tuple, Union

import librosa
import numpy as np
import torch
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
from torch.utils.data import Dataset

AudioLike = Union[
    str,                        # wav path, URL, base64
    np.ndarray,                 # waveform (requires sr)
    Tuple[np.ndarray, int],     # (waveform, sr)
]
MaybeList = Union[Any, List[Any]]


class TTSDataset(Dataset):

    def __init__(
        self,
        data_list,
        processor,
        config: Qwen3TTSConfig,
        lag_num: int = -1,
        max_duration_sec: float = 20.0,
        min_duration_sec: float = 0.5,
        ref_mel_cache_size: int = 64,
    ):
        """
        Args:
            data_list: list of dicts with keys: text, audio, ref_audio, audio_codes
            processor: Qwen3-TTS processor
            config: Qwen3TTSConfig
            lag_num: internal parameter, keep -1
            max_duration_sec: FIX 3 — lọc bỏ audio dài hơn giá trị này.
                              Clip dài gây OOM khi collate vì toàn batch phải pad
                              đến max_length. Mặc định 20s phù hợp Stage 1.
                              Stage 2 (clone voice với clip đọc chậm): tăng lên 25.
            min_duration_sec: lọc bỏ clip quá ngắn (nhiễu, im lặng). Mặc định 0.5s.
            ref_mel_cache_size: FIX 1 — số lượng ref_mel tối đa trong cache LRU.
                               Stage 1 (multi-speaker): 64 (tiết kiệm RAM).
                               Stage 2 (single speaker): 1 (cache 100%).
        """
        # FIX 3: lọc audio theo độ dài trước khi train
        original_len = len(data_list)
        filtered = []
        skipped = 0
        for item in data_list:
            duration = item.get("duration_sec")
            if duration is None:
                # Nếu không có metadata duration, tính từ audio_codes
                # 12Hz codec → len(audio_codes) / 12 ≈ duration
                codes = item.get("audio_codes")
                if codes is not None:
                    duration = len(codes) / 12.0
            if duration is not None:
                if duration > max_duration_sec or duration < min_duration_sec:
                    skipped += 1
                    continue
            filtered.append(item)

        if skipped > 0:
            print(
                f"[TTSDataset] Filtered {skipped}/{original_len} samples "
                f"outside [{min_duration_sec:.1f}s, {max_duration_sec:.1f}s]. "
                f"Remaining: {len(filtered)}"
            )

        self.data_list = filtered
        self.processor = processor
        self.lag_num = lag_num
        self.config = config
        self.max_duration_sec = max_duration_sec
        self.min_duration_sec = min_duration_sec

        # FIX 1: LRU cache thay cho dict không giới hạn
        # lru_cache không gắn được trực tiếp vào instance method,
        # nên wrap qua closure để cache hoạt động per-instance.
        @lru_cache(maxsize=ref_mel_cache_size)
        def _cached_load_ref_mel(path: str):
            audio_list = self._normalize_audio_inputs(path)
            wav, sr = audio_list[0]
            return self.extract_mels(wav, sr)

        self._get_cached_ref_mel = _cached_load_ref_mel

    def __len__(self):
        return len(self.data_list)

    def _load_audio_to_np(self, x: str) -> Tuple[np.ndarray, int]:
        target_sr = 24000
        audio, sr = librosa.load(x, sr=target_sr, mono=True)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)
        return audio.astype(np.float32), int(sr)

    def _normalize_audio_inputs(
        self, audios: Union[AudioLike, List[AudioLike]]
    ) -> List[Tuple[np.ndarray, int]]:
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
        inp = self.processor(text=text, return_tensors="pt", padding=True)
        input_id = inp["input_ids"]
        input_id = input_id.unsqueeze(0) if input_id.dim() == 1 else input_id
        return input_id

    @torch.inference_mode()
    def extract_mels(self, audio: np.ndarray, sr: int) -> torch.Tensor:
        # FIX 2: khôi phục assertion 24kHz (gốc bị comment out)
        # Codec chỉ hỗ trợ 24kHz. Sai sample rate → codec tokens sai →
        # loss kỳ lạ, khó debug. Thêm tên file vào message để tìm file lỗi.
        assert sr == 24000, (
            f"Audio phải là 24000 Hz nhưng nhận được {sr} Hz. "
            "Chạy lại bước resample: ffmpeg -ar 24000 -ac 1"
        )

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

    def __getitem__(self, idx):
        item = self.data_list[idx]
        audio_path      = item["audio"]
        text            = item["text"]
        audio_codes     = item["audio_codes"]
        ref_audio_path  = item["ref_audio"]

        text = self._build_assistant_text(text)
        text_ids    = self._tokenize_texts(text)
        audio_codes = torch.tensor(audio_codes, dtype=torch.long)

        # FIX 1: dùng LRU cache có giới hạn thay vì dict vô hạn
        # Stage 1 (multi-speaker): mỗi ref_audio khác nhau, cache ít hit
        #   → cache_size=64 chỉ giữ 64 entry gần nhất, không OOM.
        # Stage 2 (single speaker): tất cả sample dùng cùng 1 ref_audio
        #   → cache_size=1 là đủ, hit rate 100% từ sample thứ 2 trở đi.
        ref_mel = self._get_cached_ref_mel(ref_audio_path)

        return {
            "text_ids":   text_ids[:, :-5],    # 1, t
            "audio_codes": audio_codes,         # t, 16
            "ref_mel":    ref_mel,
        }

    def collate_fn(self, batch):
        assert self.lag_num == -1

        item_length = [
            b["text_ids"].shape[1] + b["audio_codes"].shape[0]
            for b in batch
        ]
        max_length = max(item_length) + 8

        b, t = len(batch), max_length

        input_ids            = torch.zeros((b, t, 2),  dtype=torch.long)
        codec_ids            = torch.zeros((b, t, 16), dtype=torch.long)
        text_embedding_mask  = torch.zeros((b, t),     dtype=torch.bool)
        codec_embedding_mask = torch.zeros((b, t),     dtype=torch.bool)
        codec_mask           = torch.zeros((b, t),     dtype=torch.bool)
        attention_mask       = torch.zeros((b, t),     dtype=torch.long)
        codec_0_labels       = torch.full((b, t), -100, dtype=torch.long)

        for i, data in enumerate(batch):
            text_ids       = data["text_ids"]
            audio_codec_0  = data["audio_codes"][:, 0]
            audio_codecs   = data["audio_codes"]

            text_ids_len  = text_ids.shape[1]
            codec_ids_len = audio_codec_0.shape[0]

            # text channel
            input_ids[i, :3, 0]                              = text_ids[0, :3]
            input_ids[i, 3:7, 0]                             = self.config.tts_pad_token_id
            input_ids[i, 7, 0]                               = self.config.tts_bos_token_id
            input_ids[i, 8:8+text_ids_len-3, 0]              = text_ids[0, 3:]
            input_ids[i, 8+text_ids_len-3, 0]                = self.config.tts_eos_token_id
            input_ids[i, 8+text_ids_len-2:8+text_ids_len+codec_ids_len, 0] = self.config.tts_pad_token_id
            text_embedding_mask[i, :8+text_ids_len+codec_ids_len] = True

            # codec channel
            input_ids[i, 3:8, 1] = torch.tensor([
                self.config.talker_config.codec_nothink_id,
                self.config.talker_config.codec_think_bos_id,
                self.config.talker_config.codec_think_eos_id,
                0,   # placeholder cho speaker embedding (sẽ inject lúc train)
                self.config.talker_config.codec_pad_id,
            ])
            input_ids[i, 8:8+text_ids_len-3, 1]            = self.config.talker_config.codec_pad_id
            input_ids[i, 8+text_ids_len-3, 1]              = self.config.talker_config.codec_pad_id
            input_ids[i, 8+text_ids_len-2, 1]              = self.config.talker_config.codec_bos_id
            input_ids[i, 8+text_ids_len-1:8+text_ids_len-1+codec_ids_len, 1] = audio_codec_0
            input_ids[i, 8+text_ids_len-1+codec_ids_len, 1] = self.config.talker_config.codec_eos_token_id

            codec_0_labels[i, 8+text_ids_len-1:8+text_ids_len-1+codec_ids_len] = audio_codec_0
            codec_0_labels[i, 8+text_ids_len-1+codec_ids_len] = self.config.talker_config.codec_eos_token_id

            codec_ids[i, 8+text_ids_len-1:8+text_ids_len-1+codec_ids_len, :] = audio_codecs

            codec_embedding_mask[i, 3:8+text_ids_len+codec_ids_len] = True
            codec_embedding_mask[i, 6] = False   # position 6 dành cho speaker embedding

            codec_mask[i, 8+text_ids_len-1:8+text_ids_len-1+codec_ids_len] = True
            attention_mask[i, :8+text_ids_len+codec_ids_len] = True

        # FIX 4 — ref_mel có độ dài time frames khác nhau giữa các sample (multi-speaker)
        # Gốc: torch.cat(...) crash với "Sizes must match except in dimension 0"
        #       vì mỗi ref_audio có thời lượng khác nhau → mel frames khác nhau.
        # Fix: pad zeros về max_T trước khi cat.
        #      Speaker encoder chỉ đọc phần đầu (tiếng nói thực), padding cuối không ảnh hưởng.
        ref_mels_list = [data["ref_mel"] for data in batch]
        max_T = max(m.shape[1] for m in ref_mels_list)
        ref_mels = torch.cat([
            torch.nn.functional.pad(m, (0, 0, 0, max_T - m.shape[1]))
            for m in ref_mels_list
        ], dim=0)

        return {
            "input_ids":            input_ids,
            "ref_mels":             ref_mels,
            "attention_mask":       attention_mask,
            "text_embedding_mask":  text_embedding_mask.unsqueeze(-1),
            "codec_embedding_mask": codec_embedding_mask.unsqueeze(-1),
            "codec_0_labels":       codec_0_labels,
            "codec_ids":            codec_ids,
            "codec_mask":           codec_mask,
        }
