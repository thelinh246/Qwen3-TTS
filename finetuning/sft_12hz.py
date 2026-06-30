# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
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
import argparse
import json
import math
import os
import re
import shutil
from pathlib import Path

import torch
from accelerate import Accelerator
from dataset import TTSDataset
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig, get_scheduler

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

target_speaker_embedding = None
def train():
    global target_speaker_embedding

    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--output_model_path", type=str, default="output")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--speaker_name", type=str, default="speaker_test")
    parser.add_argument("--save_interval", type=int, default=1, help="Save a checkpoint every N epochs.")
    parser.add_argument("--save_last/--no-save_last", dest="save_last", default=True, action=argparse.BooleanOptionalAction, help="Always save the final epoch checkpoint.")
    parser.add_argument("--log_interval", type=int, default=10, help="Log loss every N steps.")
    parser.add_argument("--logging_dir", type=str, default=None, help="Accelerate logging directory (e.g., runs/run1/logs).")
    parser.add_argument(
        "--resume/--no-resume",
        dest="resume",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Resume from the latest checkpoint in output_model_path if available.",
    )
    parser.add_argument(
        "--accelerate-trackers/--no-accelerate-trackers",
        dest="accelerate_trackers",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Enable Accelerate tracker integrations (default: disabled).",
    )
    parser.add_argument(
        "--flash-attn/--no-flash-attn",
        dest="flash_attn",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Enable FlashAttention-2 (default: enabled).",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        choices=["cosine", "linear", "constant", "constant_with_warmup"],
        help="Learning rate scheduler type.",
    )
    parser.add_argument("--warmup_steps", type=int, default=0, help="Number of warmup steps.")
    parser.add_argument("--warmup_ratio", type=float, default=0.0, help="Warmup ratio of total steps (overrides warmup_steps if > 0).")
    args = parser.parse_args()

    accel_logging_dir = args.logging_dir or str(Path(args.output_model_path) / "logs")
    grad_accum_steps = 4
    if args.accelerate_trackers:
        try:
            from accelerate import ProjectConfiguration  # type: ignore
            project_config = ProjectConfiguration(project_dir=str(args.output_model_path), logging_dir=accel_logging_dir)
            accelerator = Accelerator(
                gradient_accumulation_steps=grad_accum_steps,
                mixed_precision="bf16",
                log_with="tensorboard",
                project_config=project_config,
            )
        except Exception:
            # Older accelerate: fall back without tracker integration.
            accelerator = Accelerator(
                gradient_accumulation_steps=grad_accum_steps,
                mixed_precision="bf16",
            )
    else:
        accelerator = Accelerator(
            gradient_accumulation_steps=grad_accum_steps,
            mixed_precision="bf16",
        )
    tb_writer = None
    if SummaryWriter is not None and accelerator.is_main_process:
        tb_log_dir = Path(args.output_model_path) / "tb"
        tb_writer = SummaryWriter(log_dir=str(tb_log_dir))

    def _find_latest_checkpoint(output_dir: str):
        root = Path(output_dir)
        if not root.exists():
            return None, None
        candidates = []
        for item in root.iterdir():
            if item.is_dir():
                m = re.match(r"checkpoint-epoch-(\d+)$", item.name)
                if m:
                    candidates.append((int(m.group(1)), item))
        if not candidates:
            return None, None
        candidates.sort(key=lambda x: x[0])
        return candidates[-1][1], candidates[-1][0]

    start_epoch = 0
    global_step = 0
    resume_checkpoint = None
    MODEL_PATH = args.init_model_path
    if args.resume:
        latest_ckpt, latest_epoch = _find_latest_checkpoint(args.output_model_path)
        if latest_ckpt is not None:
            MODEL_PATH = str(latest_ckpt)
            start_epoch = latest_epoch + 1
            resume_checkpoint = str(latest_ckpt)

    attn_impl = "flash_attention_2" if args.flash_attn else "eager"
    AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
    config_override = None
    config_path = Path(MODEL_PATH) / "config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if cfg.get("tts_model_type") != "base":
            config_override = AutoConfig.from_pretrained(MODEL_PATH)
            config_override.tts_model_type = "base"
    qwen3tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        config=config_override,
    )
    config = config_override or AutoConfig.from_pretrained(MODEL_PATH)

    train_jsonl_path = Path(args.train_jsonl)
    train_base_dir = train_jsonl_path.parent

    train_data = open(args.train_jsonl).readlines()
    train_data = [json.loads(line) for line in train_data]
    for item in train_data:
        for key in ("audio", "ref_audio"):
            if key in item and item[key]:
                path = Path(item[key])
                if not path.is_absolute():
                    candidate = train_base_dir / path
                    if candidate.exists():
                        path = candidate
                item[key] = str(path)

        ref_audio = item.get("ref_audio")
        audio = item.get("audio")
        if ref_audio and audio:
            ref_audio_path = Path(ref_audio)
            if not ref_audio_path.exists():
                audio_parent = Path(audio).parent
                fallback = audio_parent / ref_audio_path
                if fallback.exists():
                    item["ref_audio"] = str(fallback)
    dataset = TTSDataset(train_data, qwen3tts.processor, config)
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn)

    optimizer = AdamW(qwen3tts.model.parameters(), lr=args.lr, weight_decay=0.01)
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / grad_accum_steps)
    max_train_steps = args.num_epochs * num_update_steps_per_epoch
    warmup_steps = args.warmup_steps
    if args.warmup_ratio and args.warmup_ratio > 0:
        warmup_steps = int(max_train_steps * args.warmup_ratio)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_train_steps,
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        qwen3tts.model, optimizer, train_dataloader, lr_scheduler
    )

    if args.resume and resume_checkpoint is not None:
        trainer_state_path = os.path.join(resume_checkpoint, "trainer_state.json")
        optimizer_state_path = os.path.join(resume_checkpoint, "optimizer.pt")
        scheduler_state_path = os.path.join(resume_checkpoint, "scheduler.pt")
        if os.path.exists(trainer_state_path):
            with open(trainer_state_path, "r", encoding="utf-8") as f:
                trainer_state = json.load(f)
            start_epoch = max(start_epoch, int(trainer_state.get("next_epoch", start_epoch)))
            global_step = int(trainer_state.get("global_step", 0))
        if os.path.exists(optimizer_state_path):
            optimizer_state = torch.load(optimizer_state_path, map_location="cpu")
            try:
                optimizer.load_state_dict(optimizer_state)
            except Exception:
                accelerator.print("Warning: failed to load optimizer state; continuing with fresh optimizer.")
        if os.path.exists(scheduler_state_path):
            scheduler_state = torch.load(scheduler_state_path, map_location="cpu")
            try:
                lr_scheduler.load_state_dict(scheduler_state)
            except Exception:
                accelerator.print("Warning: failed to load scheduler state; continuing with fresh scheduler.")

    num_epochs = args.num_epochs
    model.train()

    def _save_checkpoint(epoch_idx: int):
        output_dir = os.path.join(args.output_model_path, f"checkpoint-epoch-{epoch_idx}")
        shutil.copytree(MODEL_PATH, output_dir, dirs_exist_ok=True)

        input_config_file = os.path.join(MODEL_PATH, "config.json")
        output_config_file = os.path.join(output_dir, "config.json")
        with open(input_config_file, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        config_dict["tts_model_type"] = "custom_voice"
        talker_config = config_dict.get("talker_config", {})
        talker_config["spk_id"] = {
            args.speaker_name: 3000
        }
        talker_config["spk_is_dialect"] = {
            args.speaker_name: False
        }
        config_dict["talker_config"] = talker_config

        with open(output_config_file, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

        unwrapped_model = accelerator.unwrap_model(model)
        state_dict = {k: v.detach().to("cpu") for k, v in unwrapped_model.state_dict().items()}

        weight = state_dict['talker.model.codec_embedding.weight']
        state_dict['talker.model.codec_embedding.weight'][3000] = target_speaker_embedding[0].detach().to(weight.device).to(weight.dtype)
        save_path = os.path.join(output_dir, "model.safetensors")
        save_file(state_dict, save_path)

        optimizer_state_path = os.path.join(output_dir, "optimizer.pt")
        torch.save(optimizer.state_dict(), optimizer_state_path)
        scheduler_state_path = os.path.join(output_dir, "scheduler.pt")
        torch.save(lr_scheduler.state_dict(), scheduler_state_path)
        trainer_state_path = os.path.join(output_dir, "trainer_state.json")
        with open(trainer_state_path, "w", encoding="utf-8") as f:
            json.dump({"next_epoch": epoch_idx + 1, "global_step": global_step}, f, indent=2)


    for epoch in range(start_epoch, num_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):

                input_ids = batch['input_ids']
                codec_ids = batch['codec_ids']
                ref_mels = batch['ref_mels']
                text_embedding_mask = batch['text_embedding_mask']
                codec_embedding_mask = batch['codec_embedding_mask']
                attention_mask = batch['attention_mask']
                codec_0_labels = batch['codec_0_labels']
                codec_mask = batch['codec_mask']

                speaker_embedding = model.speaker_encoder(ref_mels.to(model.device).to(model.dtype)).detach()
                if target_speaker_embedding is None:
                    target_speaker_embedding = speaker_embedding

                input_text_ids = input_ids[:, :, 0]
                input_codec_ids = input_ids[:, :, 1]

                input_text_embedding = model.talker.text_projection(
                    model.talker.model.text_embedding(input_text_ids)
                ) * text_embedding_mask
                input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
                input_codec_embedding[:, 6, :] = speaker_embedding

                input_embeddings = input_text_embedding + input_codec_embedding

                for i in range(1, 16):
                    codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
                    codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
                    input_embeddings = input_embeddings + codec_i_embedding

                outputs = model.talker(
                    inputs_embeds=input_embeddings[:, :-1, :],
                    attention_mask=attention_mask[:, :-1],
                    labels=codec_0_labels[:, 1:],
                    output_hidden_states=True
                )

                hidden_states = outputs.hidden_states[0][-1]
                talker_hidden_states = hidden_states[codec_mask[:, 1:]]
                talker_codec_ids = codec_ids[codec_mask]

                sub_talker_logits, sub_talker_loss = model.talker.forward_sub_talker_finetune(talker_codec_ids, talker_hidden_states)

                loss = outputs.loss + sub_talker_loss

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                if accelerator.sync_gradients:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

            if step % args.log_interval == 0:
                accelerator.print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")
                if tb_writer is not None:
                    tb_writer.add_scalar("train/loss", loss.item(), global_step)
                    tb_writer.add_scalar("train/lr", lr_scheduler.get_last_lr()[0], global_step)

        if accelerator.is_main_process and (epoch % args.save_interval == 0):
            _save_checkpoint(epoch)

    if accelerator.is_main_process and args.save_last:
        final_epoch = num_epochs - 1
        if final_epoch >= start_epoch and (final_epoch % args.save_interval != 0):
            _save_checkpoint(final_epoch)

    if tb_writer is not None:
        tb_writer.flush()
        tb_writer.close()

if __name__ == "__main__":
    train()
