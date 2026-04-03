# coding=utf-8
# Copyright 2026 The Alibaba Qwen team (modified with LoRA support for Vietnamese).
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import shutil

import torch
from accelerate import Accelerator
from dataset import TTSDataset
from peft import LoraConfig, get_peft_model
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig

target_speaker_embedding = None


def train():
    global target_speaker_embedding

    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    parser.add_argument("--output_model_path", type=str, default="output")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--speaker_name", type=str, default="speaker_test")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--no_lora", dest="use_lora", action="store_false")
    args = parser.parse_args()

    accelerator = Accelerator(
        gradient_accumulation_steps=4,
        mixed_precision="bf16",
        log_with="tensorboard",
        project_dir="./logs"
    )

    MODEL_PATH = args.init_model_path

    qwen3tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    config = AutoConfig.from_pretrained(MODEL_PATH)

    # Freeze toàn bộ model
    for param in qwen3tts.model.parameters():
        param.requires_grad = False

    if args.use_lora:
        accelerator.print("Using LoRA finetuning...")

        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=args.lora_dropout,
            bias="none",
            # KHÔNG dùng modules_to_save vì codec/text embedding là ModuleList
        )

        qwen3tts.model.talker = get_peft_model(qwen3tts.model.talker, lora_config)
        qwen3tts.model.talker.print_trainable_parameters()

        # Unfreeze embeddings bằng cách tìm theo tên - không phụ thuộc vào path
        for name, param in qwen3tts.model.talker.named_parameters():
            if 'codec_embedding' in name or 'text_embedding' in name:
                param.requires_grad = True
                accelerator.print(f"Unfrozen: {name}")

    else:
        accelerator.print("Using full finetuning on talker...")
        for param in qwen3tts.model.talker.parameters():
            param.requires_grad = True

    total_params = sum(p.numel() for p in qwen3tts.model.parameters())
    trainable_params = sum(p.numel() for p in qwen3tts.model.parameters() if p.requires_grad)
    accelerator.print(f"Total: {total_params/1e6:.1f}M | Trainable: {trainable_params/1e6:.1f}M ({100*trainable_params/total_params:.2f}%)")

    train_data = open(args.train_jsonl).readlines()
    train_data = [json.loads(line) for line in train_data]
    dataset = TTSDataset(train_data, qwen3tts.processor, config)
    train_dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, qwen3tts.model.parameters()),
        lr=args.lr,
        weight_decay=0.01
    )

    # Debug: tìm speaker_encoder
    print('=== qwen3tts attributes ===')
    for attr in dir(qwen3tts):
        if not attr.startswith('_'):
            val = getattr(qwen3tts, attr, None)
            if val is not None and hasattr(val, 'parameters'):
                print(f'  qwen3tts.{attr}: {type(val).__name__}')
    print('=== qwen3tts.model attributes ===')
    for name, module in qwen3tts.model.named_children():
        print(f'  {name}: {type(module).__name__}')
    import sys; sys.exit(0)
    # Lưu reference speaker_encoder trước khi accelerator wrap
    speaker_encoder = qwen3tts.model.speaker_encoder

    model, optimizer, train_dataloader = accelerator.prepare(
        qwen3tts.model, optimizer, train_dataloader
    )
    speaker_encoder = speaker_encoder.to(accelerator.device)

    model.train()

    for epoch in range(args.num_epochs):
        total_loss = 0.0
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

                # Unwrap một lần, dùng cho tất cả
                _model = accelerator.unwrap_model(model)
                _talker_for_cond = _model.talker.base_model.model
                talker_inner = _talker_for_cond.model          # Qwen3TTSTalkerModel: text_embedding, codec_embedding
                talker_cond  = _talker_for_cond.code_predictor # CodePredictor: get_input_embeddings, config

                speaker_embedding = speaker_encoder(
                    ref_mels.to(accelerator.device).to(torch.bfloat16)
                ).detach()

                if target_speaker_embedding is None:
                    target_speaker_embedding = speaker_embedding

                input_text_ids = input_ids[:, :, 0]
                input_codec_ids = input_ids[:, :, 1]

                input_text_embedding = talker_inner.text_embedding(input_text_ids) * text_embedding_mask
                input_codec_embedding = talker_inner.codec_embedding(input_codec_ids) * codec_embedding_mask

                talker_hidden_size = input_codec_embedding.shape[-1]
                if input_text_embedding.shape[-1] != talker_hidden_size:
                    input_text_embedding = input_text_embedding[..., :talker_hidden_size]

                input_codec_embedding[:, 6, :] = speaker_embedding
                input_embeddings = input_text_embedding + input_codec_embedding

                for i in range(1, 16):
                    codec_i_embedding = talker_cond.get_input_embeddings()[i - 1](codec_ids[:, :, i])
                    codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
                    if input_embeddings.shape[-1] != codec_i_embedding.shape[-1]:
                        pad_size = input_embeddings.shape[-1] - codec_i_embedding.shape[-1]
                        codec_i_embedding = torch.nn.functional.pad(codec_i_embedding, (0, pad_size))
                    input_embeddings = input_embeddings + codec_i_embedding

                outputs = model.talker(
                    inputs_embeds=input_embeddings[:, :-1, :],
                    attention_mask=attention_mask[:, :-1],
                    labels=codec_0_labels[:, 1:],
                    output_hidden_states=True
                )

                hidden_states = outputs.hidden_states[0][-1]
                talker_hidden_states = hidden_states[codec_mask[:, :-1]]
                talker_codec_ids = codec_ids[codec_mask]

                sub_talker_hidden_size = talker_cond.config.hidden_size
                if talker_hidden_states.shape[-1] != sub_talker_hidden_size:
                    talker_hidden_states = talker_hidden_states[..., :sub_talker_hidden_size]

                sub_talker_logits, sub_talker_loss = _talker_for_cond.forward_sub_talker_finetune(
                    talker_codec_ids, talker_hidden_states
                )

                loss = outputs.loss + 0.3 * sub_talker_loss
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, model.parameters()), 1.0
                    )

                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()
            if step % 10 == 0:
                accelerator.print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_dataloader)
        accelerator.print(f"Epoch {epoch} done | Avg Loss: {avg_loss:.4f}")

        if accelerator.is_main_process:
            output_dir = os.path.join(args.output_model_path, f"checkpoint-epoch-{epoch}")
            shutil.copytree(MODEL_PATH, output_dir, dirs_exist_ok=True)

            with open(os.path.join(MODEL_PATH, "config.json"), 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            config_dict["tts_model_type"] = "custom_voice"
            talker_config = config_dict.get("talker_config", {})
            talker_config["spk_id"] = {args.speaker_name: 3000}
            talker_config["spk_is_dialect"] = {args.speaker_name: False}
            config_dict["talker_config"] = talker_config
            with open(os.path.join(output_dir, "config.json"), 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)

            unwrapped_model = accelerator.unwrap_model(model)

            if args.use_lora:
                import copy
                # Deep copy để không làm hỏng model đang train
                talker_copy = copy.deepcopy(unwrapped_model.talker)
                talker_merged = talker_copy.merge_and_unload()
                state_dict = {k: v.detach().to("cpu") for k, v in talker_merged.state_dict().items()}
                # Thêm prefix talker. và các key khác (speaker_encoder không cần save)
                state_dict = {"talker." + k: v for k, v in state_dict.items()}
            else:
                state_dict = {k: v.detach().to("cpu") for k, v in unwrapped_model.state_dict().items()}
                for k in [k for k in state_dict if k.startswith("speaker_encoder")]:
                    del state_dict[k]

            weight = state_dict['talker.model.codec_embedding.weight']
            state_dict['talker.model.codec_embedding.weight'][3000] = (
                target_speaker_embedding[0].detach().to(weight.device).to(weight.dtype)
            )

            save_file(state_dict, os.path.join(output_dir, "model.safetensors"))
            accelerator.print(f"Saved to {output_dir}")


if __name__ == "__main__":
    train()
