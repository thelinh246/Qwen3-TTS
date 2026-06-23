# coding=utf-8
# Stage 1: Dạy Qwen3-TTS học tiếng Việt (ngôn ngữ + phát âm)
# Mục tiêu: học text->codec mapping cho tiếng Việt, KHÔNG phải clone giọng đọc.
#
# ═══════════════════════════════════════════════════════════════
#  BUG FIXES áp dụng từ cộng đồng (so với official sft_12hz.py)
# ═══════════════════════════════════════════════════════════════
#  #1  Double label-shift (PR #178, CHƯA merge upstream)
#      Nguyên nhân: code gốc shift labels thủ công (labels[:, 1:]) rồi
#      truyền vào HF ForCausalLM vốn shift thêm một lần nữa → double shift.
#      Triệu chứng: giọng đọc nhanh dần sau mỗi epoch đến mức không nghe được.
#      Fix: bỏ labels= khỏi model.talker(), tính loss bằng F.cross_entropy.
#
#  #2  Thiếu text_projection (PR #188, ĐÃ merge tại commit 680d4e9)
#      Nguyên nhân: inference dùng text_projection sau text_embedding,
#      nhưng training không có bước này → train/infer mismatch.
#      Triệu chứng: crash hoặc embedding sai silently trên 1.7B.
#      Fix: gọi _talker_for_cond.text_projection() nếu tồn tại.
#
#  #3  LR mặc định quá cao (2e-5 → dùng 2e-6)
#      Nguyên nhân: LR cao → không hội tụ, sinh pure noise, không có EOS.
#
# ═══════════════════════════════════════════════════════════════
#  SO SÁNH Stage 1 (học ngôn ngữ) vs Stage 2 (clone voice)
# ═══════════════════════════════════════════════════════════════
#  Stage 1 (file này):
#   - Dataset: nhiều speaker, nhiều câu tiếng Việt đa dạng
#   - ref_audio: lấy từ chính mẫu đó (mỗi sample một ref khác nhau)
#   - KHÔNG lưu speaker embedding vào codec_embedding.weight[3000]
#   - KHÔNG ghi spk_id vào config
#   - Huấn luyện: toàn bộ talker (text_embedding + transformer)
#
#  Stage 2 (train_stage2_voice.py):
#   - Dataset: 10-30 phút audio một người nói duy nhất
#   - ref_audio: CỐ ĐỊNH một file ref của người đó
#   - LƯU speaker embedding vào codec_embedding.weight[spk_slot]
#   - Ghi spk_id vào config để inference dùng generate_custom_voice()
#   - Bắt đầu từ checkpoint Stage 1
# ═══════════════════════════════════════════════════════════════

import argparse
import copy
import json
import os
import shutil

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from dataset import TTSDataset
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from transformers import AutoConfig


# ──────────────────────────────────────────────────────────────
# Args
# ──────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Stage 1: Qwen3-TTS Vietnamese language learning")

    # Paths
    p.add_argument("--init_model_path", type=str,
                   default="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                   help="Model gốc để finetune. Dùng 1.7B cho kết quả tốt hơn.")
    p.add_argument("--output_model_path", type=str, default="output_stage1_vi")
    p.add_argument("--train_jsonl", type=str, required=True,
                   help="JSONL với các trường: text (tiếng Việt), audio, ref_audio. "
                        "Mỗi dòng nên là một speaker khác nhau để học ngôn ngữ tốt hơn.")
    p.add_argument("--val_jsonl", type=str, default=None,
                   help="JSONL validation (khuyến khích dùng để theo dõi overfitting).")
    p.add_argument("--resume_from_checkpoint", type=str, default=None,
                   help="Thư mục checkpoint để resume training (vd: output_stage1_vi/checkpoint-epoch-10). "
                        "Tự động load model weights + optimizer + scheduler + global_step. "
                        "Nếu chỉ muốn warm-start (không tiếp tục optimizer), dùng --init_model_path thay vì flag này.")

    # Training hyperparams
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-6,
                   help="LR được validate bởi cộng đồng. KHÔNG dùng 2e-5 (sinh noise).")
    p.add_argument("--num_epochs", type=int, default=15,
                   help="Stage 1 cần nhiều epoch hơn Stage 2. "
                        "Val loss tăng = overfitting, dừng lại.")
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--save_every_n_epochs", type=int, default=2)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--attn_impl", type=str, default="sdpa",
                   choices=["sdpa", "flash_attention_2", "eager"],
                   help="flash_attention_2 nhanh hơn ~40%% trên A100/H100. "
                        "Cần: pip install flash-attn --no-build-isolation")

    # Loss weight
    p.add_argument("--sub_talker_loss_weight", type=float, default=0.0,
                   help="Trọng số loss của codec residuals (sub-talker).")

    return p.parse_args()


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def patch_and_load_config(model_path, accelerator):
    """Đảm bảo tts_model_type=base để khởi tạo speaker_encoder.
    Chỉ thực hiện ở main process để tránh race condition khi multi-GPU."""
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if cfg.get("tts_model_type") != "base":
        if accelerator.is_main_process:
            accelerator.print(
                f"[Config] Patching tts_model_type: {cfg.get('tts_model_type')} -> base"
            )
            cfg["tts_model_type"] = "base"
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2, ensure_ascii=False)
        # Các process khác chờ main process ghi xong
        accelerator.wait_for_everyone()
        # Reload sau khi patch
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

    return cfg


def get_talker_inner(model, use_lora: bool):
    """Trả về (talker_for_cond, talker_inner, talker_cond) theo đúng path
    cho cả LoRA và full finetune.

    BUG FIX: code gốc luôn dùng .base_model.model ngay cả khi không có LoRA
    → AttributeError khi use_lora=False.
    """
    if use_lora:
        # PeftModel wraps: model.talker.base_model.model = Qwen3TTSTalker
        talker_for_cond = model.talker.base_model.model
    else:
        # Raw talker = Qwen3TTSTalker trực tiếp
        talker_for_cond = model.talker

    talker_inner = talker_for_cond.model           # Qwen3TTSTalkerModel (text_embedding, codec_embedding)
    talker_cond  = talker_for_cond.code_predictor  # CodePredictor (sub-talker)
    return talker_for_cond, talker_inner, talker_cond


def build_input_embeddings(
    input_ids, codec_ids, ref_mels, codec_mask,
    text_embedding_mask, codec_embedding_mask,
    talker_for_cond, talker_inner, talker_cond,
    speaker_encoder, device,
):
    """Xây dựng input_embeddings và lấy speaker_embedding.

    BUG FIX #2 (text_projection, PR #188):
    Official code bỏ qua bước text_projection ở training, trong khi inference
    luôn dùng nó → train/inference mismatch → embedding sai.
    Fix: gọi text_projection nếu layer đó tồn tại.

    Stage 1 vs Stage 2:
    Stage 1: mỗi sample có ref_audio riêng (multi-speaker)
             → speaker_embedding thay đổi theo từng sample trong batch.
    Stage 2: toàn bộ dataset dùng một ref_audio cố định
             → speaker_embedding là hằng số.
    """
    with torch.no_grad():
        # Lấy speaker embedding từ ref_mel của mỗi sample trong batch
        raw = speaker_encoder(ref_mels.to(device).to(torch.bfloat16))

        # BUG FIX: speaker_encoder có thể trả về tuple HOẶC tensor trực tiếp.
        # - Nếu trả tuple (tensor, ...): [0] → [B, D] ✓
        # - Nếu trả tensor [B, D] trực tiếp: [0] → [D] (chỉ sample đầu!) ✗
        #   → for loop chạy theo dim D (~512) thay vì dim B (~4) → IndexError
        # Fix: kiểm tra kiểu trả về, sau đó đảm bảo shape luôn là [B, D].
        if isinstance(raw, (tuple, list)):
            spk = raw[0].detach()   # tuple → lấy phần tử đầu: [B, D]
        else:
            spk = raw.detach()      # tensor trực tiếp: [B, D]

        # Đảm bảo shape [B, D] bất kể encoder trả về gì
        B = input_ids.shape[0]
        if spk.dim() == 1:
            # [D] → expand thành [B, D] (một embedding cho tất cả batch items)
            spk = spk.unsqueeze(0).expand(B, -1)
        elif spk.dim() == 3:
            # [B, T, D] → pool theo T → [B, D]
            spk = spk.mean(dim=1)
        # spk.shape == [B, D] ✓

        speaker_embedding = spk  # [B, D]

    # Text embedding
    input_text_embedding = talker_inner.text_embedding(input_ids[:, :, 0])

    # BUG FIX #2: Apply text_projection nếu có (đã merge vào upstream commit 680d4e9)
    # Nếu checkpoint chưa có fix này, lệnh below vẫn an toàn nhờ hasattr check
    if hasattr(talker_for_cond, "text_projection") and talker_for_cond.text_projection is not None:
        input_text_embedding = talker_for_cond.text_projection(input_text_embedding)

    input_text_embedding = input_text_embedding * text_embedding_mask

    # Codec embedding
    input_codec_embedding = talker_inner.codec_embedding(input_ids[:, :, 1]) * codec_embedding_mask

    # Căn chỉnh hidden size (0.6B vs 1.7B có hidden size khác nhau)
    talker_hidden_size = input_codec_embedding.shape[-1]
    if input_text_embedding.shape[-1] != talker_hidden_size:
        input_text_embedding = input_text_embedding[..., :talker_hidden_size]

    # Inject speaker embedding tại vị trí 6 trong sequence (speaker conditioning slot)
    # Dùng direct slice assignment thay vì for loop:
    #   input_codec_embedding[:, 6, :] là [B, D]
    #   speaker_embedding là [B, D]
    #   → mỗi sample nhận đúng embedding của mình, không cần vòng lặp
    input_codec_embedding[:, 6, :] = speaker_embedding.to(input_codec_embedding.dtype)

    # Kết hợp text + codec embeddings
    input_embeddings = input_text_embedding + input_codec_embedding

    # Thêm codec residual embeddings từ sub-talker (tracks 1-15)
    for i in range(1, 16):
        codec_i_emb = talker_cond.get_input_embeddings()[i - 1](codec_ids[:, :, i])
        codec_i_emb = codec_i_emb * codec_mask.unsqueeze(-1)
        if input_embeddings.shape[-1] != codec_i_emb.shape[-1]:
            pad = input_embeddings.shape[-1] - codec_i_emb.shape[-1]
            codec_i_emb = F.pad(codec_i_emb, (0, pad))
        input_embeddings = input_embeddings + codec_i_emb

    return input_embeddings, speaker_embedding


def compute_loss(
    model, input_embeddings, attention_mask, codec_0_labels,
    codec_ids, codec_mask, talker_for_cond, talker_cond,
    sub_talker_loss_weight, use_lora,
):
    """Tính loss với BUG FIX #1 (double label-shift, PR #178).

    Official sft_12hz.py:
        model.talker(inputs_embeds=emb[:, :-1], labels=labels[:, 1:])
        → HF ForCausalLMLoss shift thêm lần nữa → predictions lệch 2 token
        → giọng đọc nhanh dần sau mỗi epoch.

    Fix: KHÔNG truyền labels= vào model.talker().
         Thay vào đó, lấy logits rồi dùng F.cross_entropy với single shift.
    """
    outputs = model.talker(
        inputs_embeds=input_embeddings[:, :-1, :],  # seq-1 tokens làm input
        attention_mask=attention_mask[:, :-1],
        output_hidden_states=True,
        # KHÔNG truyền labels= ở đây để tránh double-shift
    )

    # BUG FIX #1: Single shift thủ công với F.cross_entropy
    # logits[i] dự đoán token[i+1], nên so với labels[1:]
    logits  = outputs.logits                  # [B, seq-1, vocab_size]
    targets = codec_0_labels[:, 1:].long()    # [B, seq-1]

    main_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=-100,
    )

    # Sub-talker loss (codec residuals 1-15)
    # Qwen3TTS talker trả về hidden_states với cấu trúc LỒNG NHAU:
    #   outputs.hidden_states    = (inner_tuple, None, ...)
    #   outputs.hidden_states[0] = (emb, layer1, ..., last_layer)  ← tuple các layer
    #   outputs.hidden_states[-1]= None  ← đây là lý do crash!
    # Fix: dùng [0][-1] để lấy last layer của inner tuple, shape [B, seq-1, hidden]
    inner = outputs.hidden_states[0]   # tuple các layer hidden states
    if isinstance(inner, (tuple, list)):
        last_hidden = inner[-1]        # last transformer layer [B, seq-1, hidden]
    else:
        last_hidden = inner            # fallback nếu không phải tuple

    # BUG FIX: dùng codec_mask[:, :-1] (seq-1) nhất quán với last_hidden
    talker_hidden_states = last_hidden[codec_mask[:, :-1]]          # [N, hidden]
    talker_codec_ids     = codec_ids[:, :-1, :][codec_mask[:, :-1]] # [N, 16]
    # Cả hai đều được mask với codec_mask[:, :-1] → N nhất quán, không crash

    sub_talker_hidden_size = talker_cond.config.hidden_size
    if talker_hidden_states.shape[-1] != sub_talker_hidden_size:
        talker_hidden_states = talker_hidden_states[..., :sub_talker_hidden_size]

    _, sub_talker_loss = talker_for_cond.forward_sub_talker_finetune(
        talker_codec_ids, talker_hidden_states
    )

    total_loss = main_loss + sub_talker_loss_weight * sub_talker_loss
    return total_loss, main_loss, sub_talker_loss


def save_checkpoint(
    accelerator, model, optimizer, scheduler,
    args, config_dict, epoch, global_step, best_val_loss,
):
    """Lưu checkpoint sau mỗi epoch.

    Lưu 2 thứ:
      - model.safetensors : model weights cho inference
      - training_state.pt : optimizer + scheduler + step cho resume training
    """
    output_dir = os.path.join(args.output_model_path, f"checkpoint-epoch-{epoch}")
    os.makedirs(output_dir, exist_ok=True)
    shutil.copytree(args.init_model_path, output_dir, dirs_exist_ok=True)

    cfg = dict(config_dict)
    cfg["tts_model_type"] = "base"
    cfg["stage1_vi"] = True
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    unwrapped = accelerator.unwrap_model(model)
    state_dict = {
        k: v.detach().cpu()
        for k, v in unwrapped.state_dict().items()
        if not k.startswith("speaker_encoder")
    }
    save_file(state_dict, os.path.join(output_dir, "model.safetensors"))

    # Lưu training state để resume đúng cách (optimizer + scheduler + step)
    # Thiếu phần này → khi resume optimizer khởi động lại từ đầu (mất momentum/variance)
    training_state = {
        "epoch":                epoch,
        "global_step":          global_step,
        "best_val_loss":        best_val_loss,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "args_lr":              args.lr,
    }
    torch.save(training_state, os.path.join(output_dir, "training_state.pt"))

    accelerator.print(f"[Checkpoint] Epoch {epoch}, step {global_step} -> {output_dir}")
    return output_dir


def run_validation(model, val_dataloader, speaker_encoder, args, accelerator):
    """Chạy validation và trả về avg loss."""
    model.eval()
    total_val_loss = 0.0
    steps = 0

    with torch.no_grad():
        for batch in val_dataloader:
            _model = accelerator.unwrap_model(model)
            talker_for_cond, talker_inner, talker_cond = get_talker_inner(
                _model, use_lora=False  # Stage 1 = full finetuning
            )

            input_embeddings, _ = build_input_embeddings(
                input_ids=batch["input_ids"],
                codec_ids=batch["codec_ids"],
                ref_mels=batch["ref_mels"],
                codec_mask=batch["codec_mask"],
                text_embedding_mask=batch["text_embedding_mask"],
                codec_embedding_mask=batch["codec_embedding_mask"],
                talker_for_cond=talker_for_cond,
                talker_inner=talker_inner,
                talker_cond=talker_cond,
                speaker_encoder=speaker_encoder,
                device=accelerator.device,
            )

            loss, _, _ = compute_loss(
                model=model,
                input_embeddings=input_embeddings,
                attention_mask=batch["attention_mask"],
                codec_0_labels=batch["codec_0_labels"],
                codec_ids=batch["codec_ids"],
                codec_mask=batch["codec_mask"],
                talker_for_cond=talker_for_cond,
                talker_cond=talker_cond,
                sub_talker_loss_weight=args.sub_talker_loss_weight,
                use_lora=False,
            )
            total_val_loss += loss.item()
            steps += 1

    model.train()
    return total_val_loss / max(steps, 1)


# ──────────────────────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────────────────────
def train():
    args = parse_args()
    set_seed(args.seed)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum_steps,
        mixed_precision="bf16",
        log_with="tensorboard",
        project_dir="./logs_stage1_vi",
    )
    accelerator.print("=" * 60)
    accelerator.print("Stage 1: Vietnamese TTS Language Learning")
    accelerator.print(f"  Model : {args.init_model_path}")
    accelerator.print(f"  LR    : {args.lr}  (validated: 2e-5 gây noise)")
    accelerator.print(f"  Epochs: {args.num_epochs}")
    accelerator.print(f"  Eff. batch: {args.batch_size} x {args.grad_accum_steps} "
                      f"= {args.batch_size * args.grad_accum_steps}")
    accelerator.print("=" * 60)

    # ── 1. Load model ────────────────────────────────────────────
    config_dict = patch_and_load_config(args.init_model_path, accelerator)

    qwen3tts = Qwen3TTSModel.from_pretrained(
        args.init_model_path,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_impl,
    )
    config = AutoConfig.from_pretrained(args.init_model_path)

    # ── 2. Freeze speaker_encoder ────────────────────────────────
    # Speaker encoder đã được pretrain tốt, không cần train lại cho Stage 1
    speaker_encoder = qwen3tts.model.speaker_encoder
    assert speaker_encoder is not None, (
        "speaker_encoder là None! Kiểm tra tts_model_type=base trong config."
    )
    speaker_encoder.eval()
    for p in speaker_encoder.parameters():
        p.requires_grad = False

    # ── 3. Unfreeze talker (học ngôn ngữ = train toàn bộ talker) ─
    # Stage 1 dùng full finetuning (không LoRA) để học tiếng Việt tốt nhất.
    # Đặc biệt: text_embedding PHẢI được train để học từ vựng tiếng Việt.
    for p in qwen3tts.model.talker.parameters():
        p.requires_grad = True

    total     = sum(p.numel() for p in qwen3tts.model.parameters())
    trainable = sum(p.numel() for p in qwen3tts.model.parameters() if p.requires_grad)
    accelerator.print(
        f"[Model] Total: {total/1e6:.1f}M | "
        f"Trainable: {trainable/1e6:.1f}M ({100*trainable/total:.2f}%)"
    )

    # ── 4. Dataset & DataLoader ──────────────────────────────────
    def load_jsonl(path):
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    train_data = load_jsonl(args.train_jsonl)
    accelerator.print(f"[Data] Train samples: {len(train_data)}")

    # Kiểm tra format JSONL
    sample = train_data[0]
    assert "text" in sample and "audio" in sample and "ref_audio" in sample, (
        "Mỗi dòng JSONL cần có: 'text', 'audio', 'ref_audio'. "
        f"Dòng đầu tiên chỉ có: {list(sample.keys())}"
    )

    dataset    = TTSDataset(train_data, qwen3tts.processor, config)
    train_dl   = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_dl = None
    if args.val_jsonl and os.path.exists(args.val_jsonl):
        val_data = load_jsonl(args.val_jsonl)
        val_ds   = TTSDataset(val_data, qwen3tts.processor, config)
        val_dl   = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=val_ds.collate_fn,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        accelerator.print(f"[Data] Val samples  : {len(val_data)}")
    else:
        accelerator.print("[Data] Val: không có (khuyến khích thêm để theo dõi overfitting)")

    # ── 5. Optimizer + Scheduler ─────────────────────────────────
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, qwen3tts.model.parameters()),
        lr=args.lr,
        weight_decay=0.01,
        betas=(0.9, 0.98),
    )

    total_update_steps = (len(train_dl) * args.num_epochs) // args.grad_accum_steps
    cosine_steps       = max(1, total_update_steps - args.warmup_steps)
    scheduler          = CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=args.lr * 0.1)

    # ── 6. Accelerate prepare ────────────────────────────────────
    prepare_args = [qwen3tts.model, optimizer, train_dl, scheduler]
    if val_dl:
        prepare_args.append(val_dl)

    prepared = accelerator.prepare(*prepare_args)

    if val_dl:
        model, optimizer, train_dl, scheduler, val_dl = prepared
    else:
        model, optimizer, train_dl, scheduler = prepared

    # speaker_encoder không qua accelerator.prepare (frozen, chỉ cần .to(device))
    speaker_encoder = speaker_encoder.to(accelerator.device)
    model.train()

    global_step = 0
    best_val_loss = float("inf")
    resume_epoch = 0

    # ── 6b. Resume từ checkpoint (load optimizer + scheduler + step) ──
    if args.resume_from_checkpoint:
        state_path = os.path.join(args.resume_from_checkpoint, "training_state.pt")
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location="cpu")
            resume_epoch  = state["epoch"] + 1          # bắt đầu từ epoch kế tiếp
            global_step   = state["global_step"]
            best_val_loss = state.get("best_val_loss", float("inf"))
            optimizer.load_state_dict(state["optimizer_state_dict"])
            scheduler.load_state_dict(state["scheduler_state_dict"])
            accelerator.print(
                f"[Resume] Loaded checkpoint từ {args.resume_from_checkpoint} | "
                f"epoch={state['epoch']}, step={global_step}, "
                f"best_val={best_val_loss:.4f}, saved_lr={state.get('args_lr', '?')}"
            )
        else:
            accelerator.print(
                f"[Resume] Không tìm thấy training_state.pt trong {args.resume_from_checkpoint}. "
                "Chỉ dùng model weights (optimizer sẽ khởi động từ đầu)."
            )

    # Unwrap một lần ngoài loop để tối ưu hiệu năng
    _model = accelerator.unwrap_model(model)
    talker_for_cond, talker_inner, talker_cond = get_talker_inner(_model, use_lora=False)

    # ── 7. Training loop ─────────────────────────────────────────
    for epoch in range(resume_epoch, args.num_epochs):
        model.train()
        ep_total = ep_main = ep_sub = 0.0

        for step, batch in enumerate(train_dl):
            with accelerator.accumulate(model):

                input_embeddings, _ = build_input_embeddings(
                    input_ids=batch["input_ids"],
                    codec_ids=batch["codec_ids"],
                    ref_mels=batch["ref_mels"],
                    codec_mask=batch["codec_mask"],
                    text_embedding_mask=batch["text_embedding_mask"],
                    codec_embedding_mask=batch["codec_embedding_mask"],
                    talker_for_cond=talker_for_cond,
                    talker_inner=talker_inner,
                    talker_cond=talker_cond,
                    speaker_encoder=speaker_encoder,
                    device=accelerator.device,
                )

                loss, main_loss, sub_loss = compute_loss(
                    model=model,
                    input_embeddings=input_embeddings,
                    attention_mask=batch["attention_mask"],
                    codec_0_labels=batch["codec_0_labels"],
                    codec_ids=batch["codec_ids"],
                    codec_mask=batch["codec_mask"],
                    talker_for_cond=talker_for_cond,
                    talker_cond=talker_cond,
                    sub_talker_loss_weight=args.sub_talker_loss_weight,
                    use_lora=False,
                )

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, model.parameters()),
                        args.max_grad_norm,
                    )
                    global_step += 1

                    # LR warmup → cosine decay
                    if global_step <= args.warmup_steps:
                        warmup_lr = args.lr * global_step / max(args.warmup_steps, 1)
                        for pg in optimizer.param_groups:
                            pg["lr"] = warmup_lr
                    else:
                        scheduler.step()

                optimizer.step()
                optimizer.zero_grad()

            ep_total += loss.item()
            ep_main  += main_loss.item()
            ep_sub   += sub_loss.item()

            if step % 20 == 0:
                cur_lr = optimizer.param_groups[0]["lr"]
                accelerator.print(
                    f"[E{epoch} S{step}/{len(train_dl)}] "
                    f"loss={loss.item():.4f} "
                    f"(main={main_loss.item():.4f} sub={sub_loss.item():.4f}) "
                    f"lr={cur_lr:.2e}"
                )

        n = len(train_dl)
        accelerator.print(
            f"[Epoch {epoch}] Train avg — "
            f"total={ep_total/n:.4f}  main={ep_main/n:.4f}  sub={ep_sub/n:.4f}"
        )

        # Validation
        if val_dl:
            val_loss = run_validation(model, val_dl, speaker_encoder, args, accelerator)
            accelerator.print(f"[Epoch {epoch}] Val loss = {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                accelerator.print(f"  ✓ Best val loss improved → {val_loss:.4f}")
            elif val_loss > best_val_loss * 1.05:
                accelerator.print(
                    f"  ⚠ Val loss tăng ({val_loss:.4f} > {best_val_loss:.4f} * 1.05). "
                    "Cân nhắc early stopping."
                )

        # Save checkpoint
        if accelerator.is_main_process and (epoch + 1) % args.save_every_n_epochs == 0:
            ckpt = save_checkpoint(
                accelerator, model, optimizer, scheduler,
                args, config_dict, epoch, global_step, best_val_loss,
            )
            accelerator.print(f"[Save] Checkpoint saved: {ckpt}")

    # Save final checkpoint
    if accelerator.is_main_process:
        ckpt = save_checkpoint(
            accelerator, model, optimizer, scheduler,
            args, config_dict, epoch="final", global_step=global_step,
            best_val_loss=best_val_loss,
        )
        accelerator.print(f"[Done] Final checkpoint: {ckpt}")

    accelerator.print("Stage 1 training complete.")
    accelerator.print(
        "Tiếp theo: dùng checkpoint tốt nhất làm --init_model_path cho Stage 2 (clone voice)."
    )


if __name__ == "__main__":
    train()
