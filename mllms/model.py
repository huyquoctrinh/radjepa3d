"""3D Medical MLLM: Vision Encoder + Projector + LLM with LoRA."""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_SRC, os.path.join(_SRC, "utils")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from vjepa_model import MambaEncoder
from load_checkpoint import build_vjepa_config_from

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model


class VisionProjector(nn.Module):

    def __init__(self, config):
        super().__init__()
        if config.projector_type == "linear":
            self.proj = nn.Linear(config.vision_dim, config.llm_dim)
        else:
            self.proj = nn.Sequential(
                nn.Linear(config.vision_dim, config.projector_hidden),
                nn.GELU(),
                nn.Linear(config.projector_hidden, config.llm_dim),
            )

    def forward(self, x):
        return self.proj(x)


def clip_contrastive_loss(vision_embeds, text_embeds, temperature=0.07):
    """CLIP-style symmetric contrastive loss.

    vision_embeds: (B, dim) — mean-pooled projected vision features
    text_embeds: (B, dim) — mean-pooled text embeddings
    """
    v = F.normalize(vision_embeds, dim=-1)
    t = F.normalize(text_embeds, dim=-1)

    # (B, B) similarity matrix
    logits = v @ t.T / temperature

    B = logits.shape[0]
    labels = torch.arange(B, device=logits.device)

    loss_v2t = F.cross_entropy(logits, labels)
    loss_t2v = F.cross_entropy(logits.T, labels)

    return (loss_v2t + loss_t2v) / 2


class Med3DMLLM(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Vision encoder from V-JEPA
        vjepa_cfg = build_vjepa_config_from(config)
        self.vision_encoder = MambaEncoder(vjepa_cfg)

        # Pool vision tokens: 1024 -> vision_pool_tokens (keep spatial detail)
        self.n_vision_tokens = getattr(config, "vision_pool_tokens", 768)
        self.vision_pool = nn.AdaptiveAvgPool1d(self.n_vision_tokens)

        # Projector
        self.projector = VisionProjector(config)

        # LLM
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.llm_name, trust_remote_code=True,
        )
        self.llm = AutoModelForCausalLM.from_pretrained(
            config.llm_name,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        special_tokens = {
            "additional_special_tokens": [
                config.vision_start_token,
                config.vision_end_token,
            ]
        }
        self.tokenizer.add_special_tokens(special_tokens)
        self.llm.resize_token_embeddings(len(self.tokenizer))

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.vision_start_id = self.tokenizer.convert_tokens_to_ids(config.vision_start_token)
        self.vision_end_id = self.tokenizer.convert_tokens_to_ids(config.vision_end_token)

    def load_vision_checkpoint(self, path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        key = "target_encoder" if "target_encoder" in ckpt else "context_encoder"
        missing, unexpected = self.vision_encoder.load_state_dict(ckpt[key], strict=False)
        print(f"Loaded vision encoder from {key} in {path} "
              f"(missing={list(missing)}, unexpected={list(unexpected)})")

    def apply_lora(self):
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=list(self.config.lora_target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.llm = get_peft_model(self.llm, lora_config)
        self.llm.print_trainable_parameters()

    def freeze_vision(self):
        for p in self.vision_encoder.parameters():
            p.requires_grad = False

    def unfreeze_vision(self):
        for p in self.vision_encoder.parameters():
            p.requires_grad = True

    def freeze_llm(self):
        for p in self.llm.parameters():
            p.requires_grad = False

    def setup_stage1(self):
        self.freeze_vision()
        self.freeze_llm()
        for p in self.projector.parameters():
            p.requires_grad = True

    def setup_stage2(self):
        self.unfreeze_vision()
        for p in self.projector.parameters():
            p.requires_grad = True
        self.apply_lora()

    def get_trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    def encode_volume(self, volumes):
        with torch.set_grad_enabled(self.vision_encoder.blocks[0].norm.weight.requires_grad):
            vision_tokens = self.vision_encoder(volumes)  # (B, 1024, 384)
        # Pool: (B, 1024, 384) -> (B, n_vision_tokens, 384)
        pooled = self.vision_pool(vision_tokens.transpose(1, 2)).transpose(1, 2)
        projected = self.projector(pooled)  # (B, n_vision_tokens, llm_dim)
        return projected

    def _get_text_embed(self, input_ids, attention_mask):
        """Get mean-pooled text embeddings from LLM's input embeddings (no vision tokens)."""
        text_embeds = self.llm.get_input_embeddings()(input_ids)  # (B, seq_len, llm_dim)

        # Mask out special vision tokens and padding
        mask = attention_mask.clone()
        mask[input_ids == self.vision_start_id] = 0
        mask[input_ids == self.vision_end_id] = 0

        # Mean pool over valid text tokens
        mask_expanded = mask.unsqueeze(-1).float()
        pooled = (text_embeds * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        return pooled  # (B, llm_dim)

    def forward(self, volumes, input_ids, attention_mask, labels, stage=1):
        """
        Returns dict with 'loss', 'lm_loss', 'contrastive_loss' (stage 1)
        or just 'loss', 'lm_loss' (stage 2)
        """
        B = volumes.shape[0]

        vision_embeds = self.encode_volume(volumes)  # (B, 1024, llm_dim)
        text_embeds = self.llm.get_input_embeddings()(input_ids)

        new_embeds_list = []
        new_labels_list = []
        new_mask_list = []

        for i in range(B):
            ids = input_ids[i]
            start_pos = (ids == self.vision_start_id).nonzero(as_tuple=True)[0]
            end_pos = (ids == self.vision_end_id).nonzero(as_tuple=True)[0]

            if len(start_pos) == 0 or len(end_pos) == 0:
                new_embeds_list.append(text_embeds[i])
                new_labels_list.append(labels[i])
                new_mask_list.append(attention_mask[i])
                continue

            s = start_pos[0].item()
            e = end_pos[0].item()

            pre = text_embeds[i, :s + 1]
            post = text_embeds[i, e:]
            vis = vision_embeds[i]

            combined = torch.cat([pre, vis, post], dim=0)
            new_embeds_list.append(combined)

            n_vis = vis.shape[0]
            pre_labels = labels[i, :s + 1]
            vis_labels = torch.full((n_vis,), -100, device=labels.device, dtype=labels.dtype)
            post_labels = labels[i, e:]
            new_labels_list.append(torch.cat([pre_labels, vis_labels, post_labels]))

            pre_mask = attention_mask[i, :s + 1]
            vis_mask = torch.ones(n_vis, device=attention_mask.device, dtype=attention_mask.dtype)
            post_mask = attention_mask[i, e:]
            new_mask_list.append(torch.cat([pre_mask, vis_mask, post_mask]))

        max_len = max(e.shape[0] for e in new_embeds_list)
        llm_dim = text_embeds.shape[-1]

        padded_embeds = torch.zeros(B, max_len, llm_dim, device=text_embeds.device, dtype=text_embeds.dtype)
        padded_labels = torch.full((B, max_len), -100, device=labels.device, dtype=labels.dtype)
        padded_mask = torch.zeros(B, max_len, device=attention_mask.device, dtype=attention_mask.dtype)

        for i in range(B):
            L = new_embeds_list[i].shape[0]
            padded_embeds[i, :L] = new_embeds_list[i]
            padded_labels[i, :L] = new_labels_list[i]
            padded_mask[i, :L] = new_mask_list[i]

        outputs = self.llm(
            inputs_embeds=padded_embeds,
            attention_mask=padded_mask,
            labels=padded_labels,
        )
        lm_loss = outputs.loss

        result = {"lm_loss": lm_loss}

        # CLIP-style contrastive loss (stage 1)
        if stage == 1 and self.config.contrastive_lambda > 0:
            # Vision: mean-pool projected tokens
            vision_pooled = vision_embeds.mean(dim=1)  # (B, llm_dim)

            # Text: mean-pool LLM input embeddings (text-only, no vision tokens)
            text_pooled = self._get_text_embed(input_ids, attention_mask)  # (B, llm_dim)

            c_loss = clip_contrastive_loss(
                vision_pooled, text_pooled,
                temperature=self.config.contrastive_temperature,
            )
            result["contrastive_loss"] = c_loss
            result["loss"] = lm_loss + self.config.contrastive_lambda * c_loss
        else:
            result["contrastive_loss"] = torch.tensor(0.0, device=lm_loss.device)
            result["loss"] = lm_loss

        return result

    @torch.no_grad()
    def generate(self, volumes, prompt, max_new_tokens=256):
        self.eval()

        messages = [{"role": "user", "content":
                     f"{self.config.vision_start_token}{self.config.vision_end_token}\n{prompt}"}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(volumes.device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            vision_embeds = self.encode_volume(volumes)

            input_ids = inputs["input_ids"]
            text_embeds = self.llm.get_input_embeddings()(input_ids)

            s = (input_ids[0] == self.vision_start_id).nonzero(as_tuple=True)[0][0].item()
            e = (input_ids[0] == self.vision_end_id).nonzero(as_tuple=True)[0][0].item()

            pre = text_embeds[0, :s + 1]
            post = text_embeds[0, e:]
            vis = vision_embeds[0]
            combined = torch.cat([pre, vis, post], dim=0).unsqueeze(0)
            attn_mask = torch.ones(1, combined.shape[1], device=volumes.device)

            outputs = self.llm.generate(
                inputs_embeds=combined,
                attention_mask=attn_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
