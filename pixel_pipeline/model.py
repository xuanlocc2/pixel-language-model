from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class VisualContinuationConfig:
    vocab_size: int
    target_patch_width: int = 4
    encoder_patch_width: int = 8
    image_height: int = 32
    num_languages: int = 5
    d_model: int = 384
    nhead: int = 8
    encoder_layers: int = 6
    decoder_layers: int = 6
    dim_feedforward: int = 1536
    dropout: float = 0.1
    max_encoder_tokens: int = 2048
    max_decoder_tokens: int = 512
    max_width_bucket: int = 2048
    pad_token_id: int = 0
    bos_token_id: int = 1


@dataclass
class Pix2StructContinuationConfig(VisualContinuationConfig):
    pix2struct_model_name: str = "google/pix2struct-base"
    pix2struct_max_patches: int = 1024
    freeze_vision_encoder: bool = True


class VisualContinuationTransformer(nn.Module):
    def __init__(self, config: VisualContinuationConfig) -> None:
        super().__init__()
        self.config = config

        self.image_proj = nn.Conv2d(
            1,
            config.d_model,
            kernel_size=(config.image_height, config.encoder_patch_width),
            stride=(config.image_height, config.encoder_patch_width),
        )
        self.own_segment = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.pair_segment = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.condition_token = nn.Parameter(torch.zeros(1, 1, config.d_model))

        self.encoder_pos = nn.Embedding(config.max_encoder_tokens, config.d_model)
        self.decoder_pos = nn.Embedding(config.max_decoder_tokens, config.d_model)
        self.language_embed = nn.Embedding(config.num_languages, config.d_model)
        self.width_embed = nn.Embedding(config.max_width_bucket + 1, config.d_model)
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.encoder_layers,
            norm=nn.LayerNorm(config.d_model),
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.decoder_layers,
            norm=nn.LayerNorm(config.d_model),
        )
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight

        self.dropout = nn.Dropout(config.dropout)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    nn.init.zeros_(module.weight[module.padding_idx])
            elif isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.own_segment, std=0.02)
        nn.init.normal_(self.pair_segment, std=0.02)
        nn.init.normal_(self.condition_token, std=0.02)

    def _pad_image_width(self, images: torch.Tensor) -> torch.Tensor:
        width = int(images.shape[-1])
        remainder = width % self.config.encoder_patch_width
        if remainder == 0:
            return images
        pad_right = self.config.encoder_patch_width - remainder
        return F.pad(images, (0, pad_right, 0, 0), value=0.0)

    def _encode_image_tokens(
        self,
        images: torch.Tensor,
        widths: torch.Tensor,
        segment: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        images = self._pad_image_width(images)
        tokens = self.image_proj(images).squeeze(2).transpose(1, 2)
        n_tokens = tokens.shape[1]
        if n_tokens > self.config.max_encoder_tokens:
            raise ValueError(
                f"Encoder sequence has {n_tokens} tokens, "
                f"but max_encoder_tokens={self.config.max_encoder_tokens}"
            )

        positions = torch.arange(n_tokens, device=tokens.device)
        tokens = tokens + self.encoder_pos(positions).unsqueeze(0) + segment

        lengths = torch.div(
            widths + self.config.encoder_patch_width - 1,
            self.config.encoder_patch_width,
            rounding_mode="floor",
        )
        arange = torch.arange(n_tokens, device=tokens.device).unsqueeze(0)
        valid_mask = arange < lengths.unsqueeze(1)
        return tokens, valid_mask

    def encode(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        own_tokens, own_valid = self._encode_image_tokens(
            batch["context_images"],
            batch["context_widths"],
            self.own_segment,
        )
        pair_tokens, pair_valid = self._encode_image_tokens(
            batch["paired_context_images"],
            batch["paired_context_widths"],
            self.pair_segment,
        )

        language_ids = batch["language_ids"]
        max_widths = batch["max_widths"].clamp(0, self.config.max_width_bucket)
        cond = (
            self.condition_token.expand(language_ids.shape[0], 1, -1)
            + self.language_embed(language_ids).unsqueeze(1)
            + self.width_embed(max_widths).unsqueeze(1)
        )
        cond_valid = torch.ones(
            (language_ids.shape[0], 1), dtype=torch.bool, device=language_ids.device
        )

        memory = torch.cat([cond, own_tokens, pair_tokens], dim=1)
        valid = torch.cat([cond_valid, own_valid, pair_valid], dim=1)
        key_padding_mask = ~valid
        memory = self.encoder(memory, src_key_padding_mask=key_padding_mask)
        return memory, key_padding_mask

    def decode(
        self,
        decoder_input_ids: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        seq_len = decoder_input_ids.shape[1]
        if seq_len > self.config.max_decoder_tokens:
            raise ValueError(
                f"Decoder sequence has {seq_len} tokens, "
                f"but max_decoder_tokens={self.config.max_decoder_tokens}"
            )
        positions = torch.arange(seq_len, device=decoder_input_ids.device)
        tgt = self.token_embed(decoder_input_ids) + self.decoder_pos(positions).unsqueeze(0)
        tgt = self.dropout(tgt)
        causal_mask = torch.triu(
            torch.ones(
                (seq_len, seq_len),
                dtype=torch.bool,
                device=decoder_input_ids.device,
            ),
            diagonal=1,
        )
        decoded = self.decoder(
            tgt,
            memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=decoder_input_ids.eq(self.config.pad_token_id),
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.lm_head(decoded)

    def forward(self, batch: dict) -> dict:
        memory, memory_key_padding_mask = self.encode(batch)
        logits = self.decode(
            batch["decoder_input_ids"],
            memory,
            memory_key_padding_mask,
        )
        out = {"logits": logits}
        if "labels" in batch:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                batch["labels"].reshape(-1),
                ignore_index=-100,
            )
            out["loss"] = loss
        return out

    @torch.no_grad()
    def generate(self, batch: dict, max_steps: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        memory, memory_key_padding_mask = self.encode(batch)
        widths = batch["max_widths"]
        target_lengths = torch.div(
            widths + self.config.target_patch_width - 1,
            self.config.target_patch_width,
            rounding_mode="floor",
        )
        batch_max_steps = int(target_lengths.max().item())
        if max_steps is not None:
            batch_max_steps = min(batch_max_steps, int(max_steps))
        if batch_max_steps > self.config.max_decoder_tokens:
            raise ValueError(
                f"Generation needs {batch_max_steps} steps, "
                f"but max_decoder_tokens={self.config.max_decoder_tokens}"
            )

        generated = torch.full(
            (widths.shape[0], 1),
            self.config.bos_token_id,
            dtype=torch.long,
            device=widths.device,
        )
        outputs = []
        for _ in range(batch_max_steps):
            logits = self.decode(generated, memory, memory_key_padding_mask)
            next_ids = logits[:, -1].argmax(dim=-1)
            outputs.append(next_ids)
            generated = torch.cat([generated, next_ids.unsqueeze(1)], dim=1)

        if outputs:
            token_ids = torch.stack(outputs, dim=1)
        else:
            token_ids = torch.empty((widths.shape[0], 0), dtype=torch.long, device=widths.device)
        valid = torch.arange(batch_max_steps, device=widths.device).unsqueeze(0) < target_lengths.unsqueeze(1)
        return token_ids, valid


def make_small_config(vocab_size: int, target_patch_width: int = 4) -> VisualContinuationConfig:
    return VisualContinuationConfig(
        vocab_size=vocab_size,
        target_patch_width=target_patch_width,
        encoder_patch_width=8,
        d_model=256,
        nhead=8,
        encoder_layers=4,
        decoder_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        max_encoder_tokens=2048,
        max_decoder_tokens=512,
    )


def make_4090_config(vocab_size: int, target_patch_width: int = 4) -> VisualContinuationConfig:
    return VisualContinuationConfig(
        vocab_size=vocab_size,
        target_patch_width=target_patch_width,
        encoder_patch_width=8,
        d_model=384,
        nhead=8,
        encoder_layers=6,
        decoder_layers=6,
        dim_feedforward=1536,
        dropout=0.1,
        max_encoder_tokens=2048,
        max_decoder_tokens=512,
    )


class Pix2StructVisualContinuationTransformer(nn.Module):
    def __init__(self, config: Pix2StructContinuationConfig) -> None:
        super().__init__()
        self.config = config
        try:
            from transformers import Pix2StructForConditionalGeneration
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Pix2StructVisualContinuationTransformer requires transformers. Install with: "
                "pip install transformers accelerate sentencepiece protobuf"
            ) from exc

        full_model = Pix2StructForConditionalGeneration.from_pretrained(
            config.pix2struct_model_name
        )
        self.vision_encoder = full_model.encoder
        del full_model
        if config.freeze_vision_encoder:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False

        vision_hidden_size = int(getattr(self.vision_encoder.config, "hidden_size"))
        self.vision_proj = nn.Linear(vision_hidden_size, config.d_model)

        self.own_segment = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.pair_segment = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.condition_token = nn.Parameter(torch.zeros(1, 1, config.d_model))

        self.decoder_pos = nn.Embedding(config.max_decoder_tokens, config.d_model)
        self.language_embed = nn.Embedding(config.num_languages, config.d_model)
        self.width_embed = nn.Embedding(config.max_width_bucket + 1, config.d_model)
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.decoder_layers,
            norm=nn.LayerNorm(config.d_model),
        )
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight
        self.dropout = nn.Dropout(config.dropout)
        self._reset_local_parameters()

    def _reset_local_parameters(self) -> None:
        for name, module in self.named_modules():
            if name.startswith("vision_encoder"):
                continue
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    nn.init.zeros_(module.weight[module.padding_idx])
        nn.init.normal_(self.own_segment, std=0.02)
        nn.init.normal_(self.pair_segment, std=0.02)
        nn.init.normal_(self.condition_token, std=0.02)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.config.freeze_vision_encoder:
            self.vision_encoder.eval()
        return self

    def _encode_pix2struct(
        self,
        flattened_patches: torch.Tensor,
        attention_mask: torch.Tensor,
        segment: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attention_mask = attention_mask.to(dtype=torch.long)
        with torch.set_grad_enabled(not self.config.freeze_vision_encoder):
            outputs = self.vision_encoder(
                flattened_patches=flattened_patches,
                attention_mask=attention_mask,
                return_dict=True,
            )
        hidden = self.vision_proj(outputs.last_hidden_state)
        hidden = hidden + segment
        valid = attention_mask.bool()
        return hidden, valid

    def encode(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        own_tokens, own_valid = self._encode_pix2struct(
            batch["pix_context_flattened_patches"],
            batch["pix_context_attention_mask"],
            self.own_segment,
        )
        pair_tokens, pair_valid = self._encode_pix2struct(
            batch["pix_paired_flattened_patches"],
            batch["pix_paired_attention_mask"],
            self.pair_segment,
        )

        language_ids = batch["language_ids"]
        max_widths = batch["max_widths"].clamp(0, self.config.max_width_bucket)
        cond = (
            self.condition_token.expand(language_ids.shape[0], 1, -1)
            + self.language_embed(language_ids).unsqueeze(1)
            + self.width_embed(max_widths).unsqueeze(1)
        )
        cond_valid = torch.ones(
            (language_ids.shape[0], 1), dtype=torch.bool, device=language_ids.device
        )

        memory = torch.cat([cond, own_tokens, pair_tokens], dim=1)
        valid = torch.cat([cond_valid, own_valid, pair_valid], dim=1)
        key_padding_mask = ~valid
        return memory, key_padding_mask

    def decode(
        self,
        decoder_input_ids: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        seq_len = decoder_input_ids.shape[1]
        if seq_len > self.config.max_decoder_tokens:
            raise ValueError(
                f"Decoder sequence has {seq_len} tokens, "
                f"but max_decoder_tokens={self.config.max_decoder_tokens}"
            )
        positions = torch.arange(seq_len, device=decoder_input_ids.device)
        tgt = self.token_embed(decoder_input_ids) + self.decoder_pos(positions).unsqueeze(0)
        tgt = self.dropout(tgt)
        causal_mask = torch.triu(
            torch.ones(
                (seq_len, seq_len),
                dtype=torch.bool,
                device=decoder_input_ids.device,
            ),
            diagonal=1,
        )
        decoded = self.decoder(
            tgt,
            memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=decoder_input_ids.eq(self.config.pad_token_id),
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.lm_head(decoded)

    def forward(self, batch: dict) -> dict:
        memory, memory_key_padding_mask = self.encode(batch)
        logits = self.decode(
            batch["decoder_input_ids"],
            memory,
            memory_key_padding_mask,
        )
        out = {"logits": logits}
        if "labels" in batch:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                batch["labels"].reshape(-1),
                ignore_index=-100,
            )
            out["loss"] = loss
        return out

    @torch.no_grad()
    def generate(self, batch: dict, max_steps: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        memory, memory_key_padding_mask = self.encode(batch)
        widths = batch["max_widths"]
        target_lengths = torch.div(
            widths + self.config.target_patch_width - 1,
            self.config.target_patch_width,
            rounding_mode="floor",
        )
        batch_max_steps = int(target_lengths.max().item())
        if max_steps is not None:
            batch_max_steps = min(batch_max_steps, int(max_steps))
        if batch_max_steps > self.config.max_decoder_tokens:
            raise ValueError(
                f"Generation needs {batch_max_steps} steps, "
                f"but max_decoder_tokens={self.config.max_decoder_tokens}"
            )

        generated = torch.full(
            (widths.shape[0], 1),
            self.config.bos_token_id,
            dtype=torch.long,
            device=widths.device,
        )
        outputs = []
        for _ in range(batch_max_steps):
            logits = self.decode(generated, memory, memory_key_padding_mask)
            next_ids = logits[:, -1].argmax(dim=-1)
            outputs.append(next_ids)
            generated = torch.cat([generated, next_ids.unsqueeze(1)], dim=1)

        if outputs:
            token_ids = torch.stack(outputs, dim=1)
        else:
            token_ids = torch.empty((widths.shape[0], 0), dtype=torch.long, device=widths.device)
        valid = torch.arange(batch_max_steps, device=widths.device).unsqueeze(0) < target_lengths.unsqueeze(1)
        return token_ids, valid


def make_pix2struct_config(
    vocab_size: int,
    target_patch_width: int = 4,
    model_name: str = "google/pix2struct-base",
    max_patches: int = 1024,
    freeze_vision_encoder: bool = True,
) -> Pix2StructContinuationConfig:
    return Pix2StructContinuationConfig(
        vocab_size=vocab_size,
        target_patch_width=target_patch_width,
        d_model=384,
        nhead=8,
        decoder_layers=6,
        dim_feedforward=1536,
        dropout=0.1,
        max_decoder_tokens=512,
        pix2struct_model_name=model_name,
        pix2struct_max_patches=max_patches,
        freeze_vision_encoder=freeze_vision_encoder,
    )
