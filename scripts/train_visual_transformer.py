from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pixel_pipeline import (
    PatchTokenizer,
    Pix2StructPairCollator,
    PixelPairDataset,
    TextRenderer,
    collate_pixel_pairs,
    make_pair_split,
)
from pixel_pipeline.model import (
    Pix2StructContinuationConfig,
    Pix2StructVisualContinuationTransformer,
    VisualContinuationConfig,
    VisualContinuationTransformer,
    make_4090_config,
    make_pix2struct_config,
    make_small_config,
)


DEFAULT_ROOT = Path(r"E:/Document/ML/introml-project-dkd-2526-2")
DEFAULT_FONT = DEFAULT_ROOT / "font-times-new-roman" / "font-times-new-roman" / "times.ttf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--font", type=Path, default=DEFAULT_FONT)
    parser.add_argument("--train-csv", type=Path, default=None)
    parser.add_argument("--tokenizer", type=Path, default=Path("artifacts/patch_tokenizer_w4.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/visual_transformer"))
    parser.add_argument("--model-type", choices=["conv", "pix2struct"], default="conv")
    parser.add_argument("--config", choices=["tiny", "small", "4090"], default="small")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--max-train-pairs", type=int, default=None)
    parser.add_argument("--max-val-pairs", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--pix2struct-model", default="google/pix2struct-base")
    parser.add_argument("--pix2struct-max-patches", type=int, default=512)
    parser.add_argument("--unfreeze-pix2struct", action="store_true")
    parser.add_argument("--pix2struct-white-on-black", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def fit_or_load_tokenizer(
    path: Path,
    train_df: pd.DataFrame,
    renderer: TextRenderer,
    patch_width: int = 4,
) -> PatchTokenizer:
    if path.exists():
        return PatchTokenizer.load(path)

    tokenizer = PatchTokenizer(patch_width=patch_width)
    for row in train_df.itertuples(index=False):
        image = renderer.render_target(row.target, int(row.max_width))
        tokenizer.fit_image(image)
    tokenizer.save(path)
    return tokenizer


def build_config(
    args: argparse.Namespace,
    vocab_size: int,
    target_patch_width: int,
) -> VisualContinuationConfig | Pix2StructContinuationConfig:
    if args.model_type == "pix2struct":
        config = make_pix2struct_config(
            vocab_size=vocab_size,
            target_patch_width=target_patch_width,
            model_name=args.pix2struct_model,
            max_patches=args.pix2struct_max_patches,
            freeze_vision_encoder=not args.unfreeze_pix2struct,
        )
        if args.config == "small":
            config.d_model = 256
            config.nhead = 8
            config.decoder_layers = 4
            config.dim_feedforward = 1024
        elif args.config == "tiny":
            config.d_model = 64
            config.nhead = 4
            config.decoder_layers = 1
            config.dim_feedforward = 128
        return config

    name = args.config
    if name == "4090":
        return make_4090_config(vocab_size=vocab_size, target_patch_width=target_patch_width)
    if name == "small":
        return make_small_config(vocab_size=vocab_size, target_patch_width=target_patch_width)
    return VisualContinuationConfig(
        vocab_size=vocab_size,
        target_patch_width=target_patch_width,
        encoder_patch_width=16,
        d_model=64,
        nhead=4,
        encoder_layers=1,
        decoder_layers=1,
        dim_feedforward=128,
        dropout=0.1,
        max_encoder_tokens=1024,
        max_decoder_tokens=512,
    )


def build_model(
    model_type: str,
    config: VisualContinuationConfig | Pix2StructContinuationConfig,
) -> VisualContinuationTransformer | Pix2StructVisualContinuationTransformer:
    if model_type == "pix2struct":
        return Pix2StructVisualContinuationTransformer(config)  # type: ignore[arg-type]
    return VisualContinuationTransformer(config)  # type: ignore[arg-type]


def load_init_weights(model: torch.nn.Module, path: Path, device: torch.device) -> None:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint["model"]
    try:
        model.load_state_dict(state)
        print(f"initialized model weights from={path}")
        return
    except RuntimeError as exc:
        print(f"strict init failed, trying partial shape-matched init: {exc}")

    model_state = model.state_dict()
    matched = {
        key: value
        for key, value in state.items()
        if key in model_state and tuple(model_state[key].shape) == tuple(value.shape)
    }
    missing, unexpected = model.load_state_dict(matched, strict=False)
    print(
        f"partial init from={path} matched={len(matched)} "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )


def checkpoint_model_state(
    model: torch.nn.Module,
    config: VisualContinuationConfig | Pix2StructContinuationConfig,
) -> dict[str, torch.Tensor]:
    state = model.state_dict()
    if isinstance(config, Pix2StructContinuationConfig) and config.freeze_vision_encoder:
        return {
            key: value
            for key, value in state.items()
            if not key.startswith("vision_encoder.")
        }
    return state


def load_checkpoint_model_state(
    model: torch.nn.Module,
    state: dict[str, torch.Tensor],
    strict: bool,
) -> None:
    try:
        missing, unexpected = model.load_state_dict(state, strict=strict)
    except RuntimeError:
        missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(
            f"checkpoint load non-strict: missing={len(missing)} "
            f"unexpected={len(unexpected)}"
        )


def move_batch(batch: dict, device: torch.device) -> dict:
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out


def evaluate(
    model: VisualContinuationTransformer,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                out = model(batch)
            losses.append(float(out["loss"].detach().cpu()))
    return float(np.mean(losses)) if losses else math.inf


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_val_loss: float,
    config: VisualContinuationConfig | Pix2StructContinuationConfig,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    torch.save(
        {
            "model": checkpoint_model_state(model, config),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "config": asdict(config),
            "model_type": args.model_type,
            "args": safe_args,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    amp_enabled = (not args.no_amp) and device.type == "cuda"
    print(f"device={device} amp={amp_enabled} torch={torch.__version__}")

    train_csv_path = args.train_csv or (args.root / "train.csv")
    train_csv = pd.read_csv(train_csv_path, encoding="utf-8")
    train_df, val_df, split = make_pair_split(
        train_csv,
        val_ratio=args.val_ratio,
        seed=args.seed,
        max_train_pairs=args.max_train_pairs,
        max_val_pairs=args.max_val_pairs,
    )
    split.save(args.out_dir / "pair_split.json")
    print(f"train_rows={len(train_df)} val_rows={len(val_df)}")

    renderer = TextRenderer(args.font)
    tokenizer = fit_or_load_tokenizer(args.tokenizer, train_df, renderer)
    print(f"tokenizer={args.tokenizer} vocab_size={tokenizer.vocab_size} patch_width={tokenizer.patch_width}")

    train_dataset = PixelPairDataset(train_df, renderer=renderer, tokenizer=tokenizer)
    val_dataset = PixelPairDataset(val_df, renderer=renderer, tokenizer=tokenizer)
    if args.model_type == "pix2struct":
        collator = Pix2StructPairCollator(
            model_name=args.pix2struct_model,
            max_patches=args.pix2struct_max_patches,
            invert_images=not args.pix2struct_white_on_black,
        )
    else:
        collator = collate_pixel_pairs
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=device.type == "cuda",
    )

    config = build_config(args, tokenizer.vocab_size, tokenizer.patch_width)
    model = build_model(args.model_type, config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"model_type={args.model_type} "
        f"total_params={total_params:,} trainable_params={trainable_params:,}"
    )
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    start_epoch = 1
    best_val_loss = math.inf

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        load_checkpoint_model_state(
            model,
            checkpoint["model"],
            strict=args.model_type != "pix2struct",
        )
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint.get("scaler", {}))
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_loss = float(checkpoint.get("best_val_loss", math.inf))
        print(f"resumed={args.resume} start_epoch={start_epoch} best_val_loss={best_val_loss:.4f}")
    elif args.init_checkpoint is not None:
        load_init_weights(model, args.init_checkpoint, device)

    metadata = {
        "config": asdict(config),
        "model_type": args.model_type,
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "vocab_size": tokenizer.vocab_size,
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    global_step = 0
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(train_loader, start=1):
            batch = move_batch(batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                out = model(batch)
                loss = out["loss"] / args.grad_accum

            scaler.scale(loss).backward()
            running_loss += float(out["loss"].detach().cpu())

            if step % args.grad_accum == 0 or step == len(train_loader):
                if args.clip_grad > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            if step % args.log_every == 0 or step == len(train_loader):
                mean_loss = running_loss / step
                print(
                    f"epoch={epoch} step={step}/{len(train_loader)} "
                    f"train_loss={mean_loss:.4f} ppl={math.exp(min(mean_loss, 20)):.2f}"
                )

        train_loss = running_loss / max(1, len(train_loader))
        val_loss = evaluate(model, val_loader, device, amp_enabled)
        elapsed = time.time() - t0
        print(
            f"epoch={epoch} done train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_ppl={math.exp(min(val_loss, 20)):.2f} "
            f"elapsed={elapsed:.1f}s"
        )

        save_checkpoint(
            args.out_dir / "last.pt",
            model,
            optimizer,
            scaler,
            epoch,
            best_val_loss,
            config,
            args,
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                args.out_dir / "best.pt",
                model,
                optimizer,
                scaler,
                epoch,
                best_val_loss,
                config,
                args,
            )
            print(f"saved best checkpoint val_loss={best_val_loss:.4f}")


if __name__ == "__main__":
    main()
