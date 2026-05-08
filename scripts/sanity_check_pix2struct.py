from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pixel_pipeline import (
    PatchTokenizer,
    Pix2StructPairCollator,
    PixelPairDataset,
    TextRenderer,
)
from pixel_pipeline.model import Pix2StructVisualContinuationTransformer, make_pix2struct_config


DEFAULT_ROOT = Path(r"E:/Document/ML/introml-project-dkd-2526-2")
DEFAULT_FONT = DEFAULT_ROOT / "font-times-new-roman" / "font-times-new-roman" / "times.ttf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--font", type=Path, default=DEFAULT_FONT)
    parser.add_argument("--tokenizer", type=Path, default=Path("artifacts/patch_tokenizer_w4.json"))
    parser.add_argument("--pix2struct-model", default="google/pix2struct-base")
    parser.add_argument("--pix2struct-max-patches", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def move_batch(batch: dict, device: torch.device) -> dict:
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    n_rows = max(2, int(args.batch_size))
    if n_rows % 2 != 0:
        n_rows += 1
    train = pd.read_csv(args.root / "train.csv", encoding="utf-8").head(n_rows)
    renderer = TextRenderer(args.font)
    tokenizer = PatchTokenizer.load(args.tokenizer)
    dataset = PixelPairDataset(train, renderer=renderer, tokenizer=tokenizer)
    collator = Pix2StructPairCollator(
        model_name=args.pix2struct_model,
        max_patches=args.pix2struct_max_patches,
    )
    batch = move_batch(collator([dataset[i] for i in range(len(dataset))]), device)
    config = make_pix2struct_config(
        vocab_size=tokenizer.vocab_size,
        target_patch_width=tokenizer.patch_width,
        model_name=args.pix2struct_model,
        max_patches=args.pix2struct_max_patches,
        freeze_vision_encoder=True,
    )
    config.d_model = 64
    config.nhead = 4
    config.decoder_layers = 1
    config.dim_feedforward = 128
    model = Pix2StructVisualContinuationTransformer(config).to(device)
    out = model(batch)
    generated, valid = model.generate(batch, max_steps=4)
    print(f"device={device} torch={torch.__version__}")
    print(f"context_patches={tuple(batch['pix_context_flattened_patches'].shape)}")
    print(f"paired_patches={tuple(batch['pix_paired_flattened_patches'].shape)}")
    print(f"logits={tuple(out['logits'].shape)} loss={float(out['loss'].detach().cpu()):.4f}")
    print(f"generated={tuple(generated.shape)} valid={tuple(valid.shape)}")


if __name__ == "__main__":
    main()
