from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pixel_pipeline import PatchTokenizer, PixelPairDataset, TextRenderer, collate_pixel_pairs
from pixel_pipeline.model import VisualContinuationConfig, VisualContinuationTransformer


DEFAULT_ROOT = Path(r"E:/Document/ML/introml-project-dkd-2526-2")
DEFAULT_FONT = DEFAULT_ROOT / "font-times-new-roman" / "font-times-new-roman" / "times.ttf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--font", type=Path, default=DEFAULT_FONT)
    parser.add_argument("--tokenizer", type=Path, default=Path("artifacts/patch_tokenizer_w4.json"))
    parser.add_argument("--batch-size", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train = pd.read_csv(args.root / "train.csv", encoding="utf-8").head(args.batch_size)
    renderer = TextRenderer(args.font)
    tokenizer = PatchTokenizer.load(args.tokenizer)
    dataset = PixelPairDataset(train, renderer=renderer, tokenizer=tokenizer)
    batch = collate_pixel_pairs([dataset[i] for i in range(len(dataset))])

    config = VisualContinuationConfig(
        vocab_size=tokenizer.vocab_size,
        target_patch_width=tokenizer.patch_width,
        encoder_patch_width=16,
        d_model=64,
        nhead=4,
        encoder_layers=1,
        decoder_layers=1,
        dim_feedforward=128,
        max_encoder_tokens=1024,
        max_decoder_tokens=512,
        dropout=0.0,
    )
    model = VisualContinuationTransformer(config)
    out = model(batch)
    generated, valid = model.generate(batch, max_steps=8)
    print(f"torch={torch.__version__}")
    print(f"vocab_size={tokenizer.vocab_size}")
    print(f"context_images={tuple(batch['context_images'].shape)}")
    print(f"decoder_input_ids={tuple(batch['decoder_input_ids'].shape)}")
    print(f"logits={tuple(out['logits'].shape)}")
    print(f"loss={float(out['loss'].detach().cpu()):.4f}")
    print(f"generated={tuple(generated.shape)} valid={tuple(valid.shape)}")


if __name__ == "__main__":
    main()
