from __future__ import annotations

import argparse
from pathlib import Path
import sys
import zipfile

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
    image_to_pixels,
)
from pixel_pipeline.model import (
    Pix2StructContinuationConfig,
    Pix2StructVisualContinuationTransformer,
    VisualContinuationConfig,
    VisualContinuationTransformer,
)


DEFAULT_ROOT = Path(r"E:/Document/ML/introml-project-dkd-2526-2")
DEFAULT_FONT = DEFAULT_ROOT / "font-times-new-roman" / "font-times-new-roman" / "times.ttf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--font", type=Path, default=DEFAULT_FONT)
    parser.add_argument("--input-csv", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=Path("artifacts/patch_tokenizer_w4.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/visual_submission"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--pix2struct-white-on-black", action="store_true")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def move_batch(batch: dict, device: torch.device) -> dict:
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    input_csv = args.input_csv or (args.root / "test.csv")

    device = resolve_device(args.device)
    amp_enabled = (not args.no_amp) and device.type == "cuda"
    tokenizer = PatchTokenizer.load(args.tokenizer)
    renderer = TextRenderer(args.font)

    df = pd.read_csv(input_csv, encoding="utf-8")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_type = checkpoint.get("model_type", "conv")
    if model_type == "pix2struct":
        config = Pix2StructContinuationConfig(**checkpoint["config"])
        checkpoint_args = checkpoint.get("args", {})
        white_on_black = bool(
            checkpoint_args.get(
                "pix2struct_white_on_black",
                args.pix2struct_white_on_black,
            )
        )
        collator = Pix2StructPairCollator(
            model_name=config.pix2struct_model_name,
            max_patches=config.pix2struct_max_patches,
            invert_images=not white_on_black,
        )
        model = Pix2StructVisualContinuationTransformer(config).to(device)
    else:
        config = VisualContinuationConfig(**checkpoint["config"])
        collator = collate_pixel_pairs
        model = VisualContinuationTransformer(config).to(device)

    if config.vocab_size != tokenizer.vocab_size:
        raise ValueError(
            f"Checkpoint vocab_size={config.vocab_size}, tokenizer vocab_size={tokenizer.vocab_size}"
        )
    dataset = PixelPairDataset(df, renderer=renderer, tokenizer=None, require_target=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=device.type == "cuda",
    )

    missing, unexpected = model.load_state_dict(
        checkpoint["model"],
        strict=model_type != "pix2struct",
    )
    if missing or unexpected:
        print(
            f"checkpoint load non-strict: missing={len(missing)} "
            f"unexpected={len(unexpected)}"
        )
    model.eval()
    print(f"model_type={model_type}")

    all_pixels = []
    generated_rows = []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            batch = move_batch(batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                token_ids, valid = model.generate(batch, max_steps=args.max_steps)

            token_ids = token_ids.detach().cpu().numpy()
            valid = valid.detach().cpu().numpy()
            sample_ids = batch["sample_ids"].detach().cpu().numpy()
            max_widths = batch["max_widths"].detach().cpu().numpy()

            for i, sample_id in enumerate(sample_ids):
                ids = token_ids[i][valid[i]]
                width = int(max_widths[i])
                image = tokenizer.decode_ids(ids, width=width, skip_special=False)
                pixels = image_to_pixels(image, sample_id=int(sample_id), max_width=width)
                if len(pixels) > 0:
                    all_pixels.append(pixels)
                generated_rows.append({"id": int(sample_id), "target": ""})

            print(f"generated batch={batch_index}/{len(loader)}")

    if all_pixels:
        pixel_array = np.concatenate(all_pixels)
    else:
        pixel_array = np.zeros(
            0,
            dtype=[("sample_id", np.int64), ("row_id", np.int64), ("col_id", np.int64)],
        )

    data_dir = args.out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    np.savez(data_dir / "pixels.npz", pixels=pixel_array)
    pd.DataFrame(generated_rows).sort_values("id").to_csv(
        args.out_dir / "submission.csv", index=False
    )

    zip_path = args.out_dir / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(data_dir / "pixels.npz", "data/pixels.npz")
        zf.write(args.out_dir / "submission.csv", "submission.csv")

    unique_ids = np.unique(pixel_array["sample_id"]) if len(pixel_array) else []
    print(f"pixels={len(pixel_array)} unique_ids={len(unique_ids)}")
    print(f"saved={zip_path.resolve()}")


if __name__ == "__main__":
    main()
