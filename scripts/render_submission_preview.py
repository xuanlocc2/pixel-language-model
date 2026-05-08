from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile
import zipfile

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pixel_pipeline.rendering import pixels_to_image


DEFAULT_ROOT = Path(r"E:/Document/ML/introml-project-dkd-2526-2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--submission",
        type=Path,
        required=True,
        help="Path to submission.zip, output folder, or data/pixels.npz",
    )
    parser.add_argument("--out", type=Path, default=Path("outputs/submission_preview.png"))
    parser.add_argument("--ids", nargs="*", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--cols", type=int, default=2)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--padding", type=int, default=16)
    parser.add_argument("--label-height", type=int, default=24)
    return parser.parse_args()


def resolve_pixels_path(path: Path) -> tuple[Path, tempfile.TemporaryDirectory | None]:
    path = Path(path)
    if path.is_file() and path.suffix.lower() == ".npz":
        return path, None
    if path.is_dir():
        candidate = path / "data" / "pixels.npz"
        if not candidate.exists():
            candidate = path / "pixels.npz"
        if not candidate.exists():
            raise FileNotFoundError(f"Cannot find pixels.npz inside {path}")
        return candidate, None
    if path.is_file() and path.suffix.lower() == ".zip":
        tmp = tempfile.TemporaryDirectory()
        with zipfile.ZipFile(path, "r") as zf:
            zf.extract("data/pixels.npz", tmp.name)
        return Path(tmp.name) / "data" / "pixels.npz", tmp
    raise FileNotFoundError(f"Unsupported submission path: {path}")


def load_font(size: int = 14) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def image_with_label(
    img: np.ndarray,
    label: str,
    scale: int,
    label_height: int,
    font: ImageFont.ImageFont,
) -> Image.Image:
    scaled = Image.fromarray(img, mode="L").resize(
        (img.shape[1] * scale, img.shape[0] * scale),
        resample=Image.Resampling.NEAREST,
    ).convert("RGB")
    canvas = Image.new("RGB", (scaled.width, scaled.height + label_height), "white")
    canvas.paste(scaled, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, scaled.height + 4), label, fill="black", font=font)
    return canvas


def make_contact_sheet(
    tiles: list[Image.Image],
    cols: int,
    padding: int,
    out_path: Path,
) -> None:
    cols = max(1, int(cols))
    rows = (len(tiles) + cols - 1) // cols
    col_widths = []
    row_heights = []
    for c in range(cols):
        col_tiles = [tiles[i] for i in range(c, len(tiles), cols)]
        col_widths.append(max((tile.width for tile in col_tiles), default=1))
    for r in range(rows):
        row_tiles = [tiles[i] for i in range(r * cols, min(len(tiles), (r + 1) * cols))]
        row_heights.append(max((tile.height for tile in row_tiles), default=1))

    width = sum(col_widths) + padding * (cols + 1)
    height = sum(row_heights) + padding * (rows + 1)
    sheet = Image.new("RGB", (width, height), "white")
    y = padding
    idx = 0
    for r in range(rows):
        x = padding
        for c in range(cols):
            if idx >= len(tiles):
                break
            tile = tiles[idx]
            sheet.paste(tile, (x, y))
            x += col_widths[c] + padding
            idx += 1
        y += row_heights[r] + padding
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def main() -> None:
    args = parse_args()
    pixels_path, tmp = resolve_pixels_path(args.submission)
    try:
        pixels = np.load(pixels_path)["pixels"]
        test = pd.read_csv(args.root / "test.csv", encoding="utf-8").sort_values("id")
        if args.ids:
            selected = test[test["id"].isin(args.ids)]
        else:
            selected = test.iloc[args.start : args.start + args.limit]

        font = load_font()
        tiles = []
        for row in selected.itertuples(index=False):
            img = pixels_to_image(
                pixels,
                sample_id=int(row.id),
                width=int(row.max_width),
                height=32,
            )
            label = f"id={int(row.id)}  {row.language}  max_width={int(row.max_width)}"
            tiles.append(
                image_with_label(
                    img,
                    label=label,
                    scale=args.scale,
                    label_height=args.label_height,
                    font=font,
                )
            )

        if not tiles:
            raise ValueError("No selected samples to render")
        make_contact_sheet(tiles, cols=args.cols, padding=args.padding, out_path=args.out)
        print(f"rendered={args.out.resolve()} samples={len(tiles)}")
    finally:
        if tmp is not None:
            tmp.cleanup()


if __name__ == "__main__":
    main()
