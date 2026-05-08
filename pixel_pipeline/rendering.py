from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


PIXEL_DTYPE = np.dtype(
    [("sample_id", np.int64), ("row_id", np.int64), ("col_id", np.int64)]
)


@dataclass(frozen=True)
class TextRenderer:
    font_path: str | Path
    font_size: int = 24
    image_height: int = 32
    margin_px: int = 10
    threshold: int = 128

    def __post_init__(self) -> None:
        object.__setattr__(self, "font_path", Path(self.font_path))
        font = ImageFont.truetype(str(self.font_path), self.font_size)
        object.__setattr__(self, "font", font)

    def text_bbox(self, text: str) -> tuple[int, int, int, int]:
        return self.font.getbbox(str(text))

    def text_size(self, text: str) -> tuple[int, int]:
        left, top, right, bottom = self.text_bbox(text)
        return right - left, bottom - top

    def render_text(
        self,
        text: str,
        width: int | None = None,
        binary: bool = False,
    ) -> np.ndarray:
        text = "" if text is None else str(text)
        bbox = self.text_bbox(text)
        text_width = max(0, bbox[2] - bbox[0])
        text_height = max(0, bbox[3] - bbox[1])
        image_width = int(width) if width is not None else max(1, text_width + self.margin_px)
        image_width = max(1, image_width)

        img = Image.new("L", (image_width, self.image_height), 0)
        if text:
            draw = ImageDraw.Draw(img)
            y_offset = (self.image_height - text_height) // 2 - bbox[1]
            draw.text((0, y_offset), text, fill=255, font=self.font)

        arr = np.asarray(img, dtype=np.uint8)
        if binary:
            return (arr > self.threshold).astype(np.uint8)
        return arr

    def render_context(self, text: str, binary: bool = False) -> np.ndarray:
        return self.render_text(text, width=None, binary=binary)

    def render_target(self, text: str, max_width: int, binary: bool = False) -> np.ndarray:
        return self.render_text(text, width=int(max_width), binary=binary)

    def text_to_pixels(self, text: str, sample_id: int, max_width: int) -> np.ndarray:
        arr = self.render_target(text, max_width=max_width, binary=False)
        rows, cols = np.where(arr > self.threshold)
        valid = (
            (rows >= 0)
            & (rows < self.image_height)
            & (cols >= 0)
            & (cols < int(max_width))
        )
        rows, cols = rows[valid], cols[valid]

        pixels = np.zeros(len(rows), dtype=PIXEL_DTYPE)
        pixels["sample_id"] = int(sample_id)
        pixels["row_id"] = rows.astype(np.int64)
        pixels["col_id"] = cols.astype(np.int64)
        return pixels


def pixels_to_image(
    pixels: np.ndarray,
    sample_id: int,
    width: int,
    height: int = 32,
) -> np.ndarray:
    img = np.zeros((height, int(width)), dtype=np.uint8)
    mask = pixels["sample_id"] == int(sample_id)
    rows = pixels["row_id"][mask]
    cols = pixels["col_id"][mask]
    valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < int(width))
    img[rows[valid], cols[valid]] = 255
    return img


def image_to_pixels(
    image: np.ndarray,
    sample_id: int,
    max_width: int | None = None,
    threshold: int = 128,
) -> np.ndarray:
    if image.ndim != 2:
        raise ValueError(f"Expected grayscale image [H,W], got shape={image.shape}")
    width = image.shape[1] if max_width is None else int(max_width)
    clipped = image[:, :width]
    rows, cols = np.where(clipped > threshold)
    valid = (rows >= 0) & (rows < image.shape[0]) & (cols >= 0) & (cols < width)
    rows, cols = rows[valid], cols[valid]

    pixels = np.zeros(len(rows), dtype=PIXEL_DTYPE)
    pixels["sample_id"] = int(sample_id)
    pixels["row_id"] = rows.astype(np.int64)
    pixels["col_id"] = cols.astype(np.int64)
    return pixels
