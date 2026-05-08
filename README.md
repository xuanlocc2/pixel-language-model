# Kaggle Pixel Understanding V1

Pixel-only text continuation pipeline for the Kaggle Pixel Understanding V1 project.

At inference time, the system renders the provided context into an image, encodes the image with a frozen Pix2Struct visual encoder, and autoregressively predicts visual patch tokens for the target image. It does not OCR the context into text and does not emit a raw text continuation; the final output is exported as `data/pixels.npz`.

## Main Flow

```text
context + paired context
-> render as 32px-high text images
-> Pix2Struct visual encoder
-> language/max_width conditioning
-> Transformer visual patch decoder
-> target image pixels
-> data/pixels.npz
```

## Important Files

```text
pixel_pipeline/                 Core renderer, tokenizer, dataset, models
scripts/make_augmented_csv.py    Build split-based augmentation data
scripts/train_visual_transformer.py
scripts/generate_visual_transformer.py
scripts/sanity_check_pix2struct.py
artifacts/patch_tokenizer_aug_w4.json
runs/pix2struct_finetune/best.pt
```

The Pix2Struct encoder is loaded from Hugging Face at runtime. The checkpoint stores the trained visual decoder/projection weights, so the instructor can generate predictions without retraining.

## Setup

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install numpy pandas pillow tqdm matplotlib transformers accelerate sentencepiece protobuf
```

## Paths

Expected competition data layout:

```text
data/train.csv
data/test.csv
data/font-times-new-roman/font-times-new-roman/times.ttf
```

Set environment variables:

```bash
export DATA_ROOT=/path/to/data
export FONT_PATH=$DATA_ROOT/font-times-new-roman/font-times-new-roman/times.ttf
```

For `csh/tcsh`:

```csh
setenv DATA_ROOT /path/to/data
setenv FONT_PATH ${DATA_ROOT}/font-times-new-roman/font-times-new-roman/times.ttf
```

## Generate Submission

```bash
python scripts/generate_visual_transformer.py \
  --root "$DATA_ROOT" \
  --font "$FONT_PATH" \
  --checkpoint runs/pix2struct_finetune/best.pt \
  --tokenizer artifacts/patch_tokenizer_aug_w4.json \
  --batch-size 1 \
  --out-dir outputs/final_pix2struct
```

Submit:

```text
outputs/final_pix2struct/submission.zip
```

## Training Commands

Augmented pretraining:

```bash
python scripts/train_visual_transformer.py \
  --root "$DATA_ROOT" \
  --font "$FONT_PATH" \
  --train-csv artifacts/train_aug40.csv \
  --tokenizer artifacts/patch_tokenizer_aug_w4.json \
  --model-type pix2struct \
  --config 4090 \
  --pix2struct-max-patches 512 \
  --epochs 6 \
  --batch-size 1 \
  --grad-accum 16 \
  --lr 2e-4 \
  --out-dir runs/pix2struct_aug40
```

Fine-tuning:

```bash
python scripts/train_visual_transformer.py \
  --root "$DATA_ROOT" \
  --font "$FONT_PATH" \
  --tokenizer artifacts/patch_tokenizer_aug_w4.json \
  --model-type pix2struct \
  --config 4090 \
  --pix2struct-max-patches 512 \
  --epochs 12 \
  --batch-size 1 \
  --grad-accum 16 \
  --lr 5e-5 \
  --init-checkpoint runs/pix2struct_aug40/best.pt \
  --out-dir runs/pix2struct_finetune
```
