# Kaggle Pixel Understanding V1 Checklist

## Phase 1 - Data Representation
- [x] Read PDF and inspect CSV/submission structure.
- [x] Implement deterministic Times New Roman renderer.
- [x] Implement target-image visual patch tokenizer.
- [x] Implement pair-aware train/test dataloader.
- [x] Run sanity checks: IDs, pair mapping, image sizes, tokenizer round-trip.

## Phase 2 - Baseline And Validation
- [x] Build pair-level train/validation split.
- [x] Create local validation renderer and `pixels.npz` exporter.
- [ ] Add OCR/character-level validation proxy if available.
- [ ] Reproduce a trivial baseline submission to validate packaging.

## Phase 3 - Model
- [x] Implement Visual Continuation Transformer.
- [x] Add Pix2Struct vision encoder option.
- [x] Add language and `max_width` conditioning.
- [ ] Train on exact 80/20 train rows.
- [x] Add generated split augmentation from train paragraphs.
- [ ] Tune patch width `4` vs `8`.

## Phase 4 - Final Submission
- [ ] Generate predictions for all 364 test samples.
- [ ] Reconstruct and visualize 20-30 outputs.
- [ ] Check every pixel has `0 <= row_id < 32` and `0 <= col_id < max_width`.
- [ ] Check `pixels.npz` contains all test sample IDs.
- [ ] Zip `data/pixels.npz` and placeholder `submission.csv`.
