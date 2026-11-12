# Project Context & Handover Summary: Location NER (KSE Assignment 3)

**Last Updated**: November 12, 2026  
**Objective**: Build a high-performing Named Entity Recognition (NER) pipeline to extract location entities from mixed Ukrainian/Russian text.

---

## 1. Non-Negotiable Rules & Constraints

1. **PyTorch Exclusively**: NO Keras, TensorFlow, or Chainer.
2. **Strictly Recurrent**: Bi-LSTM / Bi-GRU architectures only. **NO Transformers, BERT, RoBERTa, or Self-Attention**.
3. **Target Metric**: Exact set-intersection F1 evaluated after `hard_processing` normalization (`comp_metric`).

---

## 2. Project File Structure

```
assignment_03/
├── data/
│   ├── uk_geo_dataset.csv                    # ~1M synthetic Ukrainian texts with loc_markers
│   ├── ru_geo_dataset.csv                    # ~8M Russian texts (Nerus) with loc_markers
│   ├── labeling_sample.csv                   # 26 real-world annotated samples (benchmark/proxy)
│   ├── test.csv                              # 477 competition test texts
│   └── README.md
├── src/
│   ├── __init__.py
│   ├── preprocessing.py                      # hard_processing, comp_metric, WordTokenizer, WordVocab, CharVocab, clean_entity_boundaries
│   ├── dataset.py                            # NERDataset, NERCollate (dynamic padding, ignore_index=-100), create_dataloaders
│   └── models.py                             # RecurrentNERNet (Word Embeddings + Char-CNN + Bi-LSTM/GRU), RecurrentNERLightningModule
├── eda.ipynb                                 # Local EDA notebook (sequence lengths, imbalance, Cyrillic markers)
├── train_colab.ipynb                         # Earlier single-model Colab notebook (Runs 1-3)
├── train_bilstm_charcnn_3fold_ensemble.ipynb # Main Colab notebook: 7-class BiLSTM+CharCNN, 3-fold CV, Soft-Voting Ensemble (Kaggle: 0.58125)
├── ai_usage.txt                              # AI usage log (assignment requirement)
└── CONTEXT_HANDOVER.md                       # This handover document
```

**Google Drive layout**: `My Drive/DL/Assignment-3/`  
- `src/` — synced manually before each Colab run  
- `checkpoints/` — fold `.ckpt` files  
- `kse-ua-location-extraction-2026.zip` — competition data zip  
- `submission_*.csv` — versioned submissions  

---

## 3. Current Architecture Details (Phase 2 — 7-Class)

- **Tokenization**:
  - Word-level regex tokenizer (`WordTokenizer`) tracking character offsets `(start_char, end_char)`.
  - Character-level vocabulary (`CharVocab`, size ~182) covering all Cyrillic (UA + RU), Latin, digits, and punctuation.

- **Model (`src/models.py`)**:
  - `nn.Embedding(vocab_size=60000, embedding_dim=128)`
  - **Char-CNN**: `nn.Embedding(char_vocab_size, 32)` → `Conv1D(filters=64, kernel=3, padding=1)` → `ReLU` → `MaxPool` → 64-dim morphological feature per word.
  - Concatenation: `128 + 64 = 192`-dim input.
  - `SpatialDropout1d(0.20)`.
  - `nn.LSTM(input_size=192, hidden_size=256, num_layers=2, bidirectional=True, dropout=0.30, batch_first=True)`.
  - `nn.Linear(512, 7)` for **7-class BIO tagging**: `O, B-LOC, I-LOC, B-ORG, I-ORG, B-PER, I-PER`. Location probability = sum of softmax at indices `[1, 2]`.

- **Loss & Optimization**:
  - `CrossEntropyLoss(weight=[1.0, 15.0, 15.0, 5.0, 5.0, 5.0, 5.0], ignore_index=-100)` — LOC classes heavily upweighted; ORG/PER penalized to push org-name false positives into their own classes.
  - AdamW optimizer with `CosineAnnealingLR` scheduler.
  - 16-bit Mixed Precision training via PyTorch Lightning (`precision='16-mixed'`).
  - Gradient clipping: `gradient_clip_val=1.0`.

- **Training Config (`train_bilstm_charcnn_3fold_ensemble.ipynb`)**:
  - `N_SPLITS=3`, `MAX_EPOCHS=5`, `BATCH_SIZE=64`, `MAX_LEN=128`, `MAX_CHAR_LEN=16`.
  - Blended dataset: 300k Ukrainian + 100k Russian = 400k samples total.
  - Stratified 3-fold CV with per-fold threshold calibration on `labeling_sample.csv`.

---

## 4. Experiment Progression & Results Log

### Phase 1 — Antigravity Session (Runs 1–3)

#### Run 1 — Bi-LSTM Baseline (200k samples, class weights [1, 25, 25]):
- Synthetic val F1: **`0.7909`** (threshold `0.80`).
- Benchmark (`labeling_sample.csv`): `val_best_f1 = 0.5316`, precision `0.3333`, recall `0.6250`.
- **Issue**: Raw spans had trailing newlines/numbers (e.g. `'проспекті Академіка Глушкова, 9. \n\n❕'`).

#### Run 2 — Bi-LSTM + Boundary Cleaning (400k samples, class weights [1, 15, 15]):
- Synthetic val F1: **`0.7850`**.
- Benchmark: `val_best_f1 = 0.5556` (+2.4%), precision `0.3704`, val_loss dropped from 0.37 → 0.26.
- Threshold moved to `0.90`.
- **Fix added**: `clean_entity_boundaries()` strips trailing newlines, emojis, punctuation, trailing numbers.

#### Run 3 — Char-CNN + Bi-LSTM (400k samples, threshold sweep to 0.95):
- Synthetic val F1: **`0.8795`** (threshold `0.95`).
- **Kaggle Public Score**: **`0.54769`** ← did not beat baseline.
- **Root Cause**: Synthetic `uk_geo_dataset` is clean/unambiguous → 0.95 threshold maximises F1 there. Kaggle `test.csv` is noisy Telegram/news (identical distribution to `labeling_sample.csv`) → at 0.95 cutoff, recall collapses because lower-confidence real location tokens are dropped.

---

### Phase 2 — Web Gemini Session (3-Fold Ensemble, 7-Class)

#### Key Improvements Made:
1. **7-Class BIO Tagging**: Expanded from 3 classes (`O, B-LOC, I-LOC`) to 7 classes (`O, B-LOC, I-LOC, B-ORG, I-ORG, B-PER, I-PER`). Organisation and person names (e.g. "Edelburg Development") are now pushed to their own entity classes, reducing location false positives.
2. **Real-World Threshold Calibration**: After training each fold, threshold is tuned on `labeling_sample.csv` (matches Kaggle test distribution) instead of synthetic validation — this closes the synthetic/real distribution gap.
3. **3-Fold Soft-Voting Ensemble**: Probabilities averaged across all 3 fold models before applying `ensemble_th` (mean of per-fold calibrated thresholds). Entity deduplication via `dict.fromkeys()`.
4. **Hyperparameter Refinements**: Decoupled `RecurrentNERLightningModule` / `RecurrentNERNet` signatures; `class_weights` explicitly passed to `load_from_checkpoint()` (eliminates `TypeError` on reload).
5. **Submission Safety**: Outputs `submission_ensemble_latest.csv` to Drive — different filename prevents overwriting old runs.

#### Run 4 — 7-Class 3-Fold Ensemble:
- **Kaggle Public Score**: **`0.58125`** ✅ **Beat baseline!**

---

## 5. Known Issues & Diagnostic Notes

| Issue | Status / Fix |
|-------|-------------|
| Synthetic val threshold (0.95) → kills recall on real test data | Fixed in Phase 2 by calibrating per-fold threshold on `labeling_sample.csv` |
| Submission file overwritten on each run | Fixed: versioned filenames `submission_{run}_{score}_{ts}.csv` in `train_colab.ipynb`; `submission_ensemble_latest.csv` in ensemble notebook |
| `TypeError` on `load_from_checkpoint` | Fixed: explicit `class_weights=Config.CLASS_WEIGHTS` passed |
| Org names like "Edelburg Development" extracted as locations | Mitigated by 7-class model (ORG/PER classes absorb them); tunable via class weights |
| `SyntaxError` / `PowerShell parse errors` generating notebooks | Mitigation: use separate `.py` generator scripts, then delete after use |

---

## 6. Next Immediate Steps (Priority Order)

1. **Post-Processing & Boundary Refining**: Inspect predicted entity spans on `labeling_sample.csv` and `test.csv` for trailing punctuation, case endings, or whitespace artifacts. Extend `clean_entity_boundaries()` if needed.

2. **Ensemble Weight Tuning & Architecture Diversity**: 
   - Try **weighted** soft-voting (e.g. weight fold with best labeling_sample F1 higher).
   - Try a **Bi-GRU + Char-CNN** 3-fold model and blend with the current Bi-LSTM ensemble.

3. **Advanced Threshold & Class-Weight Grid Search**: Fine-tune individual class weights (esp. ORG/PER) and probability cutoffs on `labeling_sample.csv` to push F1 beyond `0.58125`.

4. **More Folds or More Epochs**: Increase `N_SPLITS` to 5 or bump `MAX_EPOCHS` to 8 — helps when `labeling_sample` proxy is small (26 samples).

5. **Blending More Russian Data**: The `ru_geo_dataset.csv` has ~8M samples — increasing RU sampling ratio may improve generalisation to news-domain text.

---

## 7. Hard Constraints Reminder

- PyTorch only — NO Keras, TF, Chainer.
- Recurrent only — NO Transformers / Self-Attention / BERT.
- Target metric: `comp_metric` (set-level F1 after `hard_processing` normalisation).
- Never overwrite old submissions — always use versioned filenames with timestamp + score.
- Google Drive `src/` must be manually synced before each Colab run.
- Assignment deliverable files: `src/preprocessing.py`, `src/dataset.py`, `src/models.py`, `eda.ipynb`, `train_colab.ipynb`, `ai_usage.txt`.
