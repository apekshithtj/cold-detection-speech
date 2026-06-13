# Cold Detection from Speech
**ML for Health · TU Munich · ComParE 2017 Cold Task**

Binary classification (Cold vs Non-Cold) from German speech recordings using acoustic features.

## Project Structure

```
cold_detection/
├── config.py               # All paths and hyperparameters
├── data_loader.py          # Label loading and split helpers
├── feature_extraction.py   # 257-d acoustic feature extractor (with per-file cache)
├── models.py               # SVM, Random Forest, XGBoost + UAR evaluation
├── explore.py              # Dataset EDA
├── train.py                # Training script
├── predict.py              # Test set prediction generator
├── run_pipeline.py         # End-to-end runner
├── cache/                  # Auto-created: cached .npy feature files
└── output/                 # Auto-created: saved model + predictions
```

## Python Environment

Uses `C:\Users\win10\anaconda3\envs\hybridvision\python.exe`

```
librosa==0.11.0   scikit-learn==1.8.0   xgboost==3.2.0
numpy  scipy  pandas  soundfile  imbalanced-learn  tqdm  joblib
```

## Features (257 dimensions)

| Feature Group         | Dims |
|-----------------------|------|
| MFCCs (40) mean+std   |  80  |
| Delta MFCCs mean+std  |  80  |
| Delta² MFCCs mean+std |  80  |
| Spectral centroid/BW/rolloff | 6 |
| ZCR + RMS energy      |   4  |
| F0 pitch stats        |   5  |
| HNR proxy mean+std    |   2  |
| **Total**             | **257** |

## Running

```powershell
# Activate environment
$env:PATH = "C:\Users\win10\anaconda3\envs\hybridvision;" + $env:PATH
$PYTHON = "C:\Users\win10\anaconda3\envs\hybridvision\python.exe"

# Full pipeline (explore → train → predict)
& $PYTHON run_pipeline.py

# Re-extract features from scratch
& $PYTHON run_pipeline.py --fresh

# EDA only
& $PYTHON explore.py

# Training only
& $PYTHON train.py

# Predict only (after training)
& $PYTHON predict.py
```

## Dataset

| Split | Total  | Cold (C) | Non-Cold (NC) | Imbalance |
|-------|--------|----------|---------------|-----------|
| Train |  9,505 |    970   |     8,535     |  1 : 8.8  |
| Devel |  9,596 |  1,011   |     8,585     |  1 : 8.5  |
| Test  |  9,551 |    —     |       —       |     —     |

## Evaluation

**UAR** (Unweighted Average Recall) = mean recall across both classes:

```
UAR = (Recall_Cold + Recall_NonCold) / 2
```

Chance level = 0.50. ComParE 2017 baseline SVM: ~0.60.

## Output

- `output/best_model.pkl` — best model (by devel UAR)
- `output/test_predictions.tsv` — test set predictions (file_name, prediction)
- `output/dataset_summary.json` — dataset statistics
