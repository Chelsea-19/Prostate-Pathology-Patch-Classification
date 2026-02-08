# Prostate Pathology Patch Classification

A project for 5-class prostate pathology patch classification using the UNI foundation model.

## Task Description

**Objective**: Classify 251x251 prostate histopathology patches into 5 classes:
- **Stroma** - Stromal tissue
- **Normal** - Normal epithelium  
- **G3** - Gleason Grade 3
- **G4** - Gleason Grade 4
- **G5** - Gleason Grade 5 (scarce class)

**Model**: UNI pathology foundation model (frozen ViT backbone) + lightweight classification head.

**Key Features**:
- WeightedRandomSampler to handle class imbalance (G5 is scarce)
- CosineAnnealingLR scheduler
- Per-class metrics including Sensitivity and Specificity
- Feature visualization for model interpretability

## Environment Setup

```bash
pip install -r requirements.txt
```

### HuggingFace Token (Required for UNI model)

The UNI model requires authentication. Set up your token:

```bash
# Option 1
huggingface-cli login

# Option 2
export HF_TOKEN=your_token_here
```

Get your token at: https://huggingface.co/settings/tokens

## Data Directory Structure

Organize your data as follows (paths can be customized via CLI):

```
data/
├── SetA/
│   ├── 251_Train_A/
│   │   ├── G3/
│   │   ├── G4/
│   │   ├── G5/
│   │   ├── Normal/
│   │   └── Stroma/
│   └── 251_Test_A/
│       └── (same structure)
└── SetB/
    ├── 251_Train_B/
    │   └── (same structure)
    └── 251_Test_B/
        └── (same structure)
```

**Important**: Train and validation sets must be **directory-level isolated** (no random split) to prevent data leakage.

## Training

### Quick Start

```bash
python run_train.py \
  --train_dirs data/SetA/251_Train_A data/SetB/251_Train_B \
  --val_dirs data/SetA/251_Test_A data/SetB/251_Test_B \
  --out_dir outputs
```

### Full Options

```bash
python run_train.py \
  --train_dirs data/SetA/251_Train_A data/SetB/251_Train_B \
  --val_dirs data/SetA/251_Test_A data/SetB/251_Test_B \
  --batch_size 32 \
  --epochs 15 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --dropout 0.3 \
  --use_sampler \
  --freeze_backbone \
  --seed 42 \
  --out_dir outputs
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--use_sampler` | True | Use WeightedRandomSampler for class imbalance |
| `--freeze_backbone` | True | Freeze UNI backbone, train only classifier head |
| `--lr` | 1e-4 | Initial learning rate |
| `--epochs` | 15 | Number of training epochs |
| `--dropout` | 0.3 | Dropout rate in classifier head |

## Ablation Experiments

Run ablation to compare sampler effect and learning rates:

```bash
python run_train.py --run_ablation \
  --train_dirs data/SetA/251_Train_A data/SetB/251_Train_B \
  --val_dirs data/SetA/251_Test_A data/SetB/251_Test_B \
  --out_dir outputs
```

This will run:
1. Baseline without WeightedRandomSampler
2. With WeightedRandomSampler (default)
3. lr=1e-4 vs lr=3e-4 comparison

Output: `ablation_summary.csv` and `ablation_plot.png`

## Output Files

After training, the following files are generated in `--out_dir`:

| File | Description |
|------|-------------|
| `loss_curve.png` | Training/validation loss over epochs |
| `confusion_matrix.png` | Confusion matrix heatmap |
| `metrics_overall.json` | Overall accuracy, macro-F1, weighted-F1 |
| `metrics_per_class.csv` | Per-class Precision/Recall/Specificity/F1 |
| `best_model.pth` | Best model checkpoint |
| `interpretability/class_*.png` | Feature heatmaps for each class |
| `ablation_summary.csv` | Ablation experiment results (if --run_ablation) |
| `ablation_plot.png` | Ablation comparison chart (if --run_ablation) |

## Project Structure

```
prostate_classification/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── data.py       # Dataset loading, transforms, sampler
│   ├── model.py      # UNI backbone + classification head
│   ├── train.py      # Training loop
│   ├── eval.py       # Evaluation metrics
│   ├── explain.py    # Feature visualization
│   └── utils.py      # Utilities (seed, plotting)
└── run_train.py      # Main entry point
```

## Expected Results

With default settings (15 epochs, lr=1e-4, use_sampler=True):
- **Overall Accuracy**: ~94%
- **Macro F1**: ~0.94
- **Per-class Sensitivity**: 85-100% depending on class

## Notes

- **Class Order**: `['G3', 'G4', 'G5', 'Normal', 'Stroma']` (alphabetical, from ImageFolder)
- **Reproducibility**: Fixed seed (42 by default), but some GPU non-determinism may remain
- DataLoader worker seed control is implemented for better reproducibility
