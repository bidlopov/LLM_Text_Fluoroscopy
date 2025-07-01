# Text-Fluoroscopy

This repository is based on the method described in the paper:  
**[Text Fluoroscopy: Detecting LLM-Generated Text through Intrinsic Features (EMNLP 2024)]([https://arxiv.org/abs/2405.13687](https://aclanthology.org/2024.emnlp-main.885.pdf))**

## Overview

This project uses the Text Fluoroscopy method to determine whether a text was written by a human or a language model (e.g., GPT).
It extracts features using the GTE-Qwen model and applies a trained MLP classifier for prediction.

## Functionality

- Loads a transformer model (e.g., Qwen1.5-GTE-Large)
- Extracts KL-divergence statistics from each layer
- Chooses the most informative layer automatically
- Trains a classifier (MLP)
- Predicts whether text is LLM-generated or human-written
- Saves results (including visualizations) to `assets/`

## Project Structure

```
Text-Fluoroscopy/
├── assets/
│   └── result.png
├── dataset/
│   ├── Training_Essay_Data.csv
│   └── smaller_dataset.csv
├── embedding_classify/
│   ├── train_classifier.py
│   ├── predict.py
│   └── look_inside_model.py
├── fluoroscopy/
│   ├── extract_features.py
│   └── model_loader.py
├── fluoroscopy_mlp.joblib
└── README.md
```

## Setup

Install the required packages manually:

```bash
pip install torch transformers scikit-learn pandas matplotlib
```

The model used (recommended in the original paper) is:

```
Qwen/Qwen1.5-GTE-Large
```

To load it efficiently, `accelerate` is also required:

```bash
pip install accelerate
```

## Usage

### Step 1: Train a classifier

```bash
python embedding_classify/train_classifier.py
```

This will process the dataset, extract features using `extract_features.py`, and train an MLP model saved as `fluoroscopy_mlp.joblib`.

### Step 2: Predict on new texts

```bash
python embedding_classify/predict.py
```

The prediction script:
- Loads 5 human and 5 GPT-generated texts (or more)
- Runs Fluoroscopy inference
- Outputs the probability of GPT-origin
- Saves a matplotlib graph to `assets/result.png`

### Step 3: Inspect the model

```bash
python embedding_classify/look_inside_model.py
```

Outputs the structure of the trained `MLPClassifier`.

## Notes

- The dataset used is based on: https://www.kaggle.com/datasets/sunilthite/llm-detect-ai-generated-text-dataset
- Only the columns `text` and `label` are used (`label=1` for GPT-generated).
- You can limit training to a smaller subset via `smaller_dataset.csv` for faster iteration.

## License
i dont know
