# ReCoptic

## Computer Vision for the Reconstruction of Dismembered Coptic Codices

**ReCoptic** provides a PyTorch Lightning implementation of a system for identifying whether pairs of Coptic manuscript pages originate from the same *codex*. The ultimate goal is to support the virtual reconstruction of fragmented Coptic manuscripts through visual similarity.

---

## Installation

To set up the environment:

```bash
conda create --name recoptic python=3.9
conda activate recoptic

pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 \
  --extra-index-url https://download.pytorch.org/whl/cu117

pip install -r requirements.txt
```

---

## Data Preparation

1. Download the Coptic manuscript images from the [Biblioteca Apostolica Vaticana](https://digi.vatlib.it/mss/Borg.copt).
2. Use the notebook `dataset_creator.ipynb` to organize the dataset and generate the appropriate training/validation/test splits.

---

## Training

To train the model:

```bash
python main.py \
  --annotation_path path/to/ann \
  --base_path path/to/imgs \
  --model_cfg configs/model/resnet152.yaml \
  --train_cfg configs/train/partial_infonce.yaml
```

---

## Evaluation

To evaluate a trained model:

```bash
python eval.py \
  --model_path path/to/checkpoint \
  --dataset_path path/to/ann \
  --base_path path/to/imgs
```

---
