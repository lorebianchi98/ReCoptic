import torch
import os
import argparse
import json
import yaml
import math
import time
import numpy as np
from tqdm import tqdm
from PIL import Image
from omegaconf import OmegaConf
import torchvision.transforms as T

from src.model import EmbeddingModel
from src.preprocess import crop_image
from src.util import get_couples, compute_narr


def load_model(model_path: str, device: str):
    pretrained = False if 'ckpt' in model_path else True

    if not pretrained:
        print("Loading trained model")
        model = EmbeddingModel.load_from_checkpoint(model_path, map_location=device).eval()
    else:
        print("Loading pre-trained model")
        with open(model_path, 'r') as f:
            model_cfg_dict = yaml.safe_load(f)
        model_cfg = OmegaConf.create(model_cfg_dict)
        model = EmbeddingModel.from_config(model_cfg, False).to(device).eval()

    return model, pretrained


def load_images(test_data, base_path, use_augmentation):
    augmentation = T.Compose([
        T.ColorJitter(brightness=0.2, contrast=0.4, saturation=0.2, hue=0.5),
    ])

    print("Loading images...")
    if not use_augmentation:
        pil_imgs = [crop_image(Image.open(os.path.join(base_path, img['collection'], img['filename'])).convert('RGB')) for img in tqdm(test_data['images'])]
    else:
        pil_imgs = [augmentation(crop_image(Image.open(os.path.join(base_path, img['collection'], img['filename']))).convert('RGB')) for img in tqdm(test_data['images'])]

    return pil_imgs


def extract_features(model, pil_imgs, device, batch_size=256):
    print("Preprocessing images...")
    batch_imgs = torch.stack([model.image_transforms(pil_img) for pil_img in tqdm(pil_imgs)])

    print("Extracting features...")
    start_time = time.time()
    with torch.no_grad():
        embeds_list = []
        n_imgs = batch_imgs.shape[0]
        n_batch = math.ceil(n_imgs / batch_size)
        for i in tqdm(range(n_batch)):
            start = i * batch_size
            end = start + batch_size if i < n_batch - 1 else n_imgs
            embeds = model(batch_imgs[start:end].to(device))
            embeds_list.append(embeds)

        embeds = torch.cat(embeds_list)
        embeds /= embeds.norm(dim=1, keepdim=True)
    end_time = time.time()
    print(f"Feature extraction took {end_time - start_time:.1f} seconds")
    return embeds


def compute_similarity(embeds):
    print("Computing similarity...")
    start_time = time.time()
    sims = embeds @ embeds.T
    end_time = time.time()
    print(f"Similarity computation took {end_time - start_time:.1f} seconds")
    return sims


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='checkpoints/resnet152-partial_infonce/last-v1.ckpt')
    parser.add_argument('--dataset_json', type=str, default='annotations/coptic_dataset.json')
    parser.add_argument('--base_path', type=str, default='coptic_dataset')
    parser.add_argument('--include_train_couples', type=bool, default=True)
    parser.add_argument('--include_val_couples', type=bool, default=True)
    parser.add_argument('--use_augmentation', type=bool, default=False)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model, pretrained = load_model(args.model_path, device)

    with open(args.dataset_json, 'r') as f:
        test_data = json.load(f)

    positive_couples, negative_couples, test_data = get_couples(
        test_data, 'test', args.include_train_couples, args.include_val_couples
    )
    couples = positive_couples + negative_couples

    pil_imgs = load_images(test_data, args.base_path, args.use_augmentation)
    embeds = extract_features(model, pil_imgs, device)
    sims = compute_similarity(embeds)

    # Compute ground-truth labels
    collections = [img['collection'] for img in test_data['images']]
    N = len(collections)
    true_labels = torch.zeros(N, N)

    for i in range(N):
        for j in range(N):
            if collections[i] == collections[j]:
                true_labels[i, j] = 1

    # Couples mask
    couples_mask = torch.zeros(N, N)
    couples_mask_symmetric = torch.zeros(N, N)

    if args.include_train_couples:
        adj = torch.zeros(N, N)
        for i, j in positive_couples:
            adj[i, j] = 1
            adj[j, i] = 1
        adj = adj > 0.5
        for i, j in couples:
            couples_mask[i, j] = 1
            couples_mask_symmetric[j, i] = 1
        couples_mask_symmetric = (couples_mask_symmetric + couples_mask) > 0.5
        couples_mask = couples_mask > 0.5
    else:
        couples_mask = None

    print(f"Pre-trained: {pretrained}")
    print(f"Include Train: {args.include_train_couples}")
    print(f"Include Val: {args.include_val_couples}")
    print(f"Total NARR: {compute_narr(true_labels, sims.cpu(), couples_mask):.3f}")


if __name__ == '__main__':
    main()
