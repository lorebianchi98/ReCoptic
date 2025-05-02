import torch
from src.model import EmbeddingModel

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import yaml
import torchvision.transforms as T
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import os
import json
from src.preprocess import crop_image
import math
import re
import pandas as pd


def plot_roc_auc(fpr, tpr, roc_auc):
    # Plot the ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {roc_auc:.4f}')  # ROC curve
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--')  # Diagonal line (random classifier)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()
    

def top_x_indices(sims, X):
    N = sims.shape[0]
    
    # Create a mask to exclude diagonal elements
    mask = ~torch.eye(N, dtype=torch.bool, device=sims.device)
    
    # Get the indices of the flattened tensor
    values = sims[mask]  # Exclude diagonal values
    indices = torch.nonzero(mask, as_tuple=True)  # Get (row, col) indices
    
    # Sort values in descending order
    sorted_indices = torch.argsort(values, descending=True)[:X]
    
    # Get the corresponding (row, col) pairs
    top_indices = [(indices[0][i].item(), indices[1][i].item()) for i in sorted_indices]
    
    return top_indices


def side_by_side(img1, img2):
    # Ensure both images have the same height
    h = max(img1.height, img2.height)
    img1 = img1.resize((img1.width, h))
    img2 = img2.resize((img2.width, h))
    
    # Create a new image with combined width
    new_width = img1.width + img2.width
    new_img = Image.new("RGB", (new_width, h))

    # Paste images side by side
    new_img.paste(img1, (0, 0))
    new_img.paste(img2, (img1.width, 0))

    return new_img

def top_matches(similarity_matrix, top_k=1):
    """
    Returns the indices of the top-k matching documents for each document in the similarity matrix.
    
    :param similarity_matrix: NxN tensor where entry (i, j) represents similarity between document i and document j.
    :param top_k: Number of top matches to retrieve for each document.
    :return: Tensor of shape (N, top_k) containing the indices of the top-k matches for each document.
    """
    N = similarity_matrix.shape[0]
    
    # Set diagonal to a very low value to avoid selecting itself as the top match
    similarity_matrix = similarity_matrix.clone()
    similarity_matrix.fill_diagonal_(-float('inf'))

    # Get the indices of the top-k highest similarity values
    _, top_indices = torch.topk(similarity_matrix, k=top_k, dim=1)

    return top_indices

augmentation = T.Compose([
    T.ColorJitter(brightness=0.2, contrast=0.4, saturation=0.2, hue=0.5),
])
def compute_narr(true_labels, sims, couples_mask=None):
    n = true_labels.shape[0]
    triu_indices = torch.triu_indices(n, n, offset=1)

    # Filter triu_indices with only relevant couples
    if couples_mask is not None:
        valid_pairs_mask = couples_mask[triu_indices[0], triu_indices[1]]
        triu_indices = triu_indices[:, valid_pairs_mask]

    y_true = true_labels[triu_indices[0], triu_indices[1]]
    y_scores = sims[triu_indices[0], triu_indices[1]]

    sorted_indices = torch.argsort(y_scores, descending=True)
    y_true_sorted = y_true[sorted_indices]

    positive_ranks = torch.nonzero(y_true_sorted).squeeze() + 1  # rank positions (1-based)

    if positive_ranks.numel() == 0:
        return 0.0

    k = len(positive_ranks)  # number of positives
    m = y_true.numel()       # total number of evaluated pairs

    # Compute Average Reciprocal Rank (ARR)
    arr = torch.sum(1.0 / positive_ranks.float()) / k

    # Ideal best-case and worst-case
    z1 = torch.sum(1.0 / torch.arange(1, k + 1).float()) / k
    z2 = torch.sum(1.0 / torch.arange(m - k + 1, m + 1).float()) / k

    narr = (arr - z2) / (z1 - z2)
    return narr.item()

def find_recto_verso_pairs_single_collection(filenames):
    page_map = {}

    for i, name in enumerate(filenames):
        match = re.search(r'_(\d{4})([rv])(?=[_.])', name)
        if match:
            page_num, side = match.groups()
            # key = name[:match.span()[1] - 1] + '_' + name[match.span()[1]:]
            key = name[match.span()[0] - 3:match.span()[1] - 1]
            if key not in page_map:
                page_map[key] = {}
            page_map[key][side] = name

    pairs = []
    for page_num, sides in page_map.items():
        if 'r' in sides and 'v' in sides:
            pairs.append((sides['r'], sides['v']))

    return pairs

def find_recto_verso_pairs(data):
    coll_filenames = {}
    for img in data['images']:
        coll_filenames[img['collection']] = coll_filenames.setdefault(img['collection'], []) + [img['filename']]

    coll_filenames = {k: [s for s in v] for k, v in coll_filenames.items()}
    pairs = [x for f in coll_filenames.values() for x in find_recto_verso_pairs_single_collection(f)]

    # Create a lookup dictionary for fast index retrieval
    filename_to_index = {img['filename']: idx for idx, img in enumerate(data['images'])}

    # Convert pairs to index pairs
    index_pairs = [(filename_to_index[a], filename_to_index[b]) for a, b in pairs]
    
    return index_pairs

def get_couples(data, split=None, include_train_couples=False, include_val_couples=False, verbose=False):
    splits = [split] if type(split) is not list else split
    if include_train_couples:
        splits_supp = ['train']
        if include_val_couples:
            splits_supp += ['val']
        train_positive_couples, train_negative_couples, data = get_couples(data, split=splits_supp, include_train_couples=False, verbose=False)
        splits += splits_supp
    # split_data['images'] = [elem for elem in data['images'] if split is None or elem['split'] in splits]

    categories = [elem['collection'] for elem in data['images']]
    splits_list = [elem['split'] for elem in data['images']]

    # Generate all pairs (positive and negative)
    positive_pairs = [(i, j) for i in range(len(categories)) for j in range(len(categories)) if i < j and categories[i] == categories[j] and (splits_list[i] in splits and splits_list[j] in splits)]
    negative_pairs = [(i, j) for i in range(len(categories)) for j in range(len(categories)) if i < j and categories[i] != categories[j] and (splits_list[i] in splits and splits_list[j] in splits)]

    if include_train_couples:
        positive_pairs = list(set(positive_pairs) - set(train_positive_couples))
        negative_pairs = list(set(negative_pairs) - set(train_negative_couples))
    
        # deleting recto and verso couples
        positive_pairs = list(set(positive_pairs) - set(find_recto_verso_pairs(data)))
    len_x_collection = {}
    for img in data['images']:
        len_x_collection[img['collection']] = len_x_collection.get(img['collection'], 0) + 1

    n_images = sum(len_x_collection.values())
    # print(df)
    if verbose:
        print(f"N. images: {n_images}")
        print(f"N. positives: {len(positive_pairs)}")
        print(f"N. negatives: {len(negative_pairs)}")
    else:
        return positive_pairs, negative_pairs, data