from PIL import Image
import cv2
import os
from tqdm import tqdm

import random
import torchvision.transforms as T
from torch.utils.data import Dataset
from src.preprocess import crop_image

from itertools import combinations
import random

def generate_limited_positive_pairs(categories, max_couple_per_category):
    """
    Generates limited positive pairs for each category.

    :param images: List of images (not used directly but needed for indexing)
    :param categories: List of category labels corresponding to images
    :param max_per_category: Integer limit of positive pairs per category
    :return: List of positive pairs (tuples of indices)
    """

    # Group image indices by category
    category_dict = {}
    for idx, cat in enumerate(categories):
        if cat not in category_dict:
            category_dict[cat] = []
        category_dict[cat].append(idx)

    positive_pairs = []

    # Generate limited positive pairs for each category
    for cat, indices in category_dict.items():
        all_pairs = list(combinations(indices, 2))  # Generate all possible pairs
        
        # Limit the number of pairs per category
        if len(all_pairs) > max_couple_per_category:
            all_pairs = random.sample(all_pairs, max_couple_per_category)

        positive_pairs.extend(all_pairs)

    return positive_pairs



class CopticDataset(Dataset):
    def __init__(self, 
                 data,
                 base_path,
                 split,
                 image_transforms,
                 balance_dataset=True,
                 crop_percentage=0.15,
                 collection_label=False,
                 use_preextracted_features_from=None,
                 max_couple_per_category=None,
                 keep_n_images=None):
        """
        use_prextracted_features: str, if setted, the features used will come from this field in data['images']
        """
        # Keeping only samples from the specified split
        split_data = {}
        split_data['images'] = [elem for elem in data['images'] if elem['split'] == split]
        
        if keep_n_images is not None:
            # debug purpose
            random.seed(123)
            split_data['images'] = random.sample(split_data['images'], keep_n_images)
        
        self.use_preextracted_features = use_preextracted_features_from is not None
        # Load images and categories
        if self.use_preextracted_features:
            images = [img[use_preextracted_features_from] for img in split_data['images']]
        else:
            images = [Image.fromarray(cv2.cvtColor((cv2.imread(os.path.join(base_path, elem['filepath']))), cv2.COLOR_BGR2RGB)) for elem in split_data['images']]
            
            self.crop_precentage = crop_percentage
            self.augmentation = T.Compose([
                T.ColorJitter(brightness=0.2, contrast=0.4, saturation=0.2, hue=0.5),
            ])
        self.images = images
        
        self.categories = [elem['collection'] for elem in split_data['images']]
        
        # Generate all pairs (positive and negative)
        if max_couple_per_category is not None:
            self.positive_pairs = generate_limited_positive_pairs(self.categories, max_couple_per_category)
        else:
            self.positive_pairs = [(i, j) for i in range(len(self.images)) for j in range(len(self.images)) if i < j and self.categories[i] == self.categories[j]]
        self.negative_pairs = [(i, j) for i in range(len(self.images)) for j in range(len(self.images)) if i < j and self.categories[i] != self.categories[j]]
        
        # Log statistics
        print(f"N. images: {len(images)}")
        print("Before balancing:")
        print(f"N. positives: {len(self.positive_pairs)}")
        print(f"N. negatives: {len(self.negative_pairs)}")
        
        # Apply balancing if required
        self.balance_dataset = balance_dataset
        self.balance_pairs()
        
        self.image_transforms = image_transforms
        
        self.collection_label = collection_label
        if collection_label:
            self.couples = self.positive_pairs
            coll2id = {coll: i for i, coll in enumerate(set([elem['collection'] for elem in split_data['images']]))}
            self.labels = [coll2id[elem['collection']] for elem in split_data['images']]

    def balance_pairs(self):
        """Ensures an equal number of positive and negative pairs."""
        if self.balance_dataset:
            num_positive = len(self.positive_pairs)
            num_negative = len(self.negative_pairs)
            if num_negative > num_positive:
                self.negative_pairs = random.sample(self.negative_pairs, num_positive)

        # Merge pairs without shuffling
        self.couples = self.positive_pairs + self.negative_pairs
        self.labels = [1] * len(self.positive_pairs) + [0] * len(self.negative_pairs)

    def __len__(self):
        return len(self.couples)

    def __getitem__(self, idx):
        i, j = self.couples[idx]
        if self.use_preextracted_features:
            img1 = self.images[i]
            img2 = self.images[j]
        else:
            img1 = self.__preprocess(self.images[i])
            img2 = self.__preprocess(self.images[j])
        
        if self.collection_label:
            assert self.labels[i] == self.labels[j], "Labels of positive pairs should be the same"
            label = self.labels[i]
        else:
            label = self.labels[idx] 
            
        return img1, img2, label

    def __preprocess(self, img):
        img = self.augmentation(crop_image(img, crop_percentage=self.crop_precentage))
        
        return self.image_transforms(img)