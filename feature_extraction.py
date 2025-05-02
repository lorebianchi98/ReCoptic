import json
import yaml
import argparse
import math
import os
import random
from PIL import Image
from omegaconf import OmegaConf
import torch
from tqdm import tqdm
import lightning as L
import torchvision.transforms as T

from src.model import EmbeddingModel
from src.preprocess import crop_image

def main(args):
    # Seeds everything
    random.seed(args.seed)
    L.seed_everything(args.seed, workers=True)  

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load dataset annotations
    with open(args.annotation_path, 'r') as f:
        data = json.load(f)

    # Load model configuration
    with open(args.model_cfg_path, 'r') as f:
        model_cfg_dict = yaml.safe_load(f)
    model_cfg = OmegaConf.create(model_cfg_dict)

    # Load model from model config
    model = EmbeddingModel.from_config(model_cfg)
    model = torch.compile(model)  # Apply compilation
    model.to(device)
    
    # Define the augmentation to use
    augmentation = T.Compose([
        T.ColorJitter(brightness=0.2, contrast=0.4, saturation=0.2, hue=0.5),
    ])

    n_imgs = len(data['images'])
    n_batch = math.ceil(n_imgs / args.batch_size)
    for i in tqdm(range(n_batch)):
        start = i * args.batch_size
        end = start + args.batch_size if i < n_batch - 1 else n_imgs
        batch_size_ = end - start
        
        raw_imgs = []
        for j in range(start, end):
            pil_img = Image.open(os.path.join(args.base_path, data['images'][j]['collection'], data['images'][j]['filename']))
                            
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            raw_imgs.append((augmentation(crop_image(pil_img))))

        batch_imgs = torch.stack([model.image_transforms(img) for img in raw_imgs]).to(device)
                
        with torch.no_grad():
            outs = model(batch_imgs, get_visual_backbone_feats=True)

        for j in range(batch_size_):
            data['images'][start + j][model_cfg.model.visual_backbone] = outs[j].to('cpu')
    
    # Saving the results  
    torch.save(data, args.out_path)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embedding Extraction")
    parser.add_argument("--annotation_path", type=str, default="annotations/coptic_dataset.json", help="Path to dataset annotations JSON file")
    parser.add_argument("--base_path", type=str, default="coptic_dataset", help="Path to the images")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--out_path", type=str, default="annotations/coptic_dataset.pt", help="Path to the output file")
    parser.add_argument("--model_cfg_path", type=str, default="configs/model/vitb_mlp.yaml", help="Path to model configuration YAML file")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    
    args = parser.parse_args()
    main(args)