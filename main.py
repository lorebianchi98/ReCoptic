import json
import yaml
import argparse
import os
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader
import lightning as L

from src.model import EmbeddingModel
from src.dataset import CopticDataset

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

def unfreeze_last_layers(model, unfreeze_from=-3):
    """
    Unfreezes from 'unfreeze_from' layers of ResNet while keeping the rest frozen.
    Default is to unfreeze from -3 (last 2 convolutional blocks).
    """
    if not hasattr(model, "visual_backbone"):
        raise ValueError("Model does not have a `visual_backbone` attribute.")

    # Freeze all layers first
    for param in model.visual_backbone.parameters():
        param.requires_grad = False

    for layer in model.visual_backbone[unfreeze_from:]:
        for param in layer.parameters():
            param.requires_grad = True

def extract_name_from_path(path):
    """Extract filename without extension from a given path."""
    return os.path.splitext(os.path.basename(path))[0]

def main(args):
    # Extract model and train config names to create a unique experiment name
    model_name = extract_name_from_path(args.model_cfg_path)
    train_name = extract_name_from_path(args.train_cfg_path)
    MODEL_NAME = f"{model_name}-{train_name}"

    # Define directories
    checkpoint_dir = os.path.join("checkpoints", MODEL_NAME)
    log_dir = "logs"

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Seed everything
    L.seed_everything(args.seed, workers=True)  
    
    # Load dataset annotations
    if args.annotation_path.endswith('.json'):
        with open(args.annotation_path, 'r') as f:
            data = json.load(f)
    elif args.annotation_path.endswith('.pt'):
        data = torch.load(args.annotation_path)
    else:
        raise ValueError("Unknown annotations file extension")
    
    # Load model configuration
    with open(args.model_cfg_path, 'r') as f:
        model_cfg_dict = yaml.safe_load(f)
    model_cfg = OmegaConf.create(model_cfg_dict)
    
    # Load training configuration
    with open(args.train_cfg_path, 'r') as f:
        train_cfg_dict = yaml.safe_load(f)
    train_cfg = OmegaConf.create(train_cfg_dict)
    ltype = OmegaConf.select(train_cfg, "train.ltype", default='soft_f1')
    collection_label = True if ltype == 'infonce' else False
    preextracted_features = OmegaConf.select(train_cfg, "train.preextracted_features", default=False)
    freeze_backbone = OmegaConf.select(train_cfg, "train.freeze_backbone", default=False)

    # Load model from model config
    model = EmbeddingModel.from_config(model_cfg, preextracted_features)
    model.configure_loss(ltype)
    if not preextracted_features:
        model = torch.compile(model)  # Apply compilation
        use_preextracted_features_from = None
    else:
        use_preextracted_features_from = model.visual_backbone_type
    
    # Update model hyperparameters
    model.hparams.lr = train_cfg.train.lr
    model.hparams.optimizer = train_cfg.train.get("optimizer", 'adamw')  
    model.hparams.weight_decay = train_cfg.train.get("weight_decay", 0.01)  
    model.hparams.momentum = train_cfg.train.get("momentum", 0.9)  
        
    if type(freeze_backbone) is bool:
        if freeze_backbone:
            for param in model.visual_backbone.parameters():
                param.requires_grad = False
    elif type(freeze_backbone) is int:
        print(f"Unfreezing last {freeze_backbone} blocks from the visual backbone")
        unfreeze_last_layers(model, freeze_backbone)
    
    # Load datasets
    print("Loading Training set...")
    train_dataset = CopticDataset(data, 
                                  base_path=args.base_path,
                                  split="train",
                                  image_transforms=model.image_transforms,
                                  use_preextracted_features_from=use_preextracted_features_from,
                                  collection_label=collection_label)
    print("Loading Validation set...")
    val_dataset = CopticDataset(data, 
                                base_path=args.base_path,
                                split="val",
                                image_transforms=model.image_transforms,
                                use_preextracted_features_from=use_preextracted_features_from,
                                collection_label=collection_label)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=train_cfg.train.batch_size, shuffle=True, num_workers=train_cfg.train.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=train_cfg.train.batch_size, shuffle=False, num_workers=train_cfg.train.num_workers)
    
    print(len(train_dataset))
    # Define checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        save_top_k=2,
        mode="min",
        save_last=True,
    )
    
    # Define TensorBoard logger
    logger = TensorBoardLogger(log_dir, name=MODEL_NAME)

    # Initialize Trainer
    trainer = L.Trainer(
        max_epochs=train_cfg.train.max_epochs,
        accelerator="gpu",
        accumulate_grad_batches=train_cfg.train.gradient_accumulation_steps,
        callbacks=[checkpoint_callback],
        logger=logger,
        log_every_n_steps=1,  # <-- Log at every step
        # strategy='ddp_find_unused_parameters_true'
    )
    
    # Train the model, resuming if a checkpoint is provided
    if args.resume_checkpoint and os.path.exists(args.resume_checkpoint):
        print(f"Resuming training from checkpoint: {args.resume_checkpoint}")
        trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume_checkpoint)
    else:
        trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Coptic Manuscript Embedding Model")
    parser.add_argument("--annotation_path", type=str, default="annotations/coptic_dataset.json", help="Path to dataset annotations JSON file")
    parser.add_argument("--base_path", type=str, default="coptic_dataset", help="Path to the images")
    parser.add_argument("--model_cfg_path", type=str, default="configs/model/resnet152.yaml", help="Path to model configuration YAML file")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--train_cfg_path", type=str, default="configs/train/partial_infonce.yaml", help="Path to training configuration YAML file")
    parser.add_argument("--resume_checkpoint", type=str, default=None, help="Path to checkpoint to resume training from")

    args = parser.parse_args()
    main(args)