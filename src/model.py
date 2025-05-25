import numpy as np
import torch
import torch.nn as nn
import lightning as L
import torchvision.transforms as T
import torchvision.models as models
import torch.nn.functional as F
from omegaconf import DictConfig
from src.loss import soft_f1_loss, UnidirectionalInfonce

class EmbeddingModel(L.LightningModule):
    def __init__(self, visual_backbone="dinov2_vitb14_reg", n_layers=1, act='relu', 
                 target_dim=768, hidden_layer_size=1024, resize_dim=518, 
                 preextracted_features=False, pretrained=True):
        super().__init__()
        self.save_hyperparameters()

        self.visual_backbone_type = visual_backbone
        self.preextracted_features = preextracted_features

        if 'dino_resnet50' in self.visual_backbone_type:
            self.visual_backbone = torch.hub.load('facebookresearch/dino:main', visual_backbone)
        elif 'dinov2' in self.visual_backbone_type:
            self.visual_backbone = torch.hub.load('facebookresearch/dinov2', visual_backbone)
            self.visual_backbone_embed_dim = self.visual_backbone.embed_dim
        elif 'resnet' in self.visual_backbone_type:
            resnet_model = getattr(models, self.visual_backbone_type)(pretrained=pretrained)
            self.visual_backbone = nn.Sequential(*list(resnet_model.children())[:-1])  # Remove FC layer
            self.visual_backbone_embed_dim = resnet_model.fc.in_features
        else:
            raise ValueError(f"Unsupported backbone: {self.visual_backbone_type}")

        self.image_transforms = T.Compose([
            T.Resize((resize_dim, resize_dim), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        # Define activation function
        self.act = self.get_activation(act)

        # Define MLP head
        if n_layers > 0:
            layers = []
            for i in range(n_layers):
                in_feats = self.visual_backbone_embed_dim if i == 0 else hidden_layer_size
                out_feats = target_dim if i == (n_layers - 1) else hidden_layer_size
                layers.append(nn.Linear(in_feats, out_feats, bias=True))
                if i < n_layers - 1:
                    layers.append(self.act)
            
            self.mlp_head = nn.Sequential(*layers)
        else:
            self.mlp_head = None

        if self.preextracted_features:
            del self.visual_backbone  # Remove backbone if features are precomputed


    @classmethod
    def from_config(cls, cfg: DictConfig, preextracted_features=False):
        return cls(**cfg.model, preextracted_features=preextracted_features)
    
    def forward(self, x, get_visual_backbone_feats=False):
        """Forward pass"""
        if not self.preextracted_features:
            x = self.visual_backbone(x)
            x = x.view(x.size(0), -1).contiguous()  # Flatten for ResNet
        
        if get_visual_backbone_feats:
            return x

        if self.mlp_head:
            x = self.mlp_head(x)

        return x


    def get_activation(self, act):
        """Returns the corresponding activation function."""
        activation_functions = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(negative_slope=0.01),
            'gelu': nn.GELU(),
            'elu': nn.ELU(alpha=1.0),
            'selu': nn.SELU(),
            'swish': nn.SiLU(),
            'softplus': nn.Softplus(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid()
        }
        return activation_functions.get(act.lower(), nn.ReLU())

    def compute_similarity(self, emb1, emb2):
        """Compute cosine similarity between two embeddings"""
        return (F.cosine_similarity(emb1, emb2, dim=-1) + 1) / 2

    def training_step(self, batch, batch_idx):
        img1, img2, label = batch
        if self.ltype != 'infonce' and self.ltype != 'sigmoid':
            # Generate embeddings
            embedding1 = self(img1)
            embedding2 = self(img2)
            
            # Compute similarity
            sims = self.compute_similarity(embedding1, embedding2)
            true_labels = label.float()  

        else:
            combined_imgs = torch.stack((img1, img2), dim=1)
            imgs = combined_imgs.view(-1, *img1.shape[1:]) 
            label = label.repeat_interleave(2) 
            
            # preparing the labels (1 if same collections 0 otherwise)
            N = label.shape[0]
            true_labels = (label.unsqueeze(1) == label.unsqueeze(0)).int()
            
            # getting embeddings
            embeds = self(imgs)
            
            # calculate cosine similarity matrix
            embeds = F.normalize(embeds, p=2, dim=1)
            sims = embeds @ embeds.T
            
            # deleting elements on the diagonal (self-similarity)
            if self.ltype == 'infonce':
                mask = ~torch.eye(N).bool()
                true_labels = true_labels[mask].view(N, N - 1).float()
                # set the true label for each row to 1 / n_positive
                true_labels /= true_labels.sum(dim = 1).unsqueeze(1)
                sims = sims[mask].view(N, N - 1)
            if self.ltype == 'sigmoid':
                # Extract upper triangular indices (excluding diagonal)
                idx = torch.triu_indices(N, N, offset=1)
                i, j = idx[0], idx[1]

                # Labels: 1 if same collection, 0 otherwise
                true_labels = true_labels[i, j].float()

                # Similarity logits (not passed through sigmoid, suitable for BCEWithLogitsLoss)
                logits = sims[i, j]

                # Compute class imbalance for pos_weight
                n_positive = true_labels.sum()
                n_total = true_labels.numel()
                n_negative = n_total - n_positive

                # Avoid division by zero
                pos_weight = torch.tensor([n_negative / n_positive]) if n_positive > 0 else torch.tensor([1.0])

                # Binary Cross-Entropy loss with logits
                criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(logits.device))
                loss = criterion(logits, true_labels)
        if self.ltype != 'sigmoid':
            # Compute loss
            loss = self.loss_fn(sims, true_labels)
        
        if self.ltype == "infonce":
            loss = loss / N

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss
    def validation_step(self, batch, batch_idx):
        img1, img2, label = batch

        if self.ltype != 'infonce' and self.ltype != 'sigmoid':
            # Generate embeddings
            embedding1 = self(img1)
            embedding2 = self(img2)

            # Compute similarity
            similarity = self.compute_similarity(embedding1, embedding2)

            # Compute loss
            label = label.float()
            loss = self.loss_fn(similarity, label)

        else:
            # Flatten batch
            combined_imgs = torch.stack((img1, img2), dim=1)
            imgs = combined_imgs.view(-1, *img1.shape[1:])
            label = label.repeat_interleave(2)

            N = label.shape[0]
            true_labels = (label.unsqueeze(1) == label.unsqueeze(0)).int()

            # Embeddings and similarity
            embeds = self(imgs)
            embeds = F.normalize(embeds, p=2, dim=1)
            sims = embeds @ embeds.T

            if self.ltype == 'infonce':
                mask = ~torch.eye(N).bool()
                true_labels = true_labels[mask].view(N, N - 1).float()
                true_labels /= true_labels.sum(dim=1).unsqueeze(1)
                sims = sims[mask].view(N, N - 1)
                loss = self.loss_fn(sims, true_labels)
                loss = loss / N

            elif self.ltype == 'sigmoid':
                # Upper triangular mask (excluding diagonal)
                idx = torch.triu_indices(N, N, offset=1)
                i, j = idx[0], idx[1]

                true_labels = true_labels[i, j].float()
                logits = sims[i, j]

                # Compute class imbalance
                n_positive = true_labels.sum()
                n_total = true_labels.numel()
                n_negative = n_total - n_positive

                pos_weight = torch.tensor([n_negative / n_positive]) if n_positive > 0 else torch.tensor([1.0])

                criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(logits.device))
                loss = criterion(logits, true_labels)

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        return loss


    def configure_loss(self, ltype='soft_f1'):
        self.ltype = ltype
        if ltype == 'soft_f1':
            self.loss_fn = soft_f1_loss
        elif ltype == 'BCE':
            self.loss_fn = nn.BCEWithLogitsLoss()
        elif ltype == 'infonce':
            loss_obj = UnidirectionalInfonce()
            self.loss_fn = loss_obj.forward
        elif ltype == 'sigmoid':
            self.loss_fn = None
        else:
            raise Exception("Unimplemented loss function")
        
        
            

    def configure_optimizers(self):
        if self.hparams.optimizer == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        elif self.hparams.optimizer == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr, momentum=self.hparams.momentum, weight_decay=self.hparams.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {self.hparams.optimizer}")

        return optimizer

