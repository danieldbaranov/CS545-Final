#!/usr/bin/env python3

print("importing")

import os
import random
import itertools
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from pytorch_metric_learning import losses
from pytorch_metric_learning.miners import TripletMarginMiner
from pytorch_metric_learning.samplers import MPerClassSampler
from sklearn import metrics
from tqdm import tqdm

import torch.nn.functional as F

from torch.hub import load_state_dict_from_url

from evaluate import evaluate_fast, eval_stats

torchvision_augs = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(8, fill=0),
    transforms.ColorJitter(0.15, 0.15, 0.15, 0.03),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
])

# --- DATASET ---
class FaceDataset(Dataset):
    def __init__(self, root_dir, augment=None):
        super().__init__()
        self.samples = []
        classes = sorted(
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        )
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        for cls in classes:
            cls_dir = os.path.join(root_dir, cls)
            for fname in os.listdir(cls_dir):
                if fname.endswith('.pt'):
                    path = os.path.join(cls_dir, fname)
                    label = self.class_to_idx[cls]
                    self.samples.append((path, label))
        self.labels = [lbl for _, lbl in self.samples]
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = torch.load(path)
        if self.augment:
            img = self.augment(img)
        return img, label


# --- MODEL ---
class EmbeddingNet(nn.Module):
    def __init__(self, pretrained=True, embed_dim=512):
        super().__init__()
        self.backbone = models.resnet50(pretrained=False)
        self.backbone.fc = nn.Identity()
        if pretrained:
            state_dict = torch.load('resnet50_casia_webface.pth')
            self.backbone.load_state_dict(state_dict)
            pass
        self.embed_dim = embed_dim
        self.fc = nn.Sequential(
            nn.Linear(2048, embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)


# --- TRAINING LOOP ---
def train_epoch(model, loader, optimizer, scaler, device, miner, loss_func):
    model.train()
    running_loss = 0.0
    for x, labels in tqdm(loader, desc="Training", leave=False):
        x = x.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            embeddings = model(x)
            hard_triplets = miner(embeddings, labels)
            loss = loss_func(embeddings, labels, hard_triplets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
    return running_loss / len(loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default='Dataset_preprocessed')
    parser.add_argument('--eval_dir', type=str, default='Datasets/images_tensor')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--m_per_class', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--freeze_epochs', type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(42)
    random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("loading dataset")

    # DataLoader with balanced sampler
    train_ds = FaceDataset(args.train_dir, augment=torchvision_augs)
    sampler = MPerClassSampler(train_ds.labels, m=args.m_per_class, batch_size=args.batch_size)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )

    eval_ds = FaceDataset(args.eval_dir)
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    #Model, miner, loss, optimizer, scheduler, scaler
    model = EmbeddingNet().to(device)

    """
    model_url = "https://huggingface.co/BooBooWu/Vec2Face/resolve/main/fr_weights/casia-webface-r50.pth"

    model = EmbeddingNet(pretrained=False).to(device)

    print(f"Downloading checkpoint from {model_url}")
    state_dict = load_state_dict_from_url(
        model_url,
        map_location=device,
        progress=True
    )
    model.load_state_dict(state_dict)
    print("Checkpoint loaded")
    """

    miner = TripletMarginMiner(margin=0.2, type_of_triplets='semihard')
    loss_func = losses.TripletMarginLoss(margin=0.2)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda')

    os.makedirs('Epochs', exist_ok=True)

    for epoch in range(1, args.epochs+1):
        print(f"Epoch {epoch}/{args.epochs}")
        if epoch == 4:
            miner = TripletMarginMiner(margin=0.2, type_of_triplets='hard')

        train_loss = train_epoch(model, train_loader, optimizer, scaler, device, miner, loss_func)
        print(f"Train Loss: {train_loss:.4f}")
        #scheduler.step(train_loss)
        scheduler.step()

        if epoch % 5 == 0:
            results = evaluate_fast(model, eval_loader, device)
            print(
                f" Val Acc: {results['val_acc']:.4f},"
                f" Test Acc: {results['test_acc']:.4f},"
                f" AUC: {results['test_auc']:.4f},"
                f" Thr: {results['threshold']:.4f}"
            )
            torch.save(model.state_dict(), f'Epochs/temp_epoch_{epoch}.pt')

    final_res = evaluate_fast(model, eval_loader, device)
    print("Final Evaluation:")
    print(f" Test Acc: {final_res['test_acc']:.4f}")
    print(f" Test AUC: {final_res['test_auc']:.4f}")

    torch.save(model.state_dict(), f'Epochs/model_epoch_{args.epochs}_final.pt')