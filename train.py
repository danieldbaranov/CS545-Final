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
from torchvision import models
from torchvision.models import ResNet50_Weights
from pytorch_metric_learning import losses
from pytorch_metric_learning.miners import TripletMarginMiner
from pytorch_metric_learning.samplers import MPerClassSampler
from sklearn import metrics
from tqdm import tqdm

import kornia.augmentation as K

from facenet_pytorch import InceptionResnetV1

import torch.nn.functional as F

from torch.hub import load_state_dict_from_url

kornia_augs = K.AugmentationSequential(
    K.RandomResizedCrop((224,224), scale=(0.9,1.0), ratio=(0.95,1.05), p=1.0),
    K.RandomHorizontalFlip(p=0.5),
    K.RandomRotation(8.0, p=0.5, keepdim=True),
    K.ColorJitter(0.15,0.15,0.15,0.03, p=0.7),
    K.RandomGaussianNoise(std=0.02, p=0.3),
    K.RandomMotionBlur(
        kernel_size=5,
        angle=5.0,
        direction=0.5,
        p=0.2,
        border_type='constant',
        resample='bilinear',
        keepdim=True
    ),
    K.RandomErasing(p=0.25, scale=(0.02,0.1), ratio=(0.3,3.3)),
    data_keys=["input"]
)

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
            img = img.unsqueeze(0)
            img = self.augment(img)
            img = img.squeeze(0)
        img = img.mul(2.0).sub(1.0)
        return img, label


class EmbeddingNet(nn.Module):
    def __init__(self, pretrained=True, embed_dim=512):
        super().__init__()

        backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)

        # remove classification layer
        self.backbone = nn.Sequential(
            *(list(backbone.children())[:-1])
        )
        self.embed_dim = embed_dim

        self.fc = nn.Sequential(
            nn.Linear(2048, embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim)
        ) if embed_dim else nn.Identity()

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)


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

@torch.inference_mode()
def evaluate_fast(model,
                  loader,
                  device,
                  val_split: float = 0.1,
                  cap_pairs: int = 200_000,
                  sample_per_chunk: int = 25_000,
                  vram_frac: float = 0.4,
                  max_chunk: int = 512):

    model.eval()

    embs, labels = [], []
    for x, y in tqdm(loader, desc="Embedding"):
        x = x.to(device)
        e = model(x).half()
        embs.append(e)
        labels.append(y.to(device))
    embs   = torch.cat(embs, 0)
    labels = torch.cat(labels)
    N, _   = embs.shape

    perm      = torch.randperm(N, device=device)
    split_idx = int(N * val_split)
    val_idx   = perm[:split_idx].cpu().numpy()
    test_idx  = perm[split_idx:].cpu().numpy()

    free_mem, _ = torch.cuda.mem_get_info(device)
    target_bytes = free_mem * vram_frac
    c = int(target_bytes / (N * 2))
    chunk_size = max(1, min(c, max_chunk))

    def gather_pairs(idxs):
        pos_list, neg_list = [], []
        idxs = list(idxs)
        for i in range(0, len(idxs), chunk_size):
            q = idxs[i:i+chunk_size]
            dist = torch.cdist(embs[q], embs)           # (len(q), N)
            same = labels[q].unsqueeze(1) == labels.unsqueeze(0)
            pos = dist[same]
            neg = dist[~same]
            if pos.numel() > sample_per_chunk:
                perm_p = torch.randperm(pos.numel(), device=pos.device)[:sample_per_chunk]
                pos    = pos[perm_p]
            if neg.numel() > sample_per_chunk:
                perm_n = torch.randperm(neg.numel(), device=neg.device)[:sample_per_chunk]
                neg    = neg[perm_n]
            pos_list.append(pos.cpu().numpy())
            neg_list.append(neg.cpu().numpy())
            del dist, pos, neg
            torch.cuda.empty_cache()

        pos_all = np.concatenate(pos_list)
        neg_all = np.concatenate(neg_list)
        if pos_all.size > cap_pairs:
            perm_p  = np.random.permutation(pos_all.size)[:cap_pairs]
            pos_all = pos_all.flat[perm_p]
        if neg_all.size > cap_pairs:
            perm_n  = np.random.permutation(neg_all.size)[:cap_pairs]
            neg_all = neg_all.flat[perm_n]
        return pos_all, neg_all

    pos_v, neg_v = gather_pairs(val_idx)
    y_v          = np.concatenate([np.ones(len(pos_v)), np.zeros(len(neg_v))])
    d_v          = np.concatenate([pos_v, neg_v])
    thr_grid     = np.linspace(d_v.min(), d_v.max(), 200)
    best_thr, best_acc = 0.0, 0.0
    for thr in thr_grid:
        acc = ((d_v <= thr) == y_v).mean()
        if acc > best_acc:
            best_acc, best_thr = acc, thr

    pos_t, neg_t = gather_pairs(test_idx)
    y_t          = np.concatenate([np.ones(len(pos_t)), np.zeros(len(neg_t))])
    d_t          = np.concatenate([pos_t, neg_t])
    preds_t      = (d_t <= best_thr).astype(int)
    test_acc     = (preds_t == y_t).mean()
    test_auc     = metrics.roc_auc_score(y_t, -d_t)

    tn, fp, fn, tp = metrics.confusion_matrix(y_t, preds_t, labels=[0,1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn)>0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp)>0 else 0.0
    to  = abs(fpr - fnr)

    return {
        "val_acc":   best_acc,
        "test_acc":  test_acc,
        "test_auc":  test_auc,
        "threshold": best_thr,
        "chunk_size": chunk_size,
        "FPR":       fpr,
        "FNR":       fnr,
        "TO":        to,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default='Dataset_preprocessed')
    parser.add_argument('--eval_dir', type=str, default='images_preprocessed')
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

    train_ds = FaceDataset(args.train_dir, augment=kornia_augs)
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

    model = EmbeddingNet().to(device)

    miner = TripletMarginMiner(margin=0.2, type_of_triplets='semihard')
    loss_func = losses.TripletMarginLoss(margin=0.2)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda')

    os.makedirs('Epochs', exist_ok=True)

    # Training loop
    for epoch in range(1, args.epochs+1):
        print(f"Epoch {epoch}/{args.epochs}")
        if epoch == 4:
            miner = TripletMarginMiner(margin=0.2, type_of_triplets='hard')

        train_loss = train_epoch(model, train_loader, optimizer, scaler, device, miner, loss_func)
        print(f"Train Loss: {train_loss:.4f}")
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