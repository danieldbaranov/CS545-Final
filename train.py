#!/usr/bin/env python3
"""
train_face_recognition.py

Train a ResNet-50 model for face recognition
using one or more preprocessed tensor datasets.

Assumes dataset structure within each provided path:
    dataset_path_1/
        class1/
            img1.pt
            img2.pt
            ...

Usage:
    python train_face_recognition.py preprocessed_dataset_path_1 [preprocessed_dataset_path_2 ...] \
        [--epochs N] [--batch-size N] [--lr LR] [--weight-decay WD] [--val-split SPLIT] \
        [--save-path PATH] [--no-pretrained] [--num-workers N] [--log-dir DIR]
"""
import os
import argparse
import glob
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

# --- Helper Function to find classes in directories ---
def find_classes_in_dirs(dir_list):
    all_classes = set()
    for root_dir in dir_list:
        if not os.path.isdir(root_dir):
            print(f"Warning: Provided dataset path {root_dir} is not a valid directory. Skipping.")
            continue
        dataset_name = os.path.basename(os.path.normpath(root_dir))
        for d in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, d)
            if os.path.isdir(class_dir):
                unique_class_name = f"{dataset_name}_{d}"
                all_classes.add(unique_class_name)
    classes = sorted(list(all_classes))
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    if not classes:
        raise FileNotFoundError(
            f"Could not find any class subdirectories in the provided paths: {dir_list}")
    return classes, class_to_idx

# --- Dataset from raw tensor files ---
class FaceTensorDataset(Dataset):
    def __init__(self, samples, transform=None):
        """
        samples: list of (tensor_path, label)
        transform: torchvision transforms to apply to each tensor
        """
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tensor_path, label = self.samples[idx]
        tensor = torch.load(tensor_path)
        if not isinstance(tensor, torch.FloatTensor):
            tensor = tensor.float()
        # If tensors are HWC instead of CHW, uncomment:
        # if tensor.ndim == 3 and tensor.shape[-1] in (1,3):
        #     tensor = tensor.permute(2,0,1)
        if self.transform:
            tensor = self.transform(tensor)
        return tensor, label

# --- Model Definition ---
def create_resnet_model(num_classes, pretrained=True):
    model = models.resnet50(
        weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_ftrs, num_classes)
    )
    print(f"Created ResNet-50 model with {num_classes} output classes (pretrained={pretrained}).")
    return model

# --- Training Function (with mixed precision & TensorBoard logging) ---
def train_model(model, dataloaders, criterion, optimizer, num_epochs, device, save_path, writer):
    start_time = time.time()
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    model.to(device)

    scaler = GradScaler()
    epochs_dir = os.path.join(os.path.dirname(save_path), "Epochs")
    os.makedirs(epochs_dir, exist_ok=True)

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss = 0.0
            running_corrects = 0
            total_samples = 0

            for inputs, labels in dataloaders.get(phase, []):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                if phase == 'train':
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()

                batch_size = inputs.size(0)
                running_loss += loss.item() * batch_size
                running_corrects += torch.sum(preds == labels.data).item()
                total_samples += batch_size

            epoch_loss = running_loss / total_samples if total_samples else 0.0
            epoch_acc = running_corrects / total_samples if total_samples else 0.0
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Log to history & TensorBoard
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc)
                writer.add_scalar('Loss/train', epoch_loss, epoch+1)
                writer.add_scalar('Accuracy/train', epoch_acc, epoch+1)
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc)
                writer.add_scalar('Loss/val', epoch_loss, epoch+1)
                writer.add_scalar('Accuracy/val', epoch_acc, epoch+1)
                if epoch_acc > best_val_acc:
                    best_val_acc = epoch_acc
                    print(f"Saving best model with Val Acc: {best_val_acc:.4f} to {save_path}")
                    torch.save(model.state_dict(), save_path)

        # Save per-epoch checkpoint
        epoch_path = os.path.join(epochs_dir, f"epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), epoch_path)
        print(f"Saved epoch {epoch+1} model to {epoch_path}")

    elapsed = time.time() - start_time
    print(f'\nTraining complete in {elapsed // 60:.0f}m {elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_val_acc:.4f}')
    return model, history

# --- Argument Parser ---
def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a ResNet-50 model for face recognition using preprocessed tensors'
    )
    parser.add_argument('dataset_paths', type=str, nargs='+', help='Paths to preprocessed dataset directories')
    parser.add_argument('--epochs', type=int, default=25, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training/validation')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='Weight decay for AdamW')
    parser.add_argument('--val-split', type=float, default=0.2, help='Fraction of data for validation')
    parser.add_argument('--save-path', type=str, default='best_face_recognition_model.pth', help='Path to save best model')
    parser.add_argument('--no-pretrained', action='store_true', help='Disable ImageNet pretrained weights')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of DataLoader workers')
    parser.add_argument('--log-dir', type=str, default='runs', help='TensorBoard log directory')
    return parser.parse_args()

# --- Main Execution ---
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # TensorBoard setup
    writer = SummaryWriter(log_dir=args.log_dir)

    try:
        all_classes, class_to_idx = find_classes_in_dirs(args.dataset_paths)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    num_classes = len(all_classes)
    print(f"Found {num_classes} classes: {all_classes}")

    # Gather all tensor paths and labels
    samples = []
    for root_dir in args.dataset_paths:
        dataset_name = os.path.basename(os.path.normpath(root_dir))
        for class_name in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            unique_class = f"{dataset_name}_{class_name}"
            if unique_class not in class_to_idx:
                continue
            label = class_to_idx[unique_class]
            for fpath in glob.glob(os.path.join(class_dir, '*.pt')):
                samples.append((fpath, label))
    if not samples:
        print("Error: No tensor files found in provided paths.")
        return

    random.shuffle(samples)
    num_val = int(len(samples) * args.val_split)
    train_samples = samples[num_val:]
    val_samples   = samples[:num_val]

    # Data transforms
    IMG_MEAN = [0.485, 0.456, 0.406]
    IMG_STD  = [0.229, 0.224, 0.225]
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224, scale=(0.7,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3,0.3,0.3,0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMG_MEAN, IMG_STD),
        transforms.RandomErasing(0.5),
    ])
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMG_MEAN, IMG_STD),
    ])

    # Datasets & loaders
    train_ds = FaceTensorDataset(train_samples, transform=train_transform)
    val_ds   = FaceTensorDataset(val_samples,   transform=val_transform) if val_samples else None
    dataloaders = {
        'train': DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, pin_memory=(device.type=='cuda')),
    }
    if val_ds:
        dataloaders['val'] = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                        num_workers=args.num_workers, pin_memory=(device.type=='cuda'))

    # Build model, criterion, optimizer
    model = create_resnet_model(num_classes, pretrained=not args.no_pretrained)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("Starting training...")
    trained_model, history = train_model(
        model, dataloaders, criterion, optimizer,
        args.epochs, device, args.save_path, writer)
    print("Training finished.")

    writer.close()

if __name__ == '__main__':
    main()
