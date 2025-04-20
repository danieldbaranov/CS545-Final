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
        [--save-path PATH] [--no-pretrained] [--num-workers N]
"""
import os
import argparse
import glob
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from torchvision import models, transforms

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

# --- Custom Dataset ---
class PreprocessedFaceDataset(Dataset):
    def __init__(self, root_dir, class_to_idx):
        self.root_dir = root_dir
        self.class_to_idx = class_to_idx
        self.samples = self._find_samples()
        if not self.samples:
            print(f"Warning: Found 0 files in subfolders of: {root_dir}. Supported extensions are: .pt")

    def _find_samples(self):
        samples = []
        dataset_name = os.path.basename(os.path.normpath(self.root_dir))
        for class_name in os.listdir(self.root_dir):
            target_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(target_dir):
                continue
            unique_class_name = f"{dataset_name}_{class_name}"
            if unique_class_name in self.class_to_idx:
                class_index = self.class_to_idx[unique_class_name]
                for fpath in glob.glob(os.path.join(target_dir, '*.pt')):
                    samples.append((fpath, class_index))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tensor_path, label = self.samples[idx]
        try:
            tensor = torch.load(tensor_path)
            if not isinstance(tensor, torch.FloatTensor):
                try:
                    tensor = tensor.float()
                except Exception as conv_e:
                    print(f"Warning: Could not convert tensor {tensor_path} to FloatTensor: {conv_e}")
        except FileNotFoundError:
            print(f"Error: Tensor file not found: {tensor_path}. Skipping.")
            return None, -1
        except Exception as e:
            print(f"Error loading tensor {tensor_path}: {e}")
            return None, -1
        return tensor, label

# --- Model Definition ---
def create_resnet_model(num_classes, pretrained=True):
    model = models.resnet50(
        weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
    num_ftrs = model.fc.in_features
    # Add dropout before final classification
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_ftrs, num_classes)
    )
    print(f"Created ResNet-50 model with {num_classes} output classes (pretrained={pretrained}).")
    return model

# --- Training Function ---
def train_model(model, dataloaders, criterion, optimizer, num_epochs, device, save_path):
    start_time = time.time()
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    model.to(device)

    # Define augmentation transforms for training
    # Using a fixed crop size of 224 (typical for ResNet)
    augmentation_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomErasing(p=0.5),
    ])

    # Create Epochs directory
    epochs_dir = os.path.join(os.path.dirname(save_path), "Epochs")
    os.makedirs(epochs_dir, exist_ok=True)

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            total_samples = 0
            for batch_idx, (inputs, labels) in enumerate(dataloaders.get(phase, [])):
                if inputs is None or labels is None:
                    print(f"Warning: Skipping batch {batch_idx} in {phase} due to load errors.")
                    continue
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                if phase == 'train':
                    orig_inputs = inputs
                    # Apply augmentations per sample
                    aug_inputs = torch.stack([augmentation_transforms(img) for img in orig_inputs.cpu()]).to(device)
                    inputs = torch.cat([orig_inputs, aug_inputs], dim=0)
                    labels = torch.cat([labels, labels], dim=0)

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                batch_size = inputs.size(0)
                running_loss += loss.item() * batch_size
                running_corrects += torch.sum(preds == labels.data)
                total_samples += batch_size

            if total_samples > 0:
                epoch_loss = running_loss / total_samples
                epoch_acc = running_corrects.double() / total_samples
            else:
                print(f"Warning: No samples processed in {phase} for epoch {epoch+1}.")
                epoch_loss, epoch_acc = 0.0, 0.0

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
                if epoch_acc.item() > best_val_acc:
                    best_val_acc = epoch_acc.item()
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
        description='Train a ResNet model for face recognition using preprocessed tensors'
    )
    parser.add_argument('dataset_paths', type=str, nargs='+', help='Paths to preprocessed dataset directories')
    parser.add_argument('--epochs', type=int, default=25, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training/validation')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.05, help='Weight decay for AdamW')
    parser.add_argument('--val-split', type=float, default=0.2, help='Fraction of data for validation')
    parser.add_argument('--save-path', type=str, default='best_face_recognition_model.pth', help='Path to save best model')
    parser.add_argument('--no-pretrained', action='store_true', help='Disable ImageNet pretrained weights')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of DataLoader workers')
    return parser.parse_args()

# --- Collate Function ---
def collate_fn_skip_none(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    if not batch:
        return None, None
    return torch.utils.data.dataloader.default_collate(batch)

# --- Main Execution ---
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        all_classes, class_to_idx = find_classes_in_dirs(args.dataset_paths)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    num_classes = len(all_classes)
    print(f"Found {num_classes} classes: {all_classes}")

    ds_list, total = [], 0
    for path in args.dataset_paths:
        if not os.path.isdir(path): continue
        ds = PreprocessedFaceDataset(path, class_to_idx)
        if len(ds) > 0:
            ds_list.append(ds)
            total += len(ds)
            print(f" -> {len(ds)} samples in {path}")
    if not ds_list:
        print("Error: No valid samples found.")
        return
    print(f"Total samples: {total}")

    full_ds = ConcatDataset(ds_list)
    num_val = int(args.val_split * len(full_ds))
    num_train = len(full_ds) - num_val
    if num_val > 0:
        train_ds, val_ds = random_split(full_ds, [num_train, num_val])
    else:
        train_ds, val_ds = full_ds, None

    dataloaders = {
        'train': DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.num_workers,
                             pin_memory=(device.type=='cuda'),
                             collate_fn=collate_fn_skip_none)
    }
    if val_ds:
        dataloaders['val'] = DataLoader(val_ds, batch_size=args.batch_size,
                                       shuffle=False, num_workers=args.num_workers,
                                       pin_memory=(device.type=='cuda'),
                                       collate_fn=collate_fn_skip_none)

    model = create_resnet_model(num_classes, pretrained=not args.no_pretrained)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("Starting training...")
    trained_model, history = train_model(
        model, dataloaders, criterion, optimizer,
        args.epochs, device, args.save_path)
    print("Training finished.")

if __name__ == '__main__':
    main()
