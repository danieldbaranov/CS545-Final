#!/usr/bin/env python3
"""
train.py

Train a ResNet 101 model for face recognition
using one or more preprocessed tensor datasets.

Assumes dataset structure within each provided path:
    dataset_path_1/
        class1/
            img1.pt
            img2.pt
            ...
        class2/
            img1.pt
            ...
    dataset_path_2/
        class1/ # Can be the same or different from dataset_path_1
            imgA.pt
            ...
        class3/
            imgB.pt
            ...
    ...

Usage:
    python train_face_recognition.py preprocessed_dataset_path_1 [preprocessed_dataset_path_2 ...] [--epochs N] [--batch-size N] [--lr LR] [--val-split SPLIT] [--save-path PATH] [--no-pretrained] [--num-workers N]
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from torchvision import models
import glob
import time
#from collections import defaultdict

# --- Helper Function to find classes in directories ---
def find_classes_in_dirs(dir_list):
    """
    Finds all unique class subdirectories across multiple root directories,
    prefixing each class with its dataset directory name for uniqueness.

    Args:
        dir_list (list): A list of root directory paths.

    Returns:
        tuple: (list of unique class names sorted alphabetically,
                dict mapping class name to index)
    """
    all_classes = set()
    for root_dir in dir_list:
        if not os.path.isdir(root_dir):
            print(f"Warning: Provided dataset path {root_dir} is not a valid directory. Skipping.")
            continue
        dataset_name = os.path.basename(os.path.normpath(root_dir))
        for d in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, d)
            if os.path.isdir(class_dir):
                # Prefix class name with dataset name for uniqueness
                unique_class_name = f"{dataset_name}_{d}"
                all_classes.add(unique_class_name)

    classes = sorted(list(all_classes))
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    if not classes:
         raise FileNotFoundError(f"Could not find any class subdirectories in the provided paths: {dir_list}")

    return classes, class_to_idx


# --- Custom Dataset ---
class PreprocessedFaceDataset(Dataset):
    """
    Custom PyTorch Dataset for loading preprocessed face tensors (.pt files).
    Uses a globally defined class-to-index mapping.
    """
    def __init__(self, root_dir, class_to_idx):
        """
        Args:
            root_dir (string): Directory with class subfolders containing .pt files.
            class_to_idx (dict): The global mapping from class name to index.
        """
        self.root_dir = root_dir
        self.class_to_idx = class_to_idx
        self.samples = self._find_samples()
        self.num_classes = len(class_to_idx)

        if not self.samples:
            print(f"Warning: Found 0 files in subfolders of: {root_dir}. Supported extensions are: .pt")

    def _find_samples(self):
        """Finds all .pt files within the class subdirectories of this root_dir."""
        samples = []
        dataset_name = os.path.basename(os.path.normpath(self.root_dir))
        for class_name in os.listdir(self.root_dir):
            target_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(target_dir):
                continue
            # Prefix class name with dataset name for uniqueness
            unique_class_name = f"{dataset_name}_{class_name}"
            if unique_class_name in self.class_to_idx:
                class_index = self.class_to_idx[unique_class_name]
                for fpath in glob.glob(os.path.join(target_dir, '*.pt')):
                    item = (fpath, class_index)
                    samples.append(item)
        return samples

    def __len__(self):
        """Returns the total number of samples found in this specific dataset directory."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Loads and returns a sample (tensor and label) from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (tensor, label) where tensor is the preprocessed face image
                   and label is the class index based on the global map.
                   Returns (None, -1) on error loading a tensor.
        """
        tensor_path, label = self.samples[idx]
        try:
            # Load the tensor from the .pt file
            tensor = torch.load(tensor_path)
            # Ensure tensor is float (it should be, but good practice)
            if not isinstance(tensor, torch.FloatTensor):
                 # Attempt conversion, warn if it fails but proceed
                 try:
                     tensor = tensor.float()
                 except Exception as conv_e:
                     print(f"Warning: Could not convert tensor {tensor_path} to FloatTensor: {conv_e}")

            # Basic check for tensor shape if needed (e.g., ensure 3 channels)
            # if tensor.shape[0] != 3:
            #    print(f"Warning: Tensor {tensor_path} has unexpected channel dimension: {tensor.shape}. Expected 3.")
            #    return None, -1 # Or attempt to fix/reshape

        except FileNotFoundError:
             print(f"Error: Tensor file not found: {tensor_path}. Skipping.")
             return None, -1 # Indicate an error
        except Exception as e:
            # Print error and return None if loading fails for other reasons
            print(f"Error loading tensor {tensor_path}: {e}")
            # Returning None allows the collate_fn to skip this sample
            return None, -1 # Indicate an error

        return tensor, label

# --- Model Definition ---
def create_resnet_model(num_classes, pretrained=True):
    """
    Creates a ResNet-101 model with a modified final layer.

    Args:
        num_classes (int): The total number of unique identities (classes) across all datasets.
        pretrained (bool): Whether to use weights pre-trained on ImageNet.

    Returns:
        torch.nn.Module: The ResNet model.
    """
    # Using ResNet101 as specified in the original script's docstring
    model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1 if pretrained else None)

    # Get the number of input features for the final fully connected layer
    num_ftrs = model.fc.in_features

    # Replace the final fully connected layer with a new one
    # matching the total number of face identities found
    model.fc = nn.Linear(num_ftrs, num_classes)

    print(f"Created ResNet-101 model with {num_classes} output classes (pretrained={pretrained}).")
    return model

# --- Training Function ---
def train_model(model, dataloaders, criterion, optimizer, num_epochs, device, save_path):
    """
    Trains the face recognition model. (Unchanged from original, but added comments)

    Args:
        model (torch.nn.Module): The model to train.
        dataloaders (dict): Dictionary containing 'train' and 'val' DataLoaders.
        criterion: The loss function.
        optimizer: The optimization algorithm.
        num_epochs (int): Number of epochs to train for.
        device (torch.device): The device to train on ('cuda' or 'cpu').
        save_path (str): Path to save the best model weights.

    Returns:
        tuple: (trained_model, history dict)
    """
    import os  # Ensure os is imported for directory creation
    start_time = time.time()
    best_val_acc = 0.0
    # Dictionary to store training history (loss and accuracy per epoch)
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    # Move model to the specified device (GPU or CPU)
    model.to(device)

    # Create Epochs directory if it doesn't exist
    epochs_dir = os.path.join(os.path.dirname(save_path), "Epochs")
    os.makedirs(epochs_dir, exist_ok=True)

    # Loop over the specified number of epochs
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase (so we can see how we're doing)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Training mode (so dropout/batchnorm works right)
            else:
                model.eval()   # Eval mode (turns off dropout, etc.)

            running_loss = 0.0
            running_corrects = 0
            total_samples_processed = 0 # We'll count how many samples we actually use

            # Go through all the batches from the dataloader
            # dataloaders[phase] gives us batches from the dataset
            for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):
                # If the batch is empty (like if all files failed to load), just skip it
                if inputs is None or labels is None:
                    print(f"Warning: Skipping batch {batch_idx} in {phase} phase due to all items failing load.")
                    continue

                # Move everything to the GPU (or CPU if that's all we have)
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero out the gradients before we do the backward pass
                optimizer.zero_grad()

                # Forward pass (and backward if we're training)
                # Only calculate gradients if we're in training phase
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    # Get the predicted class (the one with the highest score)
                    _, preds = torch.max(outputs, 1)
                    # Calculate the loss for this batch
                    loss = criterion(outputs, labels)

                    # If we're training, do the backward pass and update the weights
                    if phase == 'train':
                        loss.backward() # Compute gradients
                        optimizer.step() # Actually update the weights

                # Statistics calculation for the current batch
                batch_size = inputs.size(0)
                running_loss += loss.item() * batch_size
                running_corrects += torch.sum(preds == labels.data)
                total_samples_processed += batch_size

            # Calculate epoch loss and accuracy based on successfully processed samples
            if total_samples_processed > 0:
                epoch_loss = running_loss / total_samples_processed
                epoch_acc = running_corrects.double() / total_samples_processed
            else:
                # Handle case where no samples were processed in an epoch (e.g., all data failed)
                print(f"Warning: No samples processed in {phase} phase for epoch {epoch+1}.")
                epoch_loss = 0.0
                epoch_acc = 0.0

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Store history for plotting or analysis later
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                # Ensure accuracy is stored as a standard Python float
                history['train_acc'].append(epoch_acc.item() if isinstance(epoch_acc, torch.Tensor) else epoch_acc)
            else: # Validation phase
                history['val_loss'].append(epoch_loss)
                 # Ensure accuracy is stored as a standard Python float
                history['val_acc'].append(epoch_acc.item() if isinstance(epoch_acc, torch.Tensor) else epoch_acc)
                # Save the model if it has the best validation accuracy seen so far
                current_acc = epoch_acc.item() if isinstance(epoch_acc, torch.Tensor) else epoch_acc
                if current_acc > best_val_acc:
                    best_val_acc = current_acc
                    print(f"Saving best model with Val Acc: {best_val_acc:.4f} to {save_path}")
                    # Save the model's state dictionary
                    try:
                        torch.save(model.state_dict(), save_path)
                    except Exception as save_e:
                        print(f"Error saving model: {save_e}")

        # --- Save model at every epoch ---
        epoch_model_path = os.path.join(epochs_dir, f"epoch_{epoch+1}.pth")
        try:
            torch.save(model.state_dict(), epoch_model_path)
            print(f"Saved model for epoch {epoch+1} to {epoch_model_path}")
        except Exception as e:
            print(f"Error saving model for epoch {epoch+1}: {e}")

    # Calculate and print total training time
    time_elapsed = time.time() - start_time
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_val_acc:4f}')

    return model, history

# --- Argument Parser ---
def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Train a ResNet model for face recognition using preprocessed tensors from one or more datasets.'
    )
    parser.add_argument('dataset_paths', type=str, nargs='+', help='Path(s) to the root directory(ies) of the preprocessed dataset(s).')
    parser.add_argument('--epochs', type=int, default=25, help='Number of training epochs (default: 25)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training and validation (default: 32)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for the optimizer (default: 0.001)')
    parser.add_argument('--val-split', type=float, default=0.2, help='Fraction of the *combined* data to use for validation (default: 0.2)')
    parser.add_argument('--save-path', type=str, default='best_face_recognition_model.pth', help='Path to save the best model weights (default: best_face_recognition_model.pth)')
    parser.add_argument('--no-pretrained', action='store_true', help='Do not use ImageNet pretrained weights for ResNet')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of subprocesses to use for data loading (default: 4)')
    return parser.parse_args()

# --- Collate Function ---
def collate_fn_skip_none(batch):
    """
    Custom collate function that filters out samples where data loading failed.
    Needed because __getitem__ can return (None, -1) on error.
    """
    # Filter out samples where the tensor (first element) is None
    batch = list(filter(lambda x: x[0] is not None, batch))
    # If the entire batch was filtered out (e.g., all files in batch had errors)
    if not batch:
        # Return None for both inputs and labels to signal skipping this batch
        return None, None
    # Otherwise, use the default collate function on the filtered batch
    # This will stack the tensors and labels correctly.
    return torch.utils.data.dataloader.default_collate(batch)


# --- Main Execution ---
def main():
    """Main function to orchestrate data loading, model setup, and training."""
    args = parse_args()

    # --- Device Setup ---
    # Use GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading ---
    print(f"Scanning dataset paths: {args.dataset_paths}")
    # 1. Find all unique classes across all provided dataset directories
    try:
        all_classes_list, global_class_to_idx = find_classes_in_dirs(args.dataset_paths)
        num_classes = len(all_classes_list)
        print(f"Found {num_classes} unique classes across all datasets: {all_classes_list}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return # Exit if no classes found

    # 2. Create individual dataset instances using the global mapping
    datasets = []
    total_samples_found = 0
    for path in args.dataset_paths:
         if not os.path.isdir(path):
             # Warning already printed in find_classes_in_dirs, skip here
             continue
         print(f"Loading dataset from: {path}")
         try:
             # Pass the global map to each dataset instance
             dataset = PreprocessedFaceDataset(root_dir=path, class_to_idx=global_class_to_idx)
             if len(dataset) > 0:
                 datasets.append(dataset)
                 print(f" -> Found {len(dataset)} samples in {path}.")
                 total_samples_found += len(dataset)
             else:
                 # Warning printed inside PreprocessedFaceDataset if empty
                 pass
         except Exception as e:
              print(f"Error loading dataset from {path}: {e}. Skipping this path.")

    # Check if any samples were loaded at all
    if not datasets:
        print("Error: No valid samples found in any of the provided dataset paths. Exiting.")
        return
    print(f"\nTotal valid samples found across all datasets: {total_samples_found}")

    # 3. Combine individual datasets into one
    full_dataset = ConcatDataset(datasets)

    # 4. Split the combined dataset into training and validation sets
    num_samples = len(full_dataset)
    # Calculate the number of samples for validation based on the *combined* size
    val_size = int(args.val_split * num_samples)
    # Calculate the number of samples for training
    train_size = num_samples - val_size

    # Ensure both splits have samples
    if train_size <= 0 or val_size <= 0:
         # Adjust split or provide more data if this error occurs
        print(f"Error: Validation split {args.val_split} results in zero samples "
              f"for train ({train_size}) or val ({val_size}) set from the total {num_samples} samples. "
              f"Try adjusting --val-split or ensure datasets are not too small.")
        # Allow continuing if val_size is 0 (train on all data), but warn.
        if train_size <= 0:
            return # Cannot train with no training data
        else:
            print("Warning: Validation set size is 0. Training will proceed without validation.")
            # Set val_size to 0 explicitly if it was calculated as negative or zero
            val_size = 0
            # Adjust train_size to use all data if validation is impossible
            train_size = num_samples


    print(f"Splitting combined dataset: {train_size} train samples, {val_size} validation samples.")
    # Perform the random split on the concatenated dataset
    # Handle the case where val_size might be 0
    if val_size > 0:
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    else:
        # If no validation split, use the full dataset for training
        train_dataset = full_dataset
        val_dataset = None # No validation dataset


    # 5. Create DataLoaders
    # Use the custom collate_fn to handle potential loading errors within batches
    dataloaders = {}
    dataloaders['train'] = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                    num_workers=args.num_workers, collate_fn=collate_fn_skip_none,
                                    pin_memory=True if device.type == 'cuda' else False,
                                    # drop_last=True could be useful if the last batch is small and causes issues
                                    # drop_last=True
                                    )

    # Only create a validation loader if val_dataset exists
    if val_dataset:
        dataloaders['val'] = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, # No shuffle for validation
                                      num_workers=args.num_workers, collate_fn=collate_fn_skip_none,
                                      pin_memory=True if device.type == 'cuda' else False)
    else:
        # Set dataloaders['val'] to None or an empty list if no validation is performed
        # The training loop needs to handle this case.
        # For simplicity here, we'll rely on the training loop checking if 'val' key exists.
        # Let's ensure the training loop handles the absence of 'val' dataloader
        print("No validation dataset created due to split configuration.")


    print(f"DataLoaders created with num_workers={args.num_workers}.")

    # --- Model Setup ---
    # num_classes was determined earlier from all datasets
    model = create_resnet_model(num_classes=num_classes,
                                pretrained=not args.no_pretrained)
    # Move the model to the designated device
    model = model.to(device)

    # --- Loss and Optimizer ---
    # Define the loss function (Cross Entropy is standard for classification)
    criterion = nn.CrossEntropyLoss()
    # Define the optimizer (AdamW is a common choice for better regularization)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    # Example alternative: SGD with momentum
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    print("Starting training...")
    # --- Train ---
    # Modify the training loop call slightly if there's no validation set
    phases_to_run = ['train']
    if 'val' in dataloaders:
        phases_to_run.append('val')
    else:
        # Need to adjust the training loop slightly if no validation
        # For now, the loop handles it by checking if 'val' key exists in dataloaders
        pass

    trained_model, history = train_model(model, dataloaders, criterion, optimizer,
                                         args.epochs, device, args.save_path)

    print("Training finished.")

# Entry point for the script
if __name__ == '__main__':
    main()
