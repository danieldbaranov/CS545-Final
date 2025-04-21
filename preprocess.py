#!/usr/bin/env python3
"""
preprocess_faces.py

Preprocess facial images: alignment, resizing, histogram equalization, normalization.
Usage:
    python preprocess_faces.py [--image-size IMAGE_SIZE] [--margin MARGIN]
                               dataset1 dataset2 ...
"""
import os
import argparse
from PIL import Image, ImageOps
import torch
from torchvision import transforms
from facenet_pytorch import MTCNN

class preprocess:

    def __init__(self, datasets, image_size, margin):
        self.datasets = datasets
        self.image_size = image_size
        self.margin = margin
        self.mtcnn = MTCNN(image_size=self.image_size, margin=self.margin,
                           keep_all=False, post_process=False)
        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]
        self.transform = transforms.Compose(transform_list)
        self.extensions = ('.jpg', '.jpeg', '.png', '.bmp')

    def processDataset(self):
        for dataset in self.datasets:
            root = dataset.rstrip('/\\')
            dataset_name = os.path.basename(root)
            output_root = f"{dataset_name}_preprocessed"
            os.makedirs(output_root, exist_ok=True)
            print(f"Processing dataset: {dataset_name}")

            for class_name in os.listdir(root):
                class_dir = os.path.join(root, class_name)
                if not os.path.isdir(class_dir):
                    continue
                out_class_dir = os.path.join(output_root, class_name)
                os.makedirs(out_class_dir, exist_ok=True)

                for fname in os.listdir(class_dir):
                    if not fname.lower().endswith(self.extensions):
                        continue
                    input_path = os.path.join(class_dir, fname)
                    self.processPhoto(input_path, out_class_dir, fname)

            print(f"Preprocessed dataset saved to {output_root}")

    def processPhoto(self, input_path, out_class_dir, fname):
        # Generate one or more versions (loop kept for potential future use)
        for idx in range(1):
            img_tensor = preprocess_image(input_path, self.mtcnn, self.transform)
            if img_tensor is None:
                print(f"Warning: No face found in {input_path}, skipping.")
                break # Skip this file if no face is found
            base, _ = os.path.splitext(fname)
            # Use original filename base for the output tensor
            out_name = f"{base}.pt"
            torch.save(img_tensor, os.path.join(out_class_dir, out_name))

    def process_single_image(self, image_path):
        """
        Processes a single image file and returns the resulting tensor.

        Args:
            image_path (str): The path to the image file.

        Returns:
            torch.Tensor or None: The processed image tensor, or None if processing fails.
        """
        img_tensor = preprocess_image(image_path, self.mtcnn, self.transform)
        if img_tensor is None:
            print(f"Warning: Could not process image {image_path}.")
        return img_tensor


def parse_args():
    parser = argparse.ArgumentParser(
        description='Preprocess ImageFolder datasets for face recognition.'
    )
    parser.add_argument('datasets', nargs='+', help='Paths to input ImageFolder datasets')
    parser.add_argument('--image-size', type=int, default=224, help='Size to resize images (default: 224)')
    parser.add_argument('--margin', type=int, default=0, help='Margin for face cropping (default: 0)')

    return parser.parse_args()

def preprocess_image(img_path, mtcnn, transform, hist_equalize=True):
    try:
        img = Image.open(img_path).convert('RGB')
    except Exception as e:
        print(f"Error opening image {img_path}: {e}")
        return None

    # Detect & align face
    try:
        face_tensor = mtcnn(img)
    except Exception as e:
        # Handle potential errors during MTCNN processing
        print(f"Error processing image {img_path} with MTCNN: {e}")
        # Optionally try to process without MTCNN or return None
        face_tensor = None # Example: skip if MTCNN fails

    if face_tensor is None:
        # It's possible mtcnn returns None if no face is detected or an error occurred
        return None

    # Convert tensor back to PIL Image for equalization
    # Note: mtcnn output is already size [C, H, W] and potentially normalized
    # depending on post_process flag (False here). It's typically in range [0, 1] or [-1, 1].
    # We need to be careful with ToPILImage conversion.
    # Assuming mtcnn output is suitable for ToPILImage directly
    try:
        # Ensure tensor is on CPU and detach if necessary
        face_img = transforms.ToPILImage()(face_tensor.cpu().detach())
    except Exception as e:
        print(f"Error converting tensor to PIL image for {img_path}: {e}")
        return None # Cannot proceed if conversion fails

    # Histogram equalization
    if hist_equalize:
        try:
            face_img = ImageOps.equalize(face_img)
        except Exception as e:
            print(f"Error applying histogram equalization to {img_path}: {e}")
            # Decide whether to continue without equalization or return None
            # return None # Option: skip if equalization fails

    # Apply final transforms (ToTensor, Normalize)
    try:
        return transform(face_img)
    except Exception as e:
        print(f"Error applying final transforms to {img_path}: {e}")
        return None


def main():
    args = parse_args()

    # Instantiate the preprocessor
    preprocessor = preprocess(datasets=args.datasets,
                              image_size=args.image_size,
                              margin=args.margin)

    # Run the preprocessing
    preprocessor.processDataset()

if __name__ == '__main__':
    main()
