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

def parse_args():
    parser = argparse.ArgumentParser(
        description='Preprocess ImageFolder datasets for face recognition.'
    )
    parser.add_argument('datasets', nargs='+', help='Paths to input ImageFolder datasets')
    parser.add_argument('--image-size', type=int, default=224, help='Size to resize images (default: 224)')
    parser.add_argument('--margin', type=int, default=0, help='Margin for face cropping (default: 0)')
    
    return parser.parse_args()

def preprocess_image(img_path, mtcnn, transform, hist_equalize=True):
    img = Image.open(img_path).convert('RGB')

    # Detect & align face
    face_tensor = mtcnn(img)
    if face_tensor is None:
        return None
    face_img = transforms.ToPILImage()(face_tensor)

    # Histogram equalization
    if hist_equalize:
        face_img = ImageOps.equalize(face_img)

    # Apply transforms
    return transform(face_img)

def main():
    args = parse_args()

    # Initialize MTCNN for face detection/alignment
    mtcnn = MTCNN(image_size=args.image_size, margin=args.margin,
                  keep_all=False, post_process=False)

    # Build transform pipeline
    transform_list = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]
    transform = transforms.Compose(transform_list)
    extensions = ('.jpg', '.jpeg', '.png', '.bmp')

    for dataset in args.datasets:
        root = dataset.rstrip('/\\')
        dataset_name = os.path.basename(root)
        output_root = f"{dataset_name}_preprocessed"
        os.makedirs(output_root, exist_ok=True)

        for class_name in os.listdir(root):
            class_dir = os.path.join(root, class_name)
            if not os.path.isdir(class_dir):
                continue
            out_class_dir = os.path.join(output_root, class_name)
            os.makedirs(out_class_dir, exist_ok=True)

            for fname in os.listdir(class_dir):
                if not fname.lower().endswith(extensions):
                    continue
                input_path = os.path.join(class_dir, fname)

                # Generate one or more versions (not really useful right now)
                for idx in range(1):
                    img_tensor = preprocess_image(input_path, mtcnn, transform)
                    if img_tensor is None:
                        print(f"Warning: No face found in {input_path}, skipping.")
                        break
                    base, _ = os.path.splitext(fname)
                    out_name = f"{base}.pt"
                    torch.save(img_tensor, os.path.join(out_class_dir, out_name))

        print(f"Preprocessed dataset saved to {output_root}")

if __name__ == '__main__':
    main()
