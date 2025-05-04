import os
import argparse
from PIL import Image, ImageOps
import torch
from torchvision import transforms
from facenet_pytorch import MTCNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {device}")
mtcnn = MTCNN(image_size=224, keep_all=False, post_process=False, device=device)
transform_list = [
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std =[0.229,0.224,0.225]),
]
transform = transforms.Compose(transform_list)

def processPhoto(self, input_path, out_class_dir, fname):
    # I kept the loop just in case but not used
    for idx in range(1):
        img_tensor = preprocess_image(input_path, mtcnn)
        if img_tensor is None:
            print(f"Warning: No face found in {input_path}, skipping.")
            break
        base, _ = os.path.splitext(fname)

        out_name = f"{base}.pt"
        torch.save(img_tensor, os.path.join(out_class_dir, out_name))


def preprocess_image(img_path, mtcnn):
    try:
        img = Image.open(img_path).convert('RGB')
    except Exception as e:
        print(f"Error opening image {img_path}: {e}")
        return None

    try:
        face_tensor = mtcnn(img)
    except Exception as e:
        print(f"Error processing image {img_path} with MTCNN: {e}")
        face_tensor = None

    if face_tensor is None:
        return None

    try:
        face_img = transforms.ToPILImage()(face_tensor.cpu().detach())
    except Exception as e:
        print(f"Error converting tensor to PIL image for {img_path}: {e}")
        return None

    try:
        return transform(face_img)
    except Exception as e:
        print(f"Error applying final transforms to {img_path}: {e}")
        return None

def processDataset(dataset):

    extensions = ('.jpg', '.jpeg', '.png', '.bmp')

    root = dataset.rstrip('/\\')
    dataset_name = os.path.basename(root)
    output_root = f"Datasets/{dataset_name}_tensor"
    os.makedirs(output_root, exist_ok=True)
    print(f"Processing dataset: {dataset_name}")

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
            processPhoto(input_path, out_class_dir, fname)

    print(f"Preprocessed dataset saved to {output_root}")

def process_single_image(image_path):
    img_tensor = preprocess_image(image_path, mtcnn)
    if img_tensor is None:
        print(f"Warning: Could not process image {image_path}.")
    return img_tensor

def parse_args():
    parser.add_argument('dataset', nargs='+', help='Path to input ImageFolder dataset')

def main():
    args = parse_args()

    preprocessor.processDataset(dataset)

if __name__ == '__main__':
    main()
