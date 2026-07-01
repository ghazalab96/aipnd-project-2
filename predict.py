import argparse
import json
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import models


def get_args():
    parser = argparse.ArgumentParser(description="Predict flower class from an image")
    parser.add_argument("input", type=str, help="Path to input image")
    parser.add_argument("checkpoint", type=str, help="Path to saved model checkpoint (.pth)")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Return top K most likely classes (default: 5)")
    parser.add_argument("--category_names", type=str, default=None,
                        help="Path to JSON file mapping category indices to flower names")
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU for inference if available")
    return parser.parse_args()


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location="cpu")

    arch = checkpoint.get("arch", "vgg16")
    if arch == "vgg16":
        model = models.vgg16(pretrained=True)
    else:
        model = models.densenet121(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = checkpoint["classifier"]
    model.load_state_dict(checkpoint["model_state_dict"])
    model.class_to_idx = checkpoint["class_to_idx"]
    model.eval()
    return model


def process_image(image_path):
    pil_image = Image.open(image_path).convert("RGB")

    # Resize so shortest side is 256
    pil_image.thumbnail((256, 256))

    # Center-crop to 224x224
    width, height = pil_image.size
    left = (width - 224) / 2
    top = (height - 224) / 2
    pil_image = pil_image.crop((left, top, left + 224, top + 224))

    np_image = np.array(pil_image) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    np_image = np_image.transpose((2, 0, 1))  # HWC -> CHW
    return np_image


def predict(image_path, model, device, topk=5):
    np_image = process_image(image_path)
    tensor = torch.from_numpy(np_image).float().unsqueeze(0).to(device)

    model.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(tensor)

    probs = nn.Softmax(dim=1)(outputs)
    top_probs, top_indices = torch.topk(probs, topk)

    top_probs   = top_probs.cpu().numpy().flatten()
    top_indices = top_indices.cpu().numpy().flatten()

    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_classes  = [idx_to_class[idx] for idx in top_indices]
    return top_probs, top_classes


def main():
    args = get_args()
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

    model = load_checkpoint(args.checkpoint)
    probs, classes = predict(args.input, model, device, args.top_k)

    if args.category_names:
        with open(args.category_names) as f:
            cat_to_name = json.load(f)
        labels = [cat_to_name.get(c, c) for c in classes]
    else:
        labels = classes

    print(f"\nTop {args.top_k} predictions for: {args.input}\n")
    for label, prob in zip(labels, probs):
        print(f"  {label:<30} {prob*100:6.2f}%")


if __name__ == "__main__":
    main()
