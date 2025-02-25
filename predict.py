import torch
import argparse
from torchvision import models
from PIL import Image
import numpy as np

def process_image(image_path):
    image = Image.open(image_path)
    image = image.resize((256, 256)).crop((16, 16, 240, 240))
    np_image = np.array(image) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    np_image = np_image.transpose((2, 0, 1))
    return torch.tensor(np_image).float().unsqueeze(0)

def predict(args):
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.checkpoint)
    model = models.vgg16(pretrained=True) if checkpoint['arch'] == 'vgg16' else models.resnet50(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.to(device)
    model.eval()
    
    image = process_image(args.image_path).to(device)
    with torch.no_grad():
        output = model(image)
    probs, indices = torch.exp(output).topk(args.top_k)
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    classes = [idx_to_class[idx.item()] for idx in indices[0]]
    
    print("Predictions:")
    for prob, cls in zip(probs[0], classes):
        print(f"{cls}: {prob.item():.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict the class of an image')
    parser.add_argument('image_path', type=str, help='Path to image')
    parser.add_argument('checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K classes')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')
    args = parser.parse_args()
    predict(args)


# python predict.py flowers/test/1/image_06743.jpg checkpoint.pth --top_k 5 --gpu