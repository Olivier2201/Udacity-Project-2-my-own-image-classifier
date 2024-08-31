
import torch
from torchvision import models
from torch import nn
from PIL import Image
import json
import numpy as np
import argparse

# Argument parsing for command-line options
parser = argparse.ArgumentParser(description='Predict the class of an image using a pre-trained model')
parser.add_argument('image_path', type=str, help='Path to the image file')
parser.add_argument('checkpoint', type=str, help='Path to the model checkpoint')
parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
parser.add_argument('--category_names', type=str, help='Path to a JSON file mapping categories to real names')
parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')

args = parser.parse_args()

# Load the checkpoint and rebuild the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    if checkpoint['arch'] == 'vgg16_bn':
        model = models.vgg16_bn(pretrained=True)
    elif checkpoint['arch'] == 'resnet18':
        model = models.resnet18(pretrained=True)
        model.fc = checkpoint['classifier']
    else:
        raise ValueError("Model architecture not recognized. Please use 'vgg16_bn' or 'resnet18'.")
    
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def process_image(image_path):
    pil_image = Image.open(image_path)
    pil_image = pil_image.resize((256, 256))
    pil_image = pil_image.crop((16, 16, 240, 240))
    
    np_image = np.array(pil_image) / 255.0
    np_image = (np_image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    np_image = np_image.transpose((2, 0, 1))
    
    return torch.from_numpy(np_image).float()

def predict(image_path, model, topk=5):
    model.eval()
    model.to('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    
    img = process_image(image_path)
    img = img.unsqueeze(0).to('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    
    with torch.no_grad():
        output = model(img)
        ps = torch.exp(output)
        top_p, top_class = ps.topk(topk, dim=1)
        
        idx_to_class = {v: k for k, v in model.class_to_idx.items()}
        top_class = [idx_to_class[i] for i in top_class.cpu().numpy()[0]]
        
        return top_p.cpu().numpy()[0], top_class

# Load the model
model = load_checkpoint(args.checkpoint)

# Predict the class of an image
probs, classes = predict(args.image_path, model, topk=args.top_k)

# Convert class indices to names if provided
if args.category_names:
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    classes = [cat_to_name[i] for i in classes]

print(probs)
print(classes)
