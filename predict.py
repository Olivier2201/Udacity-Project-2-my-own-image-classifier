
import torch
from torchvision import models
from torch import nn, optim
from PIL import Image
import json
import numpy as np

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.vgg16_bn(pretrained=True)
    model.classifier = checkpoint['classifier']
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
    model.to('cuda')
    
    img = process_image(image_path)
    img = img.unsqueeze(0).to('cuda')
    
    with torch.no_grad():
        output = model(img)
        ps = torch.exp(output)
        top_p, top_class = ps.topk(topk, dim=1)
        
        idx_to_class = {v: k for k, v in model.class_to_idx.items()}
        top_class = [idx_to_class[i] for i in top_class.cpu().numpy()[0]]
        
        return top_p.cpu().numpy()[0], top_class

# Load the model
model = load_checkpoint('checkpoint.pth')

# Predict the class of an image
image_path = 'flowers/test/1/image_06743.jpg'
probs, classes = predict(image_path, model)

print(probs)
print(classes)
