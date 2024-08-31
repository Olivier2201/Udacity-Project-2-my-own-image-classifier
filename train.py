
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import json
import argparse

# Argument parsing for command-line options
parser = argparse.ArgumentParser(description='Train a neural network on a dataset')
parser.add_argument('--data_dir', type=str, default='flowers', help='Directory containing the dataset')
parser.add_argument('--save_dir', type=str, default='.', help='Directory to save the model checkpoint')
parser.add_argument('--arch', type=str, default='vgg16_bn', help='Model architecture: vgg16_bn, resnet18')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training')
parser.add_argument('--hidden_units', type=int, default=4096, help='Number of hidden units in classifier')
parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
parser.add_argument('--gpu', action='store_true', help='Use GPU for training')

args = parser.parse_args()

# Load data directories
train_dir = args.data_dir + '/train'
valid_dir = args.data_dir + '/valid'
test_dir = args.data_dir + '/test'

# Define transforms
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

# Load datasets
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

# Define dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

# Load model architecture based on user input
if args.arch == 'vgg16_bn':
    model = models.vgg16_bn(pretrained=True)
elif args.arch == 'resnet18':
    model = models.resnet18(pretrained=True)
    model.fc = nn.Sequential(nn.Linear(512, args.hidden_units),
                             nn.ReLU(),
                             nn.Dropout(0.2),
                             nn.Linear(args.hidden_units, 102),
                             nn.LogSoftmax(dim=1))
else:
    raise ValueError("Model architecture not recognized. Please choose 'vgg16_bn' or 'resnet18'.")

# Freeze parameters for pre-trained model
for param in model.parameters():
    param.requires_grad = False

# Define new classifier for VGG
if args.arch == 'vgg16_bn':
    model.classifier = nn.Sequential(nn.Linear(25088, args.hidden_units),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(args.hidden_units, 102),
                                     nn.LogSoftmax(dim=1))

# Define criterion and optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

# Move model to GPU if available and requested
device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
epochs = args.epochs
steps = 0
running_loss = 0
print_every = 5

for epoch in range(epochs):
    for images, labels in trainloader:
        steps += 1
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if steps % print_every == 0:
            valid_loss = 0
            accuracy = 0
            model.eval()
            
            with torch.no_grad():
                for images, labels in validloader:
                    images, labels = images.to(device), labels.to(device)
                    output = model(images)
                    valid_loss += criterion(output, labels).item()
                    
                    ps = torch.exp(output)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                  f"Validation accuracy: {accuracy/len(validloader):.3f}")
            running_loss = 0
            model.train()

# Save the checkpoint
model.class_to_idx = train_data.class_to_idx
checkpoint = {'arch': args.arch,
              'state_dict': model.state_dict(),
              'class_to_idx': model.class_to_idx,
              'classifier': model.classifier}

torch.save(checkpoint, args.save_dir + '/checkpoint.pth')
