import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
import numpy as np


# Load pre-trained ResNet-50 model
model = resnet50(pretrained=True)

# Set the model to evaluation mode
model.eval()

# Define image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load and preprocess the image
img_path = 'statue2.jpg'
img = Image.open(img_path)
img = transform(img)

# Add batch dimension
img = img.unsqueeze(0)

# Extract features using the ResNet-50 model
with torch.no_grad():
    feat = model.forward(img).squeeze()

# Convert the features to a numpy array
feat1 = feat.cpu().numpy()

# print(feat1)

img_path = 'statue1.jpg'
img = Image.open(img_path)
img = transform(img)

# Add batch dimension
img = img.unsqueeze(0)

# Extract features using the ResNet-50 model
with torch.no_grad():
    feat = model.forward(img).squeeze()

# Convert the features to a numpy array
feat2 = feat.cpu().numpy()

# print(feat2)



def calculate_similarity(first_embedding, second_embedding):
    
    dot_product = np.dot(first_embedding, second_embedding)
    norm_v1 = np.linalg.norm(first_embedding)
    norm_v2 = np.linalg.norm(second_embedding)
    cosine_similarity = dot_product / (norm_v1 * norm_v2)
    
    return cosine_similarity

similarity = calculate_similarity(feat1, feat2)
print("Similarity score : ", similarity)

