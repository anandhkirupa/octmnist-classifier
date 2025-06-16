import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from .model import SimpleCNN

CLASS_LABELS = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

def load_model(weights_path, device=None):
    """
    Load the trained CNN model from a .pt weights file.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleCNN()
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    model.to(device)
    return model

def preprocess_image(image_path):
    """
    Preprocess a grayscale image and return a normalized tensor.
    """
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

def predict(model, image_tensor, device=None, return_probs=False):
    """
    Perform forward pass and return predicted class ID and label.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()
        pred_class = int(np.argmax(probs))
        label = CLASS_LABELS[pred_class]

    if return_probs:
        return pred_class, label, probs
    return pred_class, label