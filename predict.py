import os
import sys
import torch
from PIL import Image
from torchvision import transforms
from model import get_model

def load_classes(classes_path):
    if not os.path.exists(classes_path):
        raise FileNotFoundError(f"Classes file not found at: {classes_path}")
    with open(classes_path, "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f.readlines() if line.strip()]
    return classes

def predict_image(image_path, model_path="best_model.pth", classes_path="classes.txt"):
    # 1. Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Load classes
    try:
        classes = load_classes(classes_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None
        
    num_classes = len(classes)
    
    # 3. Load model structure and weights
    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at: {model_path}. Please run train.py first.")
        return None
        
    model = get_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # 4. Image preprocessing
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        return None
        
    input_tensor = test_transform(image).unsqueeze(0).to(device)
    
    # 5. Inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
        # Get top class and probability
        conf, class_idx = torch.max(probabilities, 0)
        predicted_class = classes[class_idx.item()]
        confidence = conf.item() * 100
        
    return predicted_class, confidence

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image>")
        sys.exit(1)
        
    img_path = sys.argv[1]
    result = predict_image(img_path)
    if result:
        predicted_class, confidence = result
        print(f"\nPredicted: {predicted_class}")
        print(f"Confidence: {confidence:.2f}%")
