import torch
import torch.nn as nn
import torchvision.models as models

def get_model(num_classes=10, pretrained=True):
    """
    Returns a MobileNetV3-Large model with the final classification layer modified 
    to output the specified number of classes.
    """
    if pretrained:
        # For modern torchvision (>= 0.13)
        try:
            weights = models.MobileNet_V3_Large_Weights.DEFAULT
            model = models.mobilenet_v3_large(weights=weights)
        except AttributeError:
            # Fallback for older torchvision versions
            model = models.mobilenet_v3_large(pretrained=True)
    else:
        model = models.mobilenet_v3_large()
        
    # MobileNetV3 Large classifier has structure:
    # classifier[0]: Linear(in_features=960, out_features=1280)
    # classifier[1]: Hardswish()
    # classifier[2]: Dropout(p=0.2)
    # classifier[3]: Linear(in_features=1280, out_features=num_classes)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    
    return model

if __name__ == "__main__":
    # Small sanity check
    model = get_model(num_classes=10)
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    print("Model output shape:", out.shape)
