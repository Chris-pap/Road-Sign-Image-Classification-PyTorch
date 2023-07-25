'''Creates a pretrained mobilenetv3 model and the transforms'''

import torchvision
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


def create_mobilenetv3(num_classes, requires_grad):
    
    """
    Creates a pretrained mobilenetv3 model and the transforms
    requires_grad: True for training all layers, False for training only the last
    Returns:
        (model, train_transform, test_transform)
    """
    
    # model
    weights = torchvision.models.MobileNet_V3_Large_Weights.DEFAULT
    model = torchvision.models.mobilenet_v3_large(weights=weights)

    if requires_grad:
        for param in model.parameters():
            param.requires_grad = True
    else:
        for param in model.parameters():
            param.requires_grad = False
        
    model.classifier[3] = torch.nn.Linear(in_features=1280, out_features=num_classes, bias=True)
    
    
    # transforms
    class train_transform:
        def __init__(self):
            self.transform = A.Compose([
                A.Resize(224, 224),
                A.RandomBrightnessContrast(),
                A.RandomFog(),
                A.RandomRain(),
                A.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        def __call__(self, img):
            return self.transform(image=np.array(img))['image']


    class test_transform:
        def __init__(self):
            self.transform = A.Compose([
                A.Resize(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        def __call__(self, img):
            return self.transform(image=np.array(img))['image']

    train_transform = train_transform() 
    test_transform = test_transform()
    
    # return
    return model, train_transform, test_transform
