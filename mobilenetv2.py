import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

class FaceEmbedder(nn.Module):
    def __init__(self, embedding_dim=128):
        super(FaceEmbedder, self).__init__()
        base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.backbone = base.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base.last_channel, embedding_dim)
        self.l2norm = nn.functional.normalize

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x).view(x.size(0), -1)
        x = self.fc(x)
        x = self.l2norm(x, p=2, dim=1)  # normalize embeddings
        return x

# --------------------- Example Usage ---------------------
if __name__ == "__main__":
    model = FaceEmbedder(embedding_dim=128)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    img = Image.open("face_sample.jpg").convert("RGB")
    x = preprocess(img).unsqueeze(0)  # shape: (1, 3, 224, 224)

    with torch.no_grad():
        embedding = model(x).cpu().numpy()[0]

    print("Embedding shape:", embedding.shape)
    print("Example vector (first 5):", embedding[:5])
