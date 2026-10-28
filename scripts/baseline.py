import torch, torchvision

if __name__ == "__main()__":
    m = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    m.fc = torch.nn.Linear(m.fc.in_features, 1)   # logit: spoof vs live
