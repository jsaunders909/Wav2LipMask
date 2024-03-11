import torch

class UNET(nn.Module):

    def __init__(self):
        super(UNET, self).__init__()
        self.model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
            in_channels=15, out_channels=5, init_features=32, pretrained=True)

    def forward(self, x):
        return self.model(x)