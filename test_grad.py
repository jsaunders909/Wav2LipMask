from models.syncnet import SyncNet_color
import torch
import torch.nn as nn

device = 'cuda'


def get_entropy(p):
    a = p * torch.log2(p + 1e-6)
    b = (1 - p) * torch.log2((1 - p) + 1e-6)
    return -(a + b)


def certainty_loss(a, v):
    d = nn.functional.cosine_similarity(a, v).unsqueeze(1)
    entropy = get_entropy(d)
    return (1 - entropy).mean()

syncnet = SyncNet_color().to(device)
inputs = torch.randn((4, 15, 48, 96), requires_grad=True, device=device)
audios = torch.rand((4, 1, 80, 16), device=device)

a, v = syncnet(audios, inputs)

certainty = certainty_loss(a, v)

certainty.backward()

breakpoint()

