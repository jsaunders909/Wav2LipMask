import torch
from torch import nn
from torch.nn import functional as F
import math

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.conv import Conv2dTranspose, Conv2d, nonorm_Conv2d


class UNETMask(nn.Module):
    def __init__(self):
        super(UNETMask, self).__init__()

        self.face_encoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(3, 16, kernel_size=7, stride=1, padding=3)),  # 48,96

            nn.Sequential(Conv2d(16, 32, kernel_size=3, stride=(1, 2), padding=1),  # 48,48
                          Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 24,24
                          Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 12,12
                          Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 6,6
                          Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # 3,3
                          Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True), ),

            nn.Sequential(Conv2d(512, 512, kernel_size=3, stride=1, padding=0),  # 1, 1
                          Conv2d(512, 512, kernel_size=1, stride=1, padding=0)), ])

        self.face_decoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(512, 512, kernel_size=1, stride=1, padding=0), ),

            nn.Sequential(nn.Upsample(scale_factor=3),
                          Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),  # 3,3
                          Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True), ),

            nn.Sequential(nn.Upsample(scale_factor=2),
                          Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
                          Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True), ),  # 6, 6

            nn.Sequential(nn.Upsample(scale_factor=2),
                          Conv2d(768, 384, kernel_size=3, stride=1, padding=1),
                          Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True), ),  # 12, 12

            nn.Sequential(nn.Upsample(scale_factor=2),
                          Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
                          Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True), ),  # 24, 24

            nn.Sequential(nn.Upsample(scale_factor=2),
                          Conv2d(320, 128, kernel_size=3, stride=1, padding=1),
                          Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True), ),  # 48, 48

            nn.Sequential(nn.Upsample(scale_factor=(1, 2)),
                Conv2d(160, 64, kernel_size=3, stride=1, padding=1),
                          Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True), ), ])  # 96,96

        self.output_block = nn.Sequential(Conv2d(80, 32, kernel_size=3, stride=1, padding=1),
                                          nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
                                          nn.Sigmoid())

    def forward(self, face_sequences):
        # audio_sequences = (B, T, 1, 80, 16)
        B = face_sequences.shape[0]
        input_dim_size = len(face_sequences.size())
        if input_dim_size > 4:
            face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)

        feats = []
        x = face_sequences
        for f in self.face_encoder_blocks:
            x = f(x)
            feats.append(x)

        for f in self.face_decoder_blocks:
            x = f(x)
            try:
                x = torch.cat((x, feats[-1]), dim=1)
            except Exception as e:
                print(x.size())
                print(feats[-1].size())
                raise e

            feats.pop()

        x = self.output_block(x)

        if input_dim_size > 4:
            x = torch.split(x, B, dim=0)  # [(B, C, H, W)]
            outputs = torch.stack(x, dim=2)  # (B, C, T, H, W)

        else:
            outputs = x

        return outputs

if __name__ == '__main__':
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from models.syncnet import SyncNet_color

    logloss = nn.BCELoss()
    def cosine_loss(a, v, y):
        d = nn.functional.cosine_similarity(a, v)
        loss = logloss(d.unsqueeze(1), y)

        return loss
    def certainty_loss(a, v):
        d = nn.functional.cosine_similarity(a, v).unsqueeze(1)
        y = 0.5 * torch.ones_like(d)
        loss = logloss(d, y)
        return loss

    syncnet = SyncNet_color()
    unet = UNETMask()

    inputs = torch.zeros((4, 15, 48, 96))
    audios = torch.zeros((4, 1, 80, 16))

    B, t3, w, h = inputs.shape
    inputs_ind = inputs.reshape(B, t3//3, 3, w, h).reshape(-1, 3, w, h)
    mask = unet(inputs_ind).reshape(B, t3//3, 1, w, h).repeat(1, 1, 3, 1, 1).reshape(B, t3, w, h)
    inputs = inputs * mask

    a, v = syncnet(audios, inputs)
    loss = certainty_loss(a, v)

    print(a.shape, v.shape)
