import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

from utils.common.network_utils import Flatten, conv_output_size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConvEmbedder(nn.Module):
    def __init__(self, channels, embedding_size, args):
        super().__init__()

        activation = getattr(nn, args.embed_net['activation'])
        self.input_shape = (channels, *args.frame_size)
        self.embedding_size = embedding_size

        if args.pomdp:  # 28x28
            self.conv = nn.Sequential(
                        nn.Conv2d(channels, 32, 4, stride=2),
                        activation(),
                        nn.Conv2d(32, 64, 4, stride=2),
                        activation(),
                        nn.Conv2d(64, 128, 4, stride=2),
                        activation(),
                        nn.Conv2d(128, 256, 1, stride=2),
                        activation(),
                    )
        else:   # 64x64
            self.conv = nn.Sequential(
                nn.Conv2d(channels, 32, 4, stride=2),
                activation(),
                nn.Conv2d(32, 64, 4, stride=2),
                activation(),
                nn.Conv2d(64, 128, 4, stride=2),
                activation(),
                nn.Conv2d(128, 256, 4, stride=2),
                activation(),
                Flatten(),
            )

        # Get number of elements
        conv_output_numel, _ = conv_output_size(self.conv, self.input_shape)
        self.fc = nn.Identity() if embedding_size == conv_output_numel else nn.Linear(conv_output_numel, embedding_size)
        self.net = nn.Sequential(self.conv, self.fc)

    def forward(self, obs):
        obs = obs / 255. - 0.5
        if not isinstance(obs, torch.Tensor):
            obs = torch.FloatTensor(obs).unsqueeze(0).to(device)

        # (seq, batch, c, w, h)
        x = obs.reshape(-1, *self.input_shape)                      # -> (seq*batch, c, w, h)
        x = self.net(x)                                             # -> (seq*batch, obs embed)
        return x.reshape((*obs.shape[:-3], self.embedding_size))    # -> (seq, batch, obs embed)
