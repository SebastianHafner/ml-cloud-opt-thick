import torch

from pathlib import Path
from omegaconf import DictConfig
from torch import nn


def build_model(cfg: DictConfig) -> nn.Module:
    if cfg.model.name == 'mlp5':
        return MLP5(cfg.model.in_channels, cfg.model.out_channels, apply_relu=cfg.model.apply_relu)


def save_model(network, optimizer, epoch, cfg: DictConfig):
    model_name = cfg.model.name
    save_file = Path(cfg.paths.base_path_output) / 'models' / f'{model_name}.pt'
    checkpoint = {
        'epoch': epoch,
        'weights': network.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(checkpoint, save_file)


def load_model(cfg: DictConfig, device: torch.device):
    net = build_model(cfg)
    net.to(device)

    model_name = cfg.model.name
    net_file = Path(cfg.paths.base_path_output) / 'models' / f'{model_name}.pt'

    checkpoint = torch.load(net_file, map_location=device)
    net.load_state_dict(checkpoint['weights'])

    return net


# Simple 5-layer MLP model
class MLP5(nn.Module):
    def __init__(self, input_dim, output_dim=1, hidden_dim=64, apply_relu=True):
        super(MLP5, self).__init__()
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, hidden_dim)
        self.lin4 = nn.Linear(hidden_dim, hidden_dim)
        self.lin5 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.apply_relu = apply_relu

    def forward(self, x):
        x1 = self.lin1(x)
        x1 = self.relu(x1)
        x2 = self.lin2(x1)
        x2 = self.relu(x2)
        x3 = self.lin3(x2)
        x3 = self.relu(x3)
        x4 = self.lin4(x3)
        x4 = self.relu(x4)
        x5 = self.lin5(x4)
        if self.apply_relu:
            x5 = self.relu(x5)
        return x5
