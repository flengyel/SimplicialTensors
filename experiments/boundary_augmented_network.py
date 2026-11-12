"""Compare Gaussian regression runs with matched relative weight perturbations."""

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from simplicial_tensors.tensor_ops import bdry_mod1, degen

if __package__:
    from .bdry_network_experiment1 import OriginalNet as _OriginalNet, _relative_update
else:
    from bdry_network_experiment1 import OriginalNet as _OriginalNet, _relative_update


class OriginalNet(_OriginalNet):
    def __init__(self, input_size, hidden_sizes, output_size, negative_slope=0.01):
        super().__init__(input_size, hidden_sizes, output_size)
        self.activation = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, x):
        for layer in (self.fc1, self.fc2, self.fc3, self.fc4):
            x = self.activation(layer(x))
        return self.fc5(x)


class BoundaryAugmentedNet(OriginalNet):
    def __init__(self, input_size, hidden_sizes, output_size, boundary_scale=1e-5, negative_slope=0.01):
        super().__init__(input_size, hidden_sizes, output_size, negative_slope)
        if boundary_scale < 0:
            raise ValueError('boundary_scale must be nonnegative')
        self.boundary_scale = boundary_scale

    def integrate_boundary(self):
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name in ('fc3.weight', 'fc4.weight', 'fc5.weight') and min(param.shape) >= 2:
                    boundary = bdry_mod1(param.detach().cpu().numpy())
                    direction = degen(boundary, min(boundary.shape) // 2)
                    _relative_update(param, direction, self.boundary_scale)


class RandomTensorAugmentedNet(OriginalNet):
    def __init__(self, input_size, hidden_sizes, output_size, random_tensor_scale=1e-5,
                 negative_slope=0.01, seed=42):
        super().__init__(input_size, hidden_sizes, output_size, negative_slope)
        if random_tensor_scale < 0:
            raise ValueError('random_tensor_scale must be nonnegative')
        self.random_tensor_scale = random_tensor_scale
        self.rng = np.random.default_rng(seed)

    def integrate_random_tensor(self):
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name in ('fc3.weight', 'fc4.weight', 'fc5.weight') and min(param.shape) >= 2:
                    shape = tuple(dim - 1 for dim in param.shape)
                    random_tensor = self.rng.standard_normal(shape)
                    direction = degen(random_tensor, min(shape) // 2)
                    _relative_update(param, direction, self.random_tensor_scale)


def train_network(net, criterion, optimizer, scheduler, num_epochs=40, augment_type=None,
                  *, batch_size=32, seed=123):
    if num_epochs < 1 or batch_size < 1:
        raise ValueError('epochs and batch size must be positive')
    generator = torch.Generator().manual_seed(seed)
    parameter = next(net.parameters())
    history = []
    for epoch in range(num_epochs):
        inputs = torch.randn(batch_size, net.fc1.in_features, generator=generator).to(parameter)
        labels = torch.randn(batch_size, net.fc5.out_features, generator=generator).to(parameter)
        optimizer.zero_grad()
        loss = criterion(net(inputs), labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()
        # Mutations occur only after the forward/backward/optimizer sequence.
        if augment_type == 'boundary' and isinstance(net, BoundaryAugmentedNet):
            net.integrate_boundary()
        elif augment_type == 'random' and isinstance(net, RandomTensorAugmentedNet):
            net.integrate_random_tensor()
        if scheduler is not None:
            scheduler.step()
        history.append(float(loss.detach()))
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {history[-1]:.6f}')
    return history


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--relative-norm', type=float, default=1e-5)
    parser.add_argument('--small', action='store_true', help='Use widths 8, 12, 10, 8, 6, 3')
    args = parser.parse_args(argv)
    widths = (8, [12, 10, 8, 6], 3) if args.small else (64, [256, 128, 64, 32], 10)
    histories = {}
    for name, cls in [('baseline', OriginalNet), ('boundary', BoundaryAugmentedNet),
                      ('random', RandomTensorAugmentedNet)]:
        torch.manual_seed(args.seed)
        kwargs = {'boundary_scale': args.relative_norm} if name == 'boundary' else {}
        if name == 'random':
            kwargs = {'random_tensor_scale': args.relative_norm, 'seed': args.seed}
        net = cls(*widths, **kwargs)
        optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        print(f'Training {name}, seed={args.seed}, widths={widths}, batch_size={args.batch_size}, '
              f'relative_norm={args.relative_norm}')
        histories[name] = train_network(net, nn.MSELoss(), optimizer, scheduler, args.epochs,
                                        augment_type=name, batch_size=args.batch_size, seed=args.seed)
    return histories


if __name__ == '__main__':
    main()
