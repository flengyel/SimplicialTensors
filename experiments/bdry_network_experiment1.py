"""Train Gaussian regression prototypes with an additive boundary perturbation."""

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from simplicial_tensors.tensor_ops import bdry, degen


class OriginalNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        if len(hidden_sizes) != 4:
            raise ValueError('Four hidden layer widths are required')
        widths = [input_size, *hidden_sizes, output_size]
        for i in range(5):
            setattr(self, f'fc{i + 1}', nn.Linear(widths[i], widths[i + 1]))

    def forward(self, x):
        for layer in (self.fc1, self.fc2, self.fc3, self.fc4):
            x = torch.relu(layer(x))
        return self.fc5(x)


def _as_parameter_tensor(array, parameter):
    return torch.as_tensor(np.ascontiguousarray(array), device=parameter.device,
                           dtype=parameter.dtype)


def _relative_update(parameter, direction, scale):
    """Apply a direction with prescribed norm relative to the parameter."""
    direction = _as_parameter_tensor(direction, parameter)
    if direction.shape != parameter.shape:
        raise ValueError(f'Update shape {direction.shape} does not match {parameter.shape}')
    norm = torch.linalg.vector_norm(direction)
    if norm > 0:
        parameter.add_(direction * (scale * torch.linalg.vector_norm(parameter) / norm))


class BoundaryAugmentedNet(OriginalNet):
    def __init__(self, input_size, hidden_sizes, output_size, boundary_scale=1e-3):
        super().__init__(input_size, hidden_sizes, output_size)
        if boundary_scale < 0:
            raise ValueError('boundary_scale must be nonnegative')
        self.boundary_scale = boundary_scale

    def integrate_boundary(self):
        """Apply a detached NumPy boundary update after the optimizer step."""
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name.endswith('weight') and min(param.shape) >= 2:
                    weights = param.detach().cpu().numpy()
                    direction = degen(bdry(weights), 0)
                    _relative_update(param, direction, self.boundary_scale)


def train_network(net, criterion, optimizer, num_epochs=10, *, batch_size=32, seed=123):
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
        optimizer.step()
        if isinstance(net, BoundaryAugmentedNet):
            net.integrate_boundary()
        history.append(float(loss.detach()))
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {history[-1]:.6f}')
    return history


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--boundary-scale', type=float, default=1e-3)
    parser.add_argument('--small', action='store_true', help='Use widths 8, 12, 10, 8, 6, 3')
    args = parser.parse_args(argv)
    widths = (8, [12, 10, 8, 6], 3) if args.small else (64, [256, 128, 64, 32], 10)
    histories = {}
    for name, cls in [('baseline', OriginalNet), ('boundary', BoundaryAugmentedNet)]:
        torch.manual_seed(args.seed)
        net = cls(*widths, **({'boundary_scale': args.boundary_scale} if name == 'boundary' else {}))
        optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.01)
        print(f'Training {name}, seed={args.seed}, widths={widths}, batch_size={args.batch_size}, '
              f'boundary_scale={args.boundary_scale}')
        histories[name] = train_network(net, nn.MSELoss(), optimizer, args.epochs,
                                        batch_size=args.batch_size, seed=args.seed)
    return histories


if __name__ == '__main__':
    main()
