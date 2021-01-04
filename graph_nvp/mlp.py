import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, units, in_size=None):
        super(MLP, self).__init__()
        assert isinstance(units, (tuple, list))
        assert len(units) >= 1
        n_layers = len(units)

        units_list = [in_size] + list(units)
        layers = []
        for i in range(n_layers):
            layers.append(nn.Linear(units_list[i], units_list[i+1]))
        self.layers = nn.Sequential(*layers)
        self.n_layers = n_layers

    def forward(self, x):
        output = self.layers(x)
        return output
