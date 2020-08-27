import torch
import torch.nn as nn
import torch.nn.functional as F


class QFunc(nn.Module):
    def __init__(self, look_behind = 1, look_ahead=1, hidden_size = 20, n_layers = 2):
        super(QFunc, self).__init__()

    # Actions space is R \cross R
    # State Space is (R^3)^(look_ahead+look_behind+1)
        self.fc1 = nn.Linear((1+look_behind+look_ahead)*3 + 2, hidden_size)
        self.hidden = []
        for i in range(n_layers):
            self.hidden.append(nn.Linear(hidden_size, hidden_size))
        
        self.out_fc = nn.Linear(hidden_size, 1)
    
    def forward(self, state, action):
        inp = torch.cat([torch.flatten(state, start_dim=1), action], dim = 1)
        out = self.fc1(inp)
        out = F.relu(out)
        for hfc in self.hidden:
            out = hfc(out)
            out = F.relu(out)
        out = self.out_fc(out)

        return out