import torch
from torchviz import make_dot
from torchsummary import summary
import torch.nn as nn
import numpy as np

ACTION_SPACE = [
    (0, 0, 0), (0.6, 0, 0), (-0.6, 0, 0), (0, 0.2, 0), (0, 0, 0.8),  # (Steering Wheel, Gas, Brake)
] 

class QNetwork(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(QNetwork, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc_input_dim = self._get_conv_output(input_shape)
        self.fc1 = nn.Linear(self.fc_input_dim, 512)
        self.fc2 = nn.Linear(512, num_actions)
        self.apply(self.weights)

    def _get_conv_output(self, shape):
        o = torch.zeros(1, *shape)
        o = self.conv1(o)
        o = self.conv2(o)
        o = self.conv3(o)
        return int(np.prod(o.size()))

    @staticmethod
    def weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        output = self.fc2(x)
        return output

input_shape = (3, 84, 84)
num_actions = len(ACTION_SPACE)

model = QNetwork(input_shape, num_actions)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

dummy_input = torch.randn(1, *input_shape).to(device)

output = model(dummy_input)

graph = make_dot(output, params=dict(model.named_parameters()))
graph.render("qnetwork_architecture", format="png")

summary(model, input_size=input_shape)
