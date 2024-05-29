import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, obs_dim, act_dim, save_dir="./ppo_model"):
        super(Model, self).__init__()
        self.save_dir = save_dir

        self.cnn_base = nn.Sequential(
            nn.Conv2d(obs_dim, 32, kernel_size=(8, 8), stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(4, 4), stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate the flattened dimension after the CNN layers
        with torch.no_grad():
            self._to_linear = self._calculate_conv_output((1, obs_dim, 84, 84))

        self.v = nn.Sequential(
            nn.Linear(self._to_linear, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )

        self.fc = nn.Sequential(
            nn.Linear(self._to_linear, 100),
            nn.ReLU()
        )

        self.alpha_head = nn.Sequential(
            nn.Linear(100, act_dim),
            nn.Softplus()
        )

        self.beta_head = nn.Sequential(
            nn.Linear(100, act_dim),
            nn.Softplus()
        )

        self.apply(self._weights_init)
        self.ckpt_file = save_dir + ".pth"

    def _calculate_conv_output(self, shape):
        x = torch.zeros(shape)
        x = self.cnn_base(x)
        return x.numel()

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        x = self.cnn_base(x)
        v = self.v(x)
        x = self.fc(x)
        alpha = self.alpha_head(x) + 1
        beta = self.beta_head(x) + 1
        return (alpha, beta), v

# Example usage:
obs_dim = 1  # Number of channels in the input (grayscale image has 1 channel)
act_dim = 2  # Example action dimension
model = Model(obs_dim, act_dim)

# Test with a dummy input to ensure it works
dummy_input = torch.randn(1, obs_dim, 84, 84)
(alpha, beta), v = model(dummy_input)

print("Alpha:", alpha)
print("Beta:", beta)
print("Value:", v)
