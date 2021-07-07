import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class NoisyLinear(nn.Module):

    def __init__(self, num_in, num_out, is_training=True):
        super(NoisyLinear, self).__init__()
        self.num_in = num_in
        self.num_out = num_out
        self.is_training = is_training

        self.mu_weight = nn.Parameter(torch.FloatTensor(num_out, num_in))
        self.mu_bias = nn.Parameter(torch.FloatTensor(num_out))
        self.sigma_weight = nn.Parameter(torch.FloatTensor(num_out, num_in))
        self.sigma_bias = nn.Parameter(torch.FloatTensor(num_out))
        self.register_buffer("epsilon_weight", torch.FloatTensor(num_out, num_in)) 
        self.register_buffer("epsilon_bias", torch.FloatTensor(num_out))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x): 
        self.reset_noise()

        if self.is_training:
            weight = self.mu_weight + self.sigma_weight.mul(self.epsilon_weight.clone()) 
            bias = self.mu_bias + self.sigma_bias.mul(self.epsilon_bias.clone())
        else:
            weight = self.mu_weight
            buas = self.mu_bias

        y = F.linear(x, weight, bias)
        
        return y

    def reset_parameters(self):
        std = math.sqrt(3 / self.num_in)
        self.mu_weight.data.uniform_(-std, std)
        self.mu_bias.data.uniform_(-std,std)

        self.sigma_weight.data.fill_(0.017)
        self.sigma_bias.data.fill_(0.017)

    def reset_noise(self):
        self.epsilon_weight.data.normal_()
        self.epsilon_bias.data.normal_()

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.Dropout(0.2),
            nn.ReLU()
        )

        self.advantage = nn.Sequential(
            nn.Linear(3136, 512),
            nn.Dropout(0.2),
            # NoisyLinear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 3),
            # NoisyLinear(512, 3)
        )

        self.value = nn.Sequential(
            nn.Linear(3136, 512),
            nn.Dropout(0.2),
            # NoisyLinear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
            # NoisyLinear(512, 1)
        )


    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x,1)
        advantage = self.advantage(x)
        value     = self.value(x)
        return value + advantage  - advantage.mean()

    def get_weights(self):
        return dict_to_cpu(self.state_dict())

    def set_weights(self,weights):
        self.load_state_dict(weights)


def dict_to_cpu(dictionary):
    cpu_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, torch.Tensor):
            cpu_dict[key] = value.cpu()
        elif isinstance(value, dict):
            cpu_dict[key] = dict_to_cpu(value)
        else:
            cpu_dict[key] = value
    return cpu_dict