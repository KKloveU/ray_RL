from numpy.core.numeric import outer
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class NoisyFactorizedLinear(nn.Linear):
    """
    NoisyNet layer with factorized gaussian noise
    N.B. nn.Linear already initializes weight and bias to
    """
    def __init__(self, in_features, out_features, sigma_zero=0.4, bias=True):
        super(NoisyFactorizedLinear, self).__init__(in_features, out_features, bias=bias)
        sigma_init = sigma_zero / math.sqrt(in_features)
        self.sigma_weight = nn.Parameter(torch.Tensor(out_features, in_features).fill_(sigma_init))
        self.register_buffer("epsilon_input", torch.zeros(1, in_features))
        self.register_buffer("epsilon_output", torch.zeros(out_features, 1))
        if bias:
            self.sigma_bias = nn.Parameter(torch.Tensor(out_features).fill_(sigma_init))

    def forward(self, input):
        bias = self.bias
        func = lambda x: torch.sign(x) * torch.sqrt(torch.abs(x))

        with torch.no_grad():
            torch.randn(self.epsilon_input.size(), out=self.epsilon_input)
            torch.randn(self.epsilon_output.size(), out=self.epsilon_output)
            eps_in = func(self.epsilon_input)
            eps_out = func(self.epsilon_output)
            noise_v = torch.mul(eps_in, eps_out).detach()
        if bias is not None:
            bias = bias + self.sigma_bias * eps_out.t()
        return F.linear(input, self.weight + self.sigma_weight * noise_v, bias)

class Atten(nn.Module):
    def __init__(self):
        super().__init__()
        in_chanel=64
        out_chanel=64
        self.gamma=0.9

        self.query_conv=nn.Conv2d(in_chanel,out_chanel,3,1,(1,1))
        self.key_conv=nn.Conv2d(in_chanel,out_chanel,3,1,(1,1))
        self.value_conv=nn.Conv2d(in_chanel,out_chanel,3,1,(1,1))
        
    def forward(self,x):
        query=self.query_conv(x)

        key=self.key_conv(x)

        atten_logit=key*query.permute(0,1,3,2)
        atten=F.softmax(atten_logit,dim=1)

        value=self.value_conv(x)
        
        out=atten*value
        out=self.gamma*out+x
        return out

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),

            # nn.Dropout(0.2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            Atten(),
            # nn.Dropout(0.2),
            
        )

        self.advantage = nn.Sequential(
            # NoisyFactorizedLinear(3136, 512),
            nn.Linear(3136, 512),
            nn.Dropout(0.2),
            nn.ReLU(),
            # NoisyFactorizedLinear(512,3),
            nn.Linear(512, 3)
        )

        self.value = nn.Sequential(
            # NoisyFactorizedLinear(3136, 512),
            nn.Linear(3136, 512),
            nn.Dropout(0.2),

            nn.ReLU(),
            # NoisyFactorizedLinear(512, 1),
            nn.Linear(512, 1)
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