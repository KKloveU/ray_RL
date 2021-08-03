from numpy.core.numeric import outer
import torch
from torch.distributions.transforms import CatTransform
import torch.nn as nn
import torch.nn.functional as F
import math

from torch.nn.modules.activation import Softmax
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


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.features = nn.Sequential(
                nn.Conv2d(4, 32, kernel_size=8, stride=4),
                nn.ReLU(),

                nn.Conv2d(32, 64, kernel_size=4, stride=2),

                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
        )
        self.value = nn.Sequential(
            # NoisyFactorizedLinear(3136, 512),
            nn.Linear(3136, 512),
            nn.ReLU(),
            # NoisyFactorizedLinear(512, 1),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x,1)
        value     = self.value(x)
        return value

    def get_weights(self):
        return dict_to_cpu(self.state_dict())

    def set_weights(self,weights):
        self.load_state_dict(weights)


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.distribution = torch.distributions.Categorical
        self.features = nn.Sequential(
                nn.Conv2d(4, 32, kernel_size=8, stride=4),
                nn.ReLU(),

                nn.Conv2d(32, 64, kernel_size=4, stride=2),

                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
        )
        self.action_head = nn.Sequential(
            # NoisyFactorizedLinear(3136, 512),
            nn.Linear(3136, 512),
            nn.ReLU(),
            # NoisyFactorizedLinear(512,3),
            nn.Linear(512, 3),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x,1)
        action_prob = self.action_head(x)

        return action_prob

    def get_weights(self):
        return dict_to_cpu(self.state_dict())

    def set_weights(self,weights):
        self.load_state_dict(weights)
    
    def select_action(self,x):
        action_prob=self.forward(x)
        action=action_prob.multinomial(1)
        return action

    def get_kl(self,x):
        action_prob1=self.forward(x)
        action_prob0=action_prob1.detach()
        kl=action_prob0*(torch.log(action_prob0)-torch.log(action_prob1))
        return kl.sum(1,keepdim=True)
    
    def get_log_prob(self,x,actions):
        action_prob=self.forward(x)
        return torch.log(action_prob.gather(1,actions.long().unsqueeze(1)))

    def get_fim(self,x):
        action_prob=self.forward(x)
        M=action_prob.pow(-1).view(-1).detach()
        return M,action_prob,{}


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
