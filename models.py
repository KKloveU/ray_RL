from numpy.core.fromnumeric import shape
from numpy.core.shape_base import stack
import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class NoisyFactorizedLinear(nn.Linear):
    """
    NoisyNet layer with factorized gaussian noise
    N.B. nn.Linear already initializes weight and bias to
    """

    def __init__(self, in_features, out_features, sigma_zero=0.4, bias=True):
        super(NoisyFactorizedLinear, self).__init__(
            in_features, out_features, bias=bias)
        sigma_init = sigma_zero / math.sqrt(in_features)
        self.sigma_weight = nn.Parameter(torch.Tensor(
            out_features, in_features).fill_(sigma_init))
        self.register_buffer("epsilon_input", torch.zeros(1, in_features))
        self.register_buffer("epsilon_output", torch.zeros(out_features, 1))
        if bias:
            self.sigma_bias = nn.Parameter(
                torch.Tensor(out_features).fill_(sigma_init))

    def forward(self, input):
        bias = self.bias
        def func(x): return torch.sign(x) * torch.sqrt(torch.abs(x))

        with torch.no_grad():
            torch.randn(self.epsilon_input.size(), out=self.epsilon_input)
            torch.randn(self.epsilon_output.size(), out=self.epsilon_output)
            eps_in = func(self.epsilon_input)
            eps_out = func(self.epsilon_output)
            noise_v = torch.mul(eps_in, eps_out).detach()
        if bias is not None:
            bias = bias + self.sigma_bias * eps_out.t()
        return F.linear(input, self.weight + self.sigma_weight * noise_v, bias)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1568, 256),
            nn.ReLU()
        )
        self.gru = nn.GRU(input_size=256, hidden_size=512, batch_first=True)

        self.value = nn.Sequential(
            nn.Linear(512, 1)
        )
        self.actor = nn.Sequential(
            # NoisyFactorizedLinear(512, 3),
            nn.Linear(512, 3),
            nn.Softmax(dim=-1)
        )

    def forward(self, inputs,done):
        x, hx = inputs
        done=done.squeeze()
        hx=hx.unsqueeze(0)
        # print(((1.0-done[:,5].view(1,-1,1))).shape)
        x_list=[]
        for i in range(x.size(0)):
            x_list.append(self.features(x[i]))
        x=torch.stack(x_list)
        has_zeros=(((done[:,1:]==1.0).any(dim=0).nonzero().squeeze().cpu()))
        if has_zeros.dim() == 0:
            has_zeros = [has_zeros.item() + 1]
        else:
            has_zeros = (has_zeros + 1).numpy().tolist()
        has_zeros=[0]+has_zeros+[x.size(1)]

        # print(hx.shape)
        # print('==============',(hx*(1.0-done[:,5].view(1,-1,1))).shape)
        # has_zeros=[0,128,129]
        
        output=[]
        for i in range(len(has_zeros)-1):
            start_id=has_zeros[i]
            end_id=has_zeros[i+1]
            if start_id>=end_id:
                break
            # print(done.shape)
            # print(x[:,129])
            # print(x[:,start_id:end_id].shape)
            # print(x[:,start_id:end_id].shape)
            # print((hx*(1.0-done[:,start_id].view(1,-1,1))))
            # if len(has_zeros)>2:
            #     print(has_zeros)
            #     print(done[:,start_id],done[:,start_id+1],done[:,start_id-1],start_id)
            #     print('########################################')
            rnn,hx=self.gru(x[:,start_id:end_id],(hx*(1.0-done[:,start_id].view(1,-1,1))))
            output.append(rnn)

        x=torch.cat(output,dim=1)
        actor_list=[]
        value_list=[]
        for i in range(x.size(0)):
            actor_list.append(self.actor(x[i]))
            value_list.append(self.value(x[i]))
        # actor=torch.stack(actor_list)
        # value=torch.stack(value_list)
        return actor_list, value_list

    def step_forward(self, inputs):
        x, hx = inputs
        x = self.features(x)
        _, hx = self.gru(x.unsqueeze(1), hx)
        x = hx.squeeze(0)
        actor = self.actor(x)
        value = self.value(x)
        return actor, value, hx


    def get_weights(self):
        return dict_to_cpu(self.state_dict())

    def set_weights(self, weights):
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
