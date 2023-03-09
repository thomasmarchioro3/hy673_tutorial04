import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from torch.distributions import Uniform, Distribution, Normal

class CouplingLayer(nn.Module):

  def __init__(self, input_dim, hidden_dim, mask, num_layers=4):
    super().__init__()

    self.mask = mask

    modules = [nn.Linear(input_dim, hidden_dim), 
               nn.LeakyReLU(0.2)]
    
    for _ in range(num_layers - 2):
      modules.append(nn.Linear(hidden_dim, hidden_dim))
      modules.append(nn.LeakyReLU(0.2))
    modules.append(nn.Linear(hidden_dim, input_dim))

    self.m = nn.Sequential(*modules)

  def forward(self, x):
      x1 = self.mask * x
      x2 = (1 - self.mask) * x
      y1 = x1
      y2 = x2 + (self.m(x1) * (1 - self.mask))
      return y1 + y2
    
  # inverse mapping
  def inverse(self, x):
    y1 = self.mask * x
    y2 =(1 - self.mask) * x
    x1 = y1
    x2 = y2 - (self.m(y1) * (1 - self.mask))
    return x1 + x2
  
class ScalingLayer(nn.Module):
  def __init__(self, input_dim):
    super().__init__()
    self.log_scale_vector = nn.Parameter(torch.randn(1, input_dim, requires_grad=True))

  def forward(self, x):
    log_det_jacobian = torch.sum(self.log_scale_vector)
    log_likelihood = torch.exp(self.log_scale_vector) * x
    return log_likelihood, log_det_jacobian
  
  def inverse(self, x):
    return torch.exp(- self.log_scale_vector) * x  # we do not need the jacobian for the inverse
  

class LogisticDistribution(Distribution):
  def __init__(self):
    super().__init__()

  def log_prob(self, x):
    return -(F.softplus(x) + F.softplus(-x))

  def sample(self, size):
    z = Uniform(torch.FloatTensor([0.]), torch.FloatTensor([1.])).sample(size)

    return torch.log(z) - torch.log(1. - z)
  
class NICE(nn.Module):
  def __init__(self, input_dim, hidden_dim=1000, num_coupling_layers=3, num_layers=6, device='cpu', 
               use_scaling=True, prior_type='logistic'):
    super().__init__()

    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.num_coupling_layers = num_coupling_layers
    self.num_layers = num_layers  # number of linear layers for each coupling layer
    self.use_scaling = use_scaling
    self.prior_type = prior_type

    # alternating mask orientations for consecutive coupling layers
    masks = [self._get_mask(input_dim, orientation=(i % 2 == 0)).to(device)
                                            for i in range(num_coupling_layers)]

    self.coupling_layers = nn.ModuleList([CouplingLayer(input_dim=input_dim,
                                hidden_dim=hidden_dim,
                                mask=masks[i], num_layers=num_layers)
                              for i in range(num_coupling_layers)])

    if use_scaling:
      self.scaling_layer = ScalingLayer(input_dim=input_dim)

    if prior_type == 'logistic':
      self.prior = LogisticDistribution()
    elif prior_type == 'normal':
      self.prior = Normal(0, 1)
    else:
      print("Error: Invalid prior_type")
    self.device = device

  def forward(self, x):
    
    z = x
    for i in range(len(self.coupling_layers)):  # pass through each coupling layer
      z = self.coupling_layers[i](z)

    if self.use_scaling:
      z, log_det_jacobian = self.scaling_layer(z)
      log_likelihood = torch.sum(self.prior.log_prob(z), dim=1) + log_det_jacobian
    else:
      log_likelihood = torch.sum(self.prior.log_prob(z), dim=1) + log_det_jacobian
        

    return z, log_likelihood

  def inverse(self, z):
    x = z
    if self.use_scaling:
      x = self.scaling_layer.inverse(x)
    for i in reversed(range(len(self.coupling_layers))):  # pass through each coupling layer in reversed order
      x = self.coupling_layers[i].inverse(x)
    return x

  def sample(self, num_samples):
    z = self.prior.sample([num_samples, self.input_dim]).view(num_samples, self.input_dim)
    z = z.to(self.device)
    return self.inverse(z)

  def _get_mask(self, dim, orientation=True):
    mask = torch.zeros(dim)
    mask[::2] = 1.
    if orientation:
      mask = 1. - mask # flip mask if orientation is True
    return mask.float()
