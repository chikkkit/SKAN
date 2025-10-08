import torch.nn as nn
from .universal_KAN import UniversalKANLinear
from .basis import lshifted_softplus

# this two class is to be used when your skan layer or the whole skan model uses only one basis function
# if you want to use multiple basis functions, you can directly use the UniversalKAN and UniversalKANLinear
class SKANLinear_pure(UniversalKANLinear):
    def __init__(self, in_features, out_features, bias=True, basis_function=lshifted_softplus, 
                 sublayer_type='add', device='cpu', init_method='default'):
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        # make function list
        self.function_list = [
            {'function': basis_function, 'param_num': 1, 'sublayer_type': sublayer_type, 'node_num': out_features, 'use_bias': bias}
        ]
        super(SKANLinear_pure, self).__init__(in_features, out_features, self.function_list, 
                                             device=device, init_method=init_method)
    
    def forward(self, x):
        return super(SKANLinear_pure, self).forward(x)

class SKAN_pure(nn.Module):
    def __init__(self, layer_sizes, basis_function=lshifted_softplus, sublayer_type='add', 
                 bias=True, device='cpu', init_method='default'):
        super(SKAN_pure, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes)-1):
            self.layers.append(SKANLinear_pure(layer_sizes[i], layer_sizes[i+1], bias=bias, 
                                             basis_function=basis_function, sublayer_type=sublayer_type, 
                                             device=device, init_method=init_method))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x