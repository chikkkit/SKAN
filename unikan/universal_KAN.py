import torch
import torch.nn as nn
import numpy as np
from .basis import lshifted_softplus
    
class UniversalKANSubLayer(nn.Module):
    def __init__(self, in_features, out_features, sublayer_type='add', function=lshifted_softplus, 
                 param_num=1, use_bias=True, device='cpu', init_method='default'):
        """
        in_features: int, the number of input features

        out_features: int, the number of output features

        sublayer_type: str in ['add', 'mul'], the type of sublayer, default is 'add'

        function: function, the basis function to use, default is lshifted_softplus

        param_num: int, the number of parameters of this function

        use_bias: bool, whether to add bias to this function

        device: str, the device to use, default is 'cpu'

        init_method: str or callable, initialization method
            - if 'default': use kaiming_uniform_ initialization
            - if callable: the function should take weight tensor as input and initialize it
        """
        super(UniversalKANSubLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sublayer_type = sublayer_type
        self.function = function
        self.param_num = param_num
        self.use_bias = use_bias
        self.init_method = init_method
        
        if self.use_bias:
            self.weight = nn.ParameterList([nn.Parameter(torch.randn(self.out_features, self.in_features + 1, device=device)) for _ in range(self.param_num)])
        else:
            self.weight = nn.ParameterList([nn.Parameter(torch.randn(self.out_features, self.in_features, device=device)) for _ in range(self.param_num)])
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.param_num):
            if self.init_method == 'default':
                nn.init.kaiming_uniform_(self.weight[i], a=5 ** 0.5)
                # Initialize bias to 0 in default mode
                if self.use_bias:
                    nn.init.constant_(self.weight[i][:, -1], 0)
            elif callable(self.init_method):
                # For custom initialization, apply to the entire weight matrix including bias
                self.init_method(self.weight[i])

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.unsqueeze(1)
        # add bias
        if self.use_bias:
            x = torch.cat([x, torch.ones_like(x[..., :1])], dim=2)
        output = self.function(x, self.weight)
        if self.sublayer_type == 'add':
            return torch.sum(output, dim=-1)
        else:
            return torch.prod(output, dim=-1)
    
    # def extra_repr(self):
    #     return '+ add node, in_features={}, out_features={}, function={}'.format(self.in_features, self.out_features, self.function.__name__)
    
    # def __repr__(self):
    #     return self.extra_repr()
    
class UniversalKANLinear(nn.Module):
    def __init__(self, in_features, out_features, function_list=[], device='cpu', init_method='default'):
        """
        in_features: int, the number of input features

        out_features: int, the number of output features
        
        function_list: list, the list of basis functions
            It should be a list of function dict, each dict should have the following:
                'function': function, the basis function such as 'lshifted_softplus(x, k)'
                'param_num': int, the number of parameters of a single basis function
                'sublayer_type': str, the type of sublayer, 'add' or 'mul'
                'node_num': int, number to show how many nodes to use
                'use_bias': bool, whether to add bias to this function
            For example, if you want to use 2 basis functions lshifted_softplus and lsin, they are single parameterized functions, 
            each function only have 1 addition node and 0 multiplication node, then the function_list should be like:
                function_list = [
                    {'function': lshifted_softplus, 'param_num': 1, 'sublayer_type': 'add', 'node_num': 1, 'use_bias': True},
                    {'function': lsin, 'param_num': 1, 'sublayer_type': 'add', 'node_num': 1, 'use_bias': True}
                ]
            Note that functions and sublayers are bound, which means that a sublayer can only have one function. This is to accelerate operation
            in grid type network.

        device: str, the device to use, default is 'cpu'

        init_method: str or callable, initialization method
            - if 'default': use kaiming_uniform_ initialization
            - if callable: the function should take weight tensor as input and initialize it
        """
        
        if len(function_list) == 0:
            function_list.append({'function': lshifted_softplus, 'param_num': 1, 
                                  'sublayer_type': 'add', 'node_num': out_features, 'use_bias': True})

        super(UniversalKANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.basis_function = function_list
        self.sublayers = nn.ModuleList()
        # test if sum of node_num  is equal to out_features
        assert sum([function['node_num'] for function in self.basis_function]) == self.out_features, \
            'The sum of node_num should be equal to out_features'
        for i in range(len(self.basis_function)):
            if self.basis_function[i]['sublayer_type'] == 'mul':
                self.sublayers.append(UniversalKANSubLayer(self.in_features, self.basis_function[i]['node_num'], 'mul',
                                                 self.basis_function[i]['function'], self.basis_function[i]['param_num'], 
                                                 self.basis_function[i]['use_bias'], device=device, 
                                                 init_method=init_method))
            else:
                self.sublayers.append(UniversalKANSubLayer(self.in_features, self.basis_function[i]['node_num'], 'add',
                                                 self.basis_function[i]['function'], self.basis_function[i]['param_num'], 
                                                 self.basis_function[i]['use_bias'], device=device, 
                                                 init_method=init_method))
    
    def forward(self, x):
        return torch.cat([sublayer(x) for sublayer in self.sublayers], dim=-1)
    
    # def extra_repr(self):
    #     node_reprs = [node.extra_repr() for node in self.nodes]
    #     header = f'Uni-KAN Linear layer, grid_type, in_features={self.in_features}, out_features={self.out_features}\n'
    #     body = '{\n        ' + '\n        '.join(node_reprs) + '\n    }'
    #     return header + body
    
    # def __repr__(self):
    #     return self.extra_repr()
    

class UniversalKAN(nn.Module):
    def __init__(self, layer_sizes, function_lists=[], device='cpu', init_method='default'):
        
        super(UniversalKAN, self).__init__()
        if len(function_lists) == 0:
            for i in range(len(layer_sizes) - 1):
                function_lists.append([{'function': lshifted_softplus, 'param_num': 1, 
                                        'sublayer_type': 'add', 'node_num': layer_sizes[i+1], 'use_bias': True}])
        self.layer_sizes = layer_sizes
        self.function_lists = function_lists
        self.layers = nn.ModuleList()
        for i in range(len(self.layer_sizes) - 1):
            self.layers.append(UniversalKANLinear(self.layer_sizes[i], self.layer_sizes[i+1], 
                                                self.function_lists[i], device=device, 
                                                init_method=init_method))
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    # def extra_repr(self):
    #     layer_reprs = [layer.extra_repr() for layer in self.layers]
    #     header = f'Uni-KAN network, grid_type, layer_sizes={self.layer_sizes}\n'
    #     body = '{\n    ' + '    '.join(layer_reprs) + '\n}'
    #     return header + body
    
    # def __repr__(self):
    #     return self.extra_repr()
    
    # def get_graph(self):
    #     """
    #     This function is to generate the computation graph of a network, return 3 parameters.
    #     pos: dict, the position of each node, the key is the name of the node, the value is the position of the node. e.g.: 
    #         {'0, 1': (0.0, 0), '1, 1': (-2.0, 1), '1, 2': (-1.0, 1), '1, 3': (0.0, 1), 
    #         '1, 4': (1.0, 1), '1, 5': (2.0, 1), '0, b': (1.0, 0), '2, 1': (0.0, 2), '1, b': (3.0, 1)}
    #     edges: list, the edges between nodes, each element is a tuple, the first is the start node, the second is the end node. e.g.:
    #         [('0, 1', '1, 1'), ('0, 1', '1, 2'), ('0, 1', '1, 3'), ('0, 1', '1, 4'), ('0, 1', '1, 5'),
    #         ('0, b', '1, 1'), ('0, b', '1, 2'), ('0, b', '1, 3'), ('0, b', '1, 4'), ('0, b', '1, 5'),
    #         ('1, 1', '2, 1'), ('1, 2', '2, 1'), ('1, 3', '2, 1'), ('1, 4', '2, 1'), ('1, 5', '2, 1'), ('1, b', '2, 1')]
    #     funcs: dict, the function of each edge, the key is the edge, the value is the function. e.g.:
    #         {('0, 1', '1, 1'): <function __main__.<lambda>(x, ev=0.6927026510238647)>,
    #         ('0, 1', '1, 2'): <function __main__.<lambda>(x, ev=0.7938258051872253)>,
    #         ('0, 1', '1, 3'): <function __main__.<lambda>(x, ev=0.9977601766586304)>,
    #         ('0, 1', '1, 4'): <function __main__.<lambda>(x, ev=-0.29473429918289185)>,
    #         ('0, 1', '1, 5'): <function __main__.<lambda>(x, ev=-0.5044335126876831)>,
    #         ('0, b', '1, 1'): <function __main__.<lambda>(x, ev=-1.7965008020401)>,
    #         ('0, b', '1, 2'): <function __main__.<lambda>(x, ev=-0.9162623882293701)>,
    #         ('0, b', '1, 3'): <function __main__.<lambda>(x, ev=-0.9161001443862915)>,
    #         ('0, b', '1, 4'): <function __main__.<lambda>(x, ev=-0.06848419457674026)>,
    #         ('0, b', '1, 5'): <function __main__.<lambda>(x, ev=0.2307554930448532)>,
    #         ('1, 1', '2, 1'): <function __main__.<lambda>(x, ev=-1.398566722869873)>,
    #         ('1, 2', '2, 1'): <function __main__.<lambda>(x, ev=-0.33302581310272217)>,
    #         ('1, 3', '2, 1'): <function __main__.<lambda>(x, ev=-0.8638795018196106)>,
    #         ('1, 4', '2, 1'): <function __main__.<lambda>(x, ev=0.1741543859243393)>,
    #         ('1, 5', '2, 1'): <function __main__.<lambda>(x, ev=0.3043987452983856)>,
    #         ('1, b', '2, 1'): <function __main__.<lambda>(x, ev=-0.7070185542106628)>}
    #     """
    #     pos, edges, funcs = {}, [], {}
    #     # generate name of nodes from input layer
    #     for i in range(self.layer_sizes[0]):
    #         pos['0, {}'.format(i)] = ((-self.layer_sizes[0] + 1) / 2 + i, 0)

    #     # generate name of nodes from hidden layers and output layer
    #     layer_id_from = 1
    #     for layer in self.layers:
    #         node_id_from = 0
    #         use_bias = False
    #         for node in layer.nodes:
    #             for i in range(node.out_features):
    #                 name = f'{'+' if node.sublayer_type == 'add' else 'x'} {layer_id_from}, {node_id_from + i}'
    #                 pos[name] = ((-layer.out_features + 1) / 2 + i + node_id_from, layer_id_from)
    #             node_id_from += node.out_features
    #             if node.use_bias:
    #                 use_bias = True
    #         if use_bias:
    #             name = f'{layer_id_from-1}, b'
    #             pos[name] = ((-layer.in_features + 1) / 2 + layer.in_features, layer_id_from-1)
    #         layer_id_from += 1

    #     # generate edges and functions
    #     def find_string_in_list(A, s): # need this function because in each layer, the operation of the start node is hard to get, 
    #         for element in A:          # therefore we get name directly from the graph (that is the 'pos' variable)
    #             if s in element:
    #                 return element
    #         return None
    #     for layer in range(len(self.layer_sizes)-1):
    #         node_id_from = 0
    #         for node in self.layers[layer].nodes:
    #             for i in range(node.in_features):
    #                 for j in range(node.out_features):
    #                     start_node = find_string_in_list(pos.keys(), f'{layer}, {i}')
    #                     end_node = find_string_in_list(pos.keys(), f'{layer+1}, {node_id_from + j}')
    #                     edges.append((start_node, end_node))
    #                     learnale_params = [node.weight[index][j, i].item() for index in range(node.param_num)]
    #                     funcs[(start_node, end_node)] = lambda x, ev=learnale_params: node.function(x, ev)
    #             if node.use_bias:
    #                 for j in range(node.out_features):
    #                     start_node = find_string_in_list(pos.keys(), f'{layer}, b')
    #                     end_node = find_string_in_list(pos.keys(), f'{layer+1}, {node_id_from + j}')
    #                     edges.append((start_node, end_node))
    #                     learnale_params = [node.weight[index][j, -1].item() for index in range(node.param_num)]
    #                     funcs[(start_node, end_node)] = lambda x, ev=learnale_params: node.function(x, ev)
    #             node_id_from += node.out_features
    #     return pos, edges, funcs


# test to ensure the functionbility of the code
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # net = universal_KAN_grid.UniversalKAN([784, 100, 10], device=device) # test for default

    function_lists = [
        [
            {'function': lshifted_softplus, 'param_num': 1, 'sublayer_type': 'add', 'node_num': 1, 'use_bias': True},
            {'function': lshifted_softplus, 'param_num': 1, 'sublayer_type': 'mul', 'node_num': 1, 'use_bias': True}
            ],
        [
            {'function': lshifted_softplus, 'param_num': 1, 'sublayer_type': 'add', 'node_num': 2, 'use_bias': True}
            ] 
        ]
    net = UniversalKAN([2, 2, 2], function_lists=function_lists, device=device)
    net.get_graph()