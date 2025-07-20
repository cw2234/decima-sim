"""
Graph Convolutional Network

Propagate node features among neighbors
via parameterized message passing scheme
"""
import torch
import torch.nn as nn
from torch import Tensor


# from tf_op import glorot, zeros


class GraphCNN(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hid_dims: list,
                 output_dim: int,
                 max_depth: int,
                 act_fn: nn.Module):
        super(GraphCNN, self).__init__()

        self.input_dim = input_dim
        self.hid_dims = hid_dims
        self.output_dim = output_dim
        self.max_depth = max_depth
        self.act_fn = act_fn

        # initialize message passing transformation parameters
        # h: x -> x'
        self.prep_layer = self._build_layer(self.input_dim, self.hid_dims, self.output_dim)

        # f: x' -> e
        self.proc_layer = self._build_layer(self.output_dim, self.hid_dims, self.output_dim)

        # g: e -> e
        self.agg_layer = self._build_layer(self.output_dim, self.hid_dims, self.output_dim)

    def _build_layer(self,
                     input_dim: int,
                     hid_dims: list,
                     output_dim: int) -> nn.ModuleList:
        '''

        Args:
            input_dim: 网络输入的大小
            hid_dims: 隐藏层每层的大小
            output_dim: 网络输出的大小

        Returns:

        '''
        # Initialize the parameters
        layers = nn.ModuleList()

        curr_in_dim = input_dim

        # hidden layers
        for hid_dim in hid_dims:
            layer = nn.Linear(curr_in_dim, hid_dim)
            nn.init.xavier_uniform_(layer.weight)  # 权重置
            nn.init.zeros_(layer.bias)  # 偏置为0
            layers.append(layer)
            curr_in_dim = hid_dim

        # output layer
        out_layer = nn.Linear(curr_in_dim, output_dim)
        nn.init.xavier_uniform_(out_layer.weight)  # 权重置
        nn.init.zeros_(out_layer.bias)  # 偏置为0
        layers.append(out_layer)

        return layers

    def forward(self,
                inputs: Tensor,
                adj_mats: Tensor,
                masks: Tensor) -> Tensor:

        '''
        graph message passing

        Args:
            inputs: (total_node_num, input_dim)，(所有图的总节点数，5)
            adj_mats: list[depth]，每个元素的维度 (total_node_num, total_node_num) # message passing
            masks: list[depth]，每个元素的维度 (total_node_num, 1) # message passing

        Returns:

        '''
        # message passing among nodes
        # the information is flowing from leaves to roots
        x = inputs

        # raise x into higher dimension
        for layer in self.prep_layer:
            x = layer(x)
            x = self.act_fn(x)

        for d in range(self.max_depth):
            # work flow: index_select -> f -> masked assemble via adj_mat -> g
            y = x

            # process the features on the nodes
            for layer in self.proc_layer:
                y = layer(y)
                y = self.act_fn(y)

            # message passing
            y = torch.sparse.mm(adj_mats[d], y)

            # aggregate child features
            for layer in self.agg_layer:
                y = layer(y)
                y = self.act_fn(y)

            # remove the artifact from the bias term in g
            y = y * masks[d]

            # assemble neighboring information
            x = x + y

        return x
