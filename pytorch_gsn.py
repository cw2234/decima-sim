"""
Graph Summarization Network

Summarize node features globally
via parameterized aggregation scheme
"""

import torch
import torch.nn as nn


class GraphSNN(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hid_dims: list,
                 output_dim: int,
                 act_fn: nn.Module):
        super(GraphSNN, self).__init__()
        # on each transformation, input_dim -> (multiple) hid_dims -> output_dim
        # the global level summarization will use output from DAG level summarizaiton

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hid_dims = hid_dims
        self.act_fn = act_fn

        # DAG level and global level summarization
        self.summ_levels = 2

        # initialize summarization parameters for each hierarchy
        self.dag_layer = self._build_layer(self.input_dim, self.hid_dims, self.output_dim)

        self.global_layer = self._build_layer(self.output_dim, self.hid_dims, self.output_dim)

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

    def forward(self, inputs, summ_mats):
        '''
        graph summarization operation
        Args:
            inputs: (total_node_num, input_dim)，(所有图的总节点数，5 + 8)
            summ_mats: list[2]，有2文个(total_node_num, total_node_num) # graph summarization, hierarchical structure

        Returns:

        '''
        # summarize information in each hierarchy
        # e.g., first level summarize each individual DAG
        # second level globally summarize all DAGs
        x = inputs

        summaries = []

        # DAG level summary
        s = x
        for layer in self.dag_layer:
            s = layer(s)
            s = self.act_fn(s)

        s = torch.sparse.mm(summ_mats[0], s)
        summaries.append(s)

        # global level summary
        for layer in self.global_layer:
            s = layer(s)
            s = self.act_fn(s)

        s = torch.sparse.mm(summ_mats[1], s)
        summaries.append(s)

        return summaries
