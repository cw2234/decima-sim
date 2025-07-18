"""
Graph Summarization Network

Summarize node features globally
via parameterized aggregation scheme
"""

import torch
import torch.nn as nn
import numpy as np



class GraphSNN(nn.Module):
    def __init__(self, input_dim, hid_dims, output_dim, act_fn):
        super(GraphSNN, self).__init__()
        # on each transformation, input_dim -> (multiple) hid_dims -> output_dim
        # the global level summarization will use output from DAG level summarizaiton

        # self.inputs = inputs

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hid_dims = hid_dims

        self.act_fn = act_fn

        # DAG level and global level summarization
        self.summ_levels = 2

        # graph summarization, hierarchical structure
        # self.summ_mats = [tf.sparse_placeholder(
        #     tf.float32, [None, None]) for _ in range(self.summ_levels)]

        # initialize summarization parameters for each hierarchy
        self.dag_layers = self._build_layers(input_dim, hid_dims, output_dim)
        self.global_layers = self._build_layers(output_dim, hid_dims, output_dim)


        # graph summarization operation
        # self.summaries = self.summarize()

    def _build_layers(self, input_dim, hid_dims, output_dim):
        """构建线性层序列（替代原权重列表） hid_dims[x, y, ..]表示每一层的神经元个数"""
        layers = nn.ModuleList()

        curr_in_dim = input_dim

        # 隐藏层
        for hid_dim in hid_dims:
            layer = nn.Linear(curr_in_dim, hid_dim)
            # 初始化权重（对应原glorot和zeros） TODO: gain原代码中是1
            nn.init.xavier_uniform_(layer.weight, gain=np.sqrt(2))  # glorot初始化
            nn.init.zeros_(layer.bias)  # 偏置初始化为0
            layers.append(layer)
            curr_in_dim = hid_dim

        # 输出层
        out_layer = nn.Linear(curr_in_dim, output_dim)
        #  TODO: gain原代码中是1
        nn.init.xavier_uniform_(out_layer.weight, gain=np.sqrt(2))
        nn.init.zeros_(out_layer.bias)
        layers.append(out_layer)

        return layers

    def forward(self, inputs, summ_mats):
        # summarize information in each hierarchy
        # e.g., first level summarize each individual DAG
        # second level globally summarize all DAGs
        x = inputs

        summaries = []

        # DAG level summary
        s = x
        for layer in self.dag_layers:
            s = layer(s)
            s = self.act_fn(s)


        s = torch.sparse.mm(summ_mats[0], s)
        summaries.append(s)

        # global level summary
        for layer in self.global_layers:
            s = layer(s)
            s = self.act_fn(s)

        s = torch.sparse.mm(summ_mats[1], s)
        summaries.append(s)

        return summaries
