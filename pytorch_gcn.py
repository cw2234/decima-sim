"""
Graph Convolutional Network

Propagate node features among neighbors
via parameterized message passing scheme
"""

import torch
import torch.nn as nn
import numpy as np


class GraphCNN(nn.Module):
    def __init__(self, inputs, input_dim, hid_dims, output_dim, max_depth, act_fn):
        super(GraphCNN, self).__init__()

        # self.inputs = inputs

        self.input_dim = input_dim
        self.hid_dims = hid_dims
        self.output_dim = output_dim
        self.max_depth = max_depth
        self.act_fn = act_fn

        # message passing
        # self.adj_mats = [tf.sparse_placeholder(tf.float32, [None, None]) for _ in range(self.max_depth)]
        # self.masks = [tf.placeholder(tf.float32, [None, 1]) for _ in range(self.max_depth)]

        # initialize message passing transformation parameters
        # 定义三组网络层：prep（提升维度）、proc（节点特征处理）、agg（聚合特征）
        # h: x -> x'
        self.prep_layers = self._build_layers(input_dim, hid_dims, output_dim)

        # f: x' -> e
        self.proc_layers = self._build_layers(output_dim, hid_dims, output_dim)

        # g: e -> e
        self.agg_layers = self._build_layers(output_dim, hid_dims, output_dim)

        # # graph message passing
        # self.outputs = self.forward()

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

    def forward(self, inputs, adj_mats, masks):
        """
        inputs: 输入特征张量，形状为[batch_size, num_nodes, input_dim]
        adj_mats: 稀疏邻接矩阵列表，长度为max_depth，每个为[num_nodes, num_nodes]的稀疏张量
        masks: 掩码张量列表，长度为max_depth，每个为[num_nodes, 1]的张量
        """
        # message passing among nodes
        # the information is flowing from leaves to roots
        x = inputs

        # 第一步：提升特征维度（prep过程）
        # raise x into higher dimension
        for layer in self.prep_layers:
            x = layer(x)  # 等价于x = x @ W + b
            x = self.act_fn(x)

        # 第二步：多轮消息传递（max_depth次）
        for d in range(self.max_depth):
            # work flow: index_select -> f -> masked assemble via adj_mat -> g
            y = x  # 临时变量存储当前节点特征

            # process the features on the nodes
            # 处理节点特征（proc过程）
            for layer in self.proc_layers:
                y = layer(y)
                y = self.act_fn(y)

            # message passing
            # 消息传递：通过邻接矩阵聚合邻居信息（稀疏矩阵乘法）
            # 注意：adj_mats[d]是稀疏张量，形状为[num_nodes, num_nodes]
            # 这里需要确保y的形状为[num_nodes, feature_dim]，批量处理时可能需要调整维度
            # 若输入含batch_size，需先展平batch再相乘（根据实际场景调整）
            y = torch.sparse.mm(adj_mats[d], y)

            # aggregate child features
            # 聚合邻居特征（agg过程）
            for layer in self.agg_layers:
                y = layer(y)
                y = self.act_fn(y)

            # remove the artifact from the bias term in g
            # 应用掩码去除无效节点影响
            y = y * masks[d]  # 逐元素相乘，形状保持[num_nodes, feature_dim]

            # assemble neighboring information
            # 聚合邻居信息到当前节点
            x = x + y  # 残差连接（原代码中的x = x + y）

        return x