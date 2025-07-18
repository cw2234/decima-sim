import torch
import torch.nn as nn
import torch.nn.functional as F


def expand_act_on_state(state, sub_acts):
    # expand the state by explicitly adding in actions
    batch_size = state.shape[0]
    num_nodes = state.shape[1]
    num_features = state.shape[2]  # deterministic
    expand_dim = len(sub_acts)

    # replicate the state
    state = state.repeat(1, 1, expand_dim)
    state = state.reshape(batch_size, num_nodes * expand_dim, num_features)


    # prepare the appended sub-actions
    sub_acts = torch.tensor(sub_acts, dtype=torch.float32, device=state.device)
    sub_acts = sub_acts.reshape(1, 1, expand_dim)
    sub_acts = sub_acts.repeat(1, 1, num_nodes)
    sub_acts = sub_acts.reshape(1, num_nodes * expand_dim, 1)
    sub_acts = sub_acts.repeat(batch_size, 1, 1)


    # concatenate expanded state with sub-action features
    concat_state = torch.cat([state, sub_acts], dim=2)

    return concat_state


def glorot(shape, dtype=torch.float32):
    # Xavier Glorot & Yoshua Bengio (AISTATS 2010) initialization (Eqn 16)
    # 创建空张量
    tensor = torch.empty(shape, dtype=dtype)
    # 应用Xavier均匀分布初始化
    nn.init.xavier_uniform_(tensor)
    # 包装为可训练参数
    return nn.Parameter(tensor)



def leaky_relu(features, alpha=0.2):
    """Compute the Leaky ReLU activation function."""
    return F.leaky_relu(features, alpha)


def masked_outer_product(a, b, mask):
    """
    combine two probability distribution together
    a: batch_size * num_nodes
    b: batch_size * (num_executor_limit * num_jobs)
    """
    batch_size = a.shape[0]
    num_nodes = a.shape[1]
    num_limits = b.shape[1]

    a = a.reshape(batch_size, num_nodes, 1)
    b = b.reshape(batch_size, 1, num_limits)

    # outer matrix product
    outer_product = a * b
    outer_product = outer_product.reshape(batch_size, -1)

    # mask
    outer_product = outer_product.t()
    outer_product = outer_product[mask]
    outer_product = outer_product.t()

    return outer_product



def ones(shape, dtype=torch.float32):
    return nn.Parameter(torch.ones(shape, dtype=dtype))

def zeros(shape, dtype=torch.float32):
    return nn.Parameter(torch.zeros(shape, dtype=dtype))

