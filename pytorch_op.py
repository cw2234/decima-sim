import torch
from torch import Tensor

def expand_act_on_state(state: Tensor, sub_acts):
    # expand the state by explicitly adding in actions
    batch_size = state.shape[0]
    num_nodes = state.shape[1]
    num_features = state.shape[2]  # deterministic
    expand_dim = len(sub_acts)

    # replicate the state
    state = state.repeat(1, 1, expand_dim)  # 沿特征维度复制
    state = state.reshape(batch_size, num_nodes * expand_dim, num_features)

    # prepare the appended sub-actions
    sub_acts = torch.tensor(sub_acts, dtype=torch.float32, device=state.device)
    sub_acts = sub_acts.reshape(1, 1, expand_dim)
    sub_acts = sub_acts.repeat(1, 1, num_nodes)  # 沿节点维度复制
    sub_acts = sub_acts.reshape(1, num_nodes * expand_dim, 1)
    sub_acts = sub_acts.repeat(batch_size, 1, 1)  # 沿批次维度复制

    # concatenate expanded state with sub-action features
    concat_state = torch.concat([state, sub_acts], dim=2)

    return concat_state

