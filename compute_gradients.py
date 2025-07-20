import numpy as np
import utils
import torch
from actor_agent import ActorAgent
from sparse_op import expand_sp_mat, merge_and_extend_sp_mat


def compute_actor_gradients(actor_agent: ActorAgent, exp, batch_adv, entropy_weight):
    batch_points = utils.truncate_experiences(exp['job_state_change'])

    all_gradients = []
    all_loss = [[], [], 0]

    for b in range(len(batch_points) - 1):
        # need to do different batches because the
        # size of dags in state changes
        ba_start = batch_points[b]
        ba_end = batch_points[b + 1]

        # use a piece of experience
        node_inputs = torch.cat(exp['node_inputs'][ba_start : ba_end], dim=0)
        job_inputs = torch.cat(exp['job_inputs'][ba_start : ba_end], dim=0)
        node_act_vec = torch.cat(exp['node_act_vec'][ba_start : ba_end], dim=0)
        job_act_vec = torch.cat(exp['job_act_vec'][ba_start : ba_end], dim=0)
        node_valid_mask = torch.cat(exp['node_valid_mask'][ba_start : ba_end], dim=0)
        job_valid_mask = torch.cat(exp['job_valid_mask'][ba_start : ba_end], dim=0)
        summ_mats = exp['summ_mats'][ba_start : ba_end]
        running_dag_mats = exp['running_dag_mat'][ba_start : ba_end]
        adv = batch_adv[ba_start : ba_end, :]
        gcn_mats = exp['gcn_mats'][b]
        gcn_masks = exp['gcn_masks'][b]
        summ_backward_map = exp['dag_summ_back_mat'][b]

        # given an episode of experience (advantage computed from baseline)
        batch_size = node_act_vec.shape[0]

        # expand sparse adj_mats
        extended_gcn_mats = expand_sp_mat(gcn_mats, batch_size)

        # extended masks
        # (on the dimension according to extended adj_mat)
        extended_gcn_masks = [m.repeat(batch_size, 1) for m in gcn_masks]

        # expand sparse summ_mats
        extended_summ_mats = merge_and_extend_sp_mat(summ_mats)

        # expand sparse running_dag_mats
        extended_running_dag_mats = merge_and_extend_sp_mat(running_dag_mats)

        # compute gradient

        act_loss, loss = actor_agent(
            node_inputs, job_inputs,
            node_valid_mask, job_valid_mask,
            extended_gcn_mats, extended_gcn_masks,
            extended_summ_mats, extended_running_dag_mats,
            summ_backward_map, node_act_vec, job_act_vec,
            adv, entropy_weight)

        act_loss.backward() # 累积梯度
        # all_gradients.append(act_gradients)
        all_loss[0].append(loss[0])
        all_loss[1].append(loss[1])

    all_loss[0] = torch.sum(torch.stack(all_loss[0]))
    all_loss[1] = torch.sum(torch.stack(all_loss[1]))  # to get entropy
    all_loss[2] = torch.sum(batch_adv ** 2) # time based baseline loss

    # aggregate all gradients from the batches
    # gradients = utils.aggregate_gradients(all_gradients)

    return all_loss
