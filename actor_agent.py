import bisect

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import msg_passing_path
import pytorch_op
from gcn import GraphCNN
from gsn import GraphSNN
from spark_env.job_dag import JobDAG
from spark_env.node import Node

# 应该是policy network
class ActorNetwork(nn.Module):
    def __init__(self,
                 node_input_dim: int,
                 job_input_dim: int,
                 output_dim: int,
                 executor_levels: range,
                 act_fn: nn.Module):
        super(ActorNetwork, self).__init__()
        self.node_input_dim = node_input_dim
        self.job_input_dim = job_input_dim
        self.output_dim = output_dim
        self.executor_levels = executor_levels
        self.act_fn = act_fn

        # 在forward中将输入进行处理后的特征的长度，4层全连接
        reshape_node_dim = self.node_input_dim + 3 * self.output_dim
        self.fc_node = nn.Sequential(
            nn.Linear(reshape_node_dim, 32),
            self.act_fn,
            nn.Linear(32, 16),
            self.act_fn,
            nn.Linear(16, 8),
            self.act_fn,
            nn.Linear(8, 1),
        )

        # 在forward中将输入进行处理后的特征的长度，4层全连接
        expand_state_dim = self.job_input_dim + 2 * self.output_dim + 1
        self.fc_job = nn.Sequential(
            nn.Linear(expand_state_dim, 32),
            self.act_fn,
            nn.Linear(32, 16),
            self.act_fn,
            nn.Linear(16, 8),
            self.act_fn,
            nn.Linear(8, 1),
        )

    def forward(self,
                node_inputs: Tensor,
                gcn_outputs: Tensor,
                job_inputs: Tensor,
                gsn_dag_summary: Tensor,
                gsn_global_summary: Tensor,
                node_valid_mask: Tensor,
                job_valid_mask: Tensor,
                gsn_summ_backward_map: Tensor):
        # 返回选各个node，选哪个处理器的可能性
        # takes output from graph embedding and raw_input from environment

        batch_size = node_valid_mask.shape[0]

        # (1) reshape node inputs to batch format
        node_inputs_reshape = node_inputs.reshape(batch_size, -1, self.node_input_dim)

        # (2) reshape job inputs to batch format
        job_inputs_reshape = job_inputs.reshape(batch_size, -1, self.job_input_dim)

        # (4) reshape gcn_outputs to batch format
        gcn_outputs_reshape = gcn_outputs.reshape(batch_size, -1, self.output_dim)

        # (5) reshape gsn_dag_summary to batch format
        gsn_dag_summ_reshape = gsn_dag_summary.reshape(batch_size, -1, self.output_dim)

        gsn_summ_backward_map_extend = gsn_summ_backward_map.unsqueeze(0).repeat(batch_size, 1, 1)

        gsn_dag_summ_extend = torch.bmm(gsn_summ_backward_map_extend, gsn_dag_summ_reshape)

        # (6) reshape gsn_global_summary to batch format
        gsn_global_summ_reshape = gsn_global_summary.reshape(batch_size, -1, self.output_dim)
        gsn_global_summ_extend_job = gsn_global_summ_reshape.repeat(1, gsn_dag_summ_reshape.shape[1],
                                                                    1)  # gsn_dag_summ_reshape.shape[1]为所有图所有节点数量
        gsn_global_summ_extend_node = gsn_global_summ_reshape.repeat(1, gsn_dag_summ_extend.shape[1], 1)

        # (4) actor neural network
        # with tf.variable_scope(self.scope):
        # -- part A, the distribution over nodes --
        merge_node = torch.concat([
            node_inputs_reshape, gcn_outputs_reshape,
            gsn_dag_summ_extend,
            gsn_global_summ_extend_node],
            dim=2)

        # 经过全连接
        node_outputs: Tensor = self.fc_node(merge_node)

        # reshape the output dimension (batch_size, total_num_nodes)
        node_outputs = node_outputs.reshape(batch_size, -1)

        # valid mask on node
        node_valid_mask = (node_valid_mask - 1) * 10000.0

        # apply mask
        node_outputs = node_outputs + node_valid_mask

        # do masked softmax over nodes on the graph
        node_outputs = torch.softmax(node_outputs, dim=-1)

        # -- part B, the distribution over executor limits --
        merge_job = torch.concat([
            job_inputs_reshape,
            gsn_dag_summ_reshape,
            gsn_global_summ_extend_job], dim=2)

        expanded_state = pytorch_op.expand_act_on_state(
            merge_job, [l / 50.0 for l in self.executor_levels])

        # 经过全连接
        job_outputs: Tensor = self.fc_job(expanded_state)

        # reshape the output dimension (batch_size, num_jobs * num_exec_limits)
        job_outputs = job_outputs.reshape(batch_size, -1)

        # valid mask on job
        job_valid_mask = (job_valid_mask - 1) * 10000.0

        # apply mask
        job_outputs = job_outputs + job_valid_mask

        # reshape output dimension for softmaxing the executor limits
        # (batch_size, num_jobs, num_exec_limits)
        job_outputs = job_outputs.reshape(batch_size, -1, len(self.executor_levels))

        # do masked softmax over jobs
        job_outputs = torch.softmax(job_outputs, dim=-1)

        return node_outputs, job_outputs


class ActorAgent(nn.Module):
    def __init__(self,
                 node_input_dim: int,
                 job_input_dim: int,
                 hid_dims: list,
                 output_dim: int,
                 max_depth: int,
                 executor_levels: range,
                 eps=1e-6,
                 act_fn=nn.LeakyReLU()):

        super(ActorAgent, self).__init__()
        self.node_input_dim = node_input_dim
        self.job_input_dim = job_input_dim
        self.hid_dims = hid_dims
        self.output_dim = output_dim
        self.max_depth = max_depth
        self.executor_levels = executor_levels
        self.eps = eps
        self.act_fn = act_fn

        # for computing and storing message passing path
        self.postman = msg_passing_path.Postman()

        # input node_inputs，图神经网络层
        self.gcn_layer = GraphCNN(
            self.node_input_dim, self.hid_dims,
            self.output_dim, self.max_depth, self.act_fn)

        # input [node_input, gcn_outputs, axis=1]，图神经网络层
        self.gsn_layer = GraphSNN(
            self.node_input_dim + self.output_dim, self.hid_dims,
            self.output_dim, self.act_fn)

        # map gcn_outputs and raw_inputs to action probabilities
        # node_act_probs: [batch_size, total_num_nodes]
        # job_act_probs: [batch_size, total_num_dags]
        # 策略层
        self.actor_layer = ActorNetwork(self.node_input_dim,
                                        self.job_input_dim,
                                        self.output_dim,
                                        self.executor_levels,
                                        self.act_fn)

    # 计算损失函数
    def loss(self, summ_mats, node_act_probs, job_act_probs, node_act_vec, job_act_vec, adv, entropy_weight):
        '''

        Args:
            node_act_probs:
            job_act_probs:
            node_act_vec: Selected action for node, 0-1 vector ([batch_size, total_num_nodes])
            job_act_vec: Selected action for job, 0-1 vector ([batch_size, num_jobs, num_limits])
            adv: advantage term (from Monte Calro or critic) ([batch_size, 1])
            entropy_weight: use entropy to promote exploration, this term decays over time, float

        Returns:

        '''

        # select node action probability
        selected_node_prob = torch.sum(node_act_probs * node_act_vec, dim=1, keepdim=True)

        # select job action probability
        selected_job_prob = torch.sum(
            torch.sum(job_act_probs * job_act_vec, dim=2, keepdim=False),
            dim=1, keepdim=True
        )

        # actor loss due to advantge (negated)
        adv_loss = torch.sum(
            torch.log(selected_node_prob * selected_job_prob + self.eps) * (-adv)
        )

        # node_entropy
        node_entropy = torch.sum(
            node_act_probs * torch.log(node_act_probs + self.eps)
        )

        # prob on each job
        prob_each_job = torch.sparse.mm(summ_mats[0], node_act_probs.reshape(-1, 1)).reshape(node_act_probs.shape[0], -1)

        # job entropy
        job_entropy = torch.sum(
            prob_each_job * torch.sum(job_act_probs * torch.log(job_act_probs + self.eps), dim=2)
        )

        # entropy loss
        entropy_loss = node_entropy + job_entropy

        # normalize entropy
        entropy_loss /= (
                torch.log(torch.tensor(node_act_probs.shape[1], dtype=torch.float32)) +
                torch.log(torch.tensor(len(self.executor_levels), dtype=torch.float32)))
        # normalize over batch size (note: adv_loss is sum)
        # * tf.cast(tf.shape(self.node_act_probs)[0], tf.float32)

        # define combined loss
        act_loss = adv_loss + entropy_weight * entropy_loss
        return act_loss, adv_loss, entropy_loss

    # 这个用来算梯度的
    def forward(self, node_inputs, job_inputs,
                node_valid_mask, job_valid_mask,
                gcn_mats, gcn_masks, summ_mats,
                running_dags_mat, dag_summ_backward_map,
                node_act_vec, job_act_vec, adv, entropy_weight):
        node_act_probs, job_act_probs, node_acts, job_acts = self.predict(node_inputs, job_inputs, node_valid_mask,
                                                                          job_valid_mask, gcn_mats, gcn_masks,
                                                                          summ_mats, running_dags_mat,
                                                                          dag_summ_backward_map)

        act_loss, adv_loss, entropy_loss = self.loss([summ_mats, running_dags_mat], node_act_probs, job_act_probs, node_act_vec, job_act_vec, adv, entropy_weight)
        # 用act_loss.backward()来累积梯度
        return act_loss, [adv_loss, entropy_loss]


    def save_model(self, file_path):
        torch.save(self.state_dict(), file_path)


    def predict(self, node_inputs, job_inputs,
                node_valid_mask, job_valid_mask,
                gcn_mats, gcn_masks, summ_mats,
                running_dags_mat, dag_summ_backward_map):

        gcn_outputs = self.gcn_layer(node_inputs, gcn_mats, gcn_masks)  # 网络输出
        summarys = self.gsn_layer(torch.concat([node_inputs, gcn_outputs], dim=1), [summ_mats, running_dags_mat])

        node_act_probs, job_act_probs = self.actor_layer(
            node_inputs,
            gcn_outputs,
            job_inputs,
            summarys[0],  # dag
            summarys[1],  # global
            node_valid_mask,
            job_valid_mask,
            dag_summ_backward_map
        )

        # draw action based on the probability (from OpenAI baselines)
        # node_acts [batch_size, 1]
        # 计算对数概率
        logits = torch.log(node_act_probs)
        # 生成均匀噪声
        noise = torch.rand_like(logits)
        node_acts = torch.argmax(logits - torch.log(-torch.log(noise)), 1)

        # job_acts [batch_size, num_jobs, 1]
        logits = torch.log(job_act_probs)
        noise = torch.rand_like(logits)
        job_acts = torch.argmax(logits - torch.log(-torch.log(noise)), 2)

        return node_act_probs, job_act_probs, node_acts, job_acts

    def translate_state(self, obs):
        """
        Translate the observation to matrix form
        将一些量转成Tensor
        """
        (job_dags,
         source_job,
         num_source_exec,
         frontier_nodes,
         executor_limits,
         exec_commit,
         moving_executors,
         action_map) = obs

        # compute total number of nodes
        total_num_nodes = int(np.sum(job_dag.num_nodes for job_dag in job_dags))

        # job and node inputs to feed
        node_inputs = np.zeros([total_num_nodes, self.node_input_dim])
        job_inputs = np.zeros([len(job_dags), self.job_input_dim])

        # sort out the exec_map
        exec_map = {}
        for job_dag in job_dags:
            exec_map[job_dag] = len(job_dag.executors)
        # count in moving executors
        for node in moving_executors.moving_executors.values():
            exec_map[node.job_dag] += 1
        # count in executor commit
        for s in exec_commit.commit:
            if isinstance(s, JobDAG):
                j = s
            elif isinstance(s, Node):
                j = s.job_dag
            elif s is None:
                j = None
            else:
                print('source', s, 'unknown')
                exit(1)
            for n in exec_commit.commit[s]:
                if n is not None and n.job_dag != j:
                    exec_map[n.job_dag] += exec_commit.commit[s][n]

        # gather job level inputs
        job_idx = 0
        for job_dag in job_dags:
            # number of executors in the job
            job_inputs[job_idx, 0] = exec_map[job_dag] / 20.0
            # the current executor belongs to this job or not
            if job_dag is source_job:
                job_inputs[job_idx, 1] = 2
            else:
                job_inputs[job_idx, 1] = -2
            # number of source executors
            job_inputs[job_idx, 2] = num_source_exec / 20.0

            job_idx += 1

        # gather node level inputs
        node_idx = 0
        job_idx = 0
        for job_dag in job_dags:
            for node in job_dag.nodes:
                # copy the feature from job_input first
                node_inputs[node_idx, :3] = job_inputs[job_idx, :3]

                # work on the node
                node_inputs[node_idx, 3] = (node.num_tasks - node.next_task_idx) * node.tasks[-1].duration / 100000.0

                # number of tasks left
                node_inputs[node_idx, 4] = (node.num_tasks - node.next_task_idx) / 200.0

                node_idx += 1

            job_idx += 1

        return (torch.tensor(node_inputs, dtype=torch.float32), torch.tensor(job_inputs, dtype=torch.float32),
                job_dags, source_job, num_source_exec,
                frontier_nodes, executor_limits,
                exec_commit, moving_executors,
                exec_map, action_map)

    def get_valid_masks(self, job_dags, frontier_nodes,
                        source_job, num_source_exec, exec_map, action_map):

        job_valid_mask = np.zeros([1, len(job_dags) * len(self.executor_levels)])

        job_valid = {}  # if job is saturated, don't assign node

        base = 0
        for job_dag in job_dags:
            # new executor level depends on the source of executor
            if job_dag is source_job:
                least_exec_amount = exec_map[job_dag] - num_source_exec + 1
                # +1 because we want at least one executor
                # for this job
            else:
                least_exec_amount = exec_map[job_dag] + 1
                # +1 because of the same reason above

            assert least_exec_amount > 0
            assert least_exec_amount <= self.executor_levels[-1] + 1

            # find the index for first valid executor limit
            exec_level_idx = bisect.bisect_left(
                self.executor_levels, least_exec_amount)

            if exec_level_idx >= len(self.executor_levels):
                job_valid[job_dag] = False
            else:
                job_valid[job_dag] = True

            for l in range(exec_level_idx, len(self.executor_levels)):
                job_valid_mask[0, base + l] = 1

            base += self.executor_levels[-1]

        total_num_nodes = int(np.sum(
            job_dag.num_nodes for job_dag in job_dags))

        node_valid_mask = np.zeros([1, total_num_nodes])

        for node in frontier_nodes:
            if job_valid[node.job_dag]:
                act = action_map.inverse_map[node]
                node_valid_mask[0, act] = 1

        return torch.tensor(node_valid_mask, dtype=torch.float32), torch.tensor(job_valid_mask, dtype=torch.float32)

    def invoke_model(self, obs):
        # implement this module here for training
        # (to pick up state and action to record)
        (node_inputs,  # node input dimension: [total_num_nodes, num_features]
         job_inputs,  # job input dimension: [total_num_jobs, num_features]
         job_dags,
         source_job,
         num_source_exec,
         frontier_nodes,
         executor_limits,
         exec_commit,
         moving_executors,
         exec_map,
         action_map) = self.translate_state(obs)

        # get message passing path (with cache)
        (gcn_mats,
         gcn_masks,
         dag_summ_backward_map,
         running_dags_mat,
         job_dags_changed) = self.postman.get_msg_path(job_dags)

        # get node and job valid masks\

        node_valid_mask, job_valid_mask = self.get_valid_masks(job_dags,
                                                               frontier_nodes,
                                                               source_job,
                                                               num_source_exec,
                                                               exec_map,
                                                               action_map)

        # get summarization path that ignores finished nodes
        summ_mats = msg_passing_path.get_unfinished_nodes_summ_mat(job_dags)

        # invoke learning model
        (node_act_probs,
         job_act_probs,
         node_acts,
         job_acts) = self.predict(node_inputs,
                                  job_inputs,
                                  node_valid_mask,
                                  job_valid_mask,
                                  gcn_mats,
                                  gcn_masks,
                                  summ_mats,
                                  running_dags_mat,
                                  dag_summ_backward_map)

        return (node_acts,
                job_acts,
                node_act_probs,
                job_act_probs,
                node_inputs,
                job_inputs,
                node_valid_mask,
                job_valid_mask,
                gcn_mats,
                gcn_masks,
                summ_mats,
                running_dags_mat,
                dag_summ_backward_map,
                exec_map,
                job_dags_changed)

    def get_action(self, obs):

        # parse observation
        (job_dags,
         source_job,
         num_source_exec,
         frontier_nodes,
         executor_limits,
         exec_commit,
         moving_executors,
         action_map) = obs

        if len(frontier_nodes) == 0:
            # no action to take
            return None, num_source_exec

        # invoking the learning model
        (node_act,
         job_act,
         node_act_probs,
         job_act_probs,
         node_inputs,
         job_inputs,
         node_valid_mask,
         job_valid_mask,
         gcn_mats,
         gcn_masks,
         summ_mats,
         running_dags_mat,
         dag_summ_backward_map,
         exec_map,
         job_dags_changed) = self.invoke_model(obs)

        if sum(node_valid_mask[0, :]) == 0:
            # no node is valid to assign
            return None, num_source_exec

        # node_act should be valid
        assert node_valid_mask[0, node_act[0]] == 1

        # parse node action
        node = action_map[node_act[0].item()]

        # find job index based on node
        job_idx = job_dags.index(node.job_dag)

        # job_act should be valid
        assert job_valid_mask[0, job_act[0, job_idx] + len(self.executor_levels) * job_idx] == 1

        # find out the executor limit decision
        if node.job_dag is source_job:
            agent_exec_act = self.executor_levels[job_act[0, job_idx]] - exec_map[node.job_dag] + num_source_exec
        else:
            agent_exec_act = self.executor_levels[job_act[0, job_idx]] - exec_map[node.job_dag]

        # parse job limit action
        use_exec = min(
            node.num_tasks - node.next_task_idx - exec_commit.node_commit[node] - moving_executors.count(node),
            agent_exec_act,
            num_source_exec)

        return node, use_exec
