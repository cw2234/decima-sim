import os
# os.environ['CUDA_VISIBLE_DEVICES']=''
import time
import torch
import numpy as np
from time import localtime, strftime
from param import args
import utils
from spark_env.env import Environment
from average_reward import AveragePerStepReward
import compute_baselines
import compute_gradients
from actor_agent import ActorAgent
from torch.utils.tensorboard import SummaryWriter



def invoke_model(actor_agent: ActorAgent, obs, experience):
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
     job_dags_changed) = actor_agent.invoke_model(obs)

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
    assert job_valid_mask[0, job_act[0, job_idx] + len(actor_agent.executor_levels) * job_idx] == 1

    # find out the executor limit decision
    if node.job_dag is source_job:
        agent_exec_act = actor_agent.executor_levels[job_act[0, job_idx]] - exec_map[node.job_dag] + num_source_exec
    else:
        agent_exec_act = actor_agent.executor_levels[job_act[0, job_idx]] - exec_map[node.job_dag]

    # parse job limit action
    use_exec = min(node.num_tasks - node.next_task_idx - exec_commit.node_commit[node] - moving_executors.count(node),
                   agent_exec_act, num_source_exec)

    # for storing the action vector in experience
    node_act_vec = torch.zeros(node_act_probs.shape)
    node_act_vec[0, node_act[0]] = 1

    # for storing job index
    job_act_vec = torch.zeros(job_act_probs.shape)
    job_act_vec[0, job_idx, job_act[0, job_idx]] = 1

    # store experience
    experience['node_inputs'].append(node_inputs)
    experience['job_inputs'].append(job_inputs)
    experience['summ_mats'].append(summ_mats)
    experience['running_dag_mat'].append(running_dags_mat)
    experience['node_act_vec'].append(node_act_vec)
    experience['job_act_vec'].append(job_act_vec)
    experience['node_valid_mask'].append(node_valid_mask)
    experience['job_valid_mask'].append(job_valid_mask)
    experience['job_state_change'].append(job_dags_changed)

    if job_dags_changed:
        experience['gcn_mats'].append(gcn_mats)
        experience['gcn_masks'].append(gcn_masks)
        experience['dag_summ_back_mat'].append(dag_summ_backward_map)

    return node, use_exec


def main():
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using device: {device}")

    np.random.seed(args.seed)

    torch.manual_seed(args.seed)

    # create result and model folder
    utils.create_folder_if_not_exists(args.result_folder)
    utils.create_folder_if_not_exists(args.model_folder)

    # model evaluation seed
    torch.manual_seed(0)

    # 创建环境
    env = Environment()

    actor_agent = ActorAgent(
        args.node_input_dim,
        args.job_input_dim,
        args.hid_dims,
        args.output_dim,
        args.max_depth,
        range(1, args.exec_cap + 1))

    optimizer = torch.optim.Adam(actor_agent.parameters(), lr=args.lr)

    # tensorboard logging
    log_dir = os.path.join(args.result_folder, strftime("%Y-%m-%d-%H-%M-%S", localtime()))
    print(log_dir)
    tensorboard_writer = SummaryWriter(log_dir)



    # store average reward for computing differential rewards
    avg_reward_calculator = AveragePerStepReward(args.average_reward_storage_size)

    # initialize entropy parameters
    entropy_weight = args.entropy_weight_init

    # initialize episode reset probability
    reset_prob = args.reset_prob

    # ---- start training process ----
    for ep in range(1, args.num_ep):
        print('training epoch', ep)

        # generate max time stochastically based on reset prob
        max_time = utils.generate_coin_flips(reset_prob)

        seed = args.seed + ep
        # reset environment
        env.seed(seed)
        env.reset(max_time=max_time)

        # set up storage for experience
        experience = {
            'node_inputs': [],
            'job_inputs': [],
            'gcn_mats': [],
            'gcn_masks': [],
            'summ_mats': [],
            'running_dag_mat': [],
            'dag_summ_back_mat': [],
            'node_act_vec': [],
            'job_act_vec': [],
            'node_valid_mask': [],
            'job_valid_mask': [],
            'reward': [],
            'wall_time': [],
            'job_state_change': []
        }

        t1 = time.time()
        try:
            # run experiment
            obs = env.observe()
            done = False

            # initial time
            experience['wall_time'].append(env.wall_time.curr_time)
            while not done:
                with torch.no_grad():
                    node, use_exec = invoke_model(actor_agent, obs, experience)

                obs, reward, done = env.step(node, use_exec)

                if node is not None:
                    # valid action, store reward and time
                    experience['reward'].append(reward)
                    experience['wall_time'].append(env.wall_time.curr_time)
                elif len(experience['reward']) > 0:
                    # Note: if we skip the reward when node is None
                    # (i.e., no available actions), the sneaky
                    # agent will learn to exhaustively pick all
                    # nodes in one scheduling round, in order to
                    # avoid the negative reward
                    experience['reward'][-1] += reward
                    experience['wall_time'][-1] = env.wall_time.curr_time

            # report reward signals to master
            assert len(experience['node_inputs']) == len(experience['reward'])
            result = [experience['reward'], experience['wall_time'], len(env.finished_job_dags),
                      np.mean([j.completion_time - j.start_time for j in env.finished_job_dags]),
                      env.wall_time.curr_time >= env.max_time]

        except AssertionError:
            result = None

        if result is None:
            continue
        else:
            batch_reward, batch_time, num_finished_jobs, avg_job_duration, reset_hit = result

            diff_time = np.array(batch_time[1:]) - np.array(batch_time[:-1])

            avg_reward_calculator.add_list_filter_zero(batch_reward, diff_time)

        t2 = time.time()
        print('got reward from workers', t2 - t1, 'seconds')

        # compute differential reward
        avg_per_step_reward = avg_reward_calculator.get_avg_per_step_reward()

        if args.diff_reward_enabled:
            # differential reward mode on
            rewards = np.array([r - avg_per_step_reward * t for (r, t) in zip(batch_reward, diff_time)])
        else:
            # regular reward
            rewards = np.array([r for (r, t) in zip(batch_reward, diff_time)])

        # 累积奖励
        cum_reward = utils.discount(rewards, args.gamma)

        # compute baseline，返回的是list，但现在只有一个agent
        baselines = compute_baselines.get_piecewise_linear_fit_baseline([cum_reward], [batch_time[1:]])

        # 变成一项
        baselines = baselines[0]  # 一个agent的baseline
        # give worker back the advantage
        batch_adv = cum_reward - baselines
        batch_adv = np.reshape(batch_adv, [len(batch_adv), 1])
        batch_adv = torch.tensor(batch_adv)

        # compute gradients
        optimizer.zero_grad() # 清除梯度

        # 累积梯度
        loss = compute_gradients.compute_actor_gradients(actor_agent, experience, batch_adv,
                                                                         entropy_weight)

        t3 = time.time()
        print('advantage ready', t3 - t2, 'seconds')

        # 用于tensorboard日志
        action_loss = loss[0]
        entropy = -loss[1] / float(cum_reward.shape[0])
        value_loss = loss[2]

        t4 = time.time()
        print('worker send back gradients', t4 - t3, 'seconds')

        optimizer.step() # 应用梯度

        t5 = time.time()
        print('apply gradient', t5 - t4, 'seconds')

        # 打印到tensorboard
        if tensorboard_writer:
            tensorboard_writer.add_scalar('actor_loss', action_loss, ep)
            tensorboard_writer.add_scalar('entropy', entropy, ep)
            tensorboard_writer.add_scalar('value_loss', value_loss, ep)
            tensorboard_writer.add_scalar('episode_length', len(baselines), ep)
            tensorboard_writer.add_scalar('average_reward_per_second', avg_per_step_reward * args.reward_scale, ep)
            tensorboard_writer.add_scalar('sum_reward', cum_reward[0], ep)
            tensorboard_writer.add_scalar('reset_probability', reset_prob, ep)
            tensorboard_writer.add_scalar('num_jobs', num_finished_jobs, ep)
            tensorboard_writer.add_scalar('reset_hit', reset_hit, ep)
            tensorboard_writer.add_scalar('average_job_duration', avg_job_duration, ep)
            tensorboard_writer.add_scalar('entropy_weight', entropy_weight, ep)

        # decrease entropy weight
        entropy_weight = utils.decrease_var(entropy_weight, args.entropy_weight_min, args.entropy_weight_decay)

        # decrease reset probability
        reset_prob = utils.decrease_var(reset_prob, args.reset_prob_min, args.reset_prob_decay)

        if ep % args.model_save_interval == 0:
            actor_agent.save_model(args.model_folder + 'model_ep_' + str(ep) + '.pth')


if __name__ == '__main__':
    main()
