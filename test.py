import torch
import os
import matplotlib
matplotlib.use('agg')
from spark_env.env import Environment
from agent_for_test.spark_agent import SparkAgent
from agent_for_test.heuristic_agent import DynamicPartitionAgent
from actor_agent import ActorAgent
from spark_env.canvas import *
from param import args
import utils


# create result folder
if not os.path.exists(args.result_folder):
    os.makedirs(args.result_folder)

# tensorflo seeding
torch.manual_seed(args.seed)

# set up environment
env = Environment()

# set up agents
agents = {}

for scheme in args.test_schemes:
    if scheme == 'learn':
        agents[scheme] = ActorAgent(
            args.node_input_dim, args.job_input_dim,
            args.hid_dims, args.output_dim, args.max_depth,
            range(1, args.exec_cap + 1))
        if args.saved_model is not None:
            agents[scheme].load_state_dict(torch.load(args.saved_model, weights_only=True))
    elif scheme == 'dynamic_partition':
        agents[scheme] = DynamicPartitionAgent()
    elif scheme == 'spark_fifo':
        agents[scheme] = SparkAgent(exec_cap=args.exec_cap)
    else:
        print('scheme ' + str(scheme) + ' not recognized')
        exit(1)

# store info for all schemes
all_total_reward = {}
for scheme in args.test_schemes:
    all_total_reward[scheme] = []


for exp in range(args.num_exp):
    print('Experiment ' + str(exp + 1) + ' of ' + str(args.num_exp))

    for scheme in args.test_schemes:
        # reset environment with seed
        env.seed(args.num_ep + exp)
        env.reset()

        # load an agent
        agent = agents[scheme]

        # start experiment
        obs = env.observe()

        total_reward = 0
        done = False

        while not done:
            with torch.no_grad():
                node, use_exec = agent.get_action(obs)
            obs, reward, done = env.step(node, use_exec)
            total_reward += reward

        all_total_reward[scheme].append(total_reward)

        if args.canvs_visualization:
            visualize_dag_time_save_pdf(
                env.finished_job_dags, env.executors,
                args.result_folder + 'visualization_exp_' + str(exp) + '_scheme_' + scheme + '.png', plot_type='app')
        else:
            visualize_executor_usage(env.finished_job_dags,
                args.result_folder + 'visualization_exp_' + str(exp) + '_scheme_' + scheme + '.png')


    # plot CDF of performance

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for scheme in args.test_schemes:
        x, y = utils.compute_CDF(all_total_reward[scheme])
        ax.plot(x, y)

    plt.xlabel('Total reward')
    plt.ylabel('CDF')
    plt.legend(args.test_schemes)
    fig.savefig(args.result_folder + 'total_reward.png')

    plt.close(fig)
