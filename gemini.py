import os
import sys

# os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # This is a TF env var, but harmless to keep
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.multiprocessing as mp

import bisect
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import networkx as nx
import heapq
import itertools

############################################################################################################################
import argparse

parser = argparse.ArgumentParser(description='DAG_ML_PyTorch')

# -- Basic --
parser.add_argument('--seed', type=int, default=42,
                    help='random seed (default: 42)')
parser.add_argument('--eps', type=float, default=1e-6,
                    help='epsilon (default: 1e-6)')
# Reduced num_proc for local testing, can be increased
parser.add_argument('--num_proc', type=int, default=1,
                    help='number of processors (default: 1)')
parser.add_argument('--num_exp', type=int, default=10,
                    help='number of experiments (default: 10)')
parser.add_argument('--query_type', type=str, default='tpch',
                    help='query type (default: tpch)')
parser.add_argument('--job_folder', type=str, default='./spark_env/tpch/',
                    help='job folder path (default: ./spark_env/tpch/)')
parser.add_argument('--result_folder', type=str, default='./results/',
                    help='Result folder path (default: ./results)')
parser.add_argument('--model_folder', type=str, default='./models/',
                    help='Model folder path (default: ./models)')

# -- Environment --
parser.add_argument('--exec_cap', type=int, default=100,
                    help='Number of total executors (default: 100)')
parser.add_argument('--num_init_dags', type=int, default=10,
                    help='Number of initial DAGs in system (default: 10)')
parser.add_argument('--num_stream_dags', type=int, default=100,
                    help='number of streaming DAGs (default: 100)')
parser.add_argument('--num_stream_dags_grow', type=float, default=0.2,
                    help='growth rate of number of streaming jobs  (default: 0.2)')
parser.add_argument('--num_stream_dags_max', type=float, default=500,
                    help='maximum number of number of streaming jobs (default: 500)')
parser.add_argument('--stream_interval', type=int, default=25000,
                    help='inter job arrival time in milliseconds (default: 25000)')
parser.add_argument('--executor_data_point', type=int,
                    default=[5, 10, 20, 40, 50, 60, 80, 100], nargs='+',
                    help='Number of executors used in data collection')
parser.add_argument('--reward_scale', type=float, default=100000.0,
                    help='scale the reward to some normal values (default: 100000.0)')
parser.add_argument('--alibaba', type=bool, default=False,
                    help='Use Alibaba dags (defaule: False)')
parser.add_argument('--var_num_dags', type=bool, default=False,
                    help='Vary number of dags in batch (default: False)')
parser.add_argument('--moving_delay', type=int, default=2000,
                    help='Moving delay (milliseconds) (default: 2000)')
parser.add_argument('--warmup_delay', type=int, default=1000,
                    help='Executor warming up delay (milliseconds) (default: 1000)')
parser.add_argument('--diff_reward_enabled', type=int, default=0,
                    help='Enable differential reward (default: 0)')
parser.add_argument('--new_dag_interval', type=int, default=10000,
                    help='new DAG arrival interval (default: 10000 milliseconds)')
parser.add_argument('--new_dag_interval_noise', type=int, default=1000,
                    help='new DAG arrival interval noise (default: 1000 milliseconds)')

# -- Multi resource environment --
parser.add_argument('--exec_group_num', type=int,
                    default=[50, 50], nargs='+',
                    help='Number of executors in each type group (default: [50, 50])')
parser.add_argument('--exec_cpus', type=float,
                    default=[1.0, 1.0], nargs='+',
                    help='Amount of CPU in each type group (default: [1.0, 1.0])')
parser.add_argument('--exec_mems', type=float,
                    default=[1.0, 0.5], nargs='+',
                    help='Amount of memory in each type group (default: [1.0, 0.5])')

# -- Evaluation --
parser.add_argument('--test_schemes', type=str,
                    default=['dynamic_partition'], nargs='+',
                    help='Schemes for testing the performance')

# -- TPC-H --
parser.add_argument('--tpch_size', type=str,
                    default=['2g', '5g', '10g', '20g', '50g', '80g', '100g'], nargs='+',
                    help='Numer of TPCH queries (default: [2g, 5g, 10g, 20g, 50g, 80g, 100g])')
parser.add_argument('--tpch_num', type=int, default=22,
                    help='Numer of TPCH queries (default: 22)')

# -- Visualization --
parser.add_argument('--canvs_visualization', type=int, default=1,
                    help='Enable canvs visualization (default: 1)')
parser.add_argument('--canvas_base', type=int, default=-10,
                    help='Canvas color scale bottom (default: -10)')

# -- Learning --
parser.add_argument('--node_input_dim', type=int, default=5,
                    help='node input dimensions to graph embedding (default: 5)')
parser.add_argument('--job_input_dim', type=int, default=3,
                    help='job input dimensions to graph embedding (default: 3)')
parser.add_argument('--hid_dims', type=int, default=[16, 8], nargs='+',
                    help='hidden dimensions throughout graph embedding (default: [16, 8])')
parser.add_argument('--output_dim', type=int, default=8,
                    help='output dimensions throughout graph embedding (default: 8)')
parser.add_argument('--max_depth', type=int, default=8,
                    help='Maximum depth of root-leaf message passing (default: 8)')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate (default: 0.001)')
parser.add_argument('--ba_size', type=int, default=64,
                    help='Batch size (default: 64)')
parser.add_argument('--gamma', type=float, default=1,
                    help='discount factor (default: 1)')
parser.add_argument('--early_terminate', type=int, default=0,
                    help='Terminate the episode when stream is empty (default: 0)')
parser.add_argument('--entropy_weight_init', type=float, default=1,
                    help='Initial exploration entropy weight (default: 1)')
parser.add_argument('--entropy_weight_min', type=float, default=0.0001,
                    help='Final minimum entropy weight (default: 0.0001)')
parser.add_argument('--entropy_weight_decay', type=float, default=1e-3,
                    help='Entropy weight decay rate (default: 1e-3)')
parser.add_argument('--log_file_name', type=str, default='log',
                    help='log file name (default: log)')
parser.add_argument('--master_num_gpu', type=int, default=0,
                    help='Number of GPU cores used in master (default: 0)')
parser.add_argument('--worker_num_gpu', type=int, default=0,
                    help='Number of GPU cores used in worker (default: 0)')
parser.add_argument('--average_reward_storage_size', type=int, default=100000,
                    help='Storage size for computing average reward (default: 100000)')
parser.add_argument('--reset_prob', type=float, default=0,
                    help='Probability for episode to reset (after x seconds) (default: 0)')
parser.add_argument('--reset_prob_decay', type=float, default=0,
                    help='Decay rate of reset probability (default: 0)')
parser.add_argument('--reset_prob_min', type=float, default=0,
                    help='Minimum of decay probability (default: 0)')
parser.add_argument('--num_agents', type=int, default=4,  # Default to 4 for easier local run
                    help='Number of parallel agents (default: 16)')
parser.add_argument('--num_ep', type=int, default=10000,  # Reduced for faster run
                    help='Number of training epochs (default: 10000000)')
parser.add_argument('--learn_obj', type=str, default='mean',
                    help='Learning objective (default: mean)')
parser.add_argument('--saved_model', type=str, default=None,
                    help='Path to the saved PyTorch model (default: None)')
parser.add_argument('--check_interval', type=float, default=0.01,
                    help='interval for master to check gradient report (default: 10ms)')
parser.add_argument('--model_save_interval', type=int, default=1000,
                    help='Interval for saving PyTorch model (default: 1000)')
parser.add_argument('--num_saved_models', type=int, default=10,  # Reduced
                    help='Number of models to keep (default: 1000)')

# -- Spark interface --
parser.add_argument('--scheduler_type', type=str, default='dynamic_partition',
                    help='type of scheduling algorithm (default: dynamic_partition)')

args = parser.parse_args()

# PyTorch device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() and args.master_num_gpu > 0 else "cpu")
WORKER_DEVICE = torch.device("cuda" if torch.cuda.is_available() and args.worker_num_gpu > 0 else "cpu")


############################################################################################################################
# FRAMEWORK-INDEPENDENT UTILS (UNCHANGED)
############################################################################################################################

def aggregate_gradients(gradients):
    # This logic is adapted for PyTorch: aggregate state_dicts or gradients
    # In this new design, we average the state dicts.
    # This function is kept for conceptual reference but not used directly for grad aggregation.
    pass


def average_state_dicts(state_dicts):
    """Averages a list of state_dicts."""
    if not state_dicts:
        return None
    avg_state_dict = OrderedDict()
    # First, sum all the tensors
    for key in state_dicts[0].keys():
        # Ensure tensors are on CPU for aggregation
        avg_state_dict[key] = sum(sd[key].cpu() for sd in state_dicts)
    # Then, divide by the number of models
    for key in avg_state_dict.keys():
        avg_state_dict[key] = avg_state_dict[key] / len(state_dicts)
    return avg_state_dict


def compute_CDF(arr, num_bins=100):
    """
    usage: x, y = compute_CDF(arr):
           plt.plot(x, y)
    """
    values, base = np.histogram(arr, bins=num_bins)
    cumulative = np.cumsum(values)
    return base[:-1], cumulative / float(cumulative[-1])


def convert_indices_to_mask(indices, mask_len):
    mask = np.zeros([1, mask_len])
    for idx in indices:
        mask[0, idx] = 1
    return mask


def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def decrease_var(var, min_var, decay_rate):
    if var - decay_rate >= min_var:
        var -= decay_rate
    else:
        var = min_var
    return var


def discount(x, gamma):
    """
    Given vector x, computes a vector y such that
    y[i] = x[i] + gamma * x[i+1] + gamma^2 x[i+2] + ...
    """
    out = np.zeros(x.shape)
    if len(x) > 0:
        out[-1] = x[-1]
        for i in reversed(range(len(x) - 1)):
            out[i] = x[i] + gamma * out[i + 1]
    return out


def generate_coin_flips(p):
    # generate coin flip until first head, with Pr(head) = p
    # this follows a geometric distribution
    if p == 0:
        # infinite sequence
        return np.inf

    # use geometric distribution
    flip_counts = np.random.geometric(p)

    return flip_counts


def get_outer_product_boolean_mask(job_dags, executor_limits):
    num_nodes = sum([j.num_nodes for j in job_dags])
    num_jobs = len(job_dags)
    num_exec_limits = len(executor_limits)
    mask = np.zeros([num_nodes, num_jobs * num_exec_limits], dtype=np.bool)

    # fill in valid entries
    base = 0
    for i in range(len(job_dags)):
        job_dag = job_dags[i]
        mask[base: base + job_dag.num_nodes,
        i * num_exec_limits: (i + 1) * num_exec_limits] = True
        base += job_dag.num_nodes

    # reshape into 1D array
    mask = np.reshape(mask, [-1])

    return mask


def get_poly_baseline(polyfit_model, all_wall_time):
    # use 5th order polynomial to get a baseline
    # normalize the time
    max_time = float(max([max(wall_time) for wall_time in all_wall_time]))
    max_time = max(1, max_time)
    baselines = []
    for i in range(len(all_wall_time)):
        normalized_time = [t / max_time for t in all_wall_time[i]]
        baseline = polyfit_model[0] * np.power(normalized_time, 5) + \
                   polyfit_model[1] * np.power(normalized_time, 4) + \
                   polyfit_model[2] * np.power(normalized_time, 3) + \
                   polyfit_model[3] * np.power(normalized_time, 2) + \
                   polyfit_model[4] * np.power(normalized_time, 1) + \
                   polyfit_model[5]
        baselines.append(baseline)
    return baselines


def get_wall_time_baseline(all_cum_rewards, all_wall_time):
    # do a 5th order polynomial fit over time
    # all_cum_rewards: list of lists of cumulative rewards
    # all_wall_time:   list of lists of physical time
    assert len(all_cum_rewards) == len(all_wall_time)
    # build one list of all values
    list_cum_rewards = list(itertools.chain.from_iterable(all_cum_rewards))
    list_wall_time = list(itertools.chain.from_iterable(all_wall_time))
    assert len(list_cum_rewards) == len(list_wall_time)
    # normalize the time by the max time
    max_time = float(max(list_wall_time))
    max_time = max(1, max_time)
    list_wall_time = [t / max_time for t in list_wall_time]
    polyfit_model = np.polyfit(list_wall_time, list_cum_rewards, 5)
    baselines = get_poly_baseline(polyfit_model, all_wall_time)
    return baselines


def increase_var(var, max_var, increase_rate):
    if var + increase_rate <= max_var:
        var += increase_rate
    else:
        var = max_var
    return var


def list_to_str(lst):
    """
    convert list of number of a string with space
    """
    return ' '.join([str(e) for e in lst])


def min_nonzero(x):
    min_val = np.inf
    for i in x:
        if i != 0 and i < min_val:
            min_val = i
    return min_val


def moving_average(x, N):
    if len(x) < N:
        return np.array([])
    return np.convolve(x, np.ones((N,)) / N, mode='valid')


class OrderedSet(object):
    def __init__(self, contents=()):
        self.set = OrderedDict((c, None) for c in contents)

    def __contains__(self, item):
        return item in self.set

    def __iter__(self):
        return iter(self.set.keys())

    def __len__(self):
        return len(self.set)

    def add(self, item):
        self.set[item] = None

    def clear(self):
        self.set.clear()

    def index(self, item):
        idx = 0
        for i in self.set.keys():
            if item == i:
                break
            idx += 1
        return idx

    def pop(self):
        if not self.set:
            raise KeyError('pop from an empty OrderedSet')
        item = next(iter(self.set))
        del self.set[item]
        return item

    def remove(self, item):
        del self.set[item]

    def to_list(self):
        return [k for k in self.set]

    def update(self, contents):
        for c in contents:
            self.add(c)


def progress_bar(count, total, status='', pattern='|', back='-'):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = pattern * filled_len + back * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s  %s\r' % (bar, percents, '%', status))
    sys.stdout.flush()

    if count == total:
        print('')


class SetWithCount(object):
    """
    allow duplication in set
    """

    def __init__(self):
        self.set = {}

    def __contains__(self, item):
        return item in self.set

    def add(self, item):
        if item in self.set:
            self.set[item] += 1
        else:
            self.set[item] = 1

    def clear(self):
        self.set.clear()

    def remove(self, item):
        self.set[item] -= 1
        if self.set[item] == 0:
            del self.set[item]


def truncate_experiences(lst):
    """
    truncate experience based on a boolean list
    e.g.,    [True, False, False, True, True, False]
          -> [0, 3, 4, 6]  (6 is dummy)
    """
    batch_points = [i for i, x in enumerate(lst) if x]
    batch_points.append(len(lst))

    return batch_points


############################################################################################################################
# ENVIRONMENT CLASSES AND LOGIC (MOSTLY UNCHANGED)
############################################################################################################################
class two_way_unordered_map(object):
    def __init__(self):
        self.map = {}
        self.inverse_map = {}

    def __setitem__(self, key, value):
        self.map[key] = value
        self.inverse_map[value] = key
        # keys and values should be unique
        assert len(self.map) == len(self.inverse_map)

    def __getitem__(self, key):
        return self.map[key]

    def __len__(self):
        return len(self.map)


def compute_act_map(job_dags):
    # translate action ~ [0, num_nodes_in_all_dags) to node object
    action_map = two_way_unordered_map()
    action = 0
    for job_dag in job_dags:
        for node in job_dag.nodes:
            action_map[action] = node
            action += 1
    return action_map


def get_frontier_acts(job_dags):
    # O(num_total_nodes)
    frontier_actions = []
    base = 0
    for job_dag in job_dags:
        for node_idx in job_dag.frontier_nodes:
            frontier_actions.append(base + node_idx)
        base += job_dag.num_nodes
    return frontier_actions


def visualize_executor_usage(job_dags, file_path):
    exp_completion_time = int(np.ceil(np.max([
                                                 j.completion_time for j in job_dags if j.completion_time != np.inf] + [
                                                 0])))

    if exp_completion_time == 0:
        print("Warning: No jobs completed, skipping executor usage visualization.")
        return

    job_durations = \
        [job_dag.completion_time - \
         job_dag.start_time for job_dag in job_dags if job_dag.completed]

    executor_occupation = np.zeros(exp_completion_time)
    executor_limit = np.ones(exp_completion_time) * args.exec_cap

    num_jobs_in_system = np.zeros(exp_completion_time)

    for job_dag in job_dags:
        for node in job_dag.nodes:
            for task in node.tasks:
                if not np.isnan(task.start_time) and not np.isnan(task.finish_time):
                    start = int(task.start_time)
                    end = int(task.finish_time)
                    if end > start:
                        executor_occupation[start:end] += 1

        if job_dag.arrived and job_dag.start_time is not None:
            start = int(job_dag.start_time)
            end = int(job_dag.completion_time) if job_dag.completed else exp_completion_time
            if end > start:
                num_jobs_in_system[start:end] += 1

    executor_usage = np.sum(executor_occupation) / (np.sum(executor_limit) + args.eps)

    fig = plt.figure()

    plt.subplot(2, 1, 1)
    avg_exec_occ = moving_average(executor_occupation, 10000)
    if avg_exec_occ.size > 0:
        plt.plot(avg_exec_occ)

    plt.ylabel('Number of busy executors')
    avg_comp_time_str = str(np.mean(job_durations)) if job_durations else "N/A"
    plt.title(f'Executor usage: {executor_usage:.2f}\n'
              f'Average completion time: {avg_comp_time_str}')

    plt.subplot(2, 1, 2)
    plt.plot(num_jobs_in_system)
    plt.xlabel('Time (milliseconds)')
    plt.ylabel('Number of jobs in the system')

    fig.savefig(file_path)
    plt.close(fig)


def visualize_dag_time(job_dags, executors, plot_total_time=None, plot_type='stage'):
    dags_makespan = 0
    all_tasks = []
    # 1. compute each DAG's finish time
    # so that we can visualize it later
    dags_finish_time = []
    dags_duration = []
    for dag in job_dags:
        dag_finish_time = 0
        for node in dag.nodes:
            for task in node.tasks:
                all_tasks.append(task)
                if not np.isnan(task.finish_time) and task.finish_time > dag_finish_time:
                    dag_finish_time = task.finish_time
        if dag.completed:
            dags_finish_time.append(dag_finish_time)
            assert dag_finish_time == dag.completion_time
            dags_duration.append(dag_finish_time - dag.start_time)

    # 2. visualize them in a canvas
    if not dags_finish_time and plot_total_time is None:
        return None, [], []  # Nothing to plot

    max_finish = max(dags_finish_time) if dags_finish_time else 0
    if plot_total_time is None:
        canvas = np.ones([len(executors), int(max_finish)]) * args.canvas_base
    else:
        canvas = np.ones([len(executors), int(plot_total_time)]) * args.canvas_base

    base = 0
    bases = {}  # job_dag -> base

    job_dag_list = list(job_dags)

    for job_dag in job_dag_list:
        bases[job_dag] = base
        base += job_dag.num_nodes

    for task in all_tasks:
        if not np.isnan(task.start_time) and not np.isnan(task.finish_time):
            start_time = int(task.start_time)
            finish_time = int(task.finish_time)
            exec_id = task.executor.idx

            if finish_time > start_time and finish_time < canvas.shape[1]:
                if plot_type == 'stage':
                    canvas[exec_id, start_time: finish_time] = \
                        bases[task.node.job_dag] + task.node.idx

                elif plot_type == 'app':
                    try:
                        canvas[exec_id, start_time: finish_time] = \
                            job_dag_list.index(task.node.job_dag)
                    except ValueError:  # If job_dag is not in the list (e.g. from a previous epoch)
                        pass

    return canvas, dags_finish_time, dags_duration


def visualize_dag_time_save_pdf(
        job_dags, executors, file_path, plot_total_time=None, plot_type='stage'):
    canvas, dag_finish_time, dags_duration = \
        visualize_dag_time(job_dags, executors, plot_total_time, plot_type)

    if canvas is None: return

    fig = plt.figure()

    # canvas
    plt.imshow(canvas, interpolation='nearest', aspect='auto')
    # plt.colorbar()
    # each dag finish time
    for finish_time in dag_finish_time:
        plt.plot([finish_time, finish_time],
                 [- 0.5, len(executors) - 0.5], 'r')
    avg_duration_str = str(np.mean(dags_duration)) if dags_duration else "N/A"
    plt.title('average DAG completion time: ' + avg_duration_str)
    fig.savefig(file_path)
    plt.close(fig)


class Executor(object):

    def __init__(self, idx):
        self.idx = idx
        self.task = None
        self.node = None
        self.job_dag = None

    def detach_node(self):
        if self.node is not None and \
                self in self.node.executors:
            self.node.executors.remove(self)
        self.node = None
        self.task = None

    def detach_job(self):
        if self.job_dag is not None and \
                self in self.job_dag.executors:
            self.job_dag.executors.remove(self)
        self.job_dag = None
        self.detach_node()

    def reset(self):
        self.task = None
        self.node = None
        self.job_dag = None


class ExecutorCommit(object):
    def __init__(self):
        # {node/job_dag -> ordered{node -> amount}}
        self.commit = {}
        # {node -> amount}
        self.node_commit = {}
        # {node -> set(nodes/job_dags)}
        self.backward_map = {}

    def __getitem__(self, source):
        return self.commit[source]

    def add(self, source, node, amount):
        # source can be node or job
        # node: executors continuously free up
        # job: free executors

        # add foward connection
        if node not in self.commit[source]:
            self.commit[source][node] = 0
        # add node commit
        self.commit[source][node] += amount
        # add to record of total commit on node
        self.node_commit[node] += amount
        # add backward connection
        self.backward_map[node].add(source)

    def pop(self, source):
        # implicitly assert source in self.commit
        # implicitly assert len(self.commit[source]) > 0
        if not self.commit.get(source):
            return None

        # find the node in the map
        node = next(iter(self.commit[source]))

        # deduct one commitment
        self.commit[source][node] -= 1
        self.node_commit[node] -= 1
        assert self.commit[source][node] >= 0
        assert self.node_commit[node] >= 0

        # remove commitment on job if exhausted
        if self.commit[source][node] == 0:
            del self.commit[source][node]
            self.backward_map[node].remove(source)

        return node

    def add_job(self, job_dag):
        # add commit entry to the map
        self.commit[job_dag] = OrderedDict()
        for node in job_dag.nodes:
            self.commit[node] = OrderedDict()
            self.node_commit[node] = 0
            self.backward_map[node] = set()

    def remove_job(self, job_dag):
        # when removing jobs, the commiment should be all satisfied
        if job_dag in self.commit:
            assert len(self.commit[job_dag]) == 0
            del self.commit[job_dag]

        # clean up commitment to the job
        for node in job_dag.nodes:
            # the executors should all move out
            if node in self.commit:
                assert len(self.commit[node]) == 0
                del self.commit[node]

            if node in self.backward_map:
                for source in list(self.backward_map[node]):  # use list to avoid modification during iteration
                    # remove forward link if node exists in commit[source]
                    if source in self.commit and node in self.commit[source]:
                        del self.commit[source][node]
                # remove backward link
                del self.backward_map[node]
            # remove node commit records
            if node in self.node_commit:
                del self.node_commit[node]

    def reset(self):
        self.commit = {}
        self.node_commit = {}
        self.backward_map = {}
        # for agent to make void action
        self.commit[None] = OrderedDict()
        self.node_commit[None] = 0
        self.backward_map[None] = set()


class FreeExecutors(object):
    def __init__(self, executors):
        self.free_executors = {}
        self.free_executors[None] = OrderedSet()
        for executor in executors:
            self.free_executors[None].add(executor)

    def __getitem__(self, job):
        return self.free_executors[job]

    def contain_executor(self, job, executor):
        return job in self.free_executors and executor in self.free_executors[job]

    def pop(self, job):
        executor = next(iter(self.free_executors[job]))
        self.free_executors[job].remove(executor)
        return executor

    def add(self, job, executor):
        if job is None:
            executor.detach_job()
        else:
            executor.detach_node()
        self.free_executors.setdefault(job, OrderedSet()).add(executor)

    def remove(self, executor):
        if executor.job_dag in self.free_executors:
            if executor in self.free_executors[executor.job_dag]:
                self.free_executors[executor.job_dag].remove(executor)

    def add_job(self, job):
        self.free_executors[job] = OrderedSet()

    def remove_job(self, job):
        if job in self.free_executors:
            # put all free executors to global free pool
            for executor in self.free_executors[job]:
                executor.detach_job()
                self.free_executors[None].add(executor)
            del self.free_executors[job]

    def reset(self, executors):
        self.free_executors = {}
        self.free_executors[None] = OrderedSet()
        for executor in executors:
            self.free_executors[None].add(executor)


class MovingExecutors(object):
    def __init__(self):
        # executor -> node
        self.moving_executors = {}
        # node -> set(executors)
        self.node_track = {}

    def __contains__(self, executor):
        return executor in self.moving_executors

    def __getitem__(self, executor):
        return self.moving_executors[executor]

    def __len__(self):
        return len(self.moving_executors)

    def add(self, executor, node):
        # detach the executor from old job
        executor.detach_job()
        # keep track of moving executor
        self.moving_executors[executor] = node
        self.node_track.setdefault(node, set()).add(executor)

    def pop(self, executor):
        if executor in self.moving_executors:
            node = self.moving_executors[executor]
            if node in self.node_track and executor in self.node_track[node]:
                self.node_track[node].remove(executor)
            del self.moving_executors[executor]
        else:
            # job is completed by the time
            # executor arrives
            node = None
        return node

    def count(self, node):
        return len(self.node_track.get(node, []))

    def add_job(self, job_dag):
        for node in job_dag.nodes:
            self.node_track[node] = set()

    def remove_job(self, job_dag):
        for node in job_dag.nodes:
            if node in self.node_track:
                for executor in self.node_track[node]:
                    if executor in self.moving_executors:
                        del self.moving_executors[executor]
                del self.node_track[node]

    def reset(self):
        self.moving_executors = {}
        self.node_track = {}


class Node(object):
    def __init__(self, idx, tasks, task_duration, wall_time, np_random):
        self.idx = idx
        self.tasks = tasks
        self.wall_time = wall_time
        self.np_random = np_random

        self.task_duration = task_duration

        self.num_tasks = len(tasks)
        self.num_finished_tasks = 0
        self.next_task_idx = 0
        self.no_more_tasks = False
        self.tasks_all_done = False
        self.node_finish_time = np.inf

        self.executors = OrderedSet()

        # uninitialized
        self.parent_nodes = []
        self.child_nodes = []
        self.descendant_nodes = []
        self.job_dag = None

        self.assign_node_to_tasks()

    def assign_node_to_tasks(self):
        for task in self.tasks:
            task.node = self

    def get_node_duration(self):
        # Warning: this is slow O(num_tasks)
        # get the total duration over all tasks
        duration = 0
        for task in self.tasks:
            duration += task.get_duration()
        return duration

    def is_schedulable(self):
        if self.no_more_tasks:  # no more tasks
            return False
        if self.tasks_all_done:  # node done
            return False
        for node in self.parent_nodes:
            if not node.tasks_all_done:  # a parent node not done
                return False
        return True

    def reset(self):
        for task in self.tasks:
            task.reset()
        self.executors.clear()
        self.num_finished_tasks = 0
        self.next_task_idx = 0
        self.no_more_tasks = False
        self.tasks_all_done = False
        self.node_finish_time = np.inf

    def sample_executor_key(self, num_executors):
        if num_executors == 0:
            return args.executor_data_point[0]

        (left_exec, right_exec) = \
            self.job_dag.executor_interval_map[num_executors]

        executor_key = None

        if left_exec == right_exec:
            executor_key = left_exec

        else:
            rand_pt = self.np_random.randint(1, right_exec - left_exec + 1)
            if rand_pt <= num_executors - left_exec:
                executor_key = left_exec
            else:
                executor_key = right_exec

        if executor_key not in self.task_duration['first_wave']:
            # more executors than number of tasks in the job
            largest_key = 0
            for e in self.task_duration['first_wave']:
                if e > largest_key:
                    largest_key = e
            executor_key = largest_key

        return executor_key

    def schedule(self, executor):
        assert self.next_task_idx < self.num_tasks
        task = self.tasks[self.next_task_idx]

        # task duration is determined by wave
        num_executors = len(self.job_dag.executors)

        # sample an executor point in the data
        executor_key = self.sample_executor_key(num_executors)

        if executor.task is None or \
                executor.task.node.job_dag != task.node.job_dag:
            # the executor never runs a task in this job
            # fresh executor incurrs a warmup delay
            if len(self.task_duration['fresh_durations'][executor_key]) > 0:
                # (1) try to directly retrieve the warmup delay from data
                fresh_durations = \
                    self.task_duration['fresh_durations'][executor_key]
                i = self.np_random.randint(len(fresh_durations))
                duration = fresh_durations[i]
            else:
                # (2) use first wave but deliberately add in a warmup delay
                first_wave = \
                    self.task_duration['first_wave'][executor_key]
                i = self.np_random.randint(len(first_wave))
                duration = first_wave[i] + args.warmup_delay

        elif executor.task is not None and \
                executor.task.node == task.node and \
                len(self.task_duration['rest_wave'][executor_key]) > 0:
            # executor was working on this node
            # the task duration should be retrieved from rest wave
            rest_wave = self.task_duration['rest_wave'][executor_key]
            i = self.np_random.randint(len(rest_wave))
            duration = rest_wave[i]
        else:
            # executor is fresh to this node, use first wave
            if len(self.task_duration['first_wave'][executor_key]) > 0:
                # (1) try to retrieve first wave from data
                first_wave = \
                    self.task_duration['first_wave'][executor_key]
                i = self.np_random.randint(len(first_wave))
                duration = first_wave[i]
            else:
                # (2) first wave doesn't exist, use fresh durations instead
                # (should happen very rarely)
                fresh_durations = \
                    self.task_duration['fresh_durations'][executor_key]
                i = self.np_random.randint(len(fresh_durations))
                duration = fresh_durations[i]

        # detach the executor from old node
        # the executor can run task means it is local
        # to the job at this point
        executor.detach_node()

        # schedule the task
        task.schedule(self.wall_time.curr_time, duration, executor)

        # mark executor as running in the node
        self.executors.add(executor)
        executor.node = self

        self.next_task_idx += 1
        self.no_more_tasks = (self.next_task_idx >= self.num_tasks)

        if self.no_more_tasks:
            if self in self.job_dag.frontier_nodes:
                self.job_dag.frontier_nodes.remove(self)

        return task


class NodeDuration(object):
    # A light-weighted extra storage for node duration

    def __init__(self, node):
        self.node = node

        self.task_idx = 0  # next unscheduled task index
        self.duration = self.node.get_node_duration()

        # uninitialized when node is created
        # but can be initialized when job_dag is created
        self.descendant_work = 0  # total work of descedent nodes
        self.descendant_cp = 0  # critical path of descdent nodes


def dfs_nodes_order_by_id(node, nodes_order):
    # Depth first search by node id, use recursive search
    # this is for faithfully reproduce spark scheduling logic
    parent_id = []
    parent_map = {}
    for n in node.parent_nodes:
        parent_id.append(n.idx)
        parent_map[n.idx] = n
    parent_id = sorted(parent_id)
    for i in parent_id:
        dfs_nodes_order_by_id(parent_map[i], nodes_order)
    if node.idx not in nodes_order:
        nodes_order.append(node.idx)


class JobDAG(object):
    def __init__(self, nodes, adj_mat, name):
        # nodes: list of N nodes
        # adj_mat: N by N 0-1 adjacency matrix, e_ij = 1 -> edge from i to j
        assert len(nodes) == adj_mat.shape[0]
        assert adj_mat.shape[0] == adj_mat.shape[1]

        self.name = name

        self.nodes = nodes
        self.adj_mat = adj_mat

        self.num_nodes = len(self.nodes)
        self.num_nodes_done = 0

        # set of executors currently running on the job
        self.executors = OrderedSet()

        # the computation graph needs to be a DAG
        assert is_dag(self.num_nodes, self.adj_mat)

        # get the set of schedule nodes
        self.frontier_nodes = OrderedSet()
        for node in self.nodes:
            if node.is_schedulable():
                self.frontier_nodes.add(node)

        # assign job dag to node
        self.assign_job_dag_to_node()

        # dag is arrived
        self.arrived = False

        # dag is completed
        self.completed = False

        # dag start ime
        self.start_time = None

        # dag completion time
        self.completion_time = np.inf

        # map a executor number to an interval
        self.executor_interval_map = \
            self.get_executor_interval_map()

    def assign_job_dag_to_node(self):
        for node in self.nodes:
            node.job_dag = self

    def get_executor_interval_map(self):
        executor_interval_map = {}

        data_points = sorted(args.executor_data_point)
        if not data_points: return executor_interval_map

        # get the left most map
        for e in range(data_points[0] + 1):
            executor_interval_map[e] = (data_points[0], data_points[0])

        # get the center map
        for i in range(len(data_points) - 1):
            for e in range(data_points[i] + 1, data_points[i + 1]):
                executor_interval_map[e] = (data_points[i], data_points[i + 1])
            # at the data point
            e = data_points[i + 1]
            executor_interval_map[e] = (data_points[i + 1], data_points[i + 1])

        # get the residual map
        if args.exec_cap > data_points[-1]:
            for e in range(data_points[-1] + 1, args.exec_cap + 1):
                executor_interval_map[e] = (data_points[-1], data_points[-1])

        return executor_interval_map

    def get_nodes_duration(self):
        # Warning: this is slow O(num_nodes * num_tasks)
        # get the duration over all nodes
        duration = 0
        for node in self.nodes:
            duration += node.get_node_duration()
        return duration

    def reset(self):
        for node in self.nodes:
            node.reset()
        self.num_nodes_done = 0
        self.executors = OrderedSet()
        self.frontier_nodes = OrderedSet()
        for node in self.nodes:
            if node.is_schedulable():
                self.frontier_nodes.add(node)
        self.arrived = False
        self.completed = False
        self.start_time = None
        self.completion_time = np.inf

    def update_frontier_nodes(self, node):
        frontier_nodes_changed = False
        for child in node.child_nodes:
            if child.is_schedulable():
                # Original code had a bug: `if child.idx not in self.frontier_nodes:`
                # This check should be on the object, not index, as indices might not be unique across dags.
                if child not in self.frontier_nodes:
                    self.frontier_nodes.add(child)
                    frontier_nodes_changed = True
        return frontier_nodes_changed


def merge_job_dags(job_dags):
    # merge all DAGs into a general big DAG
    # this function will modify the original data structure
    # 1. take nodes from the natural order
    # 2. wire the parent and children across DAGs
    # 3. reconstruct adj_mat by properly connecting
    # the new edges among individual adj_mats

    total_num_nodes = sum([d.num_nodes for d in job_dags])
    nodes = []
    adj_mat = np.zeros([total_num_nodes, total_num_nodes])

    base = 0  # for figuring out new node index
    leaf_nodes = []  # leaf nodes in the current job_dag

    for job_dag in job_dags:

        num_nodes = job_dag.num_nodes

        for n in job_dag.nodes:
            n.idx += base
            nodes.append(n)

        # update the adj matrix
        adj_mat[base: base + num_nodes, \
        base: base + num_nodes] = job_dag.adj_mat

        # fundamental assumption of spark --
        # every job ends with a single final stage
        if base != 0:  # at least second job
            for i in range(num_nodes):
                if np.sum(job_dag.adj_mat[:, i]) == 0:
                    assert len(job_dag.nodes[i].parent_nodes) == 0
                    adj_mat[base - 1, base + i] = 1

        # store a set of new root nodes
        root_nodes = []
        for n in job_dag.nodes:
            if len(n.parent_nodes) == 0:
                root_nodes.append(n)

        # connect the root nodes with leaf nodes
        for root_node in root_nodes:
            for leaf_node in leaf_nodes:
                leaf_node.child_nodes.append(root_node)
                root_node.parent_nodes.append(leaf_node)

        # store a set of new leaf nodes
        leaf_nodes = []
        for n in job_dag.nodes:
            if len(n.child_nodes) == 0:
                leaf_nodes.append(n)

        # update base
        base += num_nodes

    assert len(nodes) == adj_mat.shape[0]

    merged_job_dag = JobDAG(nodes, adj_mat, "merged_dag")

    return merged_job_dag


class JobDAGDuration(object):
    # A light-weighted extra storage for job_dag duration

    def __init__(self, job_dag):
        self.job_dag = job_dag

        self.node_durations = \
            {node: NodeDuration(node) for node in self.job_dag.nodes}

        for node in self.job_dag.nodes:
            # initialize descendant nodes duration
            self.node_durations[node].descendant_work = \
                np.sum([self.node_durations[n].duration \
                        for n in node.descendant_nodes])
            # initialize descendant nodes task duration
            self.node_durations[node].descendant_cp = \
                np.sum([n.tasks[0].duration \
                        for n in node.descendant_nodes])

        self.job_dag_duration = \
            np.sum([self.node_durations[node].duration \
                    for node in self.job_dag.nodes])

        self.nodes_done = {}

    def update_duration(self):
        work_done = 0
        for node in self.job_dag.nodes:
            if node not in self.nodes_done and node.tasks_all_done:
                work_done += self.node_durations[node].duration
                self.nodes_done[node] = node
        self.job_dag_duration -= work_done


def is_dag(num_nodes, adj_mat):
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_mat[i, j] == 1:
                G.add_edge(i, j)
    return nx.is_directed_acyclic_graph(G)


class Task(object):
    def __init__(self, idx, rough_duration, wall_time):
        self.idx = idx
        self.wall_time = wall_time

        self.duration = rough_duration

        # uninitialized
        self.start_time = np.nan
        self.finish_time = np.nan
        self.executor = None
        self.node = None

    def schedule(self, start_time, duration, executor):
        assert np.isnan(self.start_time)
        assert np.isnan(self.finish_time)
        assert self.executor is None

        self.start_time = start_time
        self.duration = duration
        self.finish_time = self.start_time + duration

        # bind the executor to the task and
        # the task with the given executor
        self.executor = executor
        self.executor.task = self
        self.executor.node = self.node
        self.executor.job_dag = self.node.job_dag

    def get_duration(self):
        # get task duration lazily
        if np.isnan(self.start_time):
            # task not scheduled yet
            return self.duration
        elif self.wall_time.curr_time < self.start_time:
            # task not started yet
            return self.duration
        else:
            # task running or completed
            duration = max(0,
                           self.finish_time - self.wall_time.curr_time)
            return duration

    def reset(self):
        self.start_time = np.nan
        self.finish_time = np.nan
        self.executor = None


def load_job(file_path, query_size, query_idx, wall_time, np_random):
    try:
        query_path = file_path + query_size + '/'

        adj_mat = np.load(
            query_path + 'adj_mat_' + str(query_idx) + '.npy', allow_pickle=True)
        task_durations_raw = np.load(
            query_path + 'task_duration_' + str(query_idx) + '.npy', allow_pickle=True)

        # Handle 0-d array issue with .item()
        if task_durations_raw.shape == ():
            task_durations = task_durations_raw.item()
        else:
            task_durations = task_durations_raw

    except FileNotFoundError:
        print(f"Warning: Data for query {query_idx} size {query_size} not found at {query_path}. Skipping job.")
        return None

    assert adj_mat.shape[0] == adj_mat.shape[1]
    assert adj_mat.shape[0] == len(task_durations)

    num_nodes = adj_mat.shape[0]
    nodes = []
    for n in range(num_nodes):
        task_duration = task_durations[n]
        # Ensure there is data to process
        if not task_duration['first_wave']:
            print(f"Warning: No 'first_wave' data for node {n} in query {query_idx}. Skipping node.")
            continue  # or handle appropriately

        e = next(iter(task_duration['first_wave']))

        num_tasks = len(task_duration['first_wave'][e]) + \
                    len(task_duration['rest_wave'][e])

        # pre-process task duration
        pre_process_task_duration(task_duration)
        all_durations = \
            [i for l in task_duration['first_wave'].values() for i in l] + \
            [i for l in task_duration['rest_wave'].values() for i in l] + \
            [i for l in task_duration['fresh_durations'].values() for i in l]
        rough_duration = np.mean(all_durations) if all_durations else 100.0  # Default duration

        # generate tasks in a node
        tasks = []
        for j in range(num_tasks):
            task = Task(j, rough_duration, wall_time)
            tasks.append(task)

        # generate a node
        node = Node(n, tasks, task_duration, wall_time, np_random)
        nodes.append(node)

    # Re-map indices in case some nodes were skipped
    node_map = {old_node.idx: i for i, old_node in enumerate(nodes)}
    for node in nodes:
        node.idx = node_map[node.idx]

    # parent and child node info
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_mat[i, j] == 1:
                # Check if both nodes exist after potential skipping
                if i in node_map and j in node_map:
                    node_i = nodes[node_map[i]]
                    node_j = nodes[node_map[j]]
                    node_i.child_nodes.append(node_j)
                    node_j.parent_nodes.append(node_i)

    # Update adj_mat if nodes were skipped
    if len(nodes) != num_nodes:
        new_adj_mat = np.zeros((len(nodes), len(nodes)))
        for i, node in enumerate(nodes):
            for child in node.child_nodes:
                j = child.idx
                new_adj_mat[i, j] = 1
        adj_mat = new_adj_mat
    else:
        new_adj_mat = adj_mat

    # initialize descendant nodes
    for node in nodes:
        node.descendant_nodes = recursive_find_descendant(node)

    # generate DAG
    job_dag = JobDAG(nodes, new_adj_mat,
                     args.query_type + '-' + query_size + '-' + str(query_idx))

    return job_dag


def pre_process_task_duration(task_duration):
    # remove fresh durations from first wave
    clean_first_wave = {}
    for e in task_duration['first_wave']:
        clean_first_wave[e] = []
        fresh_durations = SetWithCount()
        # O(1) access
        if e in task_duration['fresh_durations']:
            for d in task_duration['fresh_durations'][e]:
                fresh_durations.add(d)
        for d in task_duration['first_wave'][e]:
            if d not in fresh_durations:
                clean_first_wave[e].append(d)
            else:
                # prevent duplicated fresh duration blocking first wave
                fresh_durations.remove(d)

    # fill in nearest neighour first wave
    last_first_wave = []
    for e in sorted(clean_first_wave.keys()):
        if len(clean_first_wave[e]) == 0:
            clean_first_wave[e] = last_first_wave
        last_first_wave = clean_first_wave[e]

    # swap the first wave with fresh durations removed
    task_duration['first_wave'] = clean_first_wave


def recursive_find_descendant(node, visited=None):
    if visited is None:
        visited = set()

    if node in visited:
        # This case handles cycles, but the primary guard is is_dag
        return []

    visited.add(node)

    # The original logic had a potential flaw if descendant_nodes was pre-populated
    # This revised version re-computes it cleanly every time.
    descendants = [node]
    for child_node in node.child_nodes:
        child_descendants = recursive_find_descendant(child_node, visited)
        for dn in child_descendants:
            if dn not in descendants:
                descendants.append(dn)

    # The original code modified node.descendant_nodes in place.
    # We return it instead, to be assigned by the caller.
    return descendants


def generate_alibaba_jobs(np_random, timeline, wall_time):
    print("Alibaba job generation not implemented.")
    return OrderedSet()


def generate_tpch_jobs(np_random, timeline, wall_time):
    job_dags = OrderedSet()
    t = 0

    for _ in range(args.num_init_dags):
        # generate query
        query_idx = str(np_random.randint(args.tpch_num) + 1)
        query_size = args.tpch_size[np_random.randint(len(args.tpch_size))]
        # generate job
        job_dag = load_job(
            args.job_folder, query_size, query_idx, wall_time, np_random)
        if job_dag is None: continue
        # job already arrived, put in job_dags
        job_dag.start_time = t
        job_dag.arrived = True
        job_dags.add(job_dag)

    for _ in range(args.num_stream_dags):
        # poisson process
        t += int(np_random.exponential(args.stream_interval))
        # uniform distribution
        query_size = args.tpch_size[np_random.randint(len(args.tpch_size))]
        query_idx = str(np_random.randint(args.tpch_num) + 1)
        # generate job
        job_dag = load_job(
            args.job_folder, query_size, query_idx, wall_time, np_random)
        if job_dag is None: continue
        # push into timeline
        job_dag.start_time = t
        timeline.push(t, job_dag)

    return job_dags


def generate_jobs(np_random, timeline, wall_time):
    if args.query_type == 'tpch':
        job_dags = generate_tpch_jobs(np_random, timeline, wall_time)

    elif args.query_type == 'alibaba':
        job_dags = generate_alibaba_jobs(np_random, timeline, wall_time)

    else:
        print('Invalid query type ' + args.query_type)
        exit(1)

    return job_dags


class RewardCalculator(object):
    def __init__(self):
        self.job_dags = set()
        self.prev_time = 0

    def get_reward(self, job_dags, curr_time):
        reward = 0

        # add new job into the store of jobs
        for job_dag in job_dags:
            self.job_dags.add(job_dag)

        # now for all jobs (may have completed)
        # compute the elapsed time
        if args.learn_obj == 'mean':
            for job_dag in list(self.job_dags):
                start = job_dag.start_time if job_dag.start_time is not None else 0
                reward -= (min(
                    job_dag.completion_time,
                    curr_time) - max(
                    start,
                    self.prev_time)) / \
                          args.reward_scale

                # if the job is done, remove it from the list
                if job_dag.completed:
                    self.job_dags.remove(job_dag)

        elif args.learn_obj == 'makespan':
            reward -= (curr_time - self.prev_time) / \
                      args.reward_scale

        else:
            print('Unkown learning objective')
            exit(1)

        self.prev_time = curr_time

        return reward

    def reset(self):
        self.job_dags.clear()
        self.prev_time = 0


class Timeline(object):
    def __init__(self):
        # priority queue
        self.pq = []
        # tie breaker
        self.counter = itertools.count()

    def __len__(self):
        return len(self.pq)

    def peek(self):
        if len(self.pq) > 0:
            (key, counter, item) = self.pq[0]
            return key, item
        else:
            return None, None

    def push(self, key, item):
        heapq.heappush(self.pq,
                       (key, next(self.counter), item))

    def pop(self):
        if len(self.pq) > 0:
            (key, counter, item) = heapq.heappop(self.pq)
            return key, item
        else:
            return None, None

    def reset(self):
        self.pq = []
        self.counter = itertools.count()


class WallTime(object):
    """
    A global time object distributed to all
    tasks, nodes and workers in the environment
    """

    def __init__(self):
        self.curr_time = 0.0

    def update_time(self, new_time):
        self.curr_time = new_time

    def increment_time(self, tick):
        self.curr_time += tick

    def reset(self):
        self.curr_time = 0.0


class Environment(object):
    def __init__(self):
        # isolated random number generator
        self.np_random = np.random.RandomState()

        # global timer
        self.wall_time = WallTime()

        # uses priority queue
        self.timeline = Timeline()

        # executors
        self.executors = OrderedSet()
        for exec_id in range(args.exec_cap):
            self.executors.add(Executor(exec_id))

        # free executors
        self.free_executors = FreeExecutors(self.executors)

        # moving executors
        self.moving_executors = MovingExecutors()

        # executor commit
        self.exec_commit = ExecutorCommit()

        # prevent agent keeps selecting the same node
        self.node_selected = set()

        # for computing reward at each step
        self.reward_calculator = RewardCalculator()

        # Will be populated in reset()
        self.job_dags = OrderedSet()
        self.finished_job_dags = OrderedSet()
        self.action_map = two_way_unordered_map()
        self.source_job = None
        self.num_source_exec = 0
        self.exec_to_schedule = OrderedSet()
        self.max_time = np.inf

    def add_job(self, job_dag):
        self.moving_executors.add_job(job_dag)
        self.free_executors.add_job(job_dag)
        self.exec_commit.add_job(job_dag)

    def assign_executor(self, executor, frontier_changed):
        if executor.node is not None and not executor.node.no_more_tasks:
            # keep working on the previous node
            task = executor.node.schedule(executor)
            self.timeline.push(task.finish_time, task)
        else:
            # need to move on to other nodes
            if frontier_changed:
                # frontier changed, need to consult all free executors
                # note: executor.job_dag might change after self.schedule()
                source_job = executor.job_dag
                if len(self.exec_commit[executor.node]) > 0:
                    # directly fulfill the commitment
                    self.exec_to_schedule = {executor}
                    self.schedule()
                else:
                    # free up the executor
                    self.free_executors.add(source_job, executor)
                # then consult all free executors
                self.exec_to_schedule = OrderedSet(self.free_executors[source_job])
                self.source_job = source_job
                self.num_source_exec = len(self.free_executors[source_job])
            else:
                # just need to schedule one current executor
                self.exec_to_schedule = {executor}
                # only care about executors on the node
                if len(self.exec_commit[executor.node]) > 0:
                    # directly fulfill the commitment
                    self.schedule()
                else:
                    # need to consult for ALL executors on the node
                    # Note: self.exec_to_schedule is immediate
                    #       self.num_source_exec is for commit
                    #       so len(self.exec_to_schedule) !=
                    #       self.num_source_exec can happen
                    self.source_job = executor.job_dag
                    self.num_source_exec = len(executor.node.executors)

    def backup_schedule(self, executor):
        # This function is triggered very rarely. A random policy
        # or the learned polici in early iterations might decide
        # to schedule no executors to any job. This function makes
        # sure the cluster is work conservative. Since the backup
        # policy is not strong, the learning agent should learn to
        # not rely on it.
        backup_scheduled = False
        if executor.job_dag is not None:
            # first try to schedule on current job
            for node in executor.job_dag.frontier_nodes:
                if not self.saturated(node):
                    # greedily schedule a frontier node
                    task = node.schedule(executor)
                    self.timeline.push(task.finish_time, task)
                    backup_scheduled = True
                    break
        # then try to schedule on any available node
        if not backup_scheduled:
            schedulable_nodes = self.get_frontier_nodes()
            if len(schedulable_nodes) > 0:
                node = next(iter(schedulable_nodes))
                self.timeline.push(
                    self.wall_time.curr_time + args.moving_delay, executor)
                # keep track of moving executors
                self.moving_executors.add(executor, node)
                backup_scheduled = True
        # at this point if nothing available, leave executor idle
        if not backup_scheduled:
            self.free_executors.add(executor.job_dag, executor)

    def get_frontier_nodes(self):
        # frontier nodes := unsaturated nodes with all parent nodes saturated
        frontier_nodes = OrderedSet()
        for job_dag in self.job_dags:
            for node in job_dag.nodes:
                if not node in self.node_selected and not self.saturated(node):
                    parents_saturated = True
                    for parent_node in node.parent_nodes:
                        if not parent_node.tasks_all_done:  # Changed from self.saturated for correctness
                            parents_saturated = False
                            break
                    if parents_saturated:
                        frontier_nodes.add(node)

        return frontier_nodes

    def get_executor_limits(self):
        # "minimum executor limit" for each job
        # executor limit := {job_dag -> int}
        executor_limit = {}

        for job_dag in self.job_dags:

            if self.source_job == job_dag:
                curr_exec = self.num_source_exec
            else:
                curr_exec = 0

            # note: this does not count in the commit and moving executors
            executor_limit[job_dag] = len(job_dag.executors) - curr_exec

        return executor_limit

    def observe(self):
        return self.job_dags, self.source_job, self.num_source_exec, \
            self.get_frontier_nodes(), self.get_executor_limits(), \
            self.exec_commit, self.moving_executors, self.action_map

    def saturated(self, node):
        # frontier nodes := unsaturated nodes with all parent nodes saturated
        anticipated_task_idx = node.next_task_idx + \
                               self.exec_commit.node_commit.get(node, 0) + \
                               self.moving_executors.count(node)
        # note: anticipated_task_idx can be larger than node.num_tasks
        # when the tasks finish very fast before commitments are fulfilled
        return anticipated_task_idx >= node.num_tasks

    def schedule(self):
        if not self.exec_to_schedule: return
        executor = next(iter(self.exec_to_schedule))
        source = executor.job_dag if executor.node is None else executor.node

        # schedule executors from the source until the commitment is fulfilled
        while len(self.exec_commit[source]) > 0 and \
                len(self.exec_to_schedule) > 0:

            # keep fulfilling the commitment using free executors
            node = self.exec_commit.pop(source)
            if node is None:  # Pop can return None if source is empty
                # No more commitments from this source
                # The remaining executors in exec_to_schedule should be freed
                for exec_to_free in list(self.exec_to_schedule):
                    self.free_executors.add(exec_to_free.job_dag, exec_to_free)
                    self.exec_to_schedule.remove(exec_to_free)
                break

            executor = self.exec_to_schedule.pop()

            # mark executor as in use if it was free executor previously
            if self.free_executors.contain_executor(executor.job_dag, executor):
                self.free_executors.remove(executor)

            if node is None:
                # the next node is explicitly silent, make executor ilde
                if executor.job_dag is not None and \
                        any([not n.no_more_tasks for n in \
                             executor.job_dag.nodes]):
                    # mark executor as idle in its original job
                    self.free_executors.add(executor.job_dag, executor)
                else:
                    # no where to assign, put executor in null pool
                    self.free_executors.add(None, executor)


            elif not node.no_more_tasks:
                # node is not currently saturated
                if executor.job_dag == node.job_dag:
                    # executor local to the job
                    if node in node.job_dag.frontier_nodes:
                        # node is immediately runnable
                        task = node.schedule(executor)
                        self.timeline.push(task.finish_time, task)
                    else:
                        # put executor back in the free pool
                        self.free_executors.add(executor.job_dag, executor)

                else:
                    # need to move executor
                    self.timeline.push(
                        self.wall_time.curr_time + args.moving_delay, executor)
                    # keep track of moving executors
                    self.moving_executors.add(executor, node)

            else:
                # node is already saturated, use backup logic
                self.backup_schedule(executor)

    def step(self, next_node, limit):

        # mark the node as selected
        if next_node is not None:
            assert next_node not in self.node_selected
            self.node_selected.add(next_node)

        # commit the source executor
        if not self.exec_to_schedule:  # No executors to schedule
            self.num_source_exec = 0
        else:
            executor = next(iter(self.exec_to_schedule))
            source = executor.job_dag if executor.node is None else executor.node

            # compute number of valid executors to assign
            if next_node is not None:
                use_exec = min(next_node.num_tasks - next_node.next_task_idx - \
                               self.exec_commit.node_commit.get(next_node, 0) - \
                               self.moving_executors.count(next_node), limit)
            else:
                use_exec = limit

            if use_exec > 0:
                self.exec_commit.add(source, next_node, use_exec)
                # deduct the executors that know the destination
                self.num_source_exec -= use_exec
            else:  # If no execs assigned, this round of decision is over
                self.num_source_exec = 0

            assert self.num_source_exec >= 0

        if self.num_source_exec == 0:
            # now a new scheduling round, clean up node selection
            self.node_selected.clear()
            # all commitments are made, now schedule free executors
            self.schedule()

        # Now run to the next event in the virtual timeline
        while len(self.timeline) > 0 and self.num_source_exec == 0:
            # consult agent by putting executors in source_exec

            new_time, obj = self.timeline.pop()
            self.wall_time.update_time(new_time)

            # case task: a task completion event, and frees up an executor.
            # case query: a new job arrives
            # case executor: an executor arrives at certain job

            if isinstance(obj, Task):  # task completion event
                finished_task = obj
                node = finished_task.node
                node.num_finished_tasks += 1

                # bookkeepings for node completion
                frontier_changed = False
                if node.num_finished_tasks == node.num_tasks:
                    assert not node.tasks_all_done  # only complete once
                    node.tasks_all_done = True
                    node.job_dag.num_nodes_done += 1
                    node.node_finish_time = self.wall_time.curr_time

                    frontier_changed = node.job_dag.update_frontier_nodes(node)

                # assign new destination for the job
                self.assign_executor(finished_task.executor, frontier_changed)

                # bookkeepings for job completion
                if node.job_dag.num_nodes_done == node.job_dag.num_nodes:
                    assert not node.job_dag.completed  # only complete once
                    node.job_dag.completed = True
                    node.job_dag.completion_time = self.wall_time.curr_time
                    self.remove_job(node.job_dag)

            elif isinstance(obj, JobDAG):  # new job arrival event
                job_dag = obj
                # job should be arrived at the first time
                assert not job_dag.arrived
                job_dag.arrived = True
                # inform agent about job arrival when stream is enabled
                self.job_dags.add(job_dag)
                self.add_job(job_dag)
                self.action_map = compute_act_map(self.job_dags)
                # assign free executors (if any) to the new job
                if len(self.free_executors[None]) > 0:
                    self.exec_to_schedule = \
                        OrderedSet(self.free_executors[None])
                    self.source_job = None
                    self.num_source_exec = \
                        len(self.free_executors[None])

            elif isinstance(obj, Executor):  # executor arrival event
                executor = obj
                # pop destination from the tracking record
                node = self.moving_executors.pop(executor)

                if node is not None:
                    # the job is not yet done when executor arrives
                    executor.job_dag = node.job_dag
                    node.job_dag.executors.add(executor)

                    if not node.no_more_tasks:
                        # the node is still schedulable
                        if node in node.job_dag.frontier_nodes:
                            # node is immediately runnable
                            task = node.schedule(executor)
                            self.timeline.push(task.finish_time, task)
                        else:
                            # free up the executor in this job
                            self.free_executors.add(executor.job_dag, executor)
                    else:
                        # the node is saturated or the job is done by the time executor arrives
                        self.backup_schedule(executor)
                else:  # Node/Job might be gone
                    self.backup_schedule(executor)

            else:
                print("illegal event type")
                exit(1)

        # compute reward
        reward = self.reward_calculator.get_reward(
            self.job_dags, self.wall_time.curr_time)

        # no more decision to make, jobs all done or time is up
        done = (self.num_source_exec == 0) and \
               ((len(self.timeline) == 0) or \
                (self.wall_time.curr_time >= self.max_time))

        if done and self.max_time != np.inf:
            assert self.wall_time.curr_time >= self.max_time or \
                   len(self.job_dags) == 0

        return self.observe(), reward, done

    def remove_job(self, job_dag):
        for executor in list(job_dag.executors):
            executor.detach_job()
            # Put executor back into the global free pool
            self.free_executors.add(None, executor)

        self.exec_commit.remove_job(job_dag)
        self.free_executors.remove_job(job_dag)
        self.moving_executors.remove_job(job_dag)
        if job_dag in self.job_dags:
            self.job_dags.remove(job_dag)
        self.finished_job_dags.add(job_dag)
        self.action_map = compute_act_map(self.job_dags)

    def reset(self, max_time=np.inf):
        self.max_time = max_time
        self.wall_time.reset()
        self.timeline.reset()
        self.exec_commit.reset()
        self.moving_executors.reset()
        self.reward_calculator.reset()
        self.finished_job_dags = OrderedSet()
        self.node_selected.clear()
        for executor in self.executors:
            executor.reset()
        self.free_executors.reset(self.executors)
        # generate a set of new jobs
        self.job_dags = generate_jobs(
            self.np_random, self.timeline, self.wall_time)
        # map action to dag_idx and node_idx
        self.action_map = compute_act_map(self.job_dags)
        # add initial set of jobs in the system
        for job_dag in self.job_dags:
            self.add_job(job_dag)
        # put all executors as source executors initially
        self.source_job = None
        self.num_source_exec = len(self.executors)
        self.exec_to_schedule = OrderedSet(self.executors)

        return self.observe()

    def seed(self, seed):
        self.np_random.seed(seed)


# 之后的是主要代码
############################################################################################################################
# PYTORCH MODEL AND TRAINING LOGIC
############################################################################################################################

class GraphCNN(nn.Module):
    def __init__(self, input_dim, hid_dims, output_dim, max_depth, act_fn):
        super(GraphCNN, self).__init__()
        self.input_dim = input_dim
        self.hid_dims = hid_dims
        self.output_dim = output_dim
        self.max_depth = max_depth
        self.act_fn = act_fn

        # h: x -> x'
        self.prep_layers = self._make_mlp(self.input_dim, self.hid_dims, self.output_dim)
        # f: x' -> e
        self.proc_layers = self._make_mlp(self.output_dim, self.hid_dims, self.output_dim)
        # g: e -> e
        self.agg_layers = self._make_mlp(self.output_dim, self.hid_dims, self.output_dim)

        self._initialize_weights()

    def _make_mlp(self, input_dim, hid_dims, output_dim):
        layers = []
        curr_in_dim = input_dim
        for hid_dim in hid_dims:
            layers.append(nn.Linear(curr_in_dim, hid_dim))
            layers.append(self.act_fn)
            curr_in_dim = hid_dim
        layers.append(nn.Linear(curr_in_dim, output_dim))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, adj_mats, masks):
        # x: [total_num_nodes, input_dim]
        # adj_mats: list of [total_num_nodes, total_num_nodes] sparse tensors
        # masks: list of [total_num_nodes, 1] dense tensors

        # Raise x to higher dimension
        x = self.prep_layers(x)
        x = self.act_fn(x)

        for d in range(self.max_depth):
            # Process node features
            y = self.proc_layers(x)
            y = self.act_fn(y)

            # Message passing
            y = torch.sparse.mm(adj_mats[d], y)

            # Aggregate child features
            y = self.agg_layers(y)
            y = self.act_fn(y)

            # Apply mask to remove artifact from bias term in g
            y = y * masks[d]

            # Assemble neighboring information
            x = x + y

        return x


class GraphSNN(nn.Module):
    def __init__(self, input_dim, hid_dims, output_dim, act_fn=F.leaky_relu):
        super(GraphSNN, self).__init__()
        self.input_dim = input_dim
        self.hid_dims = hid_dims
        self.output_dim = output_dim
        self.act_fn = act_fn
        self.summ_levels = 2

        # DAG level summarization
        self.dag_layers = self._make_mlp(self.input_dim, self.hid_dims, self.output_dim)
        # Global level summarization
        self.global_layers = self._make_mlp(self.output_dim, self.hid_dims, self.output_dim)

        self._initialize_weights()

    def _make_mlp(self, input_dim, hid_dims, output_dim):
        layers = []
        curr_in_dim = input_dim
        for hid_dim in hid_dims:
            layers.append(nn.Linear(curr_in_dim, hid_dim))
            layers.append(self.act_fn)
            curr_in_dim = hid_dim
        layers.append(nn.Linear(curr_in_dim, output_dim))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, summ_mats):
        # x: [total_num_nodes, input_dim]
        # summ_mats: list of 2 sparse tensors for DAG and global summary

        summaries = []

        # DAG level summary
        s = self.dag_layers(x)
        s = self.act_fn(s)
        s = torch.sparse.mm(summ_mats[0], s)
        summaries.append(s)

        # Global level summary
        s = self.global_layers(s)
        s = self.act_fn(s)
        s = torch.sparse.mm(summ_mats[1], s)
        summaries.append(s)

        return summaries


class ActorAgent(nn.Module):
    def __init__(self, node_input_dim, job_input_dim, hid_dims, output_dim,
                 max_depth, executor_levels, act_fn=nn.LeakyReLU()):
        super(ActorAgent, self).__init__()
        self.node_input_dim = node_input_dim
        self.job_input_dim = job_input_dim
        self.hid_dims = hid_dims
        self.output_dim = output_dim
        self.max_depth = max_depth
        self.executor_levels = executor_levels
        self.act_fn = act_fn

        self.gcn = GraphCNN(
            self.node_input_dim, self.hid_dims, self.output_dim,
            self.max_depth, self.act_fn)

        self.gsn = GraphSNN(
            self.node_input_dim + self.output_dim, self.hid_dims,
            self.output_dim, self.act_fn)

        # Actor network parts
        # Part A: Node selection
        node_actor_input_dim = self.node_input_dim + self.output_dim * 3  # node_in, gcn_out, dag_summary, global_summary
        self.node_actor_net = self._make_actor_mlp(node_actor_input_dim, [32, 16, 8], 1)

        # Part B: Executor limit selection
        job_feature_dim = self.job_input_dim + self.output_dim * 2  # job_in, dag_summary, global_summary
        # The executor level is added as one feature
        job_actor_input_dim = job_feature_dim + 1
        self.job_actor_net = self._make_actor_mlp(job_actor_input_dim, [32, 16, 8], 1)

        self.postman = Postman()
        self._initialize_weights()

    def _make_actor_mlp(self, input_dim, hid_dims, output_dim):
        layers = []
        curr_dim = input_dim
        for h_dim in hid_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(self.act_fn)
            curr_dim = h_dim
        layers.append(nn.Linear(curr_dim, output_dim))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, node_inputs, job_inputs, gcn_mats, gcn_masks,
                summ_mats, dag_summ_backward_map, node_valid_mask, job_valid_mask):

        batch_size = node_valid_mask.shape[0]

        # Graph embeddings
        gcn_outputs = self.gcn(node_inputs, gcn_mats, gcn_masks)
        gsn_inputs = torch.cat([node_inputs, gcn_outputs], dim=1)
        gsn_dag_summary, gsn_global_summary = self.gsn(gsn_inputs, summ_mats)

        # Reshape for batch processing
        num_nodes_per_instance = node_inputs.shape[0] // batch_size
        num_jobs_per_instance = job_inputs.shape[0] // batch_size

        node_inputs_b = node_inputs.view(batch_size, num_nodes_per_instance, -1)
        gcn_outputs_b = gcn_outputs.view(batch_size, num_nodes_per_instance, -1)

        gsn_dag_summ_b = gsn_dag_summary.view(batch_size, num_jobs_per_instance, -1)
        dag_summ_backward_map_b = dag_summ_backward_map.unsqueeze(0).expand(batch_size, -1, -1)
        gsn_dag_summ_extend = torch.bmm(dag_summ_backward_map_b, gsn_dag_summ_b)

        gsn_global_summ_b = gsn_global_summary.view(batch_size, 1, -1)
        gsn_global_summ_extend_node = gsn_global_summ_b.expand(-1, num_nodes_per_instance, -1)

        # -- Part A: Node action probabilities --
        merge_node = torch.cat([
            node_inputs_b, gcn_outputs_b,
            gsn_dag_summ_extend, gsn_global_summ_extend_node], dim=2)

        node_logits = self.node_actor_net(merge_node).squeeze(-1)  # -> [batch_size, num_nodes]

        # Apply mask
        node_logits[node_valid_mask == 0] = -1e9
        node_probs = F.softmax(node_logits, dim=-1)

        # -- Part B, the distribution over executor limits --
        job_inputs_b = job_inputs.view(batch_size, num_jobs_per_instance, -1)
        gsn_global_summ_extend_job = gsn_global_summ_b.expand(-1, num_jobs_per_instance, -1)

        merge_job = torch.cat([
            job_inputs_b, gsn_dag_summ_b, gsn_global_summ_extend_job], dim=2)

        # Expand state for each executor level action
        num_exec_levels = len(self.executor_levels)
        expanded_job_state = merge_job.unsqueeze(2).expand(-1, -1, num_exec_levels, -1)

        # Create executor level features
        exec_level_feats = torch.tensor([l / 50.0 for l in self.executor_levels], device=node_inputs.device)
        exec_level_feats = exec_level_feats.view(1, 1, num_exec_levels, 1)
        exec_level_feats = exec_level_feats.expand(batch_size, num_jobs_per_instance, -1, -1)

        # Concatenate
        job_actor_inputs = torch.cat([expanded_job_state, exec_level_feats], dim=3)

        job_logits = self.job_actor_net(job_actor_inputs).squeeze(-1)  # -> [batch_size, num_jobs, num_exec_levels]

        # Apply mask
        job_valid_mask_reshaped = job_valid_mask.view(batch_size, num_jobs_per_instance, num_exec_levels)
        job_logits[job_valid_mask_reshaped == 0] = -1e9
        job_probs = F.softmax(job_logits, dim=-1)

        return node_probs, job_probs

    def get_action(self, obs, device):
        # Parse observation and convert to tensors
        (node_inputs_np, job_inputs_np, job_dags, source_job, num_source_exec,
         frontier_nodes, executor_limits, exec_commit, moving_executors,
         exec_map, action_map) = self.translate_state(obs)

        if not frontier_nodes:
            return None, num_source_exec, None  # No action possible

        # Get sparse matrices and masks
        (gcn_mats_sp, gcn_masks_np, dag_summ_backward_map_np,
         running_dags_mat_sp, job_dags_changed) = self.postman.get_msg_path(job_dags)

        # Get valid action masks
        node_valid_mask_np, job_valid_mask_np = self.get_valid_masks(job_dags, frontier_nodes,
                                                                     source_job, num_source_exec, exec_map, action_map)

        # Get summarization path
        summ_mats_sp = get_unfinished_nodes_summ_mat(job_dags)

        # Convert all to PyTorch tensors
        node_inputs = torch.from_numpy(node_inputs_np).float().to(device)
        job_inputs = torch.from_numpy(job_inputs_np).float().to(device)
        gcn_mats = [mat.to_torch_sparse(device) for mat in gcn_mats_sp]
        gcn_masks = [torch.from_numpy(mask).float().to(device) for mask in gcn_masks_np]
        summ_mats = [summ_mats_sp.to_torch_sparse(device), running_dags_mat_sp.to_torch_sparse(device)]
        dag_summ_backward_map = torch.from_numpy(dag_summ_backward_map_np).float().to(device)
        node_valid_mask = torch.from_numpy(node_valid_mask_np).float().to(device)
        job_valid_mask = torch.from_numpy(job_valid_mask_np).float().to(device)

        # Model inference
        self.eval()
        with torch.no_grad():
            node_probs, job_probs = self.forward(
                node_inputs, job_inputs, gcn_mats, gcn_masks,
                summ_mats, dag_summ_backward_map, node_valid_mask, job_valid_mask
            )

        if torch.sum(node_valid_mask) == 0:
            return None, num_source_exec, None

        # Sample action
        node_dist = Categorical(probs=node_probs)
        node_act_tensor = node_dist.sample()
        node_act = node_act_tensor.item()

        # Get corresponding job and sample job action
        selected_node_obj = action_map[node_act]
        job_idx = list(job_dags).index(selected_node_obj.job_dag)

        job_dist = Categorical(probs=job_probs[:, job_idx, :])
        job_act_tensor = job_dist.sample()
        job_act = job_act_tensor.item()

        # Parse action to environment step arguments
        # find out the executor limit decision
        if selected_node_obj.job_dag is source_job:
            agent_exec_act = self.executor_levels[job_act] - exec_map[selected_node_obj.job_dag] + num_source_exec
        else:
            agent_exec_act = self.executor_levels[job_act] - exec_map[selected_node_obj.job_dag]

        # parse job limit action
        use_exec = min(selected_node_obj.num_tasks - selected_node_obj.next_task_idx - exec_commit.node_commit.get(
            selected_node_obj, 0) - moving_executors.count(selected_node_obj),
                       agent_exec_act, num_source_exec)

        # Pack experience for training
        exp_data = {
            "node_inputs": node_inputs_np, "job_inputs": job_inputs_np,
            "gcn_mats": gcn_mats_sp, "gcn_masks": gcn_masks_np,
            "summ_mats": summ_mats_sp, "running_dag_mat": running_dags_mat_sp,
            "dag_summ_back_mat": dag_summ_backward_map_np,
            "node_act": node_act, "job_act": job_act, "job_idx": job_idx,
            "node_valid_mask": node_valid_mask_np, "job_valid_mask": job_valid_mask_np,
            "job_state_change": job_dags_changed
        }

        return selected_node_obj, use_exec, exp_data

    def translate_state(self, obs):
        """
        Translate the observation to matrix form
        """
        job_dags, source_job, num_source_exec, \
            frontier_nodes, executor_limits, \
            exec_commit, moving_executors, action_map = obs

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
            if node.job_dag in exec_map:
                exec_map[node.job_dag] += 1

        # count in executor commit
        for s in exec_commit.commit:
            if isinstance(s, JobDAG):
                j = s
            elif isinstance(s, Node):
                j = s.job_dag
            else:  # s is None
                j = None

            for n in exec_commit.commit[s]:
                if n is not None and n.job_dag != j:
                    if n.job_dag in exec_map:
                        exec_map[n.job_dag] += exec_commit.commit[s][n]

        # gather job level inputs
        job_idx = 0
        job_dag_list = list(job_dags)
        for job_dag in job_dag_list:
            # number of executors in the job
            job_inputs[job_idx, 0] = exec_map.get(job_dag, 0) / 20.0
            # the current executor belongs to this job or not
            if job_dag is source_job:
                job_inputs[job_idx, 1] = 2
            else:
                job_inputs[job_idx, 1] = -2
            # number of source executors
            job_inputs[job_idx, 2] = num_source_exec / 20.0

            job_idx += 1

        # gather node level inputs
        node_idx_counter = 0
        job_idx = 0
        for job_dag in job_dag_list:
            for node in job_dag.nodes:
                # copy the feature from job_input first
                node_inputs[node_idx_counter, :3] = job_inputs[job_idx, :3]

                # work on the node
                # Make sure task list is not empty
                task_duration = node.tasks[-1].duration if node.tasks else 0.0
                node_inputs[node_idx_counter, 3] = \
                    (node.num_tasks - node.next_task_idx) * \
                    task_duration / 100000.0

                # number of tasks left
                node_inputs[node_idx_counter, 4] = (node.num_tasks - node.next_task_idx) / 200.0

                node_idx_counter += 1

            job_idx += 1

        return (node_inputs, job_inputs,
                job_dags, source_job, num_source_exec,
                frontier_nodes, executor_limits,
                exec_commit, moving_executors,
                exec_map, action_map)

    def get_valid_masks(self, job_dags, frontier_nodes,
                        source_job, num_source_exec, exec_map, action_map):

        num_jobs = len(job_dags)
        job_valid_mask = np.zeros([1, num_jobs * len(self.executor_levels)])

        job_valid = {}  # if job is saturated, don't assign node

        base = 0
        job_dag_list = list(job_dags)

        for i, job_dag in enumerate(job_dag_list):
            # new executor level depends on the source of executor
            current_execs = exec_map.get(job_dag, 0)
            if job_dag is source_job:
                least_exec_amount = current_execs - num_source_exec + 1
            else:
                least_exec_amount = current_execs + 1

            # find the index for first valid executor limit
            exec_level_idx = bisect.bisect_left(
                self.executor_levels, least_exec_amount)

            job_valid[job_dag] = exec_level_idx < len(self.executor_levels)

            for l in range(exec_level_idx, len(self.executor_levels)):
                job_valid_mask[0, i * len(self.executor_levels) + l] = 1

        total_num_nodes = action_map.__len__()
        node_valid_mask = np.zeros([1, total_num_nodes])

        for node in frontier_nodes:
            if job_valid.get(node.job_dag, False):
                act = action_map.inverse_map.get(node)
                if act is not None:
                    node_valid_mask[0, act] = 1

        return node_valid_mask, job_valid_mask


class SparseMat(object):
    def __init__(self, dtype, shape):
        self.dtype = dtype
        self.shape = shape
        self.row = []
        self.col = []
        self.data = []

    def add(self, row, col, data):
        self.row.append(row)
        self.col.append(col)
        self.data.append(data)

    def get_col(self):
        return np.array(self.col)

    def get_row(self):
        return np.array(self.row)

    def get_data(self):
        return np.array(self.data)

    def to_torch_sparse(self, device):
        # Use `len()` for a robust check on whether the list of indices is empty.
        # This avoids the ValueError with NumPy arrays.
        if len(self.row) == 0:  # Empty sparse matrix
            return torch.sparse_coo_tensor(torch.empty((2, 0), dtype=torch.long), [], self.shape).to(device)

        indices = torch.tensor([self.row, self.col], dtype=torch.long)
        values = torch.tensor(self.data, dtype=torch.float)
        return torch.sparse_coo_tensor(indices, values, self.shape).to(device)


def absorb_sp_mats(in_mats, depth):
    sp_mats = []

    for d in range(depth):
        row_idx, col_idx, data = [], [], []
        shape = 0
        base = 0
        for m_list in in_mats:
            m = m_list[d]
            row_idx.append(m.get_row() + base)
            col_idx.append(m.get_col() + base)
            data.append(m.get_data())
            shape += m.shape[0]
            base += m.shape[0]

        new_sp_mat = SparseMat(np.float32, (shape, shape))
        if any(len(arr) > 0 for arr in row_idx):
            new_sp_mat.row = np.hstack(row_idx)
            new_sp_mat.col = np.hstack(col_idx)
            new_sp_mat.data = np.hstack(data)

        sp_mats.append(new_sp_mat)
    return sp_mats


def merge_masks(masks):
    merged_masks = []
    for d in range(args.max_depth):
        merged_mask = []
        for mask in masks:
            merged_mask.append(mask[d:d + 1, :].transpose())

        if len(merged_mask) > 0:
            merged_mask = np.vstack(merged_mask)
        else:  # Handle case with no masks
            merged_mask = np.empty((0, 1))

        merged_masks.append(merged_mask)
    return merged_masks


class Postman(object):
    def __init__(self):
        self.reset()

    def get_msg_path(self, job_dags):
        job_dag_list = list(job_dags)
        if len(self.job_dags) != len(job_dag_list) or \
                not all(i is j for i, j in zip(self.job_dags, job_dag_list)):
            job_dags_changed = True
        else:
            job_dags_changed = False

        if job_dags_changed:
            self.msg_mats, self.msg_masks = get_msg_path(job_dag_list)
            self.dag_summ_backward_map = get_dag_summ_backward_map(job_dag_list)
            self.running_dag_mat = get_running_dag_mat(job_dag_list)
            self.job_dags = list(job_dags)

        return self.msg_mats, self.msg_masks, \
            self.dag_summ_backward_map, self.running_dag_mat, \
            job_dags_changed

    def reset(self):
        self.job_dags = []
        self.msg_mats = []
        self.msg_masks = []
        self.dag_summ_backward_map = None
        self.running_dag_mat = None


def get_msg_path(job_dags):
    if not job_dags:
        return [], []
    msg_mats, msg_masks = [], []
    for job_dag in job_dags:
        msg_mat, msg_mask = get_bottom_up_paths(job_dag)
        msg_mats.append(msg_mat)
        msg_masks.append(msg_mask)

    msg_mats = absorb_sp_mats(msg_mats, args.max_depth)
    msg_masks = merge_masks(msg_masks)
    return msg_mats, msg_masks


def get_bottom_up_paths(job_dag):
    num_nodes = job_dag.num_nodes
    msg_mats = []
    msg_masks = np.zeros([args.max_depth, num_nodes])

    if num_nodes == 0:
        for _ in range(args.max_depth):
            msg_mats.append(SparseMat(dtype=np.float32, shape=(0, 0)))
        return msg_mats, msg_masks

    # Initial frontier: nodes that are leaves or will be reached from leaves within max_depth steps
    frontier = set(n for n in job_dag.nodes if not n.child_nodes)

    # Pass messages up from leaves
    for depth in range(args.max_depth):
        new_frontier = set()
        sp_mat = SparseMat(dtype=np.float32, shape=(num_nodes, num_nodes))

        # All parents of the current frontier are candidates for the new frontier
        candidates = {p for n in frontier for p in n.parent_nodes}

        for parent in candidates:
            # A parent is ready if all its children are in the current frontier
            if all(child in frontier for child in parent.child_nodes):
                new_frontier.add(parent)
                msg_masks[depth, parent.idx] = 1
                for child in parent.child_nodes:
                    sp_mat.add(row=parent.idx, col=child.idx, data=1)

        msg_mats.append(sp_mat)

        if not new_frontier:  # No more parents to propagate to
            # Fill remaining depth with empty matrices
            for _ in range(depth + 1, args.max_depth):
                msg_mats.append(SparseMat(dtype=np.float32, shape=(num_nodes, num_nodes)))
            break

        frontier = new_frontier

    # Ensure msg_mats has max_depth elements
    while len(msg_mats) < args.max_depth:
        msg_mats.append(SparseMat(dtype=np.float32, shape=(num_nodes, num_nodes)))

    return msg_mats, msg_masks


def get_dag_summ_backward_map(job_dags):
    total_num_nodes = int(np.sum([job_dag.num_nodes for job_dag in job_dags]))
    num_jobs = len(job_dags)
    dag_summ_backward_map = np.zeros([total_num_nodes, num_jobs])
    base = 0
    for j_idx, job_dag in enumerate(job_dags):
        dag_summ_backward_map[base: base + job_dag.num_nodes, j_idx] = 1
        base += job_dag.num_nodes
    return dag_summ_backward_map


def get_running_dag_mat(job_dags):
    running_dag_mat = SparseMat(dtype=np.float32, shape=(1, len(job_dags)))
    for j_idx, job_dag in enumerate(job_dags):
        if not job_dag.completed:
            running_dag_mat.add(row=0, col=j_idx, data=1)
    return running_dag_mat


def get_unfinished_nodes_summ_mat(job_dags):
    total_num_nodes = int(np.sum([job_dag.num_nodes for job_dag in job_dags]))
    num_jobs = len(job_dags)
    summ_mat = SparseMat(dtype=np.float32, shape=(num_jobs, total_num_nodes))
    base = 0
    for j_idx, job_dag in enumerate(job_dags):
        for node in job_dag.nodes:
            if not node.tasks_all_done:
                summ_mat.add(row=j_idx, col=base + node.idx, data=1)
        base += job_dag.num_nodes
    return summ_mat


class AveragePerStepReward(object):
    def __init__(self, size):
        self.size = size
        self.count = 0
        self.reward_record = []
        self.time_record = []
        self.reward_sum = 0
        self.time_sum = 0

    def add(self, reward, time):
        if self.count >= self.size:
            stale_reward = self.reward_record.pop(0)
            stale_time = self.time_record.pop(0)
            self.reward_sum -= stale_reward
            self.time_sum -= stale_time
        else:
            self.count += 1

        self.reward_record.append(reward)
        self.time_record.append(time)
        self.reward_sum += reward
        self.time_sum += time

    def add_list(self, list_reward, list_time):
        assert len(list_reward) == len(list_time)
        for i in range(len(list_reward)):
            self.add(list_reward[i], list_time[i])

    def add_list_filter_zero(self, list_reward, list_time):
        assert len(list_reward) == len(list_time)
        for i in range(len(list_reward)):
            if list_time[i] != 0:
                self.add(list_reward[i], list_time[i])
            else:
                assert list_reward[i] == 0

    def get_avg_per_step_reward(self):
        if self.time_sum == 0: return 0.0
        return float(self.reward_sum) / float(self.time_sum)


def get_piecewise_linear_fit_baseline(all_cum_rewards, all_wall_time):
    if not all_wall_time or not any(all_wall_time): return [np.array([]) for _ in all_wall_time]

    unique_wall_time = np.unique(np.hstack([wt for wt in all_wall_time if len(wt) > 0]))
    if len(unique_wall_time) == 0: return [np.array([]) for _ in all_wall_time]

    baseline_values = {}
    for t in unique_wall_time:
        baseline = 0
        count = 0
        for i in range(len(all_wall_time)):
            if len(all_wall_time[i]) == 0: continue
            count += 1
            idx = bisect.bisect_left(all_wall_time[i], t)
            if idx == 0:
                baseline += all_cum_rewards[i][0]
            elif idx == len(all_cum_rewards[i]):
                baseline += all_cum_rewards[i][-1]
            elif all_wall_time[i][idx] == t:
                baseline += all_cum_rewards[i][idx]
            else:
                baseline += \
                    (all_cum_rewards[i][idx] - all_cum_rewards[i][idx - 1]) / \
                    (all_wall_time[i][idx] - all_wall_time[i][idx - 1] + args.eps) * \
                    (t - all_wall_time[i][idx]) + all_cum_rewards[i][idx]

        baseline_values[t] = baseline / float(count) if count > 0 else 0

    baselines = []
    for wall_time in all_wall_time:
        baseline = np.array([baseline_values[t] for t in wall_time])
        baselines.append(baseline)
    return baselines


def compute_loss(agent, experiences, batch_adv, entropy_weight, device):
    all_policy_loss = 0
    all_entropy_loss = 0
    total_steps = len(experiences)

    batch_points = truncate_experiences([exp['job_state_change'] for exp in experiences])

    for b in range(len(batch_points) - 1):
        ba_start = batch_points[b]
        ba_end = batch_points[b + 1]

        # Stack experiences for this segment
        exp_segment = experiences[ba_start:ba_end]
        node_inputs_np = np.vstack([e['node_inputs'] for e in exp_segment])
        job_inputs_np = np.vstack([e['job_inputs'] for e in exp_segment])
        node_valid_mask_np = np.vstack([e['node_valid_mask'] for e in exp_segment])
        job_valid_mask_np = np.vstack([e['job_valid_mask'] for e in exp_segment])

        node_acts = torch.tensor([e['node_act'] for e in exp_segment], dtype=torch.long, device=device)
        job_acts = torch.tensor([e['job_act'] for e in exp_segment], dtype=torch.long, device=device)
        job_indices = torch.tensor([e['job_idx'] for e in exp_segment], dtype=torch.long, device=device)

        adv = torch.from_numpy(batch_adv[ba_start:ba_end]).float().to(device)

        # Get sparse matrices for this segment (they are the same within a segment)
        ref_exp = exp_segment[0]
        gcn_mats_sp = ref_exp['gcn_mats']
        gcn_masks_np = ref_exp['gcn_masks']
        summ_mats_sp = ref_exp['summ_mats']
        running_dag_mat_sp = ref_exp['running_dag_mat']
        dag_summ_back_map_np = ref_exp['dag_summ_back_mat']

        # Convert to tensors
        node_inputs = torch.from_numpy(node_inputs_np).float().to(device)
        job_inputs = torch.from_numpy(job_inputs_np).float().to(device)
        node_valid_mask = torch.from_numpy(node_valid_mask_np).float().to(device)
        job_valid_mask = torch.from_numpy(job_valid_mask_np).float().to(device)

        gcn_mats = [m.to_torch_sparse(device) for m in gcn_mats_sp]
        gcn_masks = [torch.from_numpy(m).float().to(device) for m in gcn_masks_np]
        summ_mats = [summ_mats_sp.to_torch_sparse(device), running_dag_mat_sp.to_torch_sparse(device)]
        dag_summ_back_map = torch.from_numpy(dag_summ_back_map_np).float().to(device)

        # Forward pass to get new probabilities for the loss calculation
        node_probs, job_probs = agent(
            node_inputs, job_inputs, gcn_mats, gcn_masks,
            summ_mats, dag_summ_back_map, node_valid_mask, job_valid_mask
        )

        # Calculate losses
        # Node loss
        node_dist = Categorical(probs=node_probs)
        node_log_probs = node_dist.log_prob(node_acts)

        # Job loss
        job_probs_selected = job_probs[torch.arange(len(exp_segment)), job_indices]
        job_dist = Categorical(probs=job_probs_selected)
        job_log_probs = job_dist.log_prob(job_acts)

        # Total policy loss
        policy_loss = -torch.sum((node_log_probs + job_log_probs) * adv.squeeze())

        # Entropy
        node_entropy = node_dist.entropy().sum()
        job_entropy = job_dist.entropy().sum()
        entropy_loss = -(node_entropy + job_entropy)

        all_policy_loss += policy_loss
        all_entropy_loss += entropy_loss

    total_loss = (all_policy_loss + entropy_weight * all_entropy_loss) / total_steps
    return total_loss, all_policy_loss.item(), all_entropy_loss.item()


def train_agent(agent_id, param_queue, result_queue, adv_queue):
    torch.manual_seed(args.seed + agent_id)

    # set up environment
    env = Environment()

    # set up actor agent
    actor_agent = ActorAgent(args.node_input_dim, args.job_input_dim, args.hid_dims, args.output_dim,
                             args.max_depth, range(1, args.exec_cap + 1)).to(WORKER_DEVICE)

    optimizer = optim.Adam(actor_agent.parameters(), lr=args.lr)

    # collect experiences
    while True:
        # get parameters from master
        (actor_params, seed, max_time, entropy_weight) = param_queue.get()

        if actor_params is None:  # Shutdown signal
            break

        # synchronize model
        actor_agent.load_state_dict(actor_params)

        # reset environment
        env.seed(seed)
        obs = env.reset(max_time=max_time)

        # storage for experience from one episode
        experiences = []
        ep_rewards, ep_wall_times = [], []

        try:
            # run experiment
            done = False
            ep_wall_times.append(env.wall_time.curr_time)

            while not done:
                node, use_exec, exp_data = actor_agent.get_action(obs, WORKER_DEVICE)

                if node is None and use_exec == env.num_source_exec:
                    # No valid action, step with None
                    obs, reward, done = env.step(None, env.num_source_exec)
                else:
                    obs, reward, done = env.step(node, use_exec)
                    # Store experience only if an action was taken
                    experiences.append(exp_data)
                    ep_rewards.append(reward)
                    ep_wall_times.append(env.wall_time.curr_time)

            # Report reward signals to master
            if not experiences:  # If no actions were taken, report empty
                result_queue.put(([0], [0, 1], 0, 0.0, env.wall_time.curr_time >= env.max_time))
            else:
                avg_duration = np.mean([j.completion_time - j.start_time for j in env.finished_job_dags if
                                        j.completed]) if env.finished_job_dags else 0.0
                result_queue.put([ep_rewards, ep_wall_times, len(env.finished_job_dags),
                                  avg_duration,
                                  env.wall_time.curr_time >= env.max_time])

            # Get advantage term from master
            batch_adv = adv_queue.get()
            if batch_adv is None: continue  # Skip update if master signals an issue

            # Compute loss and update the local model
            actor_agent.train()
            optimizer.zero_grad()
            loss, policy_loss_val, entropy_loss_val = compute_loss(
                actor_agent, experiences, batch_adv, entropy_weight, WORKER_DEVICE
            )
            loss.backward()
            optimizer.step()

            # Send updated parameters back to the master
            # Detach from graph and move to CPU before sending
            state_dict_to_send = {k: v.cpu().detach() for k, v in actor_agent.state_dict().items()}
            result_queue.put(state_dict_to_send)

        except AssertionError:
            # ask the main to abort this rollout and
            # try again
            result_queue.put(None)
            # need to still get from adv_queue to
            # prevent blocking
            adv_queue.get()


def main():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # create result and model folder
    create_folder_if_not_exists(args.result_folder)
    create_folder_if_not_exists(args.model_folder)

    # Set up multiprocessing
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass

    # initialize communication queues
    params_queues = [mp.Queue(1) for _ in range(args.num_agents)]
    result_queues = [mp.Queue(2) for _ in range(args.num_agents)]  # Can hold reward AND state_dict
    adv_queues = [mp.Queue(1) for _ in range(args.num_agents)]

    # set up actor agent on master
    master_agent = ActorAgent(args.node_input_dim, args.job_input_dim, args.hid_dims, args.output_dim,
                              args.max_depth, range(1, args.exec_cap + 1)).to(DEVICE)
    if args.saved_model:
        master_agent.load_state_dict(torch.load(args.saved_model, map_location=DEVICE))

    # store average reward for computing differential rewards
    avg_reward_calculator = AveragePerStepReward(args.average_reward_storage_size)

    # initialize entropy parameters
    entropy_weight = args.entropy_weight_init
    reset_prob = args.reset_prob

    # ---- start training process ----
    print(f"Starting training with {args.num_agents} agents on device: {DEVICE}")

    # Start training agents
    agents = []
    for i in range(args.num_agents):
        p = mp.Process(target=train_agent,
                       args=(i, params_queues[i], result_queues[i], adv_queues[i]))
        p.start()
        agents.append(p)

    for ep in range(1, args.num_ep + 1):
        ep_start_time = time.time()
        print(f'\n--- Training Epoch {ep}/{args.num_ep} ---')

        # Synchronize the model parameters for each training agent
        actor_params = {k: v.cpu() for k, v in master_agent.state_dict().items()}

        # Generate max time stochastically
        max_time = generate_coin_flips(reset_prob) if reset_prob > 0 else np.inf

        # Send out parameters to training agents
        for i in range(args.num_agents):
            params_queues[i].put([actor_params, args.seed + ep, max_time, entropy_weight])

        # Storage for advantage computation
        all_rewards, all_diff_times, all_times, all_num_finished_jobs, all_avg_job_duration = [], [], [], [], []

        # Get reward from agents
        any_agent_panic = False
        for i in range(args.num_agents):
            try:
                result = result_queues[i].get(timeout=300)  # 5 min timeout
                if result is None:
                    any_agent_panic = True
                    continue

                batch_reward, batch_time, num_finished_jobs, avg_job_duration, reset_hit = result
                diff_time = np.array(batch_time[1:]) - np.array(batch_time[:-1])

                all_rewards.append(batch_reward)
                all_diff_times.append(diff_time)
                all_times.append(batch_time[1:])
                all_num_finished_jobs.append(num_finished_jobs)
                all_avg_job_duration.append(avg_job_duration)

                avg_reward_calculator.add_list_filter_zero(batch_reward, diff_time)
            except mp.TimeoutError:
                print(f"Timeout waiting for reward from agent {i}. Marking as panic.")
                any_agent_panic = True
                continue

        if any_agent_panic:
            print("An agent panicked. Skipping advantage calculation and update for this round.")
            for i in range(args.num_agents):
                adv_queues[i].put(None)
                # We still need to drain the result queue for the updated params
                try:
                    result_queues[i].get(timeout=5)
                except:
                    pass
            continue

        # Compute differential reward
        all_cum_reward = []
        avg_per_step_reward = avg_reward_calculator.get_avg_per_step_reward()
        for i in range(args.num_agents):
            rewards = np.array(all_rewards[i])
            diff_times = np.array(all_diff_times[i])
            if args.diff_reward_enabled:
                rewards = rewards - avg_per_step_reward * diff_times

            cum_reward = discount(rewards, args.gamma)
            all_cum_reward.append(cum_reward)

        # Compute baseline and advantage
        baselines = get_piecewise_linear_fit_baseline(all_cum_reward, all_times)

        # Send advantage back to workers
        for i in range(args.num_agents):
            batch_adv = all_cum_reward[i] - baselines[i]
            batch_adv = np.reshape(batch_adv, [-1, 1])
            adv_queues[i].put(batch_adv)

        # Collect updated state dicts from workers
        updated_state_dicts = []
        for i in range(args.num_agents):
            try:
                state_dict = result_queues[i].get(timeout=300)
                updated_state_dicts.append(state_dict)
            except mp.TimeoutError:
                print(f"Timeout waiting for state_dict from agent {i}. Skipping its update.")
                continue

        # Aggregate parameters by averaging
        if updated_state_dicts:
            avg_state_dict = average_state_dicts(updated_state_dicts)
            master_agent.load_state_dict(avg_state_dict)
            print(f"Master agent updated with params from {len(updated_state_dicts)} agents.")

        # Log progress
        ep_duration = time.time() - ep_start_time
        avg_total_rew = np.mean([np.sum(r) for r in all_rewards]) if all_rewards else 0
        avg_job_dur = np.mean(
            [d for d in all_avg_job_duration if d is not None and not np.isnan(d)]) if all_avg_job_duration else 0.0

        print(
            f"Epoch {ep} finished in {ep_duration:.2f}s. Avg Reward: {avg_total_rew:.2f}. Avg Job Duration: {avg_job_dur:.2f}ms.")

        # decrease entropy weight & reset probability
        entropy_weight = decrease_var(entropy_weight, args.entropy_weight_min, args.entropy_weight_decay)
        reset_prob = decrease_var(reset_prob, args.reset_prob_min, args.reset_prob_decay)

        if ep % args.model_save_interval == 0:
            model_path = os.path.join(args.model_folder, f'model_ep_{ep}.pth')
            torch.save(master_agent.state_dict(), model_path)
            print(f"Model saved to {model_path}")

    # ---- Cleanup ----
    print("Training finished. Shutting down agents.")
    for i in range(args.num_agents):
        params_queues[i].put((None, None, None, None))  # Shutdown signal
    for p in agents:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()


if __name__ == '__main__':
    # It is crucial to check for 'fork' method, which is not available on Windows
    # and can cause issues with CUDA. 'spawn' is safer.
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)
    main()