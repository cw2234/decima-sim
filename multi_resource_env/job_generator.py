import os
import pandas as pd
from param import args
import numpy as np
import utils
from multi_resource_env.job_dag import JobDAG
from multi_resource_env.node import MultiResNode as Node
from multi_resource_env.task import MultiResTask as Task



def recursive_find_descendant(node):
    if len(node.descendant_nodes) > 0:  # already visited
        return node.descendant_nodes
    else:
        node.descendant_nodes = [node]
        for child_node in node.child_nodes:  # terminate on leaves automatically
            child_descendant_nodes = recursive_find_descendant(child_node)
            for dn in child_descendant_nodes:
                if dn not in node.descendant_nodes:  # remove dual path duplicates
                    node.descendant_nodes.append(dn)
        return node.descendant_nodes


def load_job_excel(file_path, dag_id, wall_time, np_random):

    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        print(f"Excel读取错误: {e}")
        exit()

    required_cols = ['节点ID', 'Job负载', "JobPL2大小", '节点子节点编号', 'Signal', "maxDoneDelay"]
    if not all(col in df.columns for col in required_cols):
        print(f"缺少必要列: {required_cols}")
        exit()

    nodes = []
    nodes_children = []

    for _, row in df.iterrows():
        node_id = int(row["节点ID"])
        job_load = int(row["Job负载"])
        job_pl2 = int(row["JobPL2大小"])

        # 处理子节点
        children_str = str(row["节点子节点编号"])
        if children_str.strip() and children_str != "nan":
            children_ids = [int(child.strip()) for child in children_str.split(",")]
        else:
            children_ids = []

        signal = float(row["Signal"])
        max_done_delay = float(row["maxDoneDelay"])
        num_exe = [2, 5, 10, 20, 40, 50, 60, 80, 100]
        task_duration = {
            'first_wave': {e: [job_load] for e in num_exe},
            'fresh_durations': {e: [] for e in num_exe},
            'rest_wave': {e: [] for e in num_exe},

        }

        e = next(iter(task_duration['first_wave']))

        rough_duration = job_load

        # generate random memory requirement
        task_cpu = 1.0
        task_mem = job_pl2

        # generate tasks in a node
        tasks = [Task(0, task_cpu, task_mem, rough_duration, wall_time)]

        nodes.append(Node(node_id, task_cpu, task_mem, tasks, task_duration, wall_time, np_random))
        nodes_children.append(children_ids) # 记录每个结点的孩子

    # parent and child node info
    for node_id in range(len(nodes)):
        for child_id in nodes_children[node_id]:
            nodes[node_id].child_nodes.append(nodes[child_id])
            nodes[child_id].parent_nodes.append(nodes[node_id])


    # initialize descendant nodes
    for node in nodes:
        if len(node.parent_nodes) == 0:  # root
            node.descendant_nodes = recursive_find_descendant(node)

    # generate DAG
    job_dag = JobDAG(nodes, None,  str(dag_id))

    return job_dag


def generate_excel_jobs(np_random, timeline, wall_time):

    job_dags = utils.OrderedSet()
    t = 0
    dag_id = 0
    for _ in range(args.num_init_dags):
        # generate job
        job_dag = load_job_excel('DAG.xlsx', dag_id, wall_time, np_random)
        dag_id += 1
        # job already arrived, put in job_dags
        job_dag.start_time = t
        job_dag.arrived = True
        job_dags.add(job_dag)

    t = 1 * 1300000
    for _ in range(args.num_stream_dags):


        # generate job
        job_dag = load_job_excel('DAG.xlsx', dag_id, wall_time, np_random)
        dag_id += 1
        job_dag.start_time = t
        timeline.push(t, job_dag)

    t = 2 * 1300000
    for _ in range(args.num_stream_dags):
        # generate job
        job_dag = load_job_excel('DAG.xlsx', dag_id, wall_time, np_random)
        dag_id += 1
        job_dag.start_time = t
        timeline.push(t, job_dag)

    return job_dags

def generate_jobs(np_random, timeline, wall_time):

    job_dags = generate_excel_jobs(np_random, timeline, wall_time)

    return job_dags
