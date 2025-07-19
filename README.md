# Decima

Simulator part of Decima (SIGCOMM '19) https://web.mit.edu/decima

Example:

Train Decima with 50 executors, 200 streaming jobs, 25 second Poisson job arrival interval (load ~85%), stochastic termination, input-dependent baseline and average reward, run
```
python3 train.py --exec_cap 50 --num_init_dags 1 --num_stream_dags 200 --reset_prob 5e-7 --reset_prob_min 5e-8 --reset_prob_decay 4e-10 --diff_reward_enabled 1 --num_agents 16 --model_save_interval 100 --model_folder ./models/stream_200_job_diff_reward_reset_5e-7_5e-8/
```

Use `tensorboard` to monitor the training process, some screenshots of the results are in `results/`

Test Decima after 10,000 iterations with 50 executors, 5000 streaming jobs (>10x longer than training), run
```
python3 test.py --exec_cap 50 --num_init_dags 1 --num_stream_dags 5000 --canvs_visualization 0 --test_schemes dynamic_partition learn --num_exp 1 --saved_model ./models/stream_200_job_diff_reward_reset_5e-7_5e-8/model_ep_10000
```

Some example output are in `results/`

We are currently in the process of refactoring the Spark implementation.

```
conda create -n tensorflowEnv python=3.5 tensorflow=1.5 networkx matplotlib
conda install tensorflow=1.4 networkx matplotlib
conda create -n tf1_4 python=3.6 tensorflow=1.4 numpy=1.16 networkx matplotlib
```
仔细分析强化学习代码中神经网络各个输入输出的依赖关系，将项目中使用tensorflow及与tensorflow相关模块的代码改成使用pytorch2.x的代码。
需要按顺序修改
1. tf_op.py
2. gcn.py
3. gsn.py
4. sparse_op.py
5. msg_passing_path.py
使用pytorch风格的神经网络


python train.py --exec_cap 50 --num_init_dags 1 --num_stream_dags 200 --reset_prob 5e-7 --reset_prob_min 5e-8 --reset_prob_decay 4e-10 --diff_reward_enabled 1 --num_agents 1 --model_save_interval 100 --model_folder ./models/stream_200_job_diff_reward_reset_5e-7_5e-8/