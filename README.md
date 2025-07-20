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

使用pytorch 2.7
创环境
```
conda create -n tf_py12 python=3.12
conda activate tf_py12
conda install tensorflow networkx matplotlib
```

命令行参数

--exec_cap 50 --num_init_dags 3 --num_stream_dags 200 --reset_prob 5e-7 --reset_prob_min 5e-8 --reset_prob_decay 4e-10 --diff_reward_enabled 1 --model_save_interval 100 --model_folder ./models/stream_200_job_diff_reward_reset_5e-7_5e-8/
```
python train.py --exec_cap 50 --num_init_dags 1 --num_stream_dags 200 --reset_prob 5e-7 --reset_prob_min 5e-8 --reset_prob_decay 4e-10 --diff_reward_enabled 1 --num_agents 16 --model_save_interval 100 --model_folder ./models/stream_200_job_diff_reward_reset_5e-7_5e-8/
```

用tensorboard看训练情况
```
tensorboard --logdir results
```

若使用test.py时出现

> OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
> 
> OMP: Hint This means that multiple copies of the OpenMP runtime have been linked into the program. That is dangerous, since it can degrade performance or cause incorrect results. The best thing to do is to ensure that only a single OpenMP runtime is linked into the process, e.g. by avoiding static linking of the OpenMP runtime in any library. As an unsafe, unsupported, undocumented workaround you can set the environment variable KMP_DUPLICATE_LIB_OK=TRUE to allow the program to continue to execute, but that may cause crashes or silently produce incorrect results. For more information, please see http://www.intel.com/software/products/support/.
 
将miniconda3\envs\torch_gpu_py312\Library\bin下的**libiomp5md.dll**改成别的名字即可
