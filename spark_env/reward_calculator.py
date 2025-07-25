from param import args


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
                reward -= (min(job_dag.completion_time, curr_time) - max(job_dag.start_time,
                    self.prev_time)) / args.reward_scale

                # if the job is done, remove it from the list
                if job_dag.completed:
                    self.job_dags.remove(job_dag)

        elif args.learn_obj == 'makespan':
            reward -= (curr_time - self.prev_time) / args.reward_scale

        else:
            print('Unkown learning objective')
            exit(1)

        self.prev_time = curr_time

        return reward

    def reset(self):
        self.job_dags.clear()
        self.prev_time = 0
