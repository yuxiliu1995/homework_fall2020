from collections import OrderedDict
import numpy as np
import time
import pickle

import gym
import torch

from cs285.infrastructure import pytorch_util as ptu
from cs285.infrastructure.logger import Logger
from cs285.infrastructure import utils

# how many rollouts to save as videos to tensorboard
MAX_NVIDEO = 2
MAX_VIDEO_LEN = 40  # we overwrite this in the code below


class RL_Trainer(object):

    def __init__(self, params):

        #############
        ## INIT
        #############

        # Get params, create logger.
        self.params = params
        self.logger = Logger(self.params['logdir'])

        # Set random seeds
        seed = self.params['seed']
        np.random.seed(seed)
        torch.manual_seed(seed)
        ptu.init_gpu(
            use_gpu=not self.params['no_gpu'],
            gpu_id=self.params['which_gpu']
        )

        #############
        ## ENV
        #############

        # Make the gym environment
        self.env = gym.make(self.params['env_name'])
        self.env.seed(seed)

        # Maximum length for episodes
        self.params['ep_len'] = self.params['ep_len'] or self.env.spec.max_episode_steps
        MAX_VIDEO_LEN = self.params['ep_len']

        # Is the action space of the env continuous or discrete?
        discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        self.params['agent_params']['discrete'] = discrete

        # Observation and action sizes
        ob_dim = self.env.observation_space.shape[0]
        ac_dim = self.env.action_space.n if discrete else self.env.action_space.shape[0]
        self.params['agent_params']['ac_dim'] = ac_dim
        self.params['agent_params']['ob_dim'] = ob_dim

        # simulation timestep, will be used for video saving
        if 'model' in dir(self.env):
            self.fps = 1/self.env.model.opt.timestep
        else:
            self.fps = self.env.env.metadata['video.frames_per_second']

        #############
        ## AGENT
        #############

        agent_class = self.params['agent_class']
        self.agent = agent_class(self.env, self.params['agent_params'])

    def run_training_loop(self, n_iter, collect_policy, eval_policy,
                        initial_expertdata=None, relabel_with_expert=False,
                        start_relabel_with_expert=1, expert_policy=None):
        """
        :param n_iter: Number of (DAgger) iterations.
        :param collect_policy: For collecting new trajectories by rollout.
        :param eval_policy: For logging only, to evaluate how well the agent did based on its training logs.
        :param initial_expertdata: Path to expert data pkl file. Used in collect_training_trajectories.
        :param relabel_with_expert: Whether to perform DAgger.
        :param start_relabel_with_expert: Iteration number at which to start relabel with expert.
        :param expert_policy: For relabelling the collected new trajectories (if running DAgger).
        """

        # init vars at beginning of training
        self.total_envsteps = 0
        self.start_time = time.time()

        for itr in range(n_iter):
            print("\n\n********** Iteration %i ************"%itr)

            # decide if videos should be rendered/logged at this iteration
            if itr % self.params['video_log_freq'] == 0 and self.params['video_log_freq'] != -1:
                self.log_video = True
            else:
                self.log_video = False

            # decide if metrics should be logged
            if itr % self.params['scalar_log_freq'] == 0:
                self.log_metrics = True
            else:
                self.log_metrics = False

            # collect trajectories, to be used for training
            training_returns = self.collect_training_trajectories(
                itr,
                initial_expertdata,
                collect_policy,
                self.params['batch_size']
            )
            paths, envsteps_this_batch, train_video_paths = training_returns
            self.total_envsteps += envsteps_this_batch

            # relabel the collected obs with actions from a provided expert policy
            if relabel_with_expert and itr>=start_relabel_with_expert:
                paths = self.do_relabel_with_expert(expert_policy, paths)

            # add collected data to replay buffer
            self.agent.add_to_replay_buffer(paths)

            # train agent (using sampled data from replay buffer)
            training_logs = self.train_agent()

            # log/save in the end.
            if self.log_video or self.log_metrics:

                # perform logging
                print('\nBeginning logging procedure...')
                self.perform_logging(
                    itr, paths, eval_policy, train_video_paths, training_logs)

                if self.params['save_params']:
                    print('\nSaving agent params')
                    self.agent.save('{}/policy_itr_{}.pt'.format(self.params['logdir'], itr))

    ####################################
    ####################################

    def collect_training_trajectories(
            self,
            itr,
            initial_expertdata,
            collect_policy,
            batch_size,
    ):
        """
        This function is called only in run_training_loop in this module.
        If itr == 0, it simply loads the trajectories from initial_expertdata
        Otherwise, it returns some new trajectories using collect_policy.

        :param itr: The iteration index. Starts at 0.
        :param initial_expertdata: Path to expert data pkl file.
        :param collect_policy: The current policy using which we collect data.
        :param batch_size: The number of transitions we collect.

        :return:
            paths: a list trajectories
            envsteps_this_batch: the sum over the numbers of environment steps in paths
            train_video_paths: paths which also contain videos for visualization purposes
        """
        if itr == 0:
            if initial_expertdata:
                pickle_in = open(initial_expertdata, "rb")
                loaded_paths = pickle.load(pickle_in)
                return loaded_paths, 0, None
            else:
                # it's the first iteration, but you aren't loading expert data,
                # collect `self.params['batch_size_initial']`
                batch_size = self.params['batch_size_initial']

        # collect batch_size samples with collect_policy
        # each of these collected rollouts is of length self.params['ep_len']
        print("\nCollecting data to be used for training...")
        paths, envsteps_this_batch = \
            utils.sample_trajectories(env=self.env,
                                policy=collect_policy,
                                min_timesteps_per_batch=batch_size,
                                max_path_length=self.params['ep_len'])

        # collect more rollouts with collect_policy, to be saved as videos in tensorboard
        # collect MAX_NVIDEO rollouts, each of length MAX_VIDEO_LEN.
        train_video_paths = None
        if self.log_video:
            print('\nCollecting train rollouts to be used for saving videos...')
            train_video_paths = utils.sample_n_trajectories(self.env, collect_policy, MAX_NVIDEO, MAX_VIDEO_LEN, True)

        return paths, envsteps_this_batch, train_video_paths


    def train_agent(self):
        """
        Sample self.params['train_batch_size'] frames from the replay buffer of
        the agent, then train the agent upon that.
        Repeat this for self.params['num_agent_train_steps_per_iter'] steps.

        Returns
            - all_logs: the entire training log from this training.
        """
        print('\nTraining agent using sampled data from replay buffer...')
        all_logs = []
        for train_step in range(self.params['num_agent_train_steps_per_iter']):

            # Sample some data from the replay buffer of the agent.
            # sample size is
            # self.params['train_batch_size']
            ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch \
                = self.agent.sample(self.params['train_batch_size'])

            # Use the sampled data to train an agent

            train_log = self.agent.train(
                ptu.from_numpy(ob_batch),
                ptu.from_numpy(ac_batch),
                ptu.from_numpy(re_batch),
                ptu.from_numpy(next_ob_batch),
                ptu.from_numpy(terminal_batch))
            all_logs.append(train_log) # training log for debugging
        return all_logs

    def do_relabel_with_expert(self, expert_policy, paths):
        """
        Relabels collected paths with actions from expert_policy.

        Inputs
            - `expert_policy`: a policy used as reference.
            - paths: the list of paths. See `replay_buffer.py`.
        Outputs:
            - paths: the modified list of paths.
        """
        print("\nRelabelling collected observations with labels from an expert policy...")

        for path in paths:
            obs = path["observation"]
            acs = path["action"]
            for i in range(len(obs)):
                acs[i] = expert_policy.get_action(obs[i])
        return paths

    ####################################
    ####################################

    def perform_logging(self, itr, paths, eval_policy, train_video_paths, training_logs):

        # collect eval trajectories, for logging
        print("\nCollecting data for eval...")
        eval_paths, eval_envsteps_this_batch = utils.sample_trajectories(self.env, eval_policy, self.params['eval_batch_size'], self.params['ep_len'])

        # save eval rollouts as videos in tensorboard event file
        if self.log_video and train_video_paths != None:
            print('\nCollecting video rollouts eval')
            eval_video_paths = utils.sample_n_trajectories(self.env, eval_policy, MAX_NVIDEO, MAX_VIDEO_LEN, True)

            #save train/eval videos
            print('\nSaving train rollouts as videos...')
            self.logger.log_paths_as_videos(train_video_paths, itr, fps=self.fps, max_videos_to_save=MAX_NVIDEO,
                                            video_title='train_rollouts')
            self.logger.log_paths_as_videos(eval_video_paths, itr, fps=self.fps,max_videos_to_save=MAX_NVIDEO,
                                             video_title='eval_rollouts')

        # save eval metrics
        if self.log_metrics:
            # returns, for logging
            train_returns = [path["reward"].sum() for path in paths]
            eval_returns = [eval_path["reward"].sum() for eval_path in eval_paths]

            # episode lengths, for logging
            train_ep_lens = [len(path["reward"]) for path in paths]
            eval_ep_lens = [len(eval_path["reward"]) for eval_path in eval_paths]

            # decide what to log
            logs = OrderedDict()
            logs["Eval_AverageReturn"] = np.mean(eval_returns)
            logs["Eval_StdReturn"] = np.std(eval_returns)
            logs["Eval_MaxReturn"] = np.max(eval_returns)
            logs["Eval_MinReturn"] = np.min(eval_returns)
            logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens)

            logs["Train_AverageReturn"] = np.mean(train_returns)
            logs["Train_StdReturn"] = np.std(train_returns)
            logs["Train_MaxReturn"] = np.max(train_returns)
            logs["Train_MinReturn"] = np.min(train_returns)
            logs["Train_AverageEpLen"] = np.mean(train_ep_lens)

            logs["Train_EnvstepsSoFar"] = self.total_envsteps
            logs["TimeSinceStart"] = time.time() - self.start_time
            last_log = training_logs[-1]  # Only use the last log for now
            logs.update(last_log)


            if itr == 0:
                self.initial_return = np.mean(train_returns)
            logs["Initial_DataCollection_AverageReturn"] = self.initial_return

            # perform the logging
            for key, value in logs.items():
                print('{} : {}'.format(key, value))
                self.logger.log_scalar(value, key, itr)
            print('Done logging...\n\n')

            self.logger.flush()
