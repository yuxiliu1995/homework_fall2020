from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.policies.MLP_policy import MLPPolicySL
from .base_agent import BaseAgent


class BCAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super(BCAgent, self).__init__()

        # init vars
        self.env = env
        self.agent_params = agent_params

        # actor/policy
        self.actor = MLPPolicySL(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            discrete=self.agent_params['discrete'],
            learning_rate=self.agent_params['learning_rate'],
        )

        # replay buffer
        self.replay_buffer = ReplayBuffer(self.agent_params['max_replay_buffer_size'])

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        """ Update actor policy by supervised learning,
            given observations and action labels.
        - ob_no: Observations.
        - ac_na: Action lables
        - re_n: ?
        - next_ob_no: ?
        - terminal_n: ?
        """
        log = self.actor.update(ob_no, ac_na)
        return log

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_random_data(batch_size)  # HW1: you will modify this

    def save(self, path):
        return self.actor.save(path)
