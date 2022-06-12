import numpy as np

from .base_policy import BasePolicy


class MPCPolicy(BasePolicy):

    def __init__(self,
                 env,
                 ac_dim,
                 dyn_models,
                 horizon,
                 N,
                 sample_strategy='random',
                 cem_iterations=4,
                 cem_num_elites=5,
                 cem_alpha=1,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.env = env
        self.dyn_models = dyn_models
        self.horizon = horizon
        self.N = N
        self.data_statistics = None  # NOTE must be updated from elsewhere

        self.ob_dim = self.env.observation_space.shape[0]

        # action space
        self.ac_space = self.env.action_space
        self.ac_dim = ac_dim
        self.low = self.ac_space.low
        self.high = self.ac_space.high

        # Sampling strategy
        allowed_sampling = ('random', 'cem')
        assert sample_strategy in allowed_sampling, f"sample_strategy must be one of the following: {allowed_sampling}"
        self.sample_strategy = sample_strategy
        self.cem_iterations = cem_iterations
        self.cem_num_elites = cem_num_elites
        self.cem_alpha = cem_alpha

        print(f"Using action sampling strategy: {self.sample_strategy}")
        if self.sample_strategy == 'cem':
            print(f"CEM params: alpha={self.cem_alpha}, "
                + f"num_elites={self.cem_num_elites}, iterations={self.cem_iterations}")

    def sample_action_sequences(self, num_sequences, horizon, obs=None):
        if self.sample_strategy == 'random' \
            or (self.sample_strategy == 'cem' and obs is None):
            random_action_sequences = np.random.uniform(low=self.low, high=self.high, size=(num_sequences, horizon, self.ac_dim))
            return random_action_sequences
        elif self.sample_strategy == 'cem':
            for i in range(self.cem_iterations):
                if i == 0:
                    candidate_action_sequences = np.random.uniform(low=self.low, high=self.high,
                                                                   size=(num_sequences, horizon, self.ac_dim))

                    predicted_returns = self.evaluate_candidate_sequences(candidate_action_sequences, obs)
                    elite_indices = np.argpartition(predicted_returns, -self.cem_num_elites)[-self.cem_num_elites:]
                    elite = candidate_action_sequences[elite_indices, :, :]
                    mean = np.mean(elite, axis=0)
                    variance = np.var(elite, axis=0)
                else:
                    candidate_action_sequences = np.random.normal(mean, np.sqrt(variance), (num_sequences, horizon, self.ac_dim))
                    elite_indices = np.argpartition(predicted_returns, -self.cem_num_elites)[-self.cem_num_elites:]
                    elite = candidate_action_sequences[elite_indices, :, :]
                    mean = self.cem_alpha * np.mean(elite, axis=0) + (1-self.cem_alpha)*mean
                    variance = self.cem_alpha * np.var(elite, axis=0) + (1-self.cem_alpha)*variance

            cem_action = mean

            return cem_action[None]
        else:
            raise Exception(f"Invalid sample_strategy: {self.sample_strategy}")

    def evaluate_candidate_sequences(self, candidate_action_sequences, obs):
        predicted_returns = np.zeros(candidate_action_sequences.shape[0])
        for model in self.dyn_models:
            predicted_returns += self.calculate_sum_of_rewards(obs, candidate_action_sequences, model)
        predicted_returns /= len(self.dyn_models)
        return predicted_returns

    def get_action(self, obs):
        if self.data_statistics is None:
            return self.sample_action_sequences(num_sequences=1, horizon=1)[0]

        # sample random actions (N x horizon)
        candidate_action_sequences = self.sample_action_sequences(
            num_sequences=self.N, horizon=self.horizon, obs=obs)

        if candidate_action_sequences.shape[0] == 1:
            # CEM: only a single action sequence to consider; return the first action
            return candidate_action_sequences[0][0][None]
        else:
            predicted_rewards = self.evaluate_candidate_sequences(candidate_action_sequences, obs)

            # pick the action sequence and return the 1st element of that sequence
            best_action_sequence = np.argmax(predicted_rewards)
            action_to_take = candidate_action_sequences[best_action_sequence, 0, :]
            return action_to_take[None]  # Unsqueeze the first index

    def calculate_sum_of_rewards(self, obs, candidate_action_sequences, model):
        """

        :param obs: numpy array with the current observation. Shape [D_obs]
        :param candidate_action_sequences: numpy array with the candidate action
        sequences. Shape [N, H, D_action] where
            - N is the number of action sequences considered
            - H is the horizon
            - D_action is the action of the dimension
        :param model: The current dynamics model.
        :return: numpy array with the sum of rewards for each action sequence.
        The array should have shape [N].
        """
        N, H, D_action = candidate_action_sequences.shape
        obs = np.broadcast_to(obs, (N, obs.shape[0]))
        sum_of_rewards = np.zeros(N)
        for t in range(H):
            sum_of_rewards += self.env.get_reward(obs, candidate_action_sequences[:, t, :])[0]
            obs = model.get_prediction(obs, candidate_action_sequences[:, t, :], self.data_statistics)
        return sum_of_rewards
