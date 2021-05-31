# Copyright Joe Worsham 2021

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm

from joe_agents.e_greedy_policy import EGreedyPolicy
from joe_agents.er_buffer import ExperienceReplayBuffer
from joe_agents.prioritized_er_buffer import PrioritizedExperienceReplayBuffer
from joe_agents.q_network import QNetwork
from joe_agents.q_network_policy import QNetworkPolicy
from joe_agents.q_network_value_function import QNetworkValueFunction

# hyperparameter names
HIDDEN_LAYERS = "hidden_layers"
EPISODES = "episodes"
BATCH_SIZE = "batch_size"
BUFFER_SIZE = "buffer_size"
UPDATE_RATE = "update_rate"
LEARNING_RATE = "learning_rate"
DISCOUNT_RATE = "discount_rate"
EPSILON = "epsilon"
EPSILON_DECAY = "epsilon_decay"
EPSILON_DECAY_RATE = "epsilon_decay_rate"
MIN_EPSILON = "min_epsilon"
TAU = "tau"
REPLAY = "replay"
PRIORITIZED_REPLAY_DAMP = "prioritized_replay_damp"
PRIORITIZED_REPLAY_E_CONSTANT = "e_constnat"
PRIORITIZED_REPLAY_BETA_ANNEAL_RATE = "prioritized_replay_beta_anneal_rate"
LEARNING_START = "learning_start"
DOUBLE_DQN = "double_dqn"
DEULING_DQN = "deuling_dqn"

# options for enumerated configurations
REPLAY_UNIFORM = "uniform"
REPLAY_PRIORITIZED = "prioritized"

# hyperparameter default values
DEFAULT_HIDDEN_LAYERS = None
DEFAULT_EPISODES = 10000
DEFAULT_BATCH_SIZE = 128
DEFAULT_BUFFER_SIZE = 10000
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_UPDATE_RATE = 10
DEFAULT_DISCOUNT_RATE = 0.999
DEFAULT_EPSILON = 1
DEFAULT_EPSILON_DECAY = 0.99
DEFAULT_EPSILON_DECAY_RATE = 1000
DEFAULT_MIN_EPSILON = 1e-1
DEFAULT_TAU = 1e-3
DEFAULT_REPLAY = REPLAY_UNIFORM
DEFAULT_PRIORITIZED_REPLAY_DAMP = 0.1
DEFAULT_PRIORITIZED_REPLAY_E_CONSTANT = 0.01
DEFAULT_PRIORITIZED_REPLAY_BETA_ANNEAL_RATE = 1
DEFAULT_LEARNING_START = 1000
DEFAULT_DOUBLE_DQN = False
DEFAULT_DEULING_DQN = False


# default parameter collection
DEFAULT_PARAMS = {
    HIDDEN_LAYERS: DEFAULT_HIDDEN_LAYERS,
    EPISODES: DEFAULT_EPISODES,
    BATCH_SIZE: DEFAULT_BATCH_SIZE,
    BUFFER_SIZE: DEFAULT_BUFFER_SIZE,
    LEARNING_RATE: DEFAULT_LEARNING_RATE,
    UPDATE_RATE: DEFAULT_UPDATE_RATE,
    DISCOUNT_RATE: DEFAULT_DISCOUNT_RATE,
    EPSILON: DEFAULT_EPSILON,
    EPSILON_DECAY: DEFAULT_EPSILON_DECAY,
    EPSILON_DECAY_RATE: DEFAULT_EPSILON_DECAY_RATE,
    MIN_EPSILON: DEFAULT_MIN_EPSILON,
    TAU: DEFAULT_TAU,
    REPLAY: DEFAULT_REPLAY,
    PRIORITIZED_REPLAY_DAMP: DEFAULT_PRIORITIZED_REPLAY_DAMP,
    PRIORITIZED_REPLAY_E_CONSTANT: DEFAULT_PRIORITIZED_REPLAY_E_CONSTANT,
    PRIORITIZED_REPLAY_BETA_ANNEAL_RATE: DEFAULT_PRIORITIZED_REPLAY_BETA_ANNEAL_RATE,
    LEARNING_START: DEFAULT_LEARNING_START,
    DOUBLE_DQN: DEFAULT_DOUBLE_DQN,
    DEULING_DQN: DEFAULT_DEULING_DQN
}


class DqnAgent:
    """The standard Deep Q-Network.
    Employes an experience replay buffer
    and a target network to aid training.
    """
    def __init__(self, state_size: int, action_size: int, hyperparameters: dict=None) -> None:
        """Create a new DqnAgent for the specified state and action
        space with the given hyperparameters. Note: hyperparameters
        may be None and all the defaults will be used.

        Args:
            state_size (int): The number of discrete states in the environment.
            action_size (int): The number of discrete actions in the environment.
            hyperparameters (dict, optional): The set of hyperparameters for training.
        """
        self._state_size = state_size
        self._action_size = action_size
        self._hyperparameters = DEFAULT_PARAMS
        if hyperparameters is not None:
            self._hyperparameters.update(hyperparameters)
        hidden_layers = self._hyperparameters[HIDDEN_LAYERS] if HIDDEN_LAYERS in self._hyperparameters else DEFAULT_HIDDEN_LAYERS

        # verify the hardware to run the agent on
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Using {self._device} device')
        
        # instantiate the neural networks that drive decision making
        self._q_network = QNetwork(state_size, action_size, hidden_layers).to(self._device)
        self._q_target = QNetwork(state_size, action_size, hidden_layers).to(self._device)

        # create a policy around the q network which will act as our estimated optimal policy
        self._optimal_policy = QNetworkPolicy(self._q_network, self._device)

    @property
    def hyperparameters(self):
        return self._hyperparameters

    def train(self, env):
        """Train the agent on the provided environment.
        Note: the environment is expected to conform to
        the OpenAI Gym schema.

        Returns:
            tuple: the list of score histories and list of epsilon histories during training
        """
        # start by extracting all useful hyperparameters for training
        episodes = self._hyperparameters[EPISODES] if EPISODES in self._hyperparameters else DEFAULT_EPISODES
        batch_size = self._hyperparameters[BATCH_SIZE] if BATCH_SIZE in self._hyperparameters else DEFAULT_BATCH_SIZE
        buffer_size = self._hyperparameters[BUFFER_SIZE] if BUFFER_SIZE in self._hyperparameters else DEFAULT_BUFFER_SIZE
        update_rate = self._hyperparameters[UPDATE_RATE] if UPDATE_RATE in self._hyperparameters else DEFAULT_UPDATE_RATE
        alpha = self._hyperparameters[LEARNING_RATE] if LEARNING_RATE in self._hyperparameters else DEFAULT_LEARNING_RATE
        gamma = self._hyperparameters[DISCOUNT_RATE] if DISCOUNT_RATE in self._hyperparameters else DEFAULT_DISCOUNT_RATE
        epsilon = self._hyperparameters[EPSILON] if EPSILON in self._hyperparameters else DEFAULT_EPSILON
        epsilon_decay = self._hyperparameters[EPSILON_DECAY] if EPSILON_DECAY in self._hyperparameters else DEFAULT_EPSILON_DECAY
        epsilon_decay_rate = self._hyperparameters[EPSILON_DECAY_RATE] if EPSILON_DECAY_RATE in self._hyperparameters else DEFAULT_EPSILON_DECAY_RATE
        min_epsilon = self._hyperparameters[MIN_EPSILON] if MIN_EPSILON in self._hyperparameters else DEFAULT_MIN_EPSILON
        tau = self._hyperparameters[TAU] if TAU in self._hyperparameters else DEFAULT_TAU
        replay = self._hyperparameters[REPLAY] if REPLAY in self._hyperparameters else DEFAULT_REPLAY
        learning_start = self._hyperparameters[LEARNING_START] if LEARNING_START in self._hyperparameters else DEFAULT_LEARNING_START
        double_dqn = self._hyperparameters[DOUBLE_DQN] if DOUBLE_DQN in self._hyperparameters else DEFAULT_DOUBLE_DQN
        deuling_dqn = self._hyperparameters[DEULING_DQN] if DEULING_DQN in self._hyperparameters else DEFAULT_DEULING_DQN

        # create the replay buffer for training
        if replay == REPLAY_PRIORITIZED:
            a_damp = self._hyperparameters[PRIORITIZED_REPLAY_DAMP] if PRIORITIZED_REPLAY_DAMP in self._hyperparameters else DEFAULT_PRIORITIZED_REPLAY_DAMP
            e_constant = self._hyperparameters[PRIORITIZED_REPLAY_E_CONSTANT] if PRIORITIZED_REPLAY_E_CONSTANT in self._hyperparameters else DEFAULT_PRIORITIZED_REPLAY_E_CONSTANT
            beta_anneal_rate = self._hyperparameters[PRIORITIZED_REPLAY_BETA_ANNEAL_RATE] if PRIORITIZED_REPLAY_BETA_ANNEAL_RATE in self._hyperparameters else DEFAULT_PRIORITIZED_REPLAY_BETA_ANNEAL_RATE
            beta_steps = episodes / beta_anneal_rate
            value_function = QNetworkValueFunction(self._q_network, self._device)
            buffer = PrioritizedExperienceReplayBuffer(buffer_size, value_function, gamma, a_damp, e_constant)
        else:
            buffer = ExperienceReplayBuffer(buffer_size)

        # create an e-greedy policy to use while training
        # balances exploration and exploitation
        e_greedy = EGreedyPolicy(self._action_size, epsilon, self._optimal_policy)

        # the optimizer updates the neural network weights
        optimizer = optim.Adam(self._q_network.parameters(), lr=alpha)

        # go ahead and train
        global_steps = 0
        scores = []
        epsilons = []
        batch_reward_sums = []
        batch_buffer_len = []
        buffer_stats = {
            "batch_reward_sums": batch_reward_sums,
            "batch_buffer_len": batch_buffer_len
        }
        if replay == REPLAY_PRIORITIZED:
            prioritized_replay_betas = []
            buffer_stats["prioritized_replay_beta"] = prioritized_replay_betas
        for i in tqdm(range(episodes)):
            state = env.reset()
            done = False
            score = 0
            while not done:
                # pick the next random with e-greey policy and advance the environment
                action = e_greedy(state)
                next_state, reward, done, _ = env.step(action)
                score += reward
                global_steps += 1

                # log this experience to the buffer
                buffer.append(state, action, reward, next_state, done)

                # reset for the next rount
                state = next_state

                # update the weights at a fixed interval (assuming enough experiences exist)
                if global_steps % update_rate == 0 and len(buffer) >= learning_start:
                    # gather a batch of experience data to train on
                    experiences = buffer.sample(batch_size)
                    idx_start = 0
                    if replay == REPLAY_PRIORITIZED:
                        priorities = torch.from_numpy(experiences[0]).float().to(self._device)    
                        idx_start = 1
                    states = torch.from_numpy(experiences[idx_start]).float().to(self._device)
                    actions = torch.from_numpy(experiences[idx_start+1]).long().to(self._device)
                    rewards = torch.from_numpy(experiences[idx_start+2]).float().to(self._device)
                    next_states = torch.from_numpy(experiences[idx_start+3]).float().to(self._device)
                    dones = torch.from_numpy(experiences[idx_start+4]).float().to(self._device)

                    # capture buffer statistics for debugging
                    batch_reward_sums.append(np.sum(experiences[2]))
                    batch_buffer_len.append(len(buffer))

                    # train the q_network to better predict the action-values
                    # note: the estimated correct action-value comes from the target network
                    eval_policy = self._q_target if double_dqn else self._q_network
                    q_targets_next = eval_policy(next_states).detach().max(1)[0].unsqueeze(1)
                    q_targets = rewards + (gamma * q_targets_next * (1 - dones))
                    q_expected = self._q_network(states).gather(1, actions)
                    loss = F.mse_loss(q_expected, q_targets, reduction='none')
                    if replay == REPLAY_PRIORITIZED:
                        beta = (i // beta_anneal_rate) / beta_steps
                        weights = ((1 / len(buffer)) * (1/priorities))**beta
                        loss *= weights
                        prioritized_replay_betas.append(beta)
                    loss = torch.mean(loss)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # update the target network to slowly converge to the q network
                    for target_param, local_param in zip(self._q_target.parameters(), self._q_network.parameters()):
                        target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

            # log the performance and training characteristics
            scores.append(score)
            epsilons.append(e_greedy.epsilon)
            
            # decay epsilon between episodes at the specified rate
            if i % epsilon_decay_rate == 0:
                e_greedy.epsilon = max(min_epsilon, e_greedy.epsilon * epsilon_decay)

        # return training history
        return scores, epsilons, buffer_stats


    def act(self, state) -> int:
        """Compute the optimal action according to the
        agent for the given state.

        Args:
            state (np.Array): The agent's current state in the environment.

        Returns:
            int: the discrete action the agent wants to take
        """
        return self._optimal_policy(state)

    def save(self, path: str="checkpoint.pth"):
        """Save the model to disk.

        Args:
            path (str, optional): Name of agent to save. Defaults to "checkpoint.pth".
        """
        torch.save(self._q_network.state_dict(), path)

    def load(self, path: str="checkpoint.pth"):
        """Load a trained agent from disk.

        Args:
            path (str, optional): Name of the agent to load. Defaults to "checkpoint.pth".
        """
        self._q_network.load_state_dict(torch.load('checkpoint.pth'))
