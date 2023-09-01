from sb3_contrib import QRDQN
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

from sb3_contrib.qrdqn.policies import QuantileNetwork, QRDQNPolicy
from stable_baselines3.common.buffers import ReplayBuffer, DictReplayBuffer
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import FlattenExtractor
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork
from torch.utils import tensorboard
from tqdm.auto import tqdm
from .params import Params
from spr_rl.envs.spr_env import SprEnv
import csv
import sys
import torch as th
from torch import nn

# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# Progress bar code from
# https://colab.research.google.com/github/araffin/rl-tutorial-jnrr19/blob/master/4_callbacks_hyperparameter_tuning.ipynb
class ProgressBarCallback(BaseCallback):
    """
    :param pbar: (tqdm.pbar) Progress bar object
    """
    def __init__(self, pbar):
        super(ProgressBarCallback, self).__init__()
        self._pbar = pbar

    def _on_step(self):
        # Update the progress bar:
        self._pbar.n = self.num_timesteps
        self._pbar.update(0)


# this callback uses the 'with' block, allowing for correct initialisation and destruction
class ProgressBarManager(object):
    def __init__(self, total_timesteps):  # init object with total timesteps
        self.pbar = None
        self.total_timesteps = total_timesteps

    def __enter__(self):  # create the progress bar and callback, return the callback
        self.pbar = tqdm(total=self.total_timesteps)

        return ProgressBarCallback(self.pbar)

    def __exit__(self, exc_type, exc_val, exc_tb):  # close the callback
        self.pbar.n = self.total_timesteps
        self.pbar.update(0)
        self.pbar.close()

class SPRPolicy(QRDQNPolicy):
    """
    Custom policy. Exactly the same as MlpPolicy but with different NN configuration
    """

    def __init__(self, observation_space, action_space, lr_schedule, **_kwargs):

        self.params: Params = _kwargs['params']
        activ_function_name = self.params.agent_config['nn_activ']
        activation_fn = eval(activ_function_name)
        # Disable orthogonal initialization
        self.ortho_init = False
        super(SPRPolicy, self).__init__(observation_space, action_space, lr_schedule=lr_schedule,
                                        activation_fn=activation_fn)


class DRDQN_Agent:

    def __init__(self, params: Params):

        self.params: Params = params
        policy_name = self.params.agent_config['policy']
        self.policy = eval(policy_name)

    def create_model(self, n_envs=1):
        """ Create env and agent model """
        env_cls = SprEnv
        self.env = make_vec_env(env_cls, n_envs=n_envs, env_kwargs={"params": self.params}, seed=self.params.seed)
        self.model = QRDQN(
            self.policy,
            self.env,
            learning_rate=self.params.agent_config['learning_rate'],
            buffer_size=self.params.agent_config['buffer_size'],
            learning_starts=self.params.agent_config['learning_starts'],
            batch_size=self.params.agent_config['batch_size'],
            tau=self.params.agent_config['tau'],
            gamma=self.params.agent_config['gamma'],
            train_freq=self.params.agent_config['train_freq'],
            gradient_steps=self.params.agent_config['gradient_steps'],
            # replay_buffer_class=Optional[ReplayBuffer],
            # replay_buffer_kwargs={"buffer_size": 1000000},
            optimize_memory_usage=self.params.agent_config['optimize_memory_usage'],
            target_update_interval=self.params.agent_config['target_update_interval'],
            exploration_fraction=self.params.agent_config['exploration_fraction'],
            exploration_initial_eps=self.params.agent_config['exploration_initial_eps'],
            exploration_final_eps=self.params.agent_config['exploration_final_eps'],
            max_grad_norm=self.params.agent_config['max_grad_norm'],
            verbose=self.params.agent_config['verbose'],
            tensorboard_log=".\\tb\\dqn\\",
            seed=self.params.seed,
            policy_kwargs={"params": self.params}
        )

    def train(self):
        with ProgressBarManager(self.params.training_duration) as callback:
            self.model.learn(
                total_timesteps=self.params.training_duration,
                tb_log_name=self.params.tb_log_name,
                callback=callback)

    def test(self):
        self.params.test_mode = True
        obs = self.env.reset()
        self.setup_writer()
        episode = 1
        step = 0
        episode_reward = [0.0]
        done = False
        # Test for 1 episode
        while not done:
            action, _states = self.model.predict(obs)
            obs, reward, dones, info = self.env.step(action)
            episode_reward[episode - 1] += reward[0]
            if info[0]['sim_time'] >= self.params.testing_duration:
                done = True
                self.write_reward(episode, episode_reward[episode - 1])
                episode += 1
            sys.stdout.write(
                "\rTesting:" +
                f"Current Simulator Time: {info[0]['sim_time']}. Testing duration: {self.params.testing_duration}")
            sys.stdout.flush()
            step += 1
        print("")

    def save_model(self):
        """ Save the model to a zip archive """
        self.model.save(self.params.model_path)

    def load_model(self, path=None):
        """ Load the model from a zip archive """
        if path is not None:
            self.model = QRDQN.load(path)
        else:
            self.model = QRDQN.load(self.params.model_path)
            # Copy the model to the new directory
            self.model.save(self.params.model_path)

    def setup_writer(self):
        episode_reward_filename = f"{self.params.result_dir}/episode_reward.csv"
        episode_reward_header = ['episode', 'reward']
        self.episode_reward_stream = open(episode_reward_filename, 'a+', newline='')
        self.episode_reward_writer = csv.writer(self.episode_reward_stream)
        self.episode_reward_writer.writerow(episode_reward_header)

    def write_reward(self, episode, reward):
        self.episode_reward_writer.writerow([episode, reward])
