import numpy as np
from ray.rllib.evaluation.tests.test_trajectory_view_api import MyCallbacks
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import ray
from ray import tune
from ray.tune import commands
from ray.rllib.agents import trainer
import ray.rllib.agents.ppo as ppo
from ray.rllib import TorchPolicy, env
from ray.rllib.examples.custom_env import TorchCustomModel

from ray.rllib.models import ModelCatalog
from ray.rllib.models.catalog import torch
from torch.utils import tensorboard
from tqdm.auto import tqdm
from spr_rl.agent.params import Params
from ray.tune.registry import register_env
from spr_rl.envs.spr_env import SprEnv
import csv
import sys
import torch as th
from torch import nn


class PPO_Agent:

    def __init__(self, params: Params):
        self.params: Params = params
        # policy_name = self.params.agent_config['policy']
        # self.policy = eval(policy_name)

        # self.env_config = {"params": self.params}

    def create_model(self, n_envs=1):
        """ Create env and agent model """
        # ray.init()

        self.env = gym.make('SprEnv-v0', params=self.params)
        # register_env('SprEnv-v0', lambda: self.params)
        # self.env = 'SprEnv-v0'

    def train(self):

        analysis = ray.tune.run(
            ppo.PPOTrainer,
            stop={"timesteps_total": self.params.training_duration},
            config={
                "env": 'SprEnv-v0',
                "env_config": {"params": self.params},
                "framework": "torch",
                # "model": {
                #     "post_fcnet_hiddens": [64, 64],
                #     "post_fcnet_activation": "relu",
                # },
                "num_workers": 0,
                "exploration_config": {
                "type": "Curiosity",  # <- Use the Curiosity module for exploring.
                "eta": 1.0,  # Weight for intrinsic rewards before being added to extrinsic ones.
                "lr": 0.001,  # Learning rate of the curiosity (ICM) module.
                "feature_dim": 128,  # Dimensionality of the generated feature vectors.
                # Setup of the feature net (used to encode observations into feature (latent) vectors).
                "feature_net_config": {
                    "fcnet_hiddens": [],
                    "fcnet_activation": "relu",
                },
                "inverse_net_hiddens": [64],  # Hidden layers of the "inverse" model.
                "inverse_net_activation": "relu",  # Activation of the "inverse" model.
                "forward_net_hiddens": [64],  # Hidden layers of the "forward" model.
                "forward_net_activation": "relu",  # Activation of the "forward" model.
                "beta": 0.2,  # Weight for the "forward" loss (beta) over the "inverse" loss (1.0 - beta).
                # Specify, which exploration sub-type to use (usually, the algo's "default"
                # exploration, e.g. EpsilonGreedy for DQN, StochasticSampling for PG/SAC).
                "sub_exploration": {
                    "type": "StochasticSampling",
                }}
                # "callbacks": MyCallbacks,
                # "episodes_this_iter": 0
                # "lr": self.params.agent_config['learning_rate']
                # "lr": 3e-4,
                # "rollout_fragment_length": 64,
                # "sgd_minibatch_size": 64,
                # "lambda": 0.95,
                # "vf_clip_param": 0.2,
                # "vf_loss_coeff": 0.5

            },
            checkpoint_at_end=True,
            local_dir="D:\\code\\madrl-coordination\\src\\spr_rl\\agent\\"
        )
        # analysis.default_metric = "episode_reward_mean"
        # analysis.default_mode = "max"

        checkpoints = analysis.get_trial_checkpoints_paths(trial=analysis.get_best_trial('episode_reward_mean','max'),
                                                           metric='episode_reward_mean')
        # retriev the checkpoint path; we only have a single checkpoint, so take the first one
        checkpoint_path = checkpoints[0][0]

        return checkpoint_path, analysis


    def test(self):
        self.params.test_mode = True
        env = self.env
        # env = self.env
        obs = env.reset()
        self.setup_writer()
        episode = 1
        step = 0
        episode_reward = [0.0]
        done = False
        # Test for 1 episode
        while not done:
            action = self.agent.compute_action(obs)
            obs, reward, dones, info = env.step(action)
            episode_reward[episode - 1] += reward
            if info['sim_time'] >= self.params.testing_duration:
                done = True
                self.write_reward(episode, episode_reward[episode - 1])
                episode += 1
            sys.stdout.write(
                "\rTesting:" +
                f"Current Simulator Time: {info['sim_time']}. Testing duration: {self.params.testing_duration}")
            sys.stdout.flush()
            step += 1
        print("")

    def save_model(self):
        """ Save the model to a zip archive """
        # self.trainer.save(self.params.model_path)

    def load_model(self, checkpoint_path):
        """ Load the model from a zip archive """
        self.agent = ppo.PPOTrainer(config={
                "env": 'SprEnv-v0',
                "env_config": {"params": self.params},
                "framework": "torch",
                "num_workers": 0,
                "exploration_config": {
                "type": "Curiosity",  # <- Use the Curiosity module for exploring.
                "eta": 1.0,  # Weight for intrinsic rewards before being added to extrinsic ones.
                "lr": 0.001,  # Learning rate of the curiosity (ICM) module.
                "feature_dim": 128,  # Dimensionality of the generated feature vectors.
                # Setup of the feature net (used to encode observations into feature (latent) vectors).
                "feature_net_config": {
                    "fcnet_hiddens": [],
                    "fcnet_activation": "relu",
                },
                "inverse_net_hiddens": [64],  # Hidden layers of the "inverse" model.
                "inverse_net_activation": "relu",  # Activation of the "inverse" model.
                "forward_net_hiddens": [64],  # Hidden layers of the "forward" model.
                "forward_net_activation": "relu",  # Activation of the "forward" model.
                "beta": 0.2,  # Weight for the "forward" loss (beta) over the "inverse" loss (1.0 - beta).
                # Specify, which exploration sub-type to use (usually, the algo's "default"
                # exploration, e.g. EpsilonGreedy for DQN, StochasticSampling for PG/SAC).
                "sub_exploration": {
                    "type": "StochasticSampling",
                }}
                # "callbacks": MyCallbacks,
                # "episodes_this_iter": 0
                # "lr": self.params.agent_config['learning_rate']
                # "lr": 3e-4,
                # "rollout_fragment_length": 64,
                # "sgd_minibatch_size": 64,
                # "lambda": 0.95,
                # "vf_clip_param": 0.2,
                # "vf_loss_coeff": 0.5
            })
        self.agent.restore(checkpoint_path)
        # if path is not None:
        #     self.agent.restore(self.checkpoint_path)
        # else:
        #     self.agent.restore(self.checkpoint_path)
            # Copy the model to the new directory


    def setup_writer(self):
        episode_reward_filename = f"{self.params.result_dir}/episode_reward.csv"
        episode_reward_header = ['episode', 'reward']
        self.episode_reward_stream = open(episode_reward_filename, 'a+', newline='')
        self.episode_reward_writer = csv.writer(self.episode_reward_stream)
        self.episode_reward_writer.writerow(episode_reward_header)

    def write_reward(self, episode, reward):
        self.episode_reward_writer.writerow([episode, reward])


