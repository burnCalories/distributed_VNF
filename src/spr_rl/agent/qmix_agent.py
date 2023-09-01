import gym
import ray
from gym.spaces import Tuple, MultiDiscrete, Dict, Discrete, Box
from ray import tune
from ray.tune import commands
from ray.rllib.agents import trainer
import ray.rllib.agents.qmix as qmix
from ray.rllib import TorchPolicy, env
from ray.rllib.examples.custom_env import TorchCustomModel
from ray.rllib.models import ModelCatalog
from ray.rllib.models.catalog import torch
from torch.utils import tensorboard
from tqdm.auto import tqdm
from typing import Tuple

from spr_rl.agent.params import Params
from ray.tune.registry import register_env
from spr_rl.envs.spr_env import SprEnv
from ray.rllib.env.multi_agent_env import MultiAgentEnv, make_multi_agent
import csv
import sys
import torch as th
from torch import nn


class QMIX_Agent:

    def __init__(self, params: Params):
        self.params: Params = params
        # policy_name = self.params.agent_config['policy']
        # self.policy = eval(policy_name)

        # self.env_config = {"params": self.params}

    def create_model(self, n_envs=1):
        """ Create env and agent model """
        # ray.init()

        grouping = {"group_1": [0, 1]}
        obs_space = gym.spaces.Tuple([
            gym.spaces.Box(-20, 1000, shape=self.params.observation_shape),
            gym.spaces.Box(-20, 1000, shape=self.params.observation_shape)
        ]
        )
        act_space = gym.spaces.Tuple(
            [
                gym.spaces.Discrete(self.params.action_limit),
                gym.spaces.Discrete(self.params.action_limit)
            ]
        )

        self.env = register_env("SprEnv-v0",
                     lambda config: SprEnv(params=self.params).with_agent_groups(
            grouping, obs_space=obs_space, act_space=act_space))

    def train(self):

        analysis = ray.tune.run(
            qmix.QMixTrainer,
            stop={"timesteps_total": self.params.training_duration},
            config={
                "env": 'SprEnv-v0',
                "env_config": {"params": self.params, "num_agents": 2},
                "framework": "torch",
                # "model": {
                #     "post_fcnet_hiddens": [64, 64],
                #     "post_fcnet_activation": "relu",
                # },
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
        global agent
        self.params.test_mode = True
        env = self.env
        # env = self.env
        obs = env.reset()
        obs_dict = {agent: obs[i] for i, agent in enumerate(env.agents)}
        self.setup_writer()
        episode = 1
        step = 0
        episode_reward = [0.0]
        dones = {agent: False for agent in env.agents}

        # Test for 1 episode
        while not all(dones.values()):
            actions = {agent: self.agent.compute_action(obs_dict[agent])
                  for agent in env.agents}
            obs, reward, dones, infos = env.step(actions)
            obs_dict = {agent: obs[i] for i, agent in enumerate(env.agents)}
            for agent in env.agents:
                episode_reward[agent] += reward
            if infos['sim_time'] >= self.params.testing_duration:
                dones = {agent: True for agent in env.agents}
                self.write_reward(episode, episode_reward[agent])
                episode += 1
            sys.stdout.write(
                "\rTesting:" +
                f"Current Simulator Time: {infos['sim_time']}. Testing duration: {self.params.testing_duration}")
            sys.stdout.flush()
            step += 1
        print("")

    def save_model(self):
        """ Save the model to a zip archive """
        # self.trainer.save(self.params.model_path)

    def load_model(self, checkpoint_path):
        """ Load the model from a zip archive """
        self.agent = qmix.QMixTrainer(config={
                "env": 'SprEnv-v0',
                "env_config": {"params": self.params},
                "framework": "torch"
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