import click
import random

import ray

from spr_rl.agent import PPO_Agent
# from spr_rl.agent import SAC_Agent
# from spr_rl.agent import DQN_Agent
# from spr_rl.agent import DRDQN_Agent
# from spr_rl.agent import TQC_Agent
from spr_rl.agent import QMIX_Agent
from spr_rl.agent import Params
import os
import inspect
import numpy as np
from spr_rl.envs.spr_env import SprEnv
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["LOG_DIR"] = "D:\\code\\madrl-coordination\\src\\spr_rl\\agent\\"
# TUNE_DISABLE_AUTO_CALLBACK_SYNCER=1

# Click decorators
# TODO: Add testing flag (timestamp and seed). Already set in params
@click.command()
@click.argument('network', type=click.Path(exists=True))
@click.argument('agent_config', type=click.Path(exists=True))
@click.argument('simulator_config', type=click.Path(exists=True))
@click.argument('services', type=click.Path(exists=True))
@click.argument('training_duration', type=int)
@click.option('-s', '--seed', type=int, help="Set the agent's seed", default=None)
@click.option('-t', '--test', help="Path to test timestamp and seed", default=None)
@click.option('-a', '--append_test', help="test after training", is_flag=True)
@click.option('-m', '--checkpoint_path', help="path to a model zip file", default=None)
@click.option('-ss', '--sim-seed', type=int, help="simulator seed", default=None)
@click.option('-b', '--best', help="Select the best agent", is_flag=True)
def main(network, agent_config, simulator_config, services, training_duration,
         seed, test, append_test, checkpoint_path, sim_seed, best):
    """
    SPR-RL DRL Scaling and Placement main executable
    """
    # Get or set a seed
    # global checkpoint_path
    if seed is None:
        seed = random.randint(0, 9999)

    # Seed random generators
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    print(f"Creating agent with seed: {seed}")
    print(f"Using network: {network}")

    if best:
        # Just a placeholder to trick params into thinking it is in test mode
        test = "placeholder"

    # Create the parameters object
    params = Params(
        seed,
        agent_config,
        simulator_config,
        network,
        services,
        training_duration=training_duration,
        test_mode=test,
        sim_seed=sim_seed,
        best=best
    )

    # Create the agent
    agent_class = eval(params.agent_config['agent'])
    agent = agent_class(params)

    if test is None:
        # Create model and environment
        agent.create_model(n_envs=params.agent_config['n_env'])
        # Train the agent
        print(f"Training for {training_duration} steps.")
        checkpoint_path, analysis = agent.train()

        # Save the model
        print(f"Saving the model to {params.model_path}")
        print(f"Agent training ID: {params.training_id}")
        agent.save_model()

    # Check to see if testing or append test is set to test the agent
    if test is not None or append_test:
        if append_test:
            params.test_mode = True
            params.create_result_dir()
        # Create or recreate model
        agent.create_model()

        # Load weigths
        # ray.init()
        # agent.load_model('D:/code/madrl-coordination/src/spr_rl/agent/PPO_2023-05-17_22-22-01/PPO_SprEnv-v0_30cf8_00000_0_2023-05-17_22-22-02/checkpoint_000050/checkpoint-50')
        agent.load_model(checkpoint_path)
        print("Testing for 1 episode")
        agent.test()
    print("Storing reward function")
    store_reward_function(params)
    print("Done")


def store_reward_function(params: Params):
    reward_path = os.path.join(params.result_dir, "reward.py")
    with open(reward_path, 'w') as reward_file:
        reward_function = inspect.getsource(SprEnv.step)
        reward_file.write(reward_function)


if __name__ == "__main__":
    # agent_config = "D:/code/madrl-coordination/inputs/config/drl/qmix/qmix_default.yaml"
    agent_config = "D:/code/madrl-coordination/inputs/config/drl/ppo/ppo_default.yaml"
    # agent_config = "D:/code/madrl-coordination/inputs/config/drl/sac/sac_default.yaml"
    # agent_config = "D:/code/madrl-coordination/inputs/config/drl/dqn/dqn_defaul.yaml"
    # agent_config = "D:/code/madrl-coordination/inputs/config/drl/drdqn/drdqn_defaul.yaml"
    # agent_config = "D:/code/madrl-coordination/inputs/config/drl/tqc/tqc_defaul.yaml"
    network = "D:/code/madrl-coordination/inputs/networks/abilene_1-5in-1eg/abilene-in5-rand-cap0-2.graphml"
    # network = "D:/code/madrl-coordination/inputs/networks/abilene_1-5in-1eg/abilene-in8-rand-cap0-10.graphml"
    services = "D:/code/madrl-coordination/inputs/services/abc-start_delay1.yaml"
    # sim_config = "D:/code/madrl-coordination/inputs/config/simulator/mean-10.yaml"
    # sim_config = "D:/code/madrl-coordination/inputs/config/simulator/mean-10-poisson.yaml"
    # sim_config = "D:/code/madrl-coordination/inputs/config/simulator/mmpp-12-8.yaml"
    sim_config = "D:/code/madrl-coordination/inputs/config/simulator/poisson-real-world-trace.yaml"
    training_duration = "200000"
    # training_duration = "100000"
    main([network, agent_config, sim_config, services, training_duration, '-a', '-s', '2443'])
    # main([network, agent_config, sim_config, services, training_duration, '-t', '2023-05-17_22-21-46_seed2443'])

    # main([network, agent_config, sim_config, services, training_duration, '--best'])
    # main([network, agent_config, sim_config, services, training_duration, '-t', 'best',
    #       '-m', 'results/models/poisson/model.zip'])
