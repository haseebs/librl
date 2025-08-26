import random
import hydra
import torch
import numpy as np
import logging

from datetime import timedelta
from rich.pretty import pretty_repr
from timeit import default_timer as timer
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
import gymnasium
from gymnasium.spaces.utils import flatdim

from utils import utils
from utils.evaluation import evaluate_episodic
from utils.utils import prep_cfg_for_db
from experiment import ExperimentManager, Metric
from environments.factory import get_env
from models.actor import PPOActor
from models.critic import get_critic
from models.replay_buffers.factory import get_buffer
from agents.ppo import PPO
from models.replay_buffers.rollout_buffer import TorchRolloutBuffer

log = logging.getLogger(__name__)

@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    start = timer()
    if HydraConfig.get().mode.value == 2:  # check whether its a sweep
        cfg.run += HydraConfig.get().job.num
        log.info(f'Running sweep... Run ID: {cfg.run}')
    log.info(f"Output directory  : {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}")
    flattened_cfg = prep_cfg_for_db(OmegaConf.to_container(cfg), to_remove=["schema", "db"])
    log.info(pretty_repr(flattened_cfg))

    exp = ExperimentManager(cfg.db_name, flattened_cfg, cfg.db_prefix, cfg.db)
    tables = {}
    for table_name in list(cfg.schema.keys()):
        columns = cfg.schema[table_name].columns
        primary_keys = cfg.schema[table_name].primary_keys
        tables[table_name] = Metric(table_name, columns, primary_keys, exp)
    torch.set_num_threads(cfg.n_threads)
    utils.set_seed(cfg.seed)

    env = get_env(cfg.env, cfg.seed)
    test_env = get_env(cfg.env, seed=cfg.seed)
    if isinstance(env, gymnasium.Env):
        state_dim = env.observation_space.shape
        action_dim = env.action_space.shape

    actor = PPOActor(cfg.agent, 
                     env, 
                     cfg.agent.store_old_policy,
                     cfg.device
                     )
    buffer = TorchRolloutBuffer(capacity=cfg.agent.episode_steps+1,
                                seed=cfg.seed,
                                state_size=state_dim,
                                action_size=action_dim,
                                device=cfg.device,
                                )
    if isinstance(env, gymnasium.Env):
        state_dim = flatdim(env.observation_space)
        action_dim = flatdim(env.action_space)

    agent = PPO(discrete_action=False,
                            action_dim=action_dim,
                            state_dim=state_dim,
                            gamma=cfg.agent.gamma,
                            actor=actor,
                            rollout_buffer=buffer,
                            episode_steps=cfg.agent.episode_steps,
                            minibatches=cfg.agent.minibatches,
                            training_epochs=cfg.agent.training_epochs,
                            eps_clip=cfg.agent.eps_clip,
                            vloss_coef=cfg.agent.vloss_coef,
                            entropy_coef=cfg.agent.entropy_coef,
                            clip_vloss=cfg.agent.clip_vloss,
                            device=cfg.device,
                            )

    episode = 0
    episode_rewards = 0
    all_rewards = []
    obs, info = env.reset(seed=cfg.seed)
    action = env.action_space.sample()
    
    for step in range(cfg.steps):

        action, logprob, _, value = agent.get_action_and_value(obs)
        obs_next, reward, terminated, truncated, info = env.step(action)
        buffer.push(obs, action, reward, obs_next, 1 - (terminated or truncated), value, logprob)
        obs = obs_next
        episode_rewards += reward

        if terminated or truncated:
            log.info(f'step: {step}, \t episode: {episode}, \t episode reward: {episode_rewards} \t action: {action}')
            obs, info = env.reset(seed=cfg.seed)
            episode_rewards = 0
            episode += 1

        if np.mod(step+1, cfg.agent.episode_steps) == 0:
            agent.update_actor(log)
            buffer.clear()
        
        if step % cfg.evaluation_steps == 0:
            mean_reward, std_reward = evaluate_episodic(test_env,
                                                        agent,
                                                        cfg.evaluation_episodes)
            tables["returns"].add_data(
                [
                    cfg.run,
                    step,
                    episode,
                    mean_reward
                ]
            )
            all_rewards.append(mean_reward)

        if step % 10000 == 0:
            tables["returns"].commit_to_database()

    total_time = timedelta(seconds=timer() - start).seconds / 60
    auc_10 = float(np.mean(all_rewards[-int(len(all_rewards)*0.1):]))
    auc_50 = float(np.mean(all_rewards[-int(len(all_rewards)*0.5):]))
    auc_100 = float(np.mean(all_rewards))
    tables["summary"].add_data(
        [
            cfg.run,
            step,
            episode,
            auc_100,
            auc_50,
            auc_10,
            total_time
        ]
    )
    tables["returns"].commit_to_database()
    tables["summary"].commit_to_database()
    log.info(f'Total time taken: {total_time}  minutes')


if __name__ == "__main__":
    main()
