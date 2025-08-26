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

from utils import utils
from utils.evaluation import evaluate_episodic
from utils.utils import prep_cfg_for_db
from experiment import ExperimentManager, Metric
from environments.factory import get_env
from models.actor import Actor
from models.critic import get_critic
from models.replay_buffers.factory import get_buffer
from agents.factory import get_agent

log = logging.getLogger(__name__)

def evaluate_fixed_env(cfg, seed, actor):
    env = get_env(cfg, seed)
    obs, info = env.reset(seed=seed)
    mean, shape = actor.policy(torch.FloatTensor(obs).unsqueeze(0))
    return mean, shape

@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    start = timer()
    if HydraConfig.get().mode.value == 2:  # check whether its a sweep
        cfg.run += HydraConfig.get().job.num
        log.info(f'Running sweep... Run ID: {cfg.run}')
    log.info(f"Output directory  : \
             {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}")
    cfg.agent.actor.optimizer.lr = cfg.agent.actor.optimizer.critic_lr_multiplier * \
                                    cfg.agent.critic.optimizer.lr
    flattened_cfg = prep_cfg_for_db(OmegaConf.to_container(cfg),
                                    to_remove=["schema", "db"])
    log.info(pretty_repr(flattened_cfg))

    torch.set_num_threads(cfg.n_threads)
    utils.set_seed(cfg.seed)

    env = get_env(cfg.env, seed=cfg.seed)
    test_env = get_env(cfg.env, seed=cfg.seed)

    actor = Actor(cfg.agent.actor, env, cfg.agent.store_old_policy, cfg.device)
    critic = get_critic(cfg.agent.critic, env, cfg.device)
    buffer = get_buffer(cfg.agent.buffer, cfg.seed, env, cfg.device)
    agent = get_agent(cfg.agent, False, cfg.device, env, actor, critic, buffer)

    obs, info = env.reset(seed=cfg.seed)
    test_env.reset(seed=cfg.seed)
    step = 0
    all_rewards = []
    episode = 0
    episodic_reward = 0
    policy_info = []
    for step in range(cfg.steps):
        if step % 1000 == 0:
            policy_info.append(evaluate_fixed_env(cfg.env, cfg.seed, actor))
        if step < cfg.learning_starts:
            action = env.action_space.sample()
        else:
            action = agent.act(obs)
        obs_next, reward, terminated, truncated, info = env.step(action)
        # this causes gamma=0 when truncated. Do we want this behavior?
        # (instead of gamma=0 only when terminated)
        buffer.push(obs, action, reward, obs_next, 1-(terminated or truncated))
        if step > cfg.learning_starts:
            agent.update_critic()
            agent.update_actor()
        obs = obs_next
        episodic_reward += reward

        if terminated or truncated:
            obs, info = env.reset()
            log.info(f'step: {step} \t \t episode: {episode}, \
                     reward: {episodic_reward} \t action: {action}')
            episode += 1
            episodic_reward = 0


    torch.save(policy_info, cfg.agent.actor.policy.policy + ".pt")
    total_time = timedelta(seconds=timer() - start).seconds / 60
    auc_10 = float(np.mean(all_rewards[-int(len(all_rewards)*0.1):]))
    auc_50 = float(np.mean(all_rewards[-int(len(all_rewards)*0.5):]))
    auc_100 = float(np.mean(all_rewards))
    log.info(f'Total time taken: {total_time}  minutes')

if __name__ == "__main__":
    main()
