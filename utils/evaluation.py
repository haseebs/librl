import logging
import numpy as np

log = logging.getLogger(__name__)

def evaluate(env, agent, episodes):
    epsiode_rewards = []
    episode_lengths = []
    for e in range(episodes):
        obs, info = env.reset()
        length = 0
        while True:
            action = agent.act(obs, greedy=False)
            obs_next, reward, terminated, truncated, info = env.step(action)
            epsiode_rewards.append(reward)
            length += 1
            if terminated or truncated:
                obs, info = env.reset()
                episode_lengths.append(length)
                break
            obs = obs_next

def evaluate_episodic(env, agent, episodes):
    epsiode_rewards = []
    episode_lengths = []
    for e in range(episodes):
        obs, info = env.reset()
        sum_rewards = 0
        length = 0
        while True:
            action = agent.act(obs, greedy=False)
            obs_next, reward, terminated, truncated, info = env.step(action)
            length += 1
            sum_rewards += reward
            if terminated or truncated:
                obs, info = env.reset()
                episode_lengths.append(length)
                epsiode_rewards.append(sum_rewards)
                break
            obs = obs_next
    
    """
    take only last half of trajectories
    """
    # epsiode_rewards = epsiode_rewards[-10:]
    mean_reward = np.mean(epsiode_rewards)
    max_reward = np.max(epsiode_rewards)
    min_reward = np.min(epsiode_rewards)
    median_reward = np.median(epsiode_rewards)
    std_reward = np.std(epsiode_rewards)
    mean_length = np.mean(episode_lengths)
    log.info(f'evaluation \t \t mean length {int(mean_length)} \
             mean/max/min/median/std: \
             {mean_reward:.2f}/{max_reward:.2f}/{min_reward:.2f}/{median_reward:.2f}/{std_reward:.2f}')

    return mean_reward, std_reward
