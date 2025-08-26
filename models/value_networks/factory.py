import torch.nn as nn
import gymnasium
from gymnasium.spaces.utils import flatdim
from .discrete_q_net import DiscreteQ
from .double_q_net import DoubleQ
from .q_net import Q
from .v_net import V


def get_value_network(cfg: dict, env: object) -> nn.Module:
    if isinstance(env, gymnasium.Env):
        num_inputs = flatdim(env.observation_space)
        num_actions = flatdim(env.action_space)
    else:
        raise NotImplementedError

    match cfg.name:
        case "v_net":
            return V(num_inputs, cfg.hidden_dim, cfg.init, cfg.activation)
        case "q_net":
            return Q(num_inputs, num_actions, cfg.hidden_dim, cfg.init, cfg.activation)
        case "double_q_net":
            return DoubleQ(num_inputs, num_actions, cfg.hidden_dim, cfg.init, cfg.activation)
        case _:
            raise NotImplementedError
