import numpy as np
import torch
import argparse
import wandb
import time

import config
from env.envs import GriddlyGymEnv
from agents.agent import Agent
from agents.agent_dqn import AgentDQN
from agents.agent_bellman import AgentBellman
from agents.agent_a2c import AgentA2C
from utils.common import logging


# ======= Init config =======
time_stamp = time.strftime("%d%m%Y-%H%M")
parser = argparse.ArgumentParser()
parser.add_argument("--logdir", type=str, default='log-{}'.format(time_stamp))
#parser.add_argument("--config_name", type=str, default=config_spec)
args = parser.parse_args()
args = config.get_params(args)

# ====== Monitoring =======
if args.wandb_log:
    args.wandb_log = wandb.init(project='ThesisComparisons', dir='{}/monitoring'.format(args.logdir), config=args, name='qssm-{}'.format(time_stamp))

# ======= Seeds =======
if args.seed is not None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

# ======= Logs ========
print(time.ctime())
print(args)
print(args.logdir)
logging.init_dirs(args.logdir)

# ======= Environment =========
env = GriddlyGymEnv(args)
test_env = GriddlyGymEnv(args, registered=True)
action_space = env.action_space.n
obs_dims = env.observation_space.shape
print('action space ', action_space, ' obs space ', obs_dims)

# ======= Agent =======
if args.agent in ['dqn', 'q_rhe', 'dqn_plan', 'dqn_seed_rhe']:
    agent = AgentDQN(env, test_env, args)
elif args.agent in ['td_rhe']:
    agent = AgentBellman(env, test_env, args)
elif args.agent == 'a2c':
    agent = AgentA2C(env, test_env, args)
else:
    raise NotImplementedError
print('agent ', args.agent)

agent.train()