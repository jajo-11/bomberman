import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import settings
from .hyper_parameters import *
from numba import njit, float32, int32, int64, boolean

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# our pytorch neural network
class DQN(nn.Module):
  def __init__(self):
    super(DQN, self).__init__()
    self.lin = nn.Linear(FEATURE_SIZE, 512)
    self.out = nn.Linear(512, 6)

  # evaluate features x -> values for actions
  def forward(self, x):
    x = F.relu(self.lin(x))
    return self.out(x)


def setup(self):
  # select device for torch to run on
  self.device = torch.device("cuda" if torch.cuda.is_available() and self.train else "cpu")
  self.rand = np.random.default_rng()

  # init networks
  self.policy = DQN()
  self.target = DQN()

  # load network from file if it exisits
  if not os.path.isfile("Policy.pt"):
    self.logger.info("Setting up model from scratch.")
  else:
    self.policy.load_state_dict(torch.load("Policy.pt", map_location=self.device))

  # match target network to policy network
  self.target.load_state_dict(self.policy.state_dict())
  self.target.eval()

  # move network to selected device
  self.policy.to(self.device)
  self.target.to(self.device)

  # dummy call to compile the function before we are time constraint
  x, y, b = 1, 1, True
  field = np.ones((17, 17), dtype=np.int32)
  explo = np.zeros((17, 17), dtype=np.float32)
  coins = np.zeros((1, 2), dtype=np.int64)
  bombs = np.zeros((1, 2), dtype=np.int64)
  bombt = np.zeros(1, dtype=np.int64)
  othep = np.zeros((1, 2), dtype=np.int64)
  otheb = np.zeros(1, dtype=np.bool_)
  visit = np.zeros((17, 17), dtype=np.int64)
  jit_state_to_features(x, y, b, field, explo, coins, bombs, bombt, othep, otheb, visit)


def act(self, game_state: dict) -> str:
  # visited map needs to be reset every round
  if game_state["step"] == 1:
    self.visited = np.zeros((17, 17), dtype=np.int64)

  # get game state
  _, _, _, (x, y) = game_state["self"]
  state = state_to_features(game_state, self.visited)
  self.visited[x, y] += 1

  # cache last state
  self.last_state = state

  # decide if we are training or not and adjust epsilon accordingly
  r = self.rand.random()
  if self.train:
    eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1.*self.steps/EPS_DECAY)
    self.tb.add_scalar("Parameters/Epsilon", eps_threshold, self.steps)
    self.steps += 1
    hi = 6
  else:
    eps_threshold = 0.05
    # do not act randomly if bomb is nearby
    for (xx, yy), t in game_state["bombs"]:
      if np.sqrt((x-xx)**2+(x-yy)**2) <= settings.BOMB_POWER+1:
        eps_threshold = 0
        break
    # do not place bombs randomly while not learning
    hi = 5

  if r > eps_threshold:
    # query network and return argmax
    with torch.no_grad():
      a = self.policy(torch.from_numpy(state).to(self.device)).max(0)[1].item()
    return ACTIONS[a]
  else:
    # random choice
    return ACTIONS[self.rand.integers(low=0, high=hi)]

# global constant for size of the feature vector
FEATURE_SIZE = 52
def state_to_features(game_state: dict, visited) -> np.array:
  # this function takes the dict and unpacks it into a data structure that numba functions prefer
  _, _, b, (x, y) = game_state["self"]
  field = game_state["field"]
  explosion_map = game_state["explosion_map"].astype(np.float32)
  coins = np.array(game_state["coins"], dtype=np.int64)
  bombs = np.zeros((len(game_state["bombs"]), 2), dtype=np.int64)
  bombs_timer = np.zeros((bombs.shape[0]), dtype=np.int64)
  for i, ((xx, yy), t) in enumerate(game_state["bombs"]):
    bombs[i] = (xx, yy)
    bombs_timer[i] = t
  other_count = len(game_state["others"])
  other_pos = np.zeros((other_count, 2), dtype=np.int64)
  other_bomb = np.zeros(other_count, dtype=np.bool_)

  for i, (_, _, b, (xx, yy)) in enumerate(game_state["others"]):
    other_pos[i] = (xx, yy)
    other_bomb[i] = b

  # numba has limited support for empty arrays so we fill coins with values that will not impact the features
  if len(coins) == 0:
    coins = np.zeros((1, 2), dtype=np.int64)

  return jit_state_to_features(x, y, b, field, explosion_map, coins, bombs, bombs_timer, other_pos, other_bomb,
                               visited)
@njit
#@njit(float32[:](int64, int64, boolean, int32[:, :], float32[:, :], int64[:,:], int64[:,:], int64[:], int64[:,:], boolean[:], int64[:, :]))
def jit_state_to_features(x: int64, y: int64, b: boolean, field, explosion_map,
                          coins, bombs, bombs_timer, other_pos, other_bomb, visited):
  features = np.zeros(FEATURE_SIZE, dtype=np.float32)

  # calculate distance map from each of the four sourounding points of the player
  distance = np.full((field.shape[0], field.shape[1], 4), np.inf, dtype=np.float32)
  for i, (xx, yy) in enumerate([(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]):
    checked = field == -1
    stack = np.zeros((17*17, 2), np.int64)
    stack[0] = xx, yy
    start = 0
    end = 1
    distance[xx, yy, i] = 0
    checked[xx, yy] = True
    while start != end:
      xxx, yyy = stack[start]
      start += 1
      if field[xxx, yyy] == 0:
        for xxxx, yyyy in [(xxx + 1, yyy), (xxx - 1, yyy), (xxx, yyy + 1), (xxx, yyy - 1)]:
          if not checked[xxxx, yyyy]:
            checked[xxxx, yyyy] = True
            distance[xxxx, yyyy, i] = distance[xxx, yyy, i] + 1
            stack[end] = xxxx, yyyy
            end += 1

  # add features that are given for each direction
  for i, (xx, yy) in enumerate([(x+1, y), (x-1, y), (x, y+1), (x, y-1)]):
    if field[xx, yy] == 0:
      
      features[i] = explosion_map[xx, yy]
      features[47+i] = visited[xx, yy]

      for xxx, yyy in coins:
        if distance[xxx, yyy, i] < 30:
          features[4+i] = max(features[4+i], (30-distance[xxx, yyy, i])/30)

      for j, (xxx, yyy) in enumerate(bombs):
        if xxx == x and yyy == y:
          features[8] = bombs_timer[j]
        elif ((xxx == xx and yyy == yy) or (xxx == xx and xx % 2 == 1) or (yyy == yy and yy % 2 == 1))\
            and np.abs(xxx - xx) + np.abs(yyy - yy) < settings.BOMB_POWER + 2:
          features[9+i] = max(features[9+i], (settings.BOMB_TIMER + 1 - bombs_timer[j])/(settings.BOMB_TIMER + 1))
          features[13+i] = max(features[13+i], (settings.BOMB_POWER + 2 - (np.abs(xxx - xx) + np.abs(yyy - yy))))

    for xxx, yyy in other_pos:
      if (xx - xxx) ** 2 + (yy - yyy) ** 2 < 6:
        features[17 + i] = max(6-np.sqrt((xx - xxx) ** 2 + (yy - yyy) ** 2), features[17+i])

  # extend the field so we can take a 5x5 slice around the agent so he can se crates and walls
  field_ex = np.full((settings.COLS + 2, settings.ROWS + 2), -1)
  field_ex[1:-1, 1:-1] = field
  features[21:46] = field_ex[x-1:x+4, y-1:y+4].flatten()

  # tell the agent if a bomb can be placed
  features[46] = b
  # maintain the markov property so the agent has a chance of learning why he is getting punished
  features[51] = visited[x, y]

  return features
