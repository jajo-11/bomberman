import lzma
import os
import pickle
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from typing import List
import torch
import time

from tensorboardX import SummaryWriter

import events as e
from .callbacks import state_to_features, FEATURE_SIZE
from .hyper_parameters import *

ACTIONS = {"UP": 0, "RIGHT": 1, "DOWN": 2, "LEFT": 3, "WAIT": 4, "BOMB": 5}

# class for replay memory, simple implementation of a ring buffer
class Memory:
  def __init__(self, capacity, device, rng):
    self.rand = rng
    self.device = device
    self.capacity = capacity
    self.position = 0
    self.full = False # True if all values are valid memories
    self.state = np.empty((capacity, FEATURE_SIZE), dtype=np.float32)
    self.action = np.empty(capacity, dtype=np.int64)
    self.next_state = np.empty((capacity, FEATURE_SIZE), dtype=np.float32)
    self.reward = np.empty(capacity, dtype=np.float32)
    self.final = np.empty(capacity, dtype=np.bool_) # was the saved state the last state in a game?

  def push(self, s, a, r, sn, final):
    i = self.position
    self.state[i] = s
    self.action[i] = a
    self.reward[i] = r
    self.next_state[i] = sn
    self.final[i] = final
    self.position += 1
    if self.position == self.capacity:  # loop back
      self.full = True
      self.position = 0

  def sample(self, batch_size):
    if self.full:
      sel = self.rand.integers(low=0, high=self.capacity, size=batch_size)
    else:
      sel = self.rand.integers(low=0, high=self.position, size=batch_size)
    return (
      self.state[sel],
      self.action[sel],
      self.reward[sel],
      self.next_state[sel],
      self.final[sel]
    )

def optimize_model(self):
  # check if enough memories for one batch are there
  if not self.memory.full and self.memory.position < BATCH_SIZE:
    return

  # sample the memories and make them tensors on the right device
  s, a, r, sn, final = self.memory.sample(BATCH_SIZE)
  s = torch.from_numpy(s).to(self.device)
  a = torch.from_numpy(a).to(self.device)
  r = torch.from_numpy(r).to(self.device)
  sn = torch.from_numpy(sn).to(self.device)
  final = torch.from_numpy(final).to(self.device)

  # get current values of actions
  state_action_values = self.policy(s)[np.arange(0, BATCH_SIZE), a]

  # get the values of the currently most valuable actions in the policy net from the target net
  with torch.no_grad():
    next_state_values = self.policy(sn)
    next_action = torch.argmax(next_state_values, axis=1)
    next_state_values = self.target(sn)[np.arange(0, BATCH_SIZE), next_action]

  next_state_values[final] = 0  # final states have an expected future reward of 0
  expected_state_action_values = (next_state_values * GAMMA) + r

  # calculate loss
  loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

  # just to be save use no_grad
  with torch.no_grad():
    self.tb.add_scalar("Stats/Loss", loss.mean(), self.steps)

  self.optimizer.zero_grad()
  loss.backward()
  # keep parameter size in check
  for param in self.policy.parameters():
    param.grad.data.clamp_(-1, 1)
  self.optimizer.step()

def setup_training(self):
  self.tb = SummaryWriter()  # init tensorboard
  # load replay memory
  if os.path.isfile("memory.xz"):
    with lzma.open("memory.xz", "rb") as mem:
      self.memory = pickle.load(mem)
  else:
    self.memory = Memory(MEMORY_SIZE, self.device, self.rand)

  # load optimizer
  self.optimizer = optim.RMSprop(self.policy.parameters())
  if os.path.isfile("Optim.pt"):
    self.optimizer.load_state_dict(torch.load("Optim.pt", map_location=self.device))

  # load where in the eps-greedy curve we are
  if os.path.isfile("steps.pkl"):
    with open("steps.pkl", "rb") as st:
      self.steps = pickle.load(st)
  else:
    self.steps = 0

  # timer for saving
  self.last_save = time.time()
  # for logging
  self.creward = 0
  self.save_counter = 0

  # dict for enemy state
  self.enemy_visited = {}
  self.enemy_last_state = {}
  self.enemy_to_record = None


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
  # skip first event of round
  if old_game_state is None:
    return
  _, _, _, (x, y) = new_game_state["self"]
  reward = reward_from_events(self, events, x, y)
  sn = state_to_features(new_game_state, self.visited)
  s = self.last_state  # load cached state
  action = ACTIONS[self_action]  # get number representation of action
  self.memory.push(s, action, reward, sn, False)  # store event
  optimize_model(self)

  # update target network
  if self.steps % TARGET_UPDATE == 0:
    self.target.load_state_dict(self.policy.state_dict())


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
  # store last event
  _, _, _, (x, y) = last_game_state["self"]
  reward = reward_from_events(self, events, x, y)
  s = state_to_features(last_game_state, self.visited)
  sn = s
  action = ACTIONS[last_action]
  self.memory.push(s, action, reward, sn, True)

  # log results
  self.tb.add_scalar("Stats/Steps", last_game_state["step"], last_game_state["round"])
  self.tb.add_scalar("Stats/Score", last_game_state["self"][1], last_game_state["round"])
  self.tb.add_scalar("Stats/Cumulative Reward", self.creward, last_game_state["round"])
  self.creward = 0

  optimize_model(self)

  # dont miss the target update
  if self.steps % TARGET_UPDATE == 0:
    self.target.load_state_dict(self.policy.state_dict())

  # maybe save checkpoint
  t = time.time()
  # check for special files to save model or memory
  if self.last_save + 5*60 < t:
    self.last_save = t
    self.save_counter += 1
    self.tb.flush()
    # save every 15 min
    if self.save_counter == 3 or os.path.isfile(".save"):
      torch.save(self.policy.state_dict(), "Policy.pt")
      torch.save(self.policy.state_dict(), f"Policy_{time.strftime('%d_%m_%H_%M')}.pt")
      torch.save(self.optimizer.state_dict(), "Optim.pt")
      with open("steps.pkl", "wb") as st:
        pickle.dump(self.steps, st)
      with lzma.open("memory.xz", "wb") as mem:
        pickle.dump(self.memory, mem)
      self.save_counter = 0

  # i am to lazy to create a specific hook at the end of all runs so just expect that 10000 is the number of rounds
  if last_game_state["round"] == 1000000:
    torch.save(self.policy.state_dict(), "Policy.pt")
    torch.save(self.optimizer.state_dict(), "Optim.pt")
    with open("steps.pkl", "wb") as st:
      pickle.dump(self.steps, st)
    with lzma.open("memory.xz", "wb") as mem:
      pickle.dump(self.memory, mem)
    self.tb.close()

def enemy_game_events_occurred(self, enemy_name: str, old_enemy_game_state: dict, enemy_action: str,
                               enemy_game_state: dict, enemy_events: List[str]):
  # for learning with multiple agents select only one to learn from
  # to not overwhelm memeorys of our own agent with forign memories
  if self.enemy_to_record is None:
    self.enemy_to_record = enemy_name
  if self.enemy_to_record != enemy_name:
    return

  # pretty much the same as for our agent
  if enemy_game_state["step"] == 1:
    self.enemy_visited[enemy_name] = np.zeros((17, 17), dtype=np.int64)

  sn = state_to_features(enemy_game_state, self.enemy_visited[enemy_name])
  _, _, _, (x, y) = enemy_game_state["self"]
  self.enemy_visited[enemy_name][x, y] += 1

  if old_enemy_game_state is None:
    self.enemy_last_state[enemy_name] = sn
    return

  s = self.enemy_last_state[enemy_name]
  self.enemy_last_state[enemy_name] = sn

  reward = reward_from_events(self, enemy_events, x, y, enemy=True)

  if enemy_action is None:
    enemy_action = "WAIT"
  action = ACTIONS[enemy_action]

  self.memory.push(s, action, reward, sn, False)

# dict with rewards for events
game_rewards = {
  e.MOVED_LEFT: .0,
  e.MOVED_RIGHT: .0,
  e.MOVED_UP: .0,
  e.MOVED_DOWN: .0,
  e.WAITED: 0,
  e.INVALID_ACTION: .0,
  e.BOMB_DROPPED: .0,
  e.BOMB_EXPLODED: .0,
  e.CRATE_DESTROYED: 1.,
  e.COIN_FOUND: 0,
  e.COIN_COLLECTED: 5,
  e.KILLED_OPPONENT: 15,
  e.KILLED_SELF: -15,
  e.GOT_KILLED: -15,
  e.OPPONENT_ELIMINATED: 0,
  e.SURVIVED_ROUND: 15,
}

def reward_from_events(self, events: List[str], x, y, enemy=False) -> int:
  reward_sum = 0
  # apply rewards
  for event in events:
    reward_sum += game_rewards[event]
  # punish standing around and looping
  reward_sum -= (self.visited[x, y] - 1) / 4
  if not enemy:
    # update log
    self.creward += reward_sum
  return reward_sum
