"""
Replay Buffer - Stores agent experiences for learning
"""
import random
from collections import deque, namedtuple
import numpy as np

# Experience tuple: (state, action, reward, next_state, done)
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class ReplayBuffer:
    """
    Stores experiences for experience replay in RL
    Helps agent learn from past experiences
    """
    
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Store an experience"""
        self.buffer.append(Experience(state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample random batch of experiences"""
        experiences = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
        states = np.array([e.state for e in experiences])
        actions = np.array([e.action for e in experiences])
        rewards = np.array([e.reward for e in experiences])
        next_states = np.array([e.next_state for e in experiences])
        dones = np.array([e.done for e in experiences])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)