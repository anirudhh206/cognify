"""
Advanced DQN Agent - Production-Ready
Includes: Double DQN, Prioritized Experience Replay, Dueling Architecture
Location: cognify/backend/agents/learning/advanced_dqn.py
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque, namedtuple
import random


# ============================================================================
# PRIORITIZED REPLAY BUFFER
# ============================================================================

class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay
    Samples important experiences more frequently
    """
    
    def __init__(self, capacity=10000, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha  # How much prioritization (0=uniform, 1=full)
        self.beta = beta    # Importance sampling correction
        self.beta_increment = 0.001
        self.epsilon = 1e-6  # Small constant to avoid zero priority
        
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.Experience = namedtuple('Experience', 
                                     ['state', 'action', 'reward', 'next_state', 'done'])
    
    def push(self, state, action, reward, next_state, done):
        """Add experience with maximum priority"""
        max_priority = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(self.Experience(state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = self.Experience(state, action, reward, next_state, done)
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """Sample batch with priorities"""
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:len(self.buffer)]
        
        # Calculate sampling probabilities
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        
        # Get experiences
        experiences = [self.buffer[idx] for idx in indices]
        
        # Calculate importance sampling weights
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize
        
        # Increase beta over time
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Convert to arrays
        states = np.array([e.state for e in experiences])
        actions = np.array([e.action for e in experiences])
        rewards = np.array([e.reward for e in experiences])
        next_states = np.array([e.next_state for e in experiences])
        dones = np.array([e.done for e in experiences])
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices, td_errors):
        """Update priorities based on TD errors"""
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + self.epsilon
    
    def __len__(self):
        return len(self.buffer)


# ============================================================================
# DUELING Q-NETWORK
# ============================================================================

class DuelingQNetwork(nn.Module):
    """
    Dueling DQN Architecture
    Separates state value and action advantages
    """
    
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DuelingQNetwork, self).__init__()
        
        # Shared feature extractor
        self.feature = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Value stream (how good is this state?)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Advantage stream (how much better is each action?)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size)
        )
    
    def forward(self, state):
        """
        Combine value and advantage streams
        Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        """
        features = self.feature(state)
        
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine: Q = V + (A - mean(A))
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_values


# ============================================================================
# ADVANCED DQN AGENT
# ============================================================================

class AdvancedDQNAgent:
    """
    State-of-the-art DQN Agent
    Features:
    - Double DQN (reduces overestimation)
    - Dueling architecture (better value estimates)
    - Prioritized replay (learn from important experiences)
    - Gradient clipping (stable training)
    """
    
    def __init__(self, state_size, action_size, learning_rate=0.0003, use_dueling=True):
        self.state_size = state_size
        self.action_size = action_size
        self.use_dueling = use_dueling
        
        # Networks
        if use_dueling:
            self.q_network = DuelingQNetwork(state_size, action_size)
            self.target_network = DuelingQNetwork(state_size, action_size)
        else:
            self.q_network = nn.Sequential(
                nn.Linear(state_size, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, action_size)
            )
            self.target_network = nn.Sequential(
                nn.Linear(state_size, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, action_size)
            )
        
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Target network in eval mode
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.criterion = nn.SmoothL1Loss()  # Huber loss (more robust than MSE)
        
        # Hyperparameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        self.update_target_every = 100  # Soft update
        self.tau = 0.005  # Soft update parameter
        self.steps = 0
        
        # Metrics
        self.train_loss_history = []
        self.epsilon_history = []
    
    def get_action(self, state, training=True):
        """Epsilon-greedy action selection"""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def learn(self, replay_buffer, batch_size=64):
        """
        Double DQN learning with prioritized replay
        """
        if len(replay_buffer) < batch_size:
            return None
        
        # Sample from prioritized replay
        states, actions, rewards, next_states, dones, indices, weights = \
            replay_buffer.sample(batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        weights = torch.FloatTensor(weights)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Double DQN: Use online network to select action, target network to evaluate
        with torch.no_grad():
            # Online network selects best action
            next_actions = self.q_network(next_states).argmax(1)
            # Target network evaluates that action
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Calculate TD errors for prioritized replay
        td_errors = (target_q_values - current_q_values.squeeze()).detach().numpy()
        replay_buffer.update_priorities(indices, td_errors)
        
        # Weighted loss (importance sampling)
        loss = (weights * self.criterion(current_q_values.squeeze(), target_q_values)).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)  # Gradient clipping
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Soft update target network
        self.steps += 1
        if self.steps % self.update_target_every == 0:
            self._soft_update_target_network()
        
        # Track metrics
        self.train_loss_history.append(loss.item())
        self.epsilon_history.append(self.epsilon)
        
        return loss.item()
    
    def _soft_update_target_network(self):
        """Soft update: θ_target = τ*θ_local + (1-τ)*θ_target"""
        for target_param, local_param in zip(self.target_network.parameters(), 
                                             self.q_network.parameters()):
            target_param.data.copy_(self.tau * local_param.data + 
                                   (1.0 - self.tau) * target_param.data)
    
    def save(self, filepath):
        """Save model"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'train_loss_history': self.train_loss_history,
            'epsilon_history': self.epsilon_history
        }, filepath)
    
    def load(self, filepath):
        """Load model"""
        checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        self.train_loss_history = checkpoint.get('train_loss_history', [])
        self.epsilon_history = checkpoint.get('epsilon_history', [])
    
    def get_metrics(self):
        """Get training metrics"""
        return {
            'epsilon': self.epsilon,
            'steps': self.steps,
            'avg_loss': np.mean(self.train_loss_history[-100:]) if self.train_loss_history else 0,
            'architecture': 'Dueling DQN' if self.use_dueling else 'Standard DQN'
        }