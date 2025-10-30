"""
Learning module for intelligent agents
"""
from .learning_agent import LearningLogisticsAgent
from .q_network import DQNAgent, QNetwork
from .replay_buffer import ReplayBuffer

# Import advanced components if available
try:
    from .advanced_dqn import AdvancedDQNAgent, PrioritizedReplayBuffer, DuelingQNetwork
    from .improved_delay_predictor import ImprovedDelayPredictor
    __all__ = [
        'LearningLogisticsAgent', 
        'DQNAgent', 
        'QNetwork', 
        'ReplayBuffer',
        'AdvancedDQNAgent',
        'PrioritizedReplayBuffer',
        'DuelingQNetwork',
        'ImprovedDelayPredictor'
    ]
except ImportError:
    __all__ = ['LearningLogisticsAgent', 'DQNAgent', 'QNetwork', 'ReplayBuffer']