"""
Improved Learning Agent - Production Version
Uses Advanced DQN with Prioritized Replay
Location: cognify/backend/agents/learning/learning_agent.py
"""
import numpy as np
import os
import logging
import torch


class LearningLogisticsAgent:
    """
    Enhanced Learning wrapper for Logistics Agent
    Now uses advanced DQN with better performance
    """
    
    def __init__(self, state_size=6, action_size=2, use_advanced=True):
        self.state_size = state_size
        self.action_size = action_size
        self.use_advanced = use_advanced
        
        # Initialize DQN agent
        if use_advanced:
            try:
                from .advanced_dqn import AdvancedDQNAgent, PrioritizedReplayBuffer
                self.dqn_agent = AdvancedDQNAgent(state_size, action_size, use_dueling=True)
                self.replay_buffer = PrioritizedReplayBuffer(capacity=20000)
                logging.info("🚀 Using Advanced DQN (Double DQN + Dueling + Prioritized Replay)")
            except ImportError as e:
                logging.warning(f"⚠️ Could not import advanced DQN: {e}")
                logging.info("📦 Falling back to Basic DQN")
                from .q_network import DQNAgent
                from .replay_buffer import ReplayBuffer
                self.dqn_agent = DQNAgent(state_size, action_size)
                self.replay_buffer = ReplayBuffer(capacity=10000)
                self.use_advanced = False
        else:
            # Fallback to basic DQN
            from .q_network import DQNAgent
            from .replay_buffer import ReplayBuffer
            self.dqn_agent = DQNAgent(state_size, action_size)
            self.replay_buffer = ReplayBuffer(capacity=10000)
            logging.info("📦 Using Basic DQN")
        
        # Training parameters
        self.batch_size = 64  # Increased from 32
        self.train_start = 200  # Train after more experiences
        self.train_frequency = 4  # Train every N steps
        
        # Metrics
        self.total_decisions = 0
        self.correct_decisions = 0
        self.total_reward = 0
        self.reward_history = []
        self.accuracy_history = []
        
        # Model paths
        self.model_dir = "backend/agents/learning/models"
        self.model_path = os.path.join(self.model_dir, "logistics_model_v2.pth")
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Load existing model
        if os.path.exists(self.model_path):
            try:
                self.dqn_agent.load(self.model_path)
                logging.info(f"✅ Loaded advanced model from {self.model_path}")
            except Exception as e:
                logging.warning(f"⚠️ Could not load model: {e}")
                logging.info("🆕 Starting with fresh model")
        else:
            logging.info("🆕 Starting with fresh advanced model")
    
    def state_from_shipment(self, shipment_data):
        """Convert shipment data to state vector"""
        weather_map = {'clear': 0, 'light_rain': 1, 'heavy_rain': 2, 'snow': 3, 'fog': 2}
        traffic_map = {'light': 0, 'moderate': 1, 'heavy': 2, 'severe': 3}
        
        # Normalize all features to [0, 1]
        state = np.array([
            weather_map.get(shipment_data.get('weather', 'clear'), 0) / 3.0,
            traffic_map.get(shipment_data.get('traffic', 'light'), 0) / 3.0,
            min(shipment_data.get('distance_km', 100) / 500.0, 1.0),
            min(abs(shipment_data.get('delay_hours', 0)) / 48.0, 1.0),
            shipment_data.get('carrier_reliability', 0.9),
            shipment_data.get('hour', 12) / 24.0
        ], dtype=np.float32)
        
        return state
    
    def should_approve_payment(self, shipment_data, training=True):
        """
        Decide whether to approve payment using learned policy
        
        Returns:
            action (int): 0 = withhold, 1 = approve
            confidence (float): Decision confidence
        """
        state = self.state_from_shipment(shipment_data)
        action = self.dqn_agent.get_action(state, training=training)
        
        # Get Q-values for confidence
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values_tensor = self.dqn_agent.q_network(state_tensor)
            q_values = q_values_tensor.squeeze().numpy()
        
        # Calculate confidence (difference between Q-values)
        q_diff = abs(q_values[1] - q_values[0])
        confidence = min(1.0, q_diff / 2.0)  # Normalize
        confidence = max(0.5, confidence)  # At least 50%
        
        return action, confidence
    
    def record_outcome(self, shipment_data, action_taken, customer_satisfied):
        """
        Record outcome and learn from it
        
        SIMPLIFIED reward function - easier to learn
        """
        state = self.state_from_shipment(shipment_data)
        delay_hours = shipment_data.get('delay_hours', 0)
        
        # SIMPLIFIED REWARD LOGIC - Only 4 clear cases
        
        # Case 1: Customer satisfied + Approved payment = GOOD
        if customer_satisfied and action_taken == 1:
            reward = 1.0
            
        # Case 2: Customer unsatisfied + Withheld payment = GOOD  
        elif not customer_satisfied and action_taken == 0:
            reward = 1.0
            
        # Case 3: Customer satisfied + Withheld payment = BAD (too strict)
        elif customer_satisfied and action_taken == 0:
            reward = -0.5
            
        # Case 4: Customer unsatisfied + Approved payment = BAD (too lenient)
        else:  # not customer_satisfied and action_taken == 1
            reward = -1.0
        
        # Bonus/Penalty based on delay magnitude
        if delay_hours < 0:  # Early delivery
            reward += 0.2
        elif delay_hours > 36:  # Very late
            reward -= 0.3
        
        # Store in replay buffer
        next_state = state  # Episodic task
        done = True
        
        self.replay_buffer.push(state, action_taken, reward, next_state, done)
        
        # Update metrics
        self.total_decisions += 1
        self.total_reward += reward
        self.reward_history.append(reward)
        
        if reward > 0:
            self.correct_decisions += 1
        
        # Calculate accuracy
        current_accuracy = self.get_accuracy()
        self.accuracy_history.append(current_accuracy)
        
        # Train more frequently
        if len(self.replay_buffer) >= self.train_start:
            if self.total_decisions % self.train_frequency == 0:
                loss = self._train()
                if loss is not None:
                    logging.debug(f"Training loss: {loss:.4f}")
        
        # Save periodically
        if self.total_decisions % 50 == 0:
            self.save_model()
            metrics = self.get_stats()
            logging.info(
                f"📊 Step {self.total_decisions}: "
                f"Acc={metrics['accuracy']:.1f}%, "
                f"AvgReward={metrics['avg_reward']:.2f}, "
                f"ε={metrics['exploration_rate']:.1f}%"
            )
    
    def _train(self):
        """Train the neural network"""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        if self.use_advanced:
            # Advanced DQN uses prioritized replay
            loss = self.dqn_agent.learn(self.replay_buffer, self.batch_size)
        else:
            # Basic DQN
            states, actions, rewards, next_states, dones = \
                self.replay_buffer.sample(self.batch_size)
            loss = self.dqn_agent.learn(states, actions, rewards, next_states, dones, self.batch_size)
        
        return loss
    
    def save_model(self):
        """Save model with metadata"""
        self.dqn_agent.save(self.model_path)
        
        # Save metadata
        metadata_path = self.model_path.replace('.pth', '_metadata.json')
        import json
        metadata = {
            'total_decisions': self.total_decisions,
            'accuracy': self.get_accuracy(),
            'avg_reward': self.get_avg_reward(),
            'model_version': 'v2_advanced' if self.use_advanced else 'v1_basic',
            'timestamp': str(np.datetime64('now'))
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def get_accuracy(self):
        """Calculate decision accuracy"""
        if self.total_decisions == 0:
            return 0.0
        return (self.correct_decisions / self.total_decisions) * 100
    
    def get_avg_reward(self):
        """Get average reward"""
        if self.total_decisions == 0:
            return 0.0
        return self.total_reward / self.total_decisions
    
    def get_recent_performance(self, window=100):
        """Get recent performance metrics"""
        if len(self.reward_history) < window:
            window = len(self.reward_history)
        
        if window == 0:
            return {'recent_accuracy': 0, 'recent_avg_reward': 0, 'trend': 'insufficient_data'}
        
        recent_rewards = self.reward_history[-window:]
        recent_accuracy = self.accuracy_history[-window:] if self.accuracy_history else [0]
        
        # Calculate trend
        if len(recent_accuracy) > 10:
            first_half = np.mean(recent_accuracy[:len(recent_accuracy)//2])
            second_half = np.mean(recent_accuracy[len(recent_accuracy)//2:])
            trend = 'improving' if second_half > first_half else 'stable' if abs(second_half - first_half) < 2 else 'declining'
        else:
            trend = 'insufficient_data'
        
        return {
            'recent_accuracy': np.mean([1 if r > 0 else 0 for r in recent_rewards]) * 100,
            'recent_avg_reward': np.mean(recent_rewards),
            'trend': trend
        }
    
    def get_stats(self):
        """Get comprehensive statistics"""
        try:
            dqn_metrics = self.dqn_agent.get_metrics()
        except:
            dqn_metrics = {'epsilon': self.dqn_agent.epsilon, 'steps': 0, 'architecture': 'Basic DQN'}
        
        recent_perf = self.get_recent_performance()
        
        return {
            'total_decisions': self.total_decisions,
            'accuracy': self.get_accuracy(),
            'avg_reward': self.get_avg_reward(),
            'exploration_rate': dqn_metrics.get('epsilon', 0) * 100,
            'training_steps': dqn_metrics.get('steps', 0),
            'recent_accuracy': recent_perf['recent_accuracy'],
            'recent_avg_reward': recent_perf['recent_avg_reward'],
            'performance_trend': recent_perf['trend'],
            'model_type': dqn_metrics.get('architecture', 'Unknown'),
            'buffer_size': len(self.replay_buffer)
        }