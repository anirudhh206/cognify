"""
Train the learning agent on historical/simulated data
Run this from project root: python train_learning_agent.py
"""
import sys
import os
import random
from datetime import datetime

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backend.agents.learning.learning_agent import LearningLogisticsAgent


def generate_training_episode():
    """
    Generate a simulated shipment scenario for training
    
    Returns:
        shipment_data: Dictionary with shipment features
        customer_satisfied: Boolean indicating if customer was happy
    """
    # Randomize environmental factors
    weather_options = ['clear', 'light_rain', 'heavy_rain', 'snow', 'fog']
    traffic_options = ['light', 'moderate', 'heavy', 'severe']
    
    weather = random.choice(weather_options)
    traffic = random.choice(traffic_options)
    distance = random.randint(50, 500)
    delay_hours = random.uniform(-10, 50)  # Negative = early delivery
    carrier_reliability = random.uniform(0.80, 0.98)
    hour = random.randint(0, 23)
    
    shipment_data = {
        'weather': weather,
        'traffic': traffic,
        'distance_km': distance,
        'delay_hours': delay_hours,
        'carrier_reliability': carrier_reliability,
        'hour': hour
    }
    
    # Determine ground truth: Would customer be satisfied?
    # More realistic rules:
    
    # 1. Early or on-time (< 0 delay) = always satisfied
    if delay_hours < 0:
        customer_satisfied = True
    
    # 2. Minor delay (< 12 hours) = usually satisfied
    elif delay_hours < 12:
        customer_satisfied = random.random() > 0.1  # 90% satisfied
    
    # 3. Moderate delay (12-24 hours) with good excuse = often satisfied
    elif delay_hours < 24:
        has_excuse = (weather in ['heavy_rain', 'snow'] or traffic in ['heavy', 'severe'])
        if has_excuse:
            customer_satisfied = random.random() > 0.3  # 70% satisfied
        else:
            customer_satisfied = random.random() > 0.6  # 40% satisfied
    
    # 4. Severe delay (24-36 hours) with extreme conditions = maybe satisfied
    elif delay_hours < 36:
        has_extreme_excuse = (weather == 'snow' and traffic == 'severe')
        if has_extreme_excuse:
            customer_satisfied = random.random() > 0.5  # 50% satisfied
        else:
            customer_satisfied = random.random() > 0.8  # 20% satisfied
    
    # 5. Very severe delay (> 36 hours) = almost never satisfied
    else:
        customer_satisfied = random.random() > 0.9  # 10% satisfied
    
    return shipment_data, customer_satisfied


def train_agent(episodes=1000, verbose=True):
    """
    Train agent on simulated episodes
    
    Args:
        episodes: Number of training episodes
        verbose: Whether to print progress
    """
    print("="*70)
    print("🎓 TRAINING LEARNING AGENT")
    print("="*70)
    print(f"Episodes: {episodes}")
    print(f"Starting training at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Initialize learning agent
    agent = LearningLogisticsAgent()
    
    print(f"🧠 Neural Network Architecture:")
    print(f"   Input: 6 features (weather, traffic, distance, delay, carrier, time)")
    print(f"   Hidden: 2 layers × 64 neurons")
    print(f"   Output: 2 actions (approve/withhold)")
    print()
    
    print(f"📚 Training Parameters:")
    print(f"   Learning Rate: 0.001")
    print(f"   Discount Factor (γ): 0.99")
    print(f"   Initial Exploration (ε): 1.0 (100%)")
    print(f"   Batch Size: 32")
    print()
    
    print("Starting training...\n")
    
    # Training loop
    for episode in range(episodes):
        # Generate scenario
        shipment_data, customer_satisfied = generate_training_episode()
        
        # Agent makes decision (exploring or exploiting)
        action, confidence = agent.should_approve_payment(shipment_data, training=True)
        
        # Record outcome - THIS IS WHERE LEARNING HAPPENS!
        agent.record_outcome(shipment_data, action, customer_satisfied)
        
        # Print progress every 100 episodes
        if verbose and (episode + 1) % 100 == 0:
            stats = agent.get_stats()
            
            # Create progress bar
            progress = (episode + 1) / episodes
            bar_length = 30
            filled = int(bar_length * progress)
            bar = '█' * filled + '░' * (bar_length - filled)
            
            print(f"[{bar}] {episode+1}/{episodes} ({progress*100:.0f}%)")
            print(f"  📊 Accuracy: {stats['accuracy']:.1f}%")
            print(f"  🎯 Avg Reward: {stats['avg_reward']:.2f}")
            print(f"  🔍 Exploration: {stats['exploration_rate']:.1f}%")
            print()
    
    # Training complete
    print("="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    
    final_stats = agent.get_stats()
    print(f"\n📈 Final Performance Metrics:")
    print(f"   Total Decisions: {final_stats['total_decisions']}")
    print(f"   Final Accuracy: {final_stats['accuracy']:.2f}%")
    print(f"   Final Avg Reward: {final_stats['avg_reward']:.3f}")
    print(f"   Exploration Rate: {final_stats['exploration_rate']:.2f}%")
    print()
    
    print(f"💾 Model saved to: {agent.model_path}")
    print(f"🕐 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Improvement analysis
    if final_stats['accuracy'] > 85:
        print("🌟 EXCELLENT! Agent achieved >85% accuracy!")
    elif final_stats['accuracy'] > 75:
        print("✅ GOOD! Agent achieved >75% accuracy!")
    else:
        print("⚠️ Agent needs more training. Consider increasing episodes.")
    
    print("\n💡 Next Steps:")
    print("   1. Test the agent: python test_learning.py")
    print("   2. Run the agents: python backend/agents/logistics_agent.py")
    print("   3. Try the demo: python client_demo.py")
    print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Learning Logistics Agent')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of training episodes (default: 1000)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick training with 100 episodes')
    
    args = parser.parse_args()
    
    episodes = 100 if args.quick else args.episodes
    
    try:
        train_agent(episodes=episodes)
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()