"""
IMPROVED Training Script for Learning Agent
Better scenario generation for higher accuracy
Location: cognify/train_learning_agent_v2.py
"""
import sys
import os
import random
import numpy as np
from datetime import datetime
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backend.agents.learning.learning_agent import LearningLogisticsAgent

logging.basicConfig(level=logging.INFO)


def generate_realistic_scenario_v2(difficulty='mixed'):
    """
    IMPROVED: Generate more realistic training scenarios
    Better correlation between conditions and customer satisfaction
    """
    
    # More realistic distributions
    weather_options = ['clear', 'light_rain', 'heavy_rain', 'snow', 'fog']
    traffic_options = ['light', 'moderate', 'heavy', 'severe']
    
    if difficulty == 'easy':
        weather_probs = [0.8, 0.15, 0.04, 0.01, 0.0]
        traffic_probs = [0.6, 0.3, 0.08, 0.02]
    elif difficulty == 'hard':
        weather_probs = [0.2, 0.2, 0.3, 0.2, 0.1]
        traffic_probs = [0.1, 0.2, 0.4, 0.3]
    else:  # mixed/realistic
        weather_probs = [0.5, 0.25, 0.15, 0.07, 0.03]
        traffic_probs = [0.35, 0.40, 0.18, 0.07]
    
    weather = random.choices(weather_options, weights=weather_probs)[0]
    traffic = random.choices(traffic_options, weights=traffic_probs)[0]
    
    # Distance: realistic distribution (most shipments 100-300km)
    distance = int(np.random.gamma(4, 50))
    distance = max(50, min(500, distance))
    
    # Carrier: most are good, some are mediocre
    carrier_reliability = np.random.beta(10, 1.5)
    carrier_reliability = max(0.80, min(0.99, carrier_reliability))
    
    # Hour: business hours more common
    hour_probs = np.ones(24)
    hour_probs[8:18] *= 3
    hour_probs /= hour_probs.sum()
    hour = np.random.choice(24, p=hour_probs)
    
    # IMPROVED: Calculate delay with clear logic
    delay_hours = calculate_delay_v2(weather, traffic, distance, carrier_reliability, hour)
    
    shipment_data = {
        'weather': weather,
        'traffic': traffic,
        'distance_km': distance,
        'delay_hours': delay_hours,
        'carrier_reliability': carrier_reliability,
        'hour': hour
    }
    
    # IMPROVED: Determine satisfaction with clearer rules
    customer_satisfied = determine_satisfaction_v2(delay_hours, weather, traffic)
    
    return shipment_data, customer_satisfied


def calculate_delay_v2(weather, traffic, distance, carrier, hour):
    """
    IMPROVED: More deterministic delay calculation
    Makes learning easier for the agent
    """
    delay = 0.0
    
    # Weather impact (clear rules)
    weather_delays = {
        'clear': -2,
        'light_rain': 2,
        'fog': 5,
        'heavy_rain': 10,
        'snow': 18
    }
    delay += weather_delays.get(weather, 0)
    
    # Traffic impact
    traffic_delays = {
        'light': -1,
        'moderate': 3,
        'heavy': 10,
        'severe': 25
    }
    delay += traffic_delays.get(traffic, 0)
    
    # Distance impact (longer = more risk)
    if distance > 400:
        delay += random.randint(5, 15)
    elif distance > 300:
        delay += random.randint(2, 8)
    elif distance > 200:
        delay += random.randint(0, 4)
    
    # Carrier impact (IMPORTANT)
    if carrier < 0.85:
        delay += random.randint(15, 30)
    elif carrier < 0.90:
        delay += random.randint(8, 20)
    elif carrier < 0.93:
        delay += random.randint(3, 10)
    elif carrier < 0.95:
        delay += random.randint(0, 5)
    else:
        delay += random.randint(-3, 2)
    
    # Rush hour impact
    if 7 <= hour <= 9 or 17 <= hour <= 19:
        delay += random.randint(3, 8)
    elif hour < 6 or hour > 22:
        delay += random.randint(-2, 0)
    
    # Add small noise
    delay += np.random.normal(0, 2)
    
    return round(delay, 1)


def determine_satisfaction_v2(delay_hours, weather, traffic):
    """
    IMPROVED: Clearer satisfaction rules
    Makes it easier for agent to learn patterns
    """
    
    # Rule 1: Early delivery = almost always happy
    if delay_hours < -5:
        return random.random() > 0.02  # 98% satisfied
    
    # Rule 2: On time or very minor delay
    if delay_hours < 3:
        return random.random() > 0.05  # 95% satisfied
    
    # Rule 3: Minor delay (3-12 hours)
    if delay_hours < 12:
        # Excusable if bad conditions
        if weather in ['heavy_rain', 'snow'] or traffic in ['heavy', 'severe']:
            return random.random() > 0.15  # 85% satisfied
        else:
            return random.random() > 0.30  # 70% satisfied
    
    # Rule 4: Moderate delay (12-24 hours)
    if delay_hours < 24:
        if weather == 'snow' and traffic == 'severe':
            return random.random() > 0.40  # 60% satisfied (extreme conditions)
        elif weather in ['heavy_rain', 'snow'] or traffic == 'severe':
            return random.random() > 0.60  # 40% satisfied
        else:
            return random.random() > 0.80  # 20% satisfied
    
    # Rule 5: Severe delay (24-36 hours)
    if delay_hours < 36:
        if weather == 'snow' and traffic == 'severe':
            return random.random() > 0.70  # 30% satisfied
        else:
            return random.random() > 0.90  # 10% satisfied
    
    # Rule 6: Extreme delay (36+ hours)
    return random.random() > 0.95  # 5% satisfied


def train_agent_improved(episodes=2000, verbose=True):
    """
    Improved training with curriculum learning
    """
    print("="*70)
    print("🎓 IMPROVED TRAINING - LEARNING AGENT V2")
    print("="*70)
    print(f"Episodes: {episodes}")
    print(f"Strategy: Curriculum Learning (Easy → Mixed → Hard)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    agent = LearningLogisticsAgent(use_advanced=True)
    
    print("🧠 Training Strategy:")
    print("   Phase 1 (0-30%): Easy scenarios - build confidence")
    print("   Phase 2 (30-70%): Mixed scenarios - realistic cases")
    print("   Phase 3 (70-100%): Hard scenarios - edge cases")
    print()
    
    # Curriculum learning: start easy, increase difficulty
    for episode in range(episodes):
        # Determine difficulty based on progress
        progress = episode / episodes
        
        if progress < 0.3:
            difficulty = 'easy'
        elif progress < 0.7:
            difficulty = 'mixed'
        else:
            difficulty = 'hard'
        
        # Generate scenario
        shipment_data, customer_satisfied = generate_realistic_scenario_v2(difficulty)
        
        # Agent decision
        action, confidence = agent.should_approve_payment(shipment_data, training=True)
        
        # Learn from outcome
        agent.record_outcome(shipment_data, action, customer_satisfied)
        
        # Progress updates
        if verbose and (episode + 1) % 200 == 0:
            stats = agent.get_stats()
            progress_pct = (episode + 1) / episodes * 100
            
            # Progress bar
            bar_length = 30
            filled = int(bar_length * progress)
            bar = '█' * filled + '░' * (bar_length - filled)
            
            print(f"[{bar}] {episode+1}/{episodes} ({progress_pct:.0f}%)")
            print(f"  Difficulty: {difficulty.upper()}")
            print(f"  📊 Accuracy: {stats['accuracy']:.1f}%")
            print(f"  🎯 Avg Reward: {stats['avg_reward']:.3f}")
            print(f"  🔍 Exploration: {stats['exploration_rate']:.1f}%")
            
            # Show trend
            if stats['performance_trend'] == 'improving':
                print(f"  📈 Trend: IMPROVING ✅")
            elif stats['performance_trend'] == 'declining':
                print(f"  📉 Trend: DECLINING ⚠️")
            print()
    
    # Final results
    print("="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    
    final_stats = agent.get_stats()
    print(f"\n📈 Final Performance:")
    print(f"   Total Decisions: {final_stats['total_decisions']}")
    print(f"   Final Accuracy: {final_stats['accuracy']:.2f}%")
    print(f"   Final Avg Reward: {final_stats['avg_reward']:.3f}")
    print(f"   Exploration Rate: {final_stats['exploration_rate']:.2f}%")
    print(f"   Model: {final_stats['model_type']}")
    print()
    
    # Performance assessment
    acc = final_stats['accuracy']
    if acc >= 85:
        print("🌟 EXCELLENT! Agent achieved expert-level performance!")
        print("   Ready for production deployment!")
    elif acc >= 75:
        print("✅ GOOD! Agent shows strong performance.")
        print("   Consider more training for 85%+ accuracy.")
    elif acc >= 65:
        print("📊 DECENT. Agent is learning but needs improvement.")
        print("   Recommendation: Train for 3000-5000 more episodes.")
    else:
        print("⚠️ NEEDS MORE TRAINING")
        print("   Recommendation: Check reward function and train longer.")
    
    print(f"\n💾 Model saved to: {agent.model_path}")
    print(f"🕐 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    return agent


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Improved Training Script')
    parser.add_argument('--episodes', type=int, default=2000,
                       help='Number of training episodes (default: 2000)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick training with 500 episodes')
    
    args = parser.parse_args()
    
    episodes = 500 if args.quick else args.episodes
    
    try:
        trained_agent = train_agent_improved(episodes=episodes)
        print("\n🎯 Next steps:")
        print("   1. Test: python test_learning.py")
        print("   2. Run agents: python backend/agents/logistics_agent.py")
        print("   3. Try in ASI:One: 'What are learning stats?'")
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()