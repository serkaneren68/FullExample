from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from src.environments.balloon_env import BalloonShooterEnv
import time
from stable_baselines3.common.callbacks import BaseCallback
import torch

class TrainingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.start_time = time.time()
        self.best_reward = -float('inf')
        self.episode_count = 0
        self.last_bullets_used = 0
        self.last_balloons_popped = 0
        
    def _on_step(self):
        if self.n_calls % 1000 == 0:  # Log every 1000 steps
            elapsed_time = int(time.time() - self.start_time)
            stats = self.training_env.get_attr('stats')[0]
            
            # Calculate hit rate for the current interval
            bullets_used_interval = stats['bullets_used'] - self.last_bullets_used
            balloons_popped_interval = stats['balloons_popped'] - self.last_balloons_popped
            hit_rate = (balloons_popped_interval / bullets_used_interval * 100) if bullets_used_interval > 0 else 0
            
            # Update last values
            self.last_bullets_used = stats['bullets_used']
            self.last_balloons_popped = stats['balloons_popped']
            
            # Get episode info if available
            episode_info = self.locals.get('infos')[0].get('episode')
            if episode_info:
                episode_reward = episode_info['r']
                episode_length = episode_info['l']
                if episode_reward > self.best_reward:
                    self.best_reward = episode_reward
            
            print("\n=== Training Progress ===")
            print(f"Steps: {self.n_calls}")
            print(f"Episode: {stats['current_episode']}")
            print(f"Time: {elapsed_time}s")
            print(f"Current Score/Reward: {stats['score']:.2f}")
            print(f"Best Reward: {self.best_reward:.2f}")
            print(f"Recent Performance:")
            print(f"  - Balloons Popped: +{balloons_popped_interval}")
            print(f"  - Bullets Used: +{bullets_used_interval}")
            print(f"  - Hit Rate: {hit_rate:.1f}%")
            print(f"Total Stats:")
            print(f"  - Total Balloons Popped: {stats['balloons_popped']}")
            print(f"  - Total Bullets Used: {stats['bullets_used']}")
            print(f"  - Overall Hit Rate: {(stats['balloons_popped'] / stats['bullets_used'] * 100) if stats['bullets_used'] > 0 else 0:.1f}%")
            print(f"  - Remaining Balloons: {stats['remaining_balloons']}")
            print("=======================\n")
        return True

print("Starting Balloon Shooter Training...")
print("===================================")

# Create and wrap the environment
env = BalloonShooterEnv()
print("Environment created successfully")

# Vectorize the environment (required for stable-baselines3)
env = DummyVecEnv([lambda: env])
print("Environment vectorized")

# Create the model with CnnPolicy for image observations
model = PPO(
    "CnnPolicy",  # Use CNN policy for image observations
    env,
    verbose=0,  # Reduce verbosity to avoid cluttering the output
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    device="cuda" if torch.cuda.is_available() else "cpu",  # Use GPU if available
    policy_kwargs={"normalize_images": False}  # Don't normalize images as we already do it
)
print("Model created successfully")

# Create callback
callback = TrainingCallback()

print("\nStarting training...")
print("Progress will be logged every 1000 steps.")
print("===================================")

# Train the model
model.learn(total_timesteps=100000, callback=callback)

print("\nTraining completed!")
print("===================================")

# Save the model
model_path = "balloon_shooter_model"
model.save(model_path)
print(f"Model saved to {model_path}")
print("===================================\n") 