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
        self.episode_rewards = {}  # episode_number -> total_reward
        self.episode_lengths = {}  # episode_number -> length
        self.episode_bullets = {}  # episode_number -> bullets_used
        self.episode_hits = {}    # episode_number -> balloons_popped
        self.current_episode_bullets = 0
        self.current_episode_hits = 0
        
    def _on_step(self):
        if self.n_calls % 1000 == 0:  # Log every 1000 steps
            elapsed_time = int(time.time() - self.start_time)
            stats = self.training_env.get_attr('stats')[0]
            current_episode = stats['current_episode']
            
            # Episode bilgilerini al
            episode_info = self.locals.get('infos')[0]
            if 'episode' in episode_info:
                # Episode bitti, istatistikleri kaydet
                self.episode_rewards[current_episode] = episode_info['episode']['r']
                self.episode_lengths[current_episode] = episode_info['episode']['l']
                self.episode_bullets[current_episode] = stats['bullets_used'] - self.current_episode_bullets
                self.episode_hits[current_episode] = stats['balloons_popped'] - self.current_episode_hits
                
                # Yeni episode için sayaçları güncelle
                self.current_episode_bullets = stats['bullets_used']
                self.current_episode_hits = stats['balloons_popped']
            
            print("\n=== Training Progress ===")
            print(f"Steps: {self.n_calls}")
            print(f"Time: {elapsed_time}s")
            print(f"Current Episode: {current_episode}")
            
            if current_episode > 0:
                # Son biten episode'un istatistikleri
                last_episode = max(self.episode_rewards.keys()) if self.episode_rewards else 0
                if last_episode > 0:
                    print(f"\nLast Completed Episode ({last_episode}):")
                    print(f"  - Length: {self.episode_lengths[last_episode]} steps")
                    print(f"  - Total Reward: {self.episode_rewards[last_episode]:.2f}")
                    bullets = self.episode_bullets[last_episode]
                    hits = self.episode_hits[last_episode]
                    hit_rate = (hits / bullets * 100) if bullets > 0 else 0
                    print(f"  - Bullets Used: {bullets}")
                    print(f"  - Balloons Hit: {hits}")
                    print(f"  - Hit Rate: {hit_rate:.1f}%")
                
                # Mevcut episode'un devam eden istatistikleri
                print(f"\nCurrent Episode Progress:")
                current_bullets = stats['bullets_used'] - self.current_episode_bullets
                current_hits = stats['balloons_popped'] - self.current_episode_hits
                current_hit_rate = (current_hits / current_bullets * 100) if current_bullets > 0 else 0
                print(f"  - Steps: {stats['episode_steps']}")
                print(f"  - Bullets Used: {current_bullets}")
                print(f"  - Balloons Hit: {current_hits}")
                print(f"  - Current Hit Rate: {current_hit_rate:.1f}%")
                print(f"  - Remaining Balloons: {stats['remaining_balloons']}")
            
            print("\nOverall Training Stats:")
            total_bullets = stats['bullets_used']
            total_hits = stats['balloons_popped']
            overall_hit_rate = (total_hits / total_bullets * 100) if total_bullets > 0 else 0
            print(f"  - Total Episodes Completed: {len(self.episode_rewards)}")
            print(f"  - Total Steps: {self.n_calls}")
            print(f"  - Total Bullets Used: {total_bullets}")
            print(f"  - Total Balloons Hit: {total_hits}")
            print(f"  - Overall Hit Rate: {overall_hit_rate:.1f}%")
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