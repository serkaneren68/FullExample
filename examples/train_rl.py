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
        self.episode_rewards = []    # Her episode'un toplam reward'ı
        self.episode_hit_rates = []  # Her episode'un hit rate'i
        self.last_bullets = 0
        self.last_hits = 0
        
    def _on_step(self):
        if self.n_calls % 1000 == 0:  # Log every 1000 steps
            elapsed_time = int(time.time() - self.start_time)
            stats = self.training_env.get_attr('stats')[0]
            
            # Episode bilgilerini al
            episode_info = self.locals.get('infos')[0]
            if 'episode' in episode_info:  # Episode bitti
                # Episode reward'ını kaydet
                self.episode_rewards.append(episode_info['episode']['r'])
                
                # Episode'un hit rate'ini hesapla ve kaydet
                bullets_used = stats['bullets_used'] - self.last_bullets
                hits = stats['balloons_popped'] - self.last_hits
                hit_rate = (hits / bullets_used * 100) if bullets_used > 0 else 0
                self.episode_hit_rates.append(hit_rate)
                
                # Sayaçları güncelle
                self.last_bullets = stats['bullets_used']
                self.last_hits = stats['balloons_popped']
            
            print("\n=== Training Progress ===")
            print(f"Steps: {self.n_calls}")
            print(f"Time: {elapsed_time}s")
            print(f"Episodes Completed: {len(self.episode_rewards)}")
            
            if len(self.episode_rewards) > 0:
                # Son episode'un performansı
                print(f"\nLast Episode Performance:")
                print(f"  - Reward: {self.episode_rewards[-1]:.2f}")
                print(f"  - Hit Rate: {self.episode_hit_rates[-1]:.1f}%")
                
                # Son 5 episode'un ortalaması
                last_n = min(5, len(self.episode_rewards))
                avg_reward = sum(self.episode_rewards[-last_n:]) / last_n
                avg_hit_rate = sum(self.episode_hit_rates[-last_n:]) / last_n
                print(f"\nLast {last_n} Episodes Average:")
                print(f"  - Reward: {avg_reward:.2f}")
                print(f"  - Hit Rate: {avg_hit_rate:.1f}%")
                
                # En iyi performans
                best_reward = max(self.episode_rewards)
                best_hit_rate = max(self.episode_hit_rates)
                print(f"\nBest Performance:")
                print(f"  - Best Reward: {best_reward:.2f}")
                print(f"  - Best Hit Rate: {best_hit_rate:.1f}%")
            
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