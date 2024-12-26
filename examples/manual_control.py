import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environments.balloon_env import BalloonShooterEnv
import pybullet as p

def main():
    # Create and initialize environment
    env = BalloonShooterEnv()
    env.reset()
    env.add_random_balloons(10)  # Add 10 balloons
    
    # Initial action
    action = [0, 0, 0]
    
    print("\nBalloon Shooter Manual Control")
    print("=============================")
    print("Controls:")
    print("↑/↓: Vertical aim")
    print("←/→: Horizontal aim")
    print("SPACE: Fire")
    print("Q: Exit")
    print("=============================\n")
    
    try:
        while True:
            keys = p.getKeyboardEvents()
            
            # Check for Q to exit
            if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED:
                break
                
            # Handle keyboard input
            if p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN:
                action[1] += 0.1
            if p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN:
                action[1] -= 0.1
            if p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN:
                action[0] += 0.1
            if p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN:
                action[0] -= 0.1
            if p.B3G_SPACE in keys and keys[p.B3G_SPACE] & p.KEY_WAS_TRIGGERED:
                action[2] = 1
            else:
                action[2] = 0
                
            # Update simulation
            _, reward, done, info = env.step(action)
            env.render()
            
            # Print stats
            if action[2] == 1:  # Only print stats when firing
                print(f"\rScore: {env.stats['score']} | Balloons: {env.stats['remaining_balloons']} | Bullets: {env.stats['bullets_used']}", end='')
                
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        p.disconnect()

if __name__ == "__main__":
    main() 