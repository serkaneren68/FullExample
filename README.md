# Balloon Shooter Environment

A PyBullet-based 3D shooting environment where an agent controls a mounted gun to shoot moving balloons.

## Features
- Real-time 3D physics simulation using PyBullet
- Moving balloons with randomized trajectories
- Score tracking and statistics
- Manual control support using keyboard inputs

## Installation 

2. Install dependencies: 

## Usage

### Manual Control

Run the manual control example: 

Controls:
- Arrow keys: Aim the gun
- Spacebar: Fire
- ESC: Exit

### Using as a Gym Environment 

## Environment Details

### Action Space
- 3-dimensional continuous action space:
  - [0]: Horizontal aim (-1 to 1)
  - [1]: Vertical aim (-1 to 1)
  - [2]: Fire trigger (0 or 1)

### Observation Space
- 5-dimensional continuous observation space containing:
  - Current gun position
  - Nearest balloon position
  - Relative velocity

### Reward Structure
- +10 points for each balloon hit
- Time penalty to encourage quick completion 