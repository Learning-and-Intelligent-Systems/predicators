import gymnasium as gym
env = gym.make("MiniGrid-Fetch-8x8-N3-v0", render_mode="human")
observation, info = env.reset(seed=42)
import ipdb; ipdb.set_trace()
for _ in range(1000):
   action = int(input("Action: "))  # User-defined policy function
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()
env.close()

# Need a look at new region operator

