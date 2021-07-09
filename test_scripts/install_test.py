import gym
import vizdoomgym

env = gym.make('VizdoomBasic-v0')

env.reset()
done = False

for _ in range(10000):
    #while not done:
    state, reward, done, info = env.step(env.action_space.sample())
    env.render()
env.close()