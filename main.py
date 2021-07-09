import numpy as np
from stable_baselines3 import PPO
from preprocessing import create_env
from utils import save_gif

saved_model_name = 'PPO_Basic'
env_name = 'VizdoomBasic-v0'
episodes = 10
render_screen=True
record_video=True
output_gif_name = saved_model_name+'_'+env_name+'_'+str(episodes)

env = create_env(env_name, repeat=4, rescale_factor=0.5, render_screen=render_screen, record_video=record_video)

model = PPO.load('./saved_models/'+str(saved_model_name))

episode_rewards = []

for epsiode in range(1,episodes+1):
    obs = env.reset()
    total_reward = 0
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, scaled_reward, done, info = env.step(action)
        reward = env.get_original_reward()
        total_reward+=reward
    episode_rewards.append(total_reward)
    print(f"Episode: {epsiode}, \t Total reward: {episode_rewards[-1][0]}")
env.close()

if env.get_attr('record_video')[0]:
    save_gif(env, output_gif_name)

print("_____________________________________\n")
print(f"Total Episodes: {episodes}, \t Average Reward: {sum(episode_rewards)[0]/episodes}")