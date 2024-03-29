import numpy as np
from stable_baselines3 import PPO
from preprocessing import create_env
from utils import save_gif

model_name = 'PPO'
env_name = 'VizdoomCorridor-v0'
episodes = 1
render_screen=True
record_video=True
output_gif_name = model_name+'_'+env_name+'_KillingReward_'+str(episodes)+'eps'

if __name__ == '__main__':

    env = create_env(env_name, repeat=4, rescale_factor=0.5, render_screen=render_screen, record_video=record_video)

    model = PPO.load(f"./saved_models/{env_name}/{model_name}")

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