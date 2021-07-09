from stable_baselines3 import PPO
from preprocessing import create_env
from callbacks import myTensorBoardCallback
from stable_baselines3.common.evaluation import evaluate_policy

saved_model_name = 'PPO_Basic'
env_name = 'VizdoomBasic-v0'

env = create_env(env_name)

model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=f"./tensorboard/{env_name}/{saved_model_name}/")

model.learn(total_timesteps=50000, callback=myTensorBoardCallback())
model.save('./saved_models/'+str(saved_model_name))

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(f"Mean Reward: {mean_reward}, Std Reward:{std_reward}")