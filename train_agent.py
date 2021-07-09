from stable_baselines3 import PPO
from preprocessing import create_env
from callbacks import myTensorBoardCallback
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

model_name = 'PPO_Basic'
env_name = 'VizdoomBasic-v0'

env = create_env(env_name)

model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=f"./tensorboard/{env_name}/{model_name}/")

eval_callback = EvalCallback(env, best_model_save_path=f"./saved_models/{env_name}/{model_name}", \
    eval_freq=10000, n_eval_episodes=5, deterministic=True, render=False)

myTensorBoardCallback = myTensorBoardCallback(avg_reward=True)

model.learn(total_timesteps=50000, callback=[eval_callback, myTensorBoardCallback])
model.save(f"./saved_models/{env_name}/{model_name}")

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(f"Mean Reward: {mean_reward}, Std Reward:{std_reward}")