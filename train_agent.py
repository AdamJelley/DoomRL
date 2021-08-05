from stable_baselines3 import PPO
from preprocessing import create_env
from callbacks import myTensorBoardCallback
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

def train_agent(model_name, env_name, policy="CnnPolicy", frame_repeat=4, frame_rescale=0.5, train_timesteps=10000, eval_frequency=5000, eval_episodes=5, final_eval_episodes=10):
    env = create_env(env_name, repeat=frame_repeat, rescale_factor=frame_rescale)

    model = PPO(policy, env, verbose=1, tensorboard_log=f"./tensorboard/{env_name}/{model_name}/")

    eval_callback = EvalCallback(env, best_model_save_path=f"./saved_models/{env_name}/checkpoint_models", \
        eval_freq=eval_frequency, n_eval_episodes=eval_episodes, deterministic=True, render=False)

    tb_callback = myTensorBoardCallback(avg_reward=True)

    print('Agent starting training...')
    model.learn(total_timesteps=train_timesteps, callback=[eval_callback, tb_callback])
    model.save(f"./saved_models/{env_name}/{model_name}")
    print('Agent finished training!')

    print('Evaluating trained agent performance:')
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=final_eval_episodes)
    print(f"Mean Reward: {mean_reward}, Std Reward:{std_reward}")

if __name__ == '__main__':

    model_name = 'PPO'
    env_name = 'VizdoomCorridor-v0'
    policy = "CnnPolicy"

    train_agent(model_name, env_name, policy, 
                frame_repeat=4, frame_rescale=0.5, 
                train_timesteps=1000000, eval_frequency=10000, 
                eval_episodes=5, final_eval_episodes=10)