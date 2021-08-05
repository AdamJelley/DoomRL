import numpy as np
import gym
import vizdoomgym
import vizdoom as vzd
import cv2
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecTransposeImage
from stable_baselines3.common.monitor import Monitor

class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, env=None, rescale_factor=1):
        super().__init__(env)
        self.rescale_factor = rescale_factor
        self.original_shape = self.observation_space.shape
        self.new_shape = (int(self.original_shape[0]*rescale_factor), int(self.original_shape[1]*rescale_factor), self.original_shape[2])
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=self.new_shape, dtype=np.uint8)

    def observation(self, obs):
        # sourcery skip: inline-immediately-returned-variable
        resized_screen = cv2.resize(obs, (self.new_shape[1], self.new_shape[0]), interpolation=cv2.INTER_AREA)
        new_obs = np.array(resized_screen, dtype=np.uint8).reshape(self.new_shape)
        return new_obs

class RepeatAction(gym.Wrapper):
    def __init__(self, env=None, repeat=4, render_screen=False, record_video=False):
        super().__init__(env)
        self.repeat = repeat
        self.render_screen = render_screen
        self.record_video = record_video
        self.images = []
        self.killcount = 0

    def step(self, action):
        t_reward = 0.0
        done = False
        for _ in range(self.repeat):
            obs, reward, done, info = self.env.step(action)
            t_reward += reward
            # Reward shaping: Add reward of 50 for killing enemies
            if self.env.game.get_game_variable(vzd.GameVariable.KILLCOUNT)>self.killcount:
                t_reward+=50
                self.killcount=self.env.game.get_game_variable(vzd.GameVariable.KILLCOUNT)
            if self.render_screen:
                self.env.render()
            if not done and self.record_video:
                img = self.env.game.get_state().screen_buffer
                self.images.append(img)
            if done:
                break
        return obs, t_reward, done, info

    def reset(self):
        return self.env.reset()

def create_env(scenario='VizdoomBasic-v0', repeat=4, rescale_factor=0.5, render_screen=False, record_video=False):
    env = gym.make(scenario)
    env = RepeatAction(env, repeat, render_screen, record_video)
    env = PreprocessFrame(env, rescale_factor)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    env = VecTransposeImage(env)
    return env