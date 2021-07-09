from stable_baselines3.common.callbacks import BaseCallback

class myTensorBoardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard
    """

    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.total_reward=0
        self.episode_length=0

    def _on_step(self):
        unscaled_reward = self.training_env.get_original_reward()[0]
        self.total_reward+=unscaled_reward
        self.episode_length+=1
        return True

    def _on_rollout_end(self):
        self.average_policy_return = self.total_reward/self.episode_length
        self.logger.record('train/AvgStepReturn', self.average_policy_return)
        self.total_reward=0
        self.episode_length=0