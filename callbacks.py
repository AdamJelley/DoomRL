from stable_baselines3.common.callbacks import BaseCallback

class myTensorBoardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard
    """

    def __init__(self, avg_reward=True, verbose=1):
        super().__init__(verbose)
        self.avg_reward = avg_reward
        self.total_reward=0
        self.episode_length=0

    def _on_step(self):
        if self.avg_reward:
            unscaled_reward = self.training_env.get_original_reward()[0]
            self.total_reward+=unscaled_reward
            self.episode_length+=1
        return True

    def _on_rollout_end(self):
        if self.avg_reward:
            self.average_policy_return = self.total_reward/self.episode_length
            self.logger.record('train/AvgReward', self.average_policy_return)
        self.total_reward=0
        self.episode_length=0