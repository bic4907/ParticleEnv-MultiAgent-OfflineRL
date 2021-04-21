import time

from recoder import VideoRecorder
from logger import Logger
from replay_buffer import ReplayBuffer
from utils.train import set_seed_everywhere
from utils.environment import get_agent_types

from env.make_env import make_env
from env.wrapper import NormalizedEnv
from model.utils.model import *

from utils.agent import find_index

import hydra
from omegaconf import DictConfig


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'Workspace: {self.work_dir}')

        self.cfg = cfg

        self.logger = Logger(self.work_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency,
                             agent=cfg.agent.name)

        set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.discrete_action = cfg.discrete_action_space

        self.env = NormalizedEnv(make_env(cfg.env, discrete_action=self.discrete_action))
        self.env.reset()

        self.env_agent_types = get_agent_types(self.env)
        self.agent_indexes = find_index(self.env_agent_types, 'ally')
        self.adversary_indexes = find_index(self.env_agent_types, 'adversary')

        if self.discrete_action:
            cfg.agent.params.obs_dim = self.env.observation_space[0].shape[0]
            cfg.agent.params.action_dim = self.env.action_space[0].n
            cfg.agent.params.action_range = list(range(cfg.agent.params.action_dim))
        else:
            cfg.agent.params.obs_dim = self.env.observation_space[0].shape[0]
            cfg.agent.params.action_dim = self.env.action_space[0].shape[0]
            cfg.agent.params.action_range = [-1, 1]

        cfg.agent.params.agent_index = self.agent_indexes
        cfg.agent.params.critic.input_dim = cfg.agent.params.obs_dim + cfg.agent.params.action_dim

        self.agent = hydra.utils.instantiate(cfg.agent)

        self.common_reward = cfg.common_reward
        obs_shape = [len(self.env_agent_types), cfg.agent.params.obs_dim]
        action_shape = [len(self.env_agent_types), cfg.agent.params.action_dim if not self.discrete_action else 1]
        reward_shape = [len(self.env_agent_types), 1]
        dones_shape = [len(self.env_agent_types), 1]
        self.replay_buffer = ReplayBuffer(obs_shape=obs_shape,
                                          action_shape=action_shape,
                                          reward_shape=reward_shape,
                                          dones_shape=dones_shape,
                                          capacity=int(cfg.replay_buffer_capacity),
                                          device=self.device)

        self.target_workspace = cfg.target_workspace
        self.replay_buffer.load(self.target_workspace)

        self.video_recorder = VideoRecorder(self.work_dir if cfg.save_video else None)
        self.step = 0

    def evaluate(self):
        average_episode_reward = 0

        self.video_recorder.init(enabled=True)
        for episode in range(self.cfg.num_eval_episodes):
            obs = self.env.reset()
            episode_step = 0

            done = False
            episode_reward = 0
            while not done:
                action = self.agent.act(obs, sample=False)
                obs, rewards, dones, _ = self.env.step(action)

                done = True in dones
                if episode_step == self.env.episode_length:
                    done = True

                self.video_recorder.record(self.env)
                episode_reward += sum(rewards)[0]

                episode_step += 1

            average_episode_reward += episode_reward
        self.video_recorder.save(f'{self.step}.mp4')

        average_episode_reward /= self.cfg.num_eval_episodes
        self.logger.log('eval/episode_reward', average_episode_reward, self.step)
        self.logger.dump(self.step)

    def run(self):

        episode, episode_reward, done = 0, 0, True
        start_time = time.time()
        while self.step < self.cfg.num_train_steps + 1:
            if done or self.step % self.cfg.eval_frequency == 0:

                if self.step > 0:
                    self.logger.log('train/duration', time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(self.step, save=True)

                if self.step > 0 and self.step % self.cfg.eval_frequency == 0:
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()
                    start_time = time.time()

                self.logger.log('train/episode_reward', episode_reward, self.step)

                episode_reward = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)

            self.agent.update(self.replay_buffer, self.logger, self.step)

            episode_step += 1
            self.step += 1


@hydra.main(config_path='config', config_name='offline_rl')
def main(cfg: DictConfig) -> None:
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
