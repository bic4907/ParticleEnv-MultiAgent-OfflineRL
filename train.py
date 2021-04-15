import os
import time
import torch
import random

from recoder import VideoRecorder
from logger import Logger
from replay_buffer import ReplayBuffer
import utils

from utils.make_env import make_env

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

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.env = make_env()
        self.env.reset()

        cfg.agent.params.obs_dim = self.env.observation_space.shape[0]
        cfg.agent.params.action_dim = self.env.action_space.shape[0]
        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]

        self.agent = hydra.utils.instantiate(cfg.agent)
        self.replay_buffer = ReplayBuffer([6],
                                          [2],
                                          int(cfg.replay_buffer_capacity),
                                          self.device)

        self.video_recorder = VideoRecorder(self.work_dir if cfg.save_video else None)
        self.step = 0

    def evaluate(self):
        average_episode_reward = 0
        for episode in range(self.cfg.num_eval_episodes):
            obs = self.env.reset()

            self.video_recorder.init(enabled=(episode == 0))
            done = False
            episode_reward = 0
            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done, _ = self.env.step(action)
                self.video_recorder.record(self.env)
                episode_reward += reward

            average_episode_reward += episode_reward
            self.video_recorder.save(f'{self.step}.mp4')

        average_episode_reward /= self.cfg.num_eval_episodes
        self.logger.log('eval/episode_reward', average_episode_reward, self.step)
        self.logger.dump(self.step)

    def run(self):

        try:
            episode, episode_reward, done = 0, 0, True
            start_time = time.time()
            while self.step < self.cfg.num_train_steps:
                if done or self.step % self.cfg.eval_frequency == 0:
                    if self.step > 0:
                        self.logger.log('train/duration', time.time() - start_time, self.step)
                        start_time = time.time()
                        self.logger.dump(
                            self.step, save=(self.step > self.cfg.num_seed_steps))

                    # evaluate agent periodically
                    if self.step > 0 and self.step % self.cfg.eval_frequency == 0 and not isinstance(self.agent, ManualDriver):
                        self.logger.log('eval/episode', episode, self.step)
                        self.evaluate()
                        start_time = time.time()

                    self.logger.log('train/episode_reward', episode_reward,
                                    self.step)

                    obs = self.env.reset()

                    done = False
                    episode_reward = 0
                    episode_step = 0
                    episode += 1

                    self.logger.log('train/episode', episode, self.step)

                # sample action for data collection

                    if self.step < self.cfg.num_seed_steps:
                        action = [random.normalvariate(0, 0.2), random.uniform(0, 1)]
                    else:
                        with utils.eval_mode(self.agent):
                            action = self.agent.act(obs, sample=True)


                # run training update
                if self.step >= self.cfg.num_seed_steps:
                    self.agent.update(self.replay_buffer, self.logger, self.step)

                # image = self.env.render(mode='human', height=608, width=800)

                next_obs, reward, done, _ = self.env.step(action)



                # allow infinite bootstrap
                done = float(done)
                done_no_max = 0 if episode_step + 1 == self.env.episode_length else done
                episode_reward += reward

                self.replay_buffer.add(obs, action, reward, next_obs, done,
                                       done_no_max)

                obs = next_obs
                episode_step += 1
                self.step += 1

        except KeyboardInterrupt:
            self.env.stop()


@hydra.main(config_path='config', config_name='train')
def main(cfg: DictConfig) -> None:
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == '__main__':
    main()




from envs.make_env import make_env
import argparse, datetime
from tensorboardX import SummaryWriter
import numpy as np

from agents import Predators, Preyer
from normalized_env import NormalizedEnv
from utils import split_obs, merge_action
import time

def main(args):

    env = make_env('simple_tag')
    env = NormalizedEnv(env)

    kwargs = dict()
    kwargs['config'] = args

    predator_model = Predators(16, 2, num_agent=3, **kwargs)
    preyer_model = Preyer(14, 2, **kwargs)
    if args.tensorboard:
        writer = SummaryWriter(log_dir='runs/'+args.log_dir)
    episode = 0
    total_step = 0

    while episode < args.max_episodes:

        state = env.reset()
        episode += 1
        step = 0
        predator_accum_reward = []
        preyer_accum_reward = 0

        while True:
            state_predator, state_prayer = split_obs(state)

            predator_model.prep_eval()
            action_predator = predator_model.choose_action(state_predator)
            action_prayer = preyer_model.random_action()
                #action_prayer = preyer_model.choose_action(state_prayer)

            action = merge_action(action_predator, action_prayer)

            next_state, reward, done, info = env.step(action)
            step += 1
            total_step += 1

            predator_accum_reward.append(np.mean(reward[:3]))
            preyer_accum_reward = reward[3]

            if step > args.episode_length:
                done = [True, True, True, True]

            if args.render and (episode % 10 == 1):
                env.render(mode='rgb_array')

            predator_model.memory(state[:3], action[:3], reward[:3], next_state[:3], done[:3])
            # preyer_model.memory(state[3], action[3], reward[3], next_state[3], done[3])

            if len(predator_model.replay_buffer) >= args.batch_size and total_step % args.steps_per_update == 0:
                predator_model.prep_train()
                predator_model.train()
                # preyer_model.train()

            if True in done:
                predator_c_loss, predator_a_loss = predator_model.getLoss()
                preyer_c_loss, preyer_a_loss = preyer_model.getLoss()
                print("[Episode %05d] reward_predator %3.1f reward_preyer %3.1f predator_c_loss %3.1f predator_a_loss %3.1f preyer_c_loss %3.1f preyer_a_loss %3.1f" % \
                      (episode, np.mean(predator_accum_reward).item(), preyer_accum_reward, predator_c_loss, predator_a_loss, preyer_c_loss, preyer_a_loss))
                if args.tensorboard:
                    # writer.add_scalar(tag='debug/memory_length', global_step=episode, scalar_value=len(predator_model.replay_buffer))
                    # writer.add_scalar(tag='debug/predator_epsilon', global_step=episode, scalar_value=predator_model.epsilon)
                    # writer.add_scalar(tag='debug/preyer_epsilon', global_step=episode, scalar_value=preyer_model.epsilon)
                    writer.add_scalar(tag='agent/reward_predator', global_step=episode, scalar_value=np.mean(predator_accum_reward).item())
                    # writer.add_scalar(tag='perf/reward_preyer', global_step=episode, scalar_value=preyer_accum_reward)
                    if predator_c_loss and predator_a_loss:
                        writer.add_scalars('agent/predator_loss', global_step=episode, tag_scalar_dict={'actor':-predator_a_loss, 'critic':predator_c_loss})
                    # writer.add_scalar(tag='loss/preyer_c_loss', global_step=episode, scalar_value=preyer_c_loss)
                    # writer.add_scalar(tag='loss/preyer_a_loss', global_step=episode, scalar_value=preyer_a_loss)

                predator_model.reset()
                preyer_model.reset()
                break

            state = next_state
    if args.tensorboard:
        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_episodes', default=25000, type=int)
    parser.add_argument('--episode_length', default=25, type=int)
    parser.add_argument('--memory_length', default=int(1e6), type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument('--tau', default=0.01, type=float)
    parser.add_argument('--gamma', default=0.95, type=float)
    parser.add_argument('--use_cuda', default=True, type=bool)
    parser.add_argument('--a_lr', default=0.01, type=float)
    parser.add_argument('--c_lr', default=0.01, type=float)
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--render', default=False, type=bool)
    parser.add_argument('--ou_theta', default=0.15, type=float)
    parser.add_argument('--ou_mu', default=0.0, type=float)
    parser.add_argument('--ou_sigma', default=0.2, type=float)
    parser.add_argument('--epsilon_decay', default=600000, type=int)
    parser.add_argument('--reward_coef', default=1, type=float)
    parser.add_argument('--tensorboard', default=False, type=bool)
    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument('--log_dir', default=datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    args = parser.parse_args()
    main(args)