import torch
import numpy as np

from utils.misc import soft_update

from model.DDPGAgent import DDPGAgent


class DDPG(object):

    def __init__(self, name, params):

        self.name = name
        self.lr = params.lr
        self.gamma = params.gamma
        self.tau = params.tau

        self.obs_dim = params.obs_dim
        self.action_dim = params.action_dim
        self.batch_size = params.batch_size
        self.device = params.device

        self.agent_index = params.agent_index
        self.num_agents = len(self.agent_index)

        self.mse_loss = torch.nn.MSELoss()

        # Reshape critic input shape for shared observation
        params.critic.obs_dim = (self.obs_dim + self.action_dim)

        self.agents = [DDPGAgent(params) for _ in range(self.num_agents)]
        [agent.to(self.device) for agent in self.agents]

    def scale_noise(self, scale):
        for a in self.agents:
            a.scale_noise(scale)

    def reset_noise(self):
        for a in self.agents:
            a.reset_noise()

    def act(self, observations, sample=False):
        observations = torch.Tensor(observations).to(self.device)

        actions = []
        for agent, obs in zip(self.agents, observations):
            agent.eval()
            actions.append(agent.act(obs, explore=sample).squeeze())
            agent.train()
        return np.array(actions)

    def update(self, replay_buffer, logger, step):

        sample = replay_buffer.sample(self.batch_size, nth=self.agent_index)
        obses, actions, rewards, next_obses, dones = sample

        for agent_i, agent in enumerate(self.agents):

            ''' Update value '''
            agent.critic_optimizer.zero_grad()

            with torch.no_grad():
                target_actions = agent.policy(next_obses[:, agent_i])

                target_critic_in = torch.cat((next_obses[:, agent_i], target_actions), dim=1)

                target_next_q = rewards[:, agent_i] + dones[:, agent_i] * self.gamma * agent.target_critic(target_critic_in)

                print(agent.target_critic(target_next_q).shape)

            critic_in = torch.cat((obses[:, agent_i], action[:, agent_i]), dim=1).view(self.batch_size, -1)
            critic_value = agent.critic(critic_in)

            critic_loss = self.mse_loss(critic_value, target_critic_value)
            # TODO log critic loss

            critic_loss.backward()

            torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
            agent.critic_optimizer.step()

            ''' Update policy '''
            agent.policy_optimizer.zero_grad()

            action = agent.policy(obses[:, agent_i])

            critic_in = torch.cat((obses, actions), dim=2).view(self.batch_size, -1)

            actor_loss = -agent.critic(critic_in).mean()
            actor_loss += (action ** 2).mean() * 1e-3  # Action regularize
            actor_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.agents[agent_i].policy.parameters(), 0.5)
            agent.policy_optimizer.step()

            self.update_all_targets()

    def update_all_targets(self):
        for agent in self.agents:
            soft_update(agent.target_critic, agent.critic, self.tau)
            soft_update(agent.target_policy, agent.policy, self.tau)

    def save(self, filename):
        raise NotImplementedError

    def load(self, filename):
        raise NotImplementedError


    @property
    def policies(self):
        return [agent.policy for agent in self.agents]

    @property
    def target_policies(self):
        return [agent.target_policy for agent in self.agents]

    @property
    def critics(self):
        return [agent.critic for agent in self.agents]

    @property
    def target_critics(self):
        return [agent.target_critic for agent in self.agents]

    '''
    @classmethod
    def init_from_save(cls, filename):

        save_dict = torch.load(filename)
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        for a, params in zip(instance.agents, save_dict['agent_params']):
            a.load_params(params)
        return instance
    '''
