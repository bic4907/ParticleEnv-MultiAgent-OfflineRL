import torch
from utils.misc import soft_update

from model.DDPGAgent import DDPGAgent


class DDPG(object):

    def __init__(self, name, params):

        self.name = name
        self.lr = params.lr
        self.gamma = params.gamma

        self.obs_dim = params.obs_dim
        self.action_dim = params.action_dim
        self.agent_types = params.agent_types
        self.device = params.device


        self.training = False




        self.num_agents = len(self.agent_types)
        # self.alg_types = alg_types
        self.agents = [DDPGAgent(params) for _ in range(self.num_agents)]
        [agent.to(self.device) for agent in self.agents]


        # self.agent_init_params = agent_init_params
        # self.gamma = gamma
        # self.tau = tau

        # self.discrete_action = discrete_action

        # self.niter = 0

    def scale_noise(self, scale):
        for a in self.agents:
            a.scale_noise(scale)

    def reset_noise(self):
        for a in self.agents:
            a.reset_noise()

    def act(self, observations, sample=False):
        observations = torch.Tensor(observations).to(self.device)
        return [agent.act(obs, explore=sample) for agent, obs in zip(self.agents, observations)]

    def update(self, replay_buffer, logger, step):
        sample = replay_buffer.sample(100)

        for agent_i, agent in enumerate(self.agents):

            obses, actions, rewards, next_obses, dones = sample
            curr_agent = self.agents[agent_i]

            curr_agent.critic_optimizer.zero_grad()
            if self.alg_types[agent_i] == 'MADDPG':
                target_actions = [pi(nobs) for pi, nobs in zip(self.target_policies, next_obses)]
                trgt_vf_in = torch.cat((*next_obses, *target_actions), dim=1)
            else:  # DDPG
                trgt_vf_in = torch.cat((next_obses[agent_i],
                                        curr_agent.target_policy(next_obses[agent_i])),
                                       dim=1)
            target_value = (rewards[agent_i].view(-1, 1) + self.gamma *
                            curr_agent.target_critic(trgt_vf_in) *
                            (1 - dones[agent_i].view(-1, 1)))

            if self.alg_types[agent_i] == 'MADDPG':
                vf_in = torch.cat((*obses, *actions), dim=1)
            else:  # DDPG
                vf_in = torch.cat((obses[agent_i], actions[agent_i]), dim=1)
            actual_value = curr_agent.critic(vf_in)
            vf_loss = torch.nn.MSELoss(actual_value, target_value.detach())
            vf_loss.backward()

            torch.nn.utils.clip_grad_norm(curr_agent.critic.parameters(), 0.5)
            curr_agent.critic_optimizer.step()
            curr_agent.policy_optimizer.zero_grad()

            curr_pol_out = curr_agent.policy(obses[agent_i])
            curr_pol_vf_in = curr_pol_out

            if self.alg_types[agent_i] == 'MADDPG':
                all_pol_acs = []
                for i, pi, ob in zip(range(self.nagents), self.policies, obses):
                    if i == agent_i:
                        all_pol_acs.append(curr_pol_vf_in)
                    else:
                        all_pol_acs.append(pi(ob))
                vf_in = torch.cat((*obses, *all_pol_acs), dim=1)
            else:  # DDPG
                vf_in = torch.cat((obses[agent_i], curr_pol_vf_in), dim=1)

            pol_loss = -curr_agent.critic(vf_in).mean()
            pol_loss += (curr_pol_out ** 2).mean() * 1e-3
            pol_loss.backward()

            torch.nn.utils.clip_grad_norm(curr_agent.policy.parameters(), 0.5)
            curr_agent.policy_optimizer.step()

    def update_all_targets(self):
        for a in self.agents:
            soft_update(a.target_critic, a.critic, self.tau)
            soft_update(a.target_policy, a.policy, self.tau)

    def save(self, filename):
        raise NotImplementedError

    def load(self, filename):
        raise NotImplementedError

    def train(self, is_train):
        # TODO Change model training/eval switching here
        return None

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