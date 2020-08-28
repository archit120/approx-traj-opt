import torch
import numpy as np
import networks.policy
import networks.q_func

import replay_buffer

import envs.race_traj

class SoftActorCritic():
    q_net_1: networks.q_func.QFunc
    q_net_2: networks.q_func.QFunc
    policy_net: networks.policy.PolicyNet

    def __init__(self, q_network=networks.q_func.QFunc, policy_network=networks.policy.PolicyNet, 
                        tau=0.005, batch_size = 256, look_ahead=1, look_behind=1, gate_width=1, gate_height=0.5):
        super().__init__()
        self.q_net_1 = q_network()
        self.q_net_2 = q_network()

        self.target_1 = q_network()
        self.target_2 = q_network()

        self.policy_net = policy_network()
        self.replay_buffer = replay_buffer.ReplayBuffer()
        self.standard_normal = torch.distributions.Normal(0, 1)

        self.tau = tau
        self.batch_size = batch_size

        self.init_target()

        self.state_size = (1+look_ahead+look_behind)*3
        self.action_size = 2
        self.look_ahead = look_ahead
        self.look_behind = look_behind
        self.gate_width = 1
        self.gate_height = 0.5
        self.env = envs.race_traj.RaceTrajEnv(gate_height=gate_height, gate_width=gate_width)


    def get_actions(self, states):
        mean, logstd = self.policy_net.forward(states)
        eps = self.standard_normal.sample(mean.shape)
        return mean+torch.exp(logstd)*eps

    def get_logprob(self, state, action):
        mean, logstd = self.policy_net.forward(state)
        eps = (action - mean)/torch.exp(logstd)
        return self.standard_normal.log_prob(eps)

    def action_map(self, action):
        action = np.tanh(action)
        action[0] *= self.gate_width/2
        action[1] *= self.gate_height/2
        return action

    def env_iter(self):
        state = self.env.reset()
        while True:
            action = self.get_actions(torch.from_numpy(state.reshape(1, -1)))
            action = action.numpy()[0]
            state_next, reward, done, _ = self.env.step(self.action_map(action))

            self.replay_buffer.add_transition((state, action, reward, state_next, done))

            if done:
                break

            state = state_next
        
    def update_targets(self):
        for target_param, param in zip(self.target_1.parameters(), self.model.q_net_1()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

        for target_param, param in zip(self.target_2.parameters(), self.model.q_net_2()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)
    
    def init_target(self):
        for target_param, param in zip(self.target_1.parameters(), self.model.q_net_1()):
            target_param.data.copy_(param)

        for target_param, param in zip(self.target_2.parameters(), self.model.q_net_2()):
            target_param.data.copy_(param)


    def grad_iter(self):
        samples = self.replay_buffer.sample_transitions(self.batch_size)
        s_t = np.zeros((self.batch_size, self.look_ahead+1+self.look_behind, 3))
        a_t = np.zeros((self.batch_size, 2))
        rewards = np.zeros((self.batch_size, 1))
        terminal = np.zeros((self.batch_size))
        s_t1 = np.zeros((self.batch_size, self.look_ahead+1+self.look_behind, 3))
        for i in range(self.batch_size):
            s_t[i] = samples[i][0]
            a_t[i] = samples[i][1]
            rewards[i] = samples[i][2]
            s_t1[i] = samples[i][3]
            terminal[i] = samples[i][4]
        
        s_t_tensor = torch.from_numpy(s_t)
        # q_val_tensor_1  = self.


