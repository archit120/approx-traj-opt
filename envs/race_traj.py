import gym
from gym import error, spaces, utils
from gym.utils import seeding

import math
import numpy as np

from copy import deepcopy

import envs.traj_reward

class RaceTrajEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }


    def __init__(self, num_of_gates = 10, angle_restriction = math.pi*3/5, 
                        next_distance_low = 5, next_distance_high = 15, 
                        look_ahead = 1, look_behind = 1,
                        gate_width = 1, gate_height = 0.5,
                        angle_dev=0.1):

        self.num_of_gates = num_of_gates
        self.angle_restriction = angle_restriction
        self.next_distance_low = next_distance_low
        self.next_distance_high = next_distance_high
        self.look_ahead = look_ahead
        self.look_behind = look_behind
        self.gate_width = gate_width
        self.gate_height = gate_height
        self.angle_dev = angle_dev

# Action space first axis is for inplane movement and second for vertical 
        self.action_space = spaces.Box(-np.inf, np.inf, shape=(2,))
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(look_behind+1+look_ahead, 3))

        self.seed()

        self.viewer = None
        self.state = None
        self.state_i = 0
        self.entire_traj = None
        self.entire_traj_act = None
        self.steps_beyond_done = None

        self.prev_reward = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        self.state_i += 1
        done = self.state_i==len(self.entire_traj)-2
        self.modifs[self.state_i] = action

        gnew = envs.traj_reward.calc_bonus(self.entire_traj[1:], self.modifs[1:])
        reward = self.prev_reward - gnew
        self.prev_reward = gnew
        
        self.state = np.array(self.entire_traj[self.state_i-self.look_behind:self.state_i+self.look_ahead+1])

        return np.array(self.state), reward, done, {}

    def get_next_pt(self, cpt, ppt):
        grad = cpt - ppt
        thetad = math.atan2(grad[1], grad[0])
        gdp = math.sqrt(grad[0]**2+grad[1]**2)
        phid = math.atan2(grad[2], gdp)
        thetaf = thetad + np.clip(np.random.standard_normal()*self.angle_dev*self.angle_restriction/2, -self.angle_restriction/2, self.angle_restriction/2)
        phif = phid + np.clip(np.random.standard_normal()*self.angle_dev*self.angle_restriction/2, -self.angle_restriction/2, self.angle_restriction/2)
        npt = np.array([math.cos(thetaf), math.sin(thetaf), math.tan(phif)])
        npt = cpt + npt/np.linalg.norm(npt)*(np.random.uniform(math.sqrt(self.next_distance_low), math.sqrt(self.next_distance_high))**2)
        return npt


    def reset(self):

        self.entire_traj = [np.array([0, -1, 0]), np.array([0, 0, 0])]

        for i in range(self.num_of_gates+1):
            self.entire_traj.append(self.get_next_pt(self.entire_traj[-1], self.entire_traj[-2]))

        self.entire_traj = np.array(self.entire_traj)

        self.modifs = np.zeros((self.entire_traj.shape[0], 2))

        self.state_i = 2
        self.state = np.array(self.entire_traj[self.state_i-self.look_behind:self.state_i+self.look_ahead+1])
        self.steps_beyond_done = None
        self.prev_reward = envs.traj_reward.get_trajectory_snap(self.entire_traj[1:])

        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold*2
        scale = screen_width/world_width
        carty = 100 # TOP OF CART
        polewidth = 10.0
        polelen = scale * 1.0
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            axleoffset =cartheight/4.0
            cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
            pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole.set_color(.8,.6,.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5,.5,.8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0,carty), (screen_width,carty))
            self.track.set_color(0,0,0)
            self.viewer.add_geom(self.track)

        if self.state is None: return None

        x = self.state
        cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer: self.viewer.close()
