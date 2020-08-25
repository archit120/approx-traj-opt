import traj_gen
import numpy as np

from traj_gen import poly_trajectory as pt

import time

def get_knots(waypoints, scale = 10):
    total_time = 0
    for wpa, wpb in zip(waypoints[1:], waypoints[:-1]):
        total_time += np.linalg.norm(wpa-wpb)    

    knots = np.zeros((len(waypoints)))
    knots[0] = 0
    for i, (wpa, wpb) in enumerate(zip(waypoints[1:], waypoints[:-1])):
        knots[i+1] = knots[i]+np.linalg.norm(wpa-wpb)/total_time
    
    return (knots*scale).astype(np.float)


def get_trajectory_snap(waypoints, tdelta = 0.1, time_between_gates = 2):
    dim = 3
    knots = get_knots(waypoints, time_between_gates*len(waypoints))
    order = 8
    optimTarget = 'poly-coeff' #'end-derivative' 'poly-coeff'
    maxConti = 4
    objWeights = np.array([0, 0, 0, 0, 1])
    pTraj = pt.PolyTrajGen(knots, order, optimTarget, dim, maxConti)

        # 2. Pin
    Xdot = np.array([0, 0, 0])
    Xddot = np.array([0, 0, 0])

    pin_ = {'t':0, 'd':1, 'X':Xdot,}
    pTraj.addPin(pin_)
    pin_ = {'t':0, 'd':2, 'X':Xddot,}
    pTraj.addPin(pin_)


    for i, wp in enumerate(waypoints):

        pin_ = {'t':knots[i], 'd':0, 'X':wp}
        pTraj.addPin(pin_)

    # solve
    pTraj.setDerivativeObj(objWeights)
    pTraj.solve()
    rng = np.linspace(0, len(waypoints)*time_between_gates, int((len(waypoints))*time_between_gates//tdelta))
    snap = pTraj.eval(rng, 4)
    return np.linalg.norm(snap)**2

def calc_bonus(waypoints, delta, gate_width = 1, gate_height = 0.5,):
    delta[:, 1] = np.clip(delta[:, 1], -gate_height/2, gate_height/2)
    delta[:, 0] = np.clip(delta[:, 0], -gate_width/2, gate_width/2)

    dvx = np.zeros((waypoints.shape[0], 3))
    dvx[1:] = waypoints[1:]-waypoints[:-1]
    dvx[0] = waypoints[1]
    az1 = np.array([0,0,1])
    az0 = np.cross(az1, dvx)
    az0 = az0/np.linalg.norm(az0)
    az1 = np.array([az1])

    waypoints_modif = waypoints.copy() + az1*delta[:, 1:] + az0* delta[:, :1]
    return get_trajectory_snap(waypoints_modif)