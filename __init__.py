from gym.envs.registration import register

register(
    id='race-traj-v0',
    entry_point='.envs:RaceTrajEnv',
)
