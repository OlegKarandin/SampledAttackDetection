from gymnasium.envs.registration import register


def register_environments():
    register(
        id="NetEnv-v0",
        entry_point="sampleddetection.environment.gymenvs.GymNetworkEnv:GymNetworkEnv",
        max_episode_steps=100,  # CHECK: This is right
    )
