from gymnasium.envs.registration import register


def explicit_registration():
    register(
        id="NetEnv",
        entry_point="gymenvs.envs:GymSamplingEnv",
        max_episode_steps=100,  # CHECK: This is right
    )
