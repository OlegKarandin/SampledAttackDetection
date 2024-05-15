from gymnasium.envs.registration import register

from sampleddetection.utils import setup_logger

logger = setup_logger(__name__)
# Read the first line
logger.info("Creating NetEnv-v0")

with open("./meepo.txt", "a+") as f:
    f.seek(0)
    line = f.readline()
    new_line_value = int(line) + 1 if line else 1
    f.seek(0)
    f.write(str(new_line_value))
    f.truncate()


register(
    id="NetEnv-v0",
    entry_point="gymenvs.GymSamplingEnv:GymSamplingEnv",
    max_episode_steps=100,  # CHECK: This is right
)
