"""
Trying raylib for the experiments
Likely mostly to serve as a benchmark of an industrial solution
"""

import argparse
import ast
import json
import random
from argparse import ArgumentParser
from pathlib import Path
from time import time
from typing import Any, Dict, Optional, Union

import gymnasium as gym
import joblib as joblib
import numpy as np
import ray
import torch
import tqdm
from gymnasium.wrappers.normalize import NormalizeObservation
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.env.env_runner import EnvRunner
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import MultiAgentBatch, SampleBatch
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
from ray.rllib.utils.typing import EpisodeType, PolicyID
from ray.tune.registry import register_env
from wandb.sdk.wandb_run import Run

import wandb

# NOTE: Importing this is critical to load all model automatically.
from gymenvs.explicit_registration import explicit_registration
from networking.common_lingo import Attack
from networking.downstream_tasks.deepnets import Classifier
from networking.netfactories import NetworkFeatureFactory, NetworkSampleFactory
from networking.readers import NetCSVReader
from networking.samplers import WeightedSampler
from sampleddetection.datastructures import Action
from sampleddetection.environments import SamplingEnvironment
from sampleddetection.reward_signals import (
    DNN_RewardCalculator,
    RandForRewardCalculator,
)
from sampleddetection.samplers import DynamicWindowSampler
from sampleddetection.utils import (
    clear_screen,
    get_keys_of_interest,
    keychain_retrieve,
    setup_logger,
)

TYPE_CHECKING = True

if TYPE_CHECKING:
    from ray.rllib.algorithms.algorithm import Algorithm
    from ray.rllib.env.env_runner import EnvRunner
    from ray.rllib.env.env_runner_group import EnvRunnerGroup

LOCAL_KEYS_OF_INTEREST = [
    ["env_runners", "episode_reward_min"],
    ["env_runners", "episode_reward_mean"],
    ["env_runners", "episode_return_max"],
    ["info", "learner", "default_policy", "learner_stats", "cur_lr"],
    ["info", "learner", "default_policy", "learner_stats", "total_loss"],
    ["info", "learner", "default_policy", "learner_stats", "policy_loss"],
    ["info", "learner", "default_policy", "learner_stats", "vf_loss"],
]


def str_to_dict(s):
    try:
        return ast.literal_eval(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid dictionary format: {s}")


def argsies():
    ap = ArgumentParser()
    ap.add_argument(
        "--csv_path_str",
        # default="./data/mini_wednesday.csv",
        default="./data/Wednesday.csv",
        type=str,
        help="Path to where the data lies",
    )
    ap.add_argument(
        "--epochs",
        default=20,
        type=int,
        help="Training epochs",
    )
    ap.add_argument(
        "--num_possible_actions",
        default=2,
        type=int,
        help="Dimension for action vector",
    )
    ap.add_argument(
        "--paradigm_constants",
        default="./paradigm_constants.json",
        type=str,
        help="Where training/paradigm constants get stored.",
    )
    ap.add_argument(
        "--sampling_budget",
        default=12,
        type=int,
        help="How many sampling windows between window skips.",
    )
    ap.add_argument(
        "--random_seed",
        default=420,
        type=int,
        help="Seed for random generators",
    )
    # Random forst
    ap.add_argument(
        "--pretrained_ranfor",
        default="./models/multinary_detection_model.joblib",
        type=str,
        help="Pretrained random forest model",
    )
    ap.add_argument(
        "--weighted_bins_num",
        default=1600,
        type=int,
        help="Bins for the weighted algorithm",
    )

    # Prelimns
    ap.add_argument(
        "--action_idx_to_direction",
        default="{0: -1, 1: 1}",
        type=str_to_dict,
        help="Map between indices outputted by model vs values they actually represent. ",
    )

    # Logging Stuff
    ap.add_argument("-w", "--wandb", action="store_true")
    ap.add_argument(
        "--wandb_project_name",
        default="SamplingSimulations",
        help="Project name",
        type=str,
    )
    ap.add_argument("--wr_name", help="FirstWandbSample", type=str)
    ap.add_argument("--wr_notes", help="Nothing of note", type=str)

    ### Parse the argument
    args = ap.parse_args()

    assert Path(
        args.csv_path_str
    ).exists(), f"--csv_path_str {args.csv_path_str} does not exist."

    # Check on their values
    ## Add Values Manually
    with open(args.paradigm_constants, "r") as f:
        # Add the amount of observations.
        paradigm_spec_file = json.load(f)
        desired_features = paradigm_spec_file["desired_features"]
        args.obs_elements = desired_features

        # Add Actions
        actions = paradigm_spec_file["actions"]
        action_dir = {i: a for i, a in enumerate(actions)}
        args.action_dir = action_dir

    return args


def set_all_seeds(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def train_result_callback(info):
    result = info["result"]
    wandb.log()


def generate_parameterized_callback(wandb_run: Run):
    return type(
        "ParamCallback",
        (MyCallbacks,),
        {
            "wandb_run": wandb_run,
        },
    )


class MyCallbacks(DefaultCallbacks):

    LIST_OF_INTEREST = [
        ["env_runners", "episode_reward_min"],
        ["env_runners", "episode_reward_mean"],
        ["env_runners", "episode_return_max"],
        ["env_runners", "num_faulty_episodes"],
        ["info", "learner", "default_policy", "learner_stats", "cur_lr"],
        ["info", "learner", "default_policy", "learner_stats", "total_loss"],
        ["info", "learner", "default_policy", "learner_stats", "policy_loss"],
        ["info", "learner", "default_policy", "learner_stats", "vf_loss"],
        [
            "info",
            "learner",
            "default_policy",
            "num_grad_updates_lifetime",
        ],
        [
            "info",
            "learner",
            "default_policy",
            "diff_num_grad_updates_vs_sampler_policy",
        ],
        ["info", "num_agent_steps_sampled"],
        ["info", "num_agent_steps_trained"],
        ["info", "num_env_steps_trained"],
        ["agent_timesteps_total"],
        ["timers", "sample_time_ms"],
        ["num_env_steps_sampled_throughput_per_sec"],
    ]

    def __init__(self, wandb_run: Run) -> None:
        super().__init__()
        self.logger = setup_logger(__class__.__name__)
        self.wandb_run = wandb_run

    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        # you can mutate the result dict to add new fields to return
        result["callback_ok"] = True

        self.logger.debug(f"Premature results {result}")
        report_dict = {}
        for kc in self.LIST_OF_INTEREST:
            val = keychain_retrieve(result, kc)
            if val != None:
                report_dict[".".join(kc)] = val
        self.logger.debug(f"Reporting results {report_dict}")

        if self.wandb_run:
            self.wandb_run.log(report_dict)

    def on_sample_end(  # type:ignore
        self, *, worker: RolloutWorker, samples: MultiAgentBatch, **kwargs
    ):
        # Sanity Check on aount of samples. Take a look at one looks like:
        self.logger.debug(f"Policy batches look like {samples.policy_batches}")
        self.logger.debug(
            f"Observations (shape {samples.policy_batches['default_policy']['obs'].shape}) themselves: {samples.policy_batches['default_policy']['obs']}"
        )
        # TODO: Create a better assertion for this
        # assert (
        #     samples.count == 128
        # ), f"Sample count does not match 128 it is actually {samples.count}"

    def on_episode_end(
        self,
        *,
        episode: Union[EpisodeType, Episode, EpisodeV2],
        env_runner: Optional[EnvRunner] = None,
        metrics_logger: Optional[MetricsLogger] = None,
        env: Optional[gym.Env] = None,
        env_index: int,
        rl_module: Optional[RLModule] = None,
        worker: Optional["EnvRunner"] = None,
        base_env: Optional[BaseEnv] = None,
        policies: Optional[Dict[PolicyID, Policy]] = None,
        **kwargs,
    ) -> None:
        self.logger.debug(f"Episode ends")

    # def on_evaluate_start(
    #     self,
    #     *,
    #     algorithm: "Algorithm",#type:ignore
    #     metrics_logger: Optional[MetricsLogger] = None,
    #     **kwargs,
    #
    # ) -> None:
    #     pass


def env_wrapper(env) -> gym.Env:
    """
    To my understanding each process will run this independently.
    """
    # Call the registration
    explicit_registration()

    # Specify the NetworkSampleFactor
    sample_factory = NetworkSampleFactory()
    feature_factory = NetworkFeatureFactory(args.obs_elements, attacks_to_detect)

    num_features = len(args.obs_elements)

    ### In case classification learner
    # # Create the downstream classidication learner
    # classifier = Classifier(
    #     input_size=num_features, output_size=len(attacks_to_detect) + 1
    # )
    # # Create reward calculator to use
    # reward_calculator = DNN_RewardCalculator(classifier)

    # Use the random forest creation
    reward_calculator = RandForRewardCalculator(args.pretrained_ranfor)

    # CHECK: Do we want to use NoReplacementSampler or DynamicSampler?
    # Remember that NoReplacementSampler has quite the overhead
    # meta_sampler = NoReplacementSampler(data_reader, sample_factory)
    sampler = ray.get(global_sampler_ref)

    sampenv = SamplingEnvironment(
        sampler,
        reward_calculator=reward_calculator,
        feature_factory=feature_factory,
    )

    print("Trying to make NETENVE")
    env = gym.make(
        "NetEnv",
        sampling_env=sampenv,
        num_obs_elements=num_features,
        actions_max_vals=Action(60, 10),
        action_idx_to_direction=args.action_dir,
    )
    print("MANAGED TO MAKE NETENV")
    # Use wrapper to normalize the data:
    # env = NormalizeObservation(env)
    return env


if __name__ == "__main__":
    args = argsies()
    print(f"Do we have epochs? {args.epochs}")

    # Make the logger
    # logging.basicConfig(level=logging.DEBUG)
    # ray_logger = logging.getLogger("ray")
    # ray_logger.setLevel(logging.WARNING)
    logger = setup_logger(__name__)
    logger.info("Starting main part of script.")

    # Initialize Wandb Logger
    wandb_run = None
    if args.wandb:
        logger.info("🪄 Instantiating WandB")
        wandb_run = wandb.init(
            project=args.wandb_project_name,
            name=args.wr_name,
            notes=args.wr_notes,
        )

    # Define which labels one expects on the given dataset
    attacks_to_detect = (
        Attack.HULK,
        Attack.GOLDENEYE,
        Attack.SLOWLORIS,
        Attack.SLOWHTTPTEST,
        # Attack.HEARTBLEED. # Takes too long find in dataset.
    )

    # Columns to Normalize
    columns_to_normalize = (
        "fwd_pkt_len_max",
        "fwd_pkt_len_min",
        "fwd_pkt_len_mean",
        "bwd_pkt_len_max",
        "bwd_pkt_len_min",
        "bwd_pkt_len_mean",
        "flow_byts_s",
        "flow_pkts_s",
        "flow_iat_mean",
        "flow_iat_max",
        "flow_iat_min",
        "fwd_iat_mean",
        "fwd_iat_max",
        "fwd_iat_min",
        "bwd_iat_max",
        "bwd_iat_min",
        "bwd_iat_mean",
        "pkt_len_min",
        "pkt_len_max",
        "pkt_len_mean",
    )
    # Make shared CSVReader
    csv_path = Path(args.csv_path_str)
    csv_reader = NetCSVReader(csv_path)

    global_sampler = WeightedSampler(
        csv_reader, args.sampling_budget, args.weighted_bins_num
    )
    global_sampler_ref = ray.put(global_sampler)
    # csv_reader_ref = ray.put(csv_reader)

    # Make the environment
    print("Make the environment")

    register_env("WrappedNetEnv", env_wrapper)

    print("Resetting the environment")
    set_all_seeds(args.random_seed)
    algo = (
        PPOConfig()
        .env_runners(num_env_runners=1)
        .resources(num_gpus=1)
        .environment(env="WrappedNetEnv")
        .training(train_batch_size=128)  # How many steps are used to update model
        .callbacks(lambda: MyCallbacks(wandb_run))  # type : ignore
        .build()
    )
    sample_timeout_s = algo.config.get("sample_timeout_s", "Not set")
    batch_size = algo.config.get("train_batch_size", "Not set")

    # This is for trying the batches

    logger.info(f"The `sample_timeout_s` parmeter is {sample_timeout_s}")
    logger.info(f"The `batch_size` parameter is {batch_size}")
    epoch_time = []
    for i in tqdm.tqdm(range(args.epochs), desc="Training"):
        start_time = time()
        result = algo.train()
        epoch_time.append(time() - start_time)
        clear_screen()
        logger.info(f"Finished with {i}th epoch with time {epoch_time[-1]}.")
        desired_dict = get_keys_of_interest(result, LOCAL_KEYS_OF_INTEREST)
        logger.info(
            f"""
        Training (epoch {i+1}/{args.epochs}):
        ---------
        ep_reward_min: {desired_dict.get('env_runners.episode_reward_min', 'N/A')}
        ep_reward_mean: {desired_dict.get('env_runners.episode_reward_mean', 'N/A')}
        ep_reward_max: {desired_dict.get('env_runners.episode_reward_max', 'N/A')}
        total_loss: {desired_dict.get('info.learner.default_policy.learner_stats.total_loss', 'N/A')}
        policy_loss: {desired_dict.get('info.learner.default_policy.learner_stats.policy_loss', 'N/A')}
        vf_loss: {desired_dict.get('info.learner.default_policy.learner_stats.vf_loss', 'N/A')}
        """
        )
    print("Training is finished")

    # Build a Algorithm object from the config and run 1 training iteration.
    # algo = ppo.PPO(env=MetaEnv, config={"num_obs_elements": args.num_obs_elements})
    # algo.train()
