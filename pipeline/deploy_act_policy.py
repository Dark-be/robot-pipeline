import argparse
import os
import sys

from robot.utils.base.load_file import load_yaml
from robot.config._GLOBAL_CONFIG import CONFIG_DIR, POLICY_DIR
from task_env.act_env import ACTEnv

parser = argparse.ArgumentParser(description="Deploy ACT policy on a real robot.")
parser.add_argument("--base_cfg", type=str, required=True,
                    help="Path to robot config YAML (e.g. config/x-one.yml).")
args_cli = parser.parse_args()


if __name__ == "__main__":
    base_cfg = load_yaml(os.path.join(CONFIG_DIR, f'{args_cli.base_cfg}.yml'))

    env = ACTEnv(base_cfg=base_cfg)
    env.set_up()
    env.run_deployment()
