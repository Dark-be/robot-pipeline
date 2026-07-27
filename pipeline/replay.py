import argparse, os

from robot.utils.base.load_file import load_yaml
from robot.config._GLOBAL_CONFIG import CONFIG_DIR
from robot.robot import get_robot

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--base_cfg", type=str, required=True, help="config file name for data collection")
parser.add_argument("--idx", type=int, required=True, help="config file name for data collection")
parser.add_argument("--collect", action="store_true", help="enable data collection")
parser.add_argument("--collect_idx", type=int, help="required when --collect is set")
args_cli = parser.parse_args()
if args_cli.collect and args_cli.collect_idx is None:
    parser.error("--collect requires --collect_idx (e.g. --collect --collect_idx 3)")

if __name__ == "__main__":
    base_cfg = load_yaml(os.path.join(CONFIG_DIR, f'{args_cli.base_cfg}.yml'))
    os.environ["INFO_LEVEL"] = base_cfg.get("INFO_LEVEL", "INFO") # DEBUG, INFO, ERROR

    task_name = base_cfg.get("task_name")

    robot = get_robot(base_cfg)
    robot.set_up()
    robot.reset()

    save_dir = os.path.join(base_cfg["collect"].get("save_dir"), base_cfg["collect"]["task_name"], base_cfg["collect"]["type"])
    
    # keybanned = ["eef"]
    # robot.replay(
    #     data_path=os.path.join(save_dir, f"{args_cli.idx}.hdf5"),
    #     fps= base_cfg["collect"].get("save_freq", 30),
    #     key_banned=keybanned,
    #     is_collect=args_cli.collect,
    #     episode_id=args_cli.collect_idx
    # )
    df = pd.read_csv(os.path.join(save_dir, 'actions.csv'))
    actions = df.to_numpy()
    robot.replay_csv(actions)   