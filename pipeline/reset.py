import argparse, os

from utils.base.load_file import load_yaml
from config._GLOBAL_CONFIG import CONFIG_DIR
from robot import get_robot

parser = argparse.ArgumentParser()
parser.add_argument("--base_cfg", type=str, required=True, help="config file name for robot setup")
args_cli = parser.parse_args()

if __name__ == "__main__":
    base_cfg = load_yaml(os.path.join(CONFIG_DIR, f'{args_cli.base_cfg}.yml'))
    os.environ["INFO_LEVEL"] = base_cfg.get("INFO_LEVEL", "INFO") # DEBUG, INFO, ERROR
    robot = get_robot(base_cfg)
    
    robot.connect()
    robot.reset()
    robot.disconnect()