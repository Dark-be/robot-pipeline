import argparse, os
from config._GLOBAL_CONFIG import CONFIG_DIR
from utils.base.load_file import load_yaml
from utils.base.data_handler import debug_print
from task_env.collect_env import CollectEnv
# 定义命令行参数并解析
# --task_name: 任务名称，必填参数
# --base_cfg: 基础配置文件名称，必填参数
# --st_idx: 起始集数索引，默认为0，表示从第0回合开始收集数据
parser = argparse.ArgumentParser()
parser.add_argument("--base_cfg", type=str, required=True)
parser.add_argument("--st_idx", type=int, default=0, help="start episode index")
args_cli = parser.parse_args()

if __name__ == "__main__":
    # 载入config文件夹下的基础配置文件 --base_cfg
    base_cfg = load_yaml(os.path.join(CONFIG_DIR, f'{args_cli.base_cfg}.yml'))
    os.environ["INFO_LEVEL"] = base_cfg.get("INFO_LEVEL", "INFO")

    env_cfg = base_cfg.get("collect_env", {})
    START = args_cli.st_idx
    END = env_cfg.get("num_episode", 50) + START

    TASK_ENV = CollectEnv(base_cfg)
    TASK_ENV.env_setup()
    
    for episode_id in range(START, END):
        if TASK_ENV.finish_flag:
            debug_print("MAIN", "Data collection has been finished by user. Exiting main loop.", "INFO")
            break
        print(
            f"\n\033[96m══════════════════════════════════════════════\033[0m\n"
            f"\033[94m▶ Episode\033[0m  \033[97m{episode_id:>3}/{END-1:<3}\033[0m   "
            f"\033[90m(id={episode_id}, range={START}-{END-1}), start from 0\033[0m\n"
            f"\033[92m[START]\033[0m set_episode_idx -> {episode_id}\n"
        )
        TASK_ENV.set_episode_idx(episode_id)
        TASK_ENV.collect_one_episode()

        print(
            f"\033[92m[DONE ]\033[0m episode_id={episode_id}\n"
            f"\033[96m══════════════════════════════════════════════\033[0m"
        )
    TASK_ENV.env_finish()