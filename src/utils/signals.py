"""采集信号定义 —— Web 层与采集环境交互的命名信号。

用明确的信号名替代"按键字符"式的隐晦交互，
Web 层 / 采集环境之间通过这些信号名驱动。

信号 → 对应 KEY_DICT 按键语义：
    - episode_start  对应 START
    - episode_stop   对应 STOP
    - quit           对应 QUIT
    - reset          对应 RESET
"""

SIG_ENV_SETUP = "env_setup"          # 启动采集任务（连接机器人、开启多回合流程）
SIG_EPISODE_START = "episode_start"  # 开始一个回合
SIG_EPISODE_STOP = "episode_stop"    # 结束一个回合（保存）
SIG_QUIT = "quit"                    # 退出整个任务
SIG_RESET = "reset"                  # 重置机械臂（归位）
