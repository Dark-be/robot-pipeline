"""Rerun server 独立进程入口（由 utils.rerun_launcher 用 subprocess 拉起）。

为什么独立进程：在 uvicorn(uvloop) 进程内 serve_grpc 会与 rerun 的 tokio
runtime 冲突，导致 gRPC server 起不来。放到独立 Python 进程即可正常运行。

本进程负责（静态端口，与 act_env.py / 前端保持一致）:
    - serve_grpc(0.0.0.0:9876)      接收 robot.visualize() 发布的数据
    - serve_web_viewer(0.0.0.0:9090) 托管 Web Viewer（前端 iframe 嵌入）
"""

from __future__ import annotations

import os
import time

import rerun as rr

GRPC_PORT = 9876
WEB_PORT = 9090


def main() -> None:
    # 与客户端共享同一 application id 与 recording_id（由 rerun_launcher 注入），
    # 使 server 端只产生一个数据集
    rr.init(
        "robot-pipeline",
        recording_id=os.environ.get("RERUN_RECORDING_ID"),
        spawn=False,
    )
    rr.serve_grpc(
        grpc_port=GRPC_PORT,
        server_memory_limit="1GiB",
        newest_first=False,
        cors_allow_origin=["*"],
    )
    rr.serve_web_viewer(
        web_port=WEB_PORT,
        open_browser=False,
        connect_to=f"rerun+http://127.0.0.1:{GRPC_PORT}/proxy",
    )
    print(f"[rerun-server] serving gRPC :{GRPC_PORT}, web viewer :{WEB_PORT}")
    while True:  # 保持进程存活
        time.sleep(60)


if __name__ == "__main__":
    main()
