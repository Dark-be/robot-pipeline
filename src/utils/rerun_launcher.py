"""Rerun 启动器 —— 在独立子进程运行 server + viewer，本进程作为 gRPC 客户端。

- 由 CollectEnv.env_setup() 调用（enable_rerun 时）
- 幂等：重复调用只初始化一次；若 9876 已被外部服务占用则直接复用
- 前端通过 web 后端 /api/rerun 查询 viewer URL，再拼 ?url= 参数嵌入 iframe

静态端口:
    - gRPC 数据服务    :9876
    - Web Viewer      :9090
"""

from __future__ import annotations

import atexit
import os
import socket
import subprocess
import sys
import threading
import time
import uuid
from typing import Optional

GRPC_PORT = 9876
WEB_PORT = 9090

_lock = threading.Lock()
_proc: Optional[subprocess.Popen] = None
_started = False
_web_url: Optional[str] = None
_client_thread_started = False


def _get_lan_ip() -> str:
    """获取本机局域网 IP（UDP 连接技巧），失败回退 127.0.0.1。"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:  # noqa: BLE001
        return "127.0.0.1"
    finally:
        s.close()


def _port_open(port: int, host: str = "127.0.0.1", timeout: float = 0.3) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def start_rerun() -> None:
    """启动 Rerun server（独立子进程）+ 常驻 gRPC 客户端连接（幂等）。"""
    global _proc, _started, _web_url, _client_thread_started
    with _lock:
        if _started:
            return
        try:
            # 0. 生成本次启动唯一的 recording_id，子进程与客户端共享同一 store，
            #    避免 viewer 里出现多个数据集。通过环境变量传递给子进程/客户端线程。
            os.environ["RERUN_RECORDING_ID"] = str(uuid.uuid4())

            # 1. 若没有现成 server，拉起独立子进程（继承环境变量）
            if not _port_open(GRPC_PORT):
                proc_path = os.path.join(os.path.dirname(__file__), "rerun_server_proc.py")
                _proc = subprocess.Popen([sys.executable, proc_path])
                atexit.register(stop_rerun)

            # 2. 等待 gRPC 端口就绪
            deadline = time.time() + 15
            while time.time() < deadline and not _port_open(GRPC_PORT):
                time.sleep(0.3)

            # 3. 常驻 daemon 线程建立 gRPC 客户端连接。
            #    关键：不能在采集线程里 connect_grpc —— 采集线程结束后，
            #    其内部的 tokio runtime 被销毁，连接会断开。独立线程持有可保持连接。
            if not _client_thread_started:
                t = threading.Thread(target=_keep_client_connected, daemon=True)
                t.start()
                _client_thread_started = True

            _started = True
            _web_url = f"http://{_get_lan_ip()}:{WEB_PORT}"
            print(f"[rerun] serving gRPC :{GRPC_PORT}, viewer at {_web_url}")
        except Exception as exc:  # noqa: BLE001
            print(f"[rerun] start failed: {exc}")


def _keep_client_connected() -> None:
    """在常驻 daemon 线程持有 gRPC 客户端连接，让 robot.visualize() 的 rr.log 能发到 server。"""
    import rerun as rr

    try:
        rr.init(
            "robot-pipeline",
            recording_id=os.environ.get("RERUN_RECORDING_ID"),
            spawn=False,
        )
        rr.connect_grpc(f"rerun+http://127.0.0.1:{GRPC_PORT}/proxy")
        print(f"[rerun] gRPC client connected (:{GRPC_PORT})")
        while True:  # 保持连接线程存活
            time.sleep(60)
    except Exception as exc:  # noqa: BLE001
        print(f"[rerun] client thread error: {exc}")


def get_rerun_url() -> str:
    """返回 Web Viewer 地址（前端 iframe 用），未启动则为空串。"""
    return _web_url or ""


def stop_rerun() -> None:
    """停止本进程拉起的 rerun 子进程（进程退出时自动调用）。"""
    global _proc
    if _proc is not None and _proc.poll() is None:
        _proc.terminate()
        try:
            _proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _proc.kill()
    _proc = None
