import rerun as rr
import numpy as np
import time
# verifying connection to server(internal),js api error Type Error: failed to fetch
def main():
    rr.init("my_app", spawn=False)
    # 启动 gRPC 服务器（用于接收日志数据）
    # 数采端启动
    server_url = rr.serve_grpc(
        grpc_port=9876,
        server_memory_limit="1GiB",
        newest_first=False,
        cors_allow_origin=["*"]
    )
    print(f"[Rerun]gRPC 服务器已启动: {server_url}")
    # 启动 Web 浏览器界面，连接到 gRPC 服务器
    rr.serve_web_viewer(
        web_port=9090,  # Web 界面端口
        open_browser=False,  # 自动打开浏览器
        connect_to=server_url  # 连接到 gRPC 服务器
    )

    # rr.log("data", rr.Boxes3D(half_sizes=[2.0, 2.0, 1.0]))
    height = 0
    print("数据已推送到服务器")
    # print("请在浏览器中访问: http://web_server_url?url=grpc_server_url")
    print(f"请在浏览器中访问: http://10.30.20.111:9090?url=rerun+http://10.30.20.111:9876/proxy")
    print("按 Enter 键停止服务器...")
    try:
        while True:
            time.sleep(1)
            rr.log("data", rr.Boxes3D(half_sizes=[height, 2.0, 1.0]))
            height+=1
    except KeyboardInterrupt:
        rr.disconnect()
        print("\nShutting down server…")
    
    # 清理

if __name__ == "__main__":
    main()