
import os
import time
from multiprocessing import shared_memory, Process, Event
import numpy as np

from robot.controller.arm_controller import ArmController
from robot.controller.Alicia_controller import AliciaController
from robot.utils.base.data_handler import debug_print

# 利用共享内存和多进程实现 Alicia 示教臂的状态采集，避免SDK状态采集阻塞主线程，确保数据采集的实时性和稳定性。
class AliciaTeachController(ArmController):
	"""Alicia 示教臂 Controller。

	- 读取：完全参考 third_party/Alicia-D-SDK/examples/03_demo_read_state.py
	  使用 robot.get_robot_state("joint_gripper") 获取 state.angles
	- 控制：仅支持关节角度（6 DoF, 弧度）下发，不做任何 IK / pose 解算
	"""
	def __init__(self, name: str, INFO: str = "INFO"):
		super().__init__()
		self.name = name
		self.controller_type = "user_controller"
		self.INFO = INFO

		self.robot = None
		self.port: str = ""

		self.collect_process = None
		self.stop_event = Event()

		self.shm = shared_memory.SharedMemory(
    	create=True,
    	size=np.dtype(np.float32).itemsize * 13
		)
		self.shm_name = self.shm.name  

		self.shared_array = np.ndarray(
    	(13,),
    	dtype=np.float32,
    	buffer=self.shm.buf
		)
		
		self.state = {"joint": None, "gripper": None, "eef": None}

	def set_up(self, port: str, collect_frequency_hz: float = 30.0):
		"""初始化连接机器人，设置串口和夹爪型号等参数。
		:param port: 串口端口，如 /dev/ttyACM0
		"""
		self.port = port
		self.collect_frequency_hz = collect_frequency_hz

		self.collect_process = Process(target=collect_alicia, args=(
			self.shm_name,
			self.name,
			self.port,
			self.collect_frequency_hz,
			self.stop_event
		))
		self.collect_process.start()

	def set_joint(self, joint: np.array):
		pass

	def get_state(self):
		data = self.shared_array.copy()
		
		return {
      		"joint": data[:6],
      		"gripper": data[6],
			"eef": [0,0,0,0,0,0]
    	}
		
	def stop(self):
		if self.shm:
			self.shm.close()
			self.shm.unlink()
			debug_print(self.name, f"Shared memory {self.shm_name} released.", self.INFO)

		if self.stop_event:
			self.stop_event.set()
		
		if self.collect_process and self.collect_process.is_alive():
			self.collect_process.join(timeout=2.0)
			if self.collect_process.is_alive():
				debug_print(self.name, "Collect process did not terminate in time, force terminating.", "WARNING")
				self.collect_process.terminate()
				self.collect_process.join()

			debug_print(self.name, "AliciaTeachController stopped and resources cleaned up.", self.INFO)

	def disconnect(self):
		self.stop()
	
def collect_alicia(shm_name, name, port, hz, stop_event):
	alicia = None
	shm = shared_memory.SharedMemory(name=shm_name)
	try:
		alicia = AliciaController(name, INFO="DEBUG")
		alicia.set_up(port=port)

		while not stop_event.is_set():
			start_time = time.monotonic()

			data = alicia.get_state()  # 获取数据
			shared = data["joint"].tolist() + [data["gripper"]] + [0, 0, 0, 0, 0, 0]  # joint(6) + gripper(1) + eef(6)
			np.ndarray((13,), dtype=np.float32, buffer=shm.buf)[:] = shared

			frame_time = time.monotonic() - start_time
			if frame_time < 1 / hz:
				time.sleep(1 / hz - frame_time)  # 控制频率节奏
			else:
				debug_print(name, f"Data collection is running behind by {frame_time:.5f}s", "WARNING")

	except Exception as e:
		debug_print(name, f"Data collection process error: {e}", "ERROR")
	finally:
		if alicia:
			alicia.disconnect()
		if shm:
			shm.close()
		debug_print(name, "AliciaTeachController data collection process exiting.", "INFO")