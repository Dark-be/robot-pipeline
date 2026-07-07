import time
import numpy as np
import cv2
from robot.sensor.base_touch_sensor import BaseTouchSensor
from robot.utils.base.data_handler import debug_print
from multiprocessing import shared_memory, resource_tracker

class TouchSensor(BaseTouchSensor):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.sensor = None
        self.device = None
        self.output_size = (60, 80)  
        self.shm = shared_memory.SharedMemory(name="touch_sensor_data")
        resource_tracker.unregister(self.shm._name, 'shared_memory')
        self.sensor_data = np.ndarray((240, 320, 3, 4), dtype=np.float32, buffer=self.shm.buf)

    def set_up(self, device: int):
        self.device = device
        debug_print(self.name, f"Setting up TouchSensor on device {device}...", "INFO")
        self.device_index = self.device
        # self.sensor = DMSensor(device)
        # self.sensor.reset()
        # time.sleep(1)  # 等待传感器稳定

    def _resize_map(self, data):
        data = np.asarray(data)
        if data.ndim != 2:
            return data

        target_width, target_height = self.output_size
        if data.shape[1] == target_width and data.shape[0] == target_height:
            return data

        return cv2.resize(data.astype(np.float32), self.output_size, interpolation=cv2.INTER_LINEAR)

    def get_force(self):
        depth = self.sensor_data[:,:,0,self.device_index]
        shear_x = self.sensor_data[:,:,1,self.device_index]
        shear_y = self.sensor_data[:,:,2,self.device_index]

        depth_deformation = self._resize_map(depth)
        shear_x_deformation = self._resize_map(shear_x)
        shear_y_deformation = self._resize_map(shear_y)
        
        mean_depth = np.mean(depth[:,:])
        #debug_print(self.name, f"Mean depth: {mean_depth}", "INFO")
        return {
            "depth": depth_deformation,
            "shear_x": shear_x_deformation,
            "shear_y": shear_y_deformation,
            "timestamp": int(time.monotonic() * 1e6)
        }

    def cleanup(self):
        pass
    def __del__(self):
        print("Cleaning up TouchSensor resources...")
        self.cleanup()

    
    

