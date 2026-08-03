import time
import numpy as np
import pyrealsense2 as rs

from sensor.base_vision_sensor import BaseVisionSensor
from utils.base.data_handler import debug_print
import cv2

def find_device_by_serial(devices, serial):
    for i, dev in enumerate(devices):
        if dev.get_info(rs.camera_info.serial_number) == serial:
            return i
    return None

class RealsenseSensor(BaseVisionSensor):
    def __init__(self, name):
        super().__init__(name)
        self.enable_depth = False
        self.is_jpeg = False
        self.context = None
        self.devices = None
        self.pipeline = None
        self.config = None
    
    def connect(self, device, is_jpeg=False, enable_depth = False):
        self.enable_depth = enable_depth
        self.is_jpeg = is_jpeg
        
        self.context = rs.context()
        self.devices = list(self.context.query_devices())
            
        if not self.devices:
            raise RuntimeError("No RealSense devices found")
            
        serial = device
        device_idx = find_device_by_serial(self.devices, serial)
        if device_idx is None:
            raise RuntimeError(f"Could not find camera with serial number {serial}")
            
        self.pipeline = rs.pipeline()
        self.config = rs.config()
            
        self.config.enable_device(serial)
        # self.config.disable_all_streams()
        # Enable color stream only
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        if enable_depth:
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            
        try:
            self.pipeline.start(self.config)
            debug_print(self.name, f"Started camera: {self.name} (SN: {serial})", "INFO")
        except RuntimeError as e:
            raise RuntimeError(f"Error starting camera: {str(e)}")

    def get_information(self):
        image = {}
        frame = self.pipeline.wait_for_frames()

        if "color" in self.collect_info:
            color_frame = frame.get_color_frame()
            if not color_frame:
                raise RuntimeError("Failed to get color frame.")
            tmp_img = np.asanyarray(color_frame.get_data())
            if self.is_jpeg:
                # 不需要转换为 BGR 格式，因为 RealSense 输出的已经是 BGR
                image["color"] = cv2.imencode('.jpg', tmp_img)[1]
            else:
                image["color"] = tmp_img[:,:,::-1]

        if "depth" in self.collect_info:
            if not self.enable_depth:
                debug_print(self.name, f"should use set_up(enable_depth=True) to enable collecting depth image","ERROR")
                raise ValueError
            else:       
                depth_frame = frame.get_depth_frame()
                if not depth_frame:
                    raise RuntimeError("Failed to get depth frame.")
                depth_image = np.asanyarray(depth_frame.get_data()).copy()
                image["depth"] = depth_image
        
        return image

    def disconnect(self):
        try:
            self.pipeline.stop()
        except Exception as e:
            debug_print(self.name, f"Pipeline stop failed: {e}", "ERROR")

if __name__ == "__main__":
    cam = RealsenseSensor("test")
    cam.connect("419522071856")
    cam.set_collect_info(["color"])
    cam_list = []
    for i in range(1000):
        print(i)
        data = cam.get_image()
        cam_list.append(data)
        time.sleep(0.1)