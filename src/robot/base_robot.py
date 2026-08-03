from utils.base.data_handler import debug_print

class Robot:
    def __init__(self, robot_config: dict) -> None:
        self.robot_config = robot_config
        self.name = self.robot_config.get("name", "unknown_robot")
        self.bias = self.robot_config.get("bias", None)
        self.ready = False
        
        self.controllers = {}
        self.sensors = {}

        self.controller_data = {}
        self.sensor_data = {}

    # 机器人内部状态同步函数，子类可以根据需要实现具体的同步逻辑
    def sync(self):
        pass

    def is_ready(self):
        return self.ready

    def get_controller_obs(self):
        if self.controllers is not None:
            for controller in self.controllers.values():
                data = controller.get()
                self.controller_data[controller.name] = data
        return self.controller_data.copy()

    def get_sensor_obs(self):
        if self.sensors is not None:
            for sensor in self.sensors.values():
                data = sensor.get()
                self.sensor_data[sensor.name] = data
        return self.sensor_data.copy()

    def get_obs(self):
        controller_obs = self.get_controller_obs()  # 更新 controller_data
        sensor_obs = self.get_sensor_obs()      # 更新 sensor_data

        return [controller_obs, sensor_obs]

    def disconnect(self):
        for controller in self.controllers.values():
            controller.disconnect()

        debug_print(self.name, "All controllers have been disconnected.", "INFO")

        for sensor in self.sensors.values():
            sensor.disconnect()

        debug_print(self.name, "All sensors have been disconnected.", "INFO")

    def take_action(self, action: dict):
        if action is None:
            return

        for controller_name, controller_action in action.items():
            self.controllers[controller_name].take_action(controller_action)

    def connect(self):
        raise NotImplementedError("Subclasses should implement this method.")
    def get_standard_obs(self):
        raise NotImplementedError("Subclasses should implement this method.")
    def reset(self):
        raise NotImplementedError("Subclasses should implement this method.")