# 使用手册

## 自定义机器人

1. 在controller中接入自定义的机械臂，在sensor中接入自定义的传感器
2. 在src/robot中组装机器人，并设置采集数据类型
3. 在robot/\_\_init\_\_.py包中注册该机器人

## 数据采集

### 开始采集

从第start_index个episode开始采集数据，采集数据轮数在config.yml中定义

```bash
bash scripts/collect.sh task-name robot-config-name start_index
```

注意：config.yml中robot["type"]需要与机器人类名对应

## 数据回放

回放第replay_index个episode

```bash
bash scripts/replay.sh task-name robot-config-name replay_index
```

## 可视化

```bash
uv run pipeline/rerun_visual.py path/to/hdf5
```