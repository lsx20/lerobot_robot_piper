# 从全新 Ubuntu 复现当前 Piper + RH56F2 + LeRobot 功能

本文目标：在一台完全清空的 Ubuntu 机器上，从 0 配置到实现当前机器同等功能：

- Piper 机械臂 SDK 控制
- RH56F2 灵巧手 RS485 控制
- LeRobot 统一封装机械臂 + 灵巧手
- 固定点位抓球放球
- 电脑摄像头 MediaPipe 手势识别
- 浏览器可视化识别画面
- 键盘/抓娃娃模式遥控
- 健康检查、温度、电流、错误状态读取

当前已验证环境参考：

- Ubuntu 22.04 系内核：`6.8.0-124-generic`
- Python：`3.13.13`
- Conda：`26.3.2`
- `lerobot==0.5.1`
- `piper_sdk==1.0.0`
- `mediapipe==0.10.35`
- `opencv-python-headless==4.13.0.92`
- `pyserial==3.5`
- Piper CAN：`can0`, bitrate `1000000`
- RH56F2：`/dev/ttyUSB0`, baudrate `115200`, hand id `1`

## 1. Ubuntu 初始系统安装

建议安装 Ubuntu 22.04 LTS 或 24.04 LTS。安装系统时选择：

- 用户名可以仍用 `lsx`，这样路径最少改
- 勾选安装第三方驱动
- 分区至少留 80GB 空间

安装完成后先更新系统：

```bash
sudo apt update
sudo apt upgrade -y
sudo reboot
```

## 2. 基础软件与中文输入法

安装基础工具：

```bash
sudo apt update
sudo apt install -y \
  git curl wget unzip tar vim nano htop tree ripgrep \
  build-essential cmake pkg-config \
  net-tools iproute2 can-utils usbutils \
  python3 python3-pip python3-venv \
  libgl1 libglib2.0-0
```

安装中文输入法，推荐 Fcitx5：

```bash
sudo apt install -y fcitx5 fcitx5-chinese-addons fcitx5-config-qt im-config
im-config
```

在弹窗里选择 `fcitx5`，然后注销重登或重启：

```bash
sudo reboot
```

重启后打开：

```bash
fcitx5-configtool
```

添加中文输入法，比如 `Pinyin`。

## 3. 安装 Miniconda

```bash
cd ~/下载
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

安装过程中：

- 路径默认：`/home/$USER/miniconda3`
- 允许 `conda init`

重开终端后检查：

```bash
conda --version
python3 -V
```

建议创建专用环境，避免污染系统：

```bash
conda create -n piper_lerobot python=3.13 -y
conda activate piper_lerobot
python -m pip install -U pip setuptools wheel
```

如果 `mediapipe` 对 Python 3.13 安装失败，则退回 Python 3.12：

```bash
conda create -n piper_lerobot python=3.12 -y
conda activate piper_lerobot
python -m pip install -U pip setuptools wheel
```

## 4. 建工作区

```bash
mkdir -p ~/robot_ws
cd ~/robot_ws
```

## 5. 获取 Piper SDK

```bash
cd ~
git clone https://github.com/agilexrobotics/piper_sdk.git
cd ~/piper_sdk
python -m pip install -e .
```

检查：

```bash
python -c "import piper_sdk; print(piper_sdk.PiperSDKVersion.PIPER_SDK_CURRENT_VERSION)"
```

当前机器输出应为类似：

```text
PIPER_SDK_VERSION_1_0_0 (1.0.0)
```

## 6. 获取 LeRobot Piper 工程

基础拉取：

```bash
cd ~/robot_ws
git clone https://github.com/charlie8612/lerobot_robot_piper.git
cd ~/robot_ws/lerobot_robot_piper
python -m pip install -e .
```

注意：当前机器有一批本地新增脚本和 RH56F2 封装，未必全在 GitHub 远端。要完全复现当前功能，最稳方式是从当前机器复制备份包：

当前备份：

```text
/home/lsx/piper_latest_backup_20260630_161200.tar.gz
```

在旧机器打包或直接复制：

```bash
scp /home/lsx/piper_latest_backup_20260630_161200.tar.gz 用户名@新机器IP:/home/用户名/
```

新机器解压：

```bash
cd ~
tar -xzf piper_latest_backup_20260630_161200.tar.gz
```

把备份里的工程覆盖到工作区：

```bash
mkdir -p ~/robot_ws
cp -a ~/piper_latest_backup_20260630_161200/lerobot_robot_piper ~/robot_ws/
cp -a ~/piper_latest_backup_20260630_161200/piper_sdk ~/
cp -a ~/piper_latest_backup_20260630_161200/piper_ball_waypoints.json ~/
cp -a ~/piper_latest_backup_20260630_161200/test_piper_enable.py ~/
cp -a ~/piper_latest_backup_20260630_161200/test_rh56f2_hand.py ~/
cp -a ~/piper_latest_backup_20260630_161200/piper_recover_standby.py ~/
```

当前后来新增的健康检查脚本也复制：

```bash
scp /home/lsx/piper_read_health.py 用户名@新机器IP:/home/用户名/
```

然后重新安装 editable 包：

```bash
conda activate piper_lerobot
python -m pip install -e ~/piper_sdk
python -m pip install -e ~/robot_ws/lerobot_robot_piper
```

## 7. 安装 Python 依赖

```bash
conda activate piper_lerobot
python -m pip install -U pip
python -m pip install \
  lerobot==0.5.1 \
  mediapipe==0.10.35 \
  opencv-python-headless==4.13.0.92 \
  pyserial==3.5 \
  numpy==2.2.6
```

检查：

```bash
python -m pip show lerobot lerobot_robot_piper piper_sdk mediapipe opencv-python-headless pyserial numpy
```

## 8. MediaPipe 手部模型

当前工程模型路径：

```text
~/robot_ws/lerobot_robot_piper/assets/hand_landmarker.task
```

如果使用备份恢复，这个文件会一起恢复。检查：

```bash
ls -lh ~/robot_ws/lerobot_robot_piper/assets/hand_landmarker.task
```

如果没有，需要重新下载 MediaPipe Hand Landmarker `.task` 模型，并放到上面路径。

## 9. 硬件连接

Piper 机械臂：

- 电源接好
- 急停释放
- CAN 转 USB 接电脑
- CAN 线接机械臂控制口

RH56F2 灵巧手：

- 灵巧手供电
- RS485 A/B 接 USB-RS485 转换器
- USB-RS485 插电脑
- 当前默认端口：`/dev/ttyUSB0`

电脑摄像头：

- 可用内置摄像头
- 或 USB 摄像头

## 10. CAN 配置

插上 USB-CAN 后检查：

```bash
lsusb
ip link show
```

应看到 `can0`。拉起 CAN：

```bash
sudo ip link set can0 down
sudo ip link set can0 up type can bitrate 1000000
ip -details link show can0
```

正确状态应包含：

```text
can0: <NOARP,UP,LOWER_UP,ECHO>
```

可选：安装 can-utils 后监听：

```bash
candump can0
```

只监听不会控制机械臂。按 `Ctrl+C` 退出。

## 11. 串口权限

检查：

```bash
ls -l /dev/ttyUSB0
groups
```

临时授权：

```bash
sudo chmod a+rw /dev/ttyUSB0
```

永久授权：

```bash
sudo usermod -aG dialout $USER
sudo reboot
```

重启后再次检查：

```bash
groups
```

应包含 `dialout`。

## 12. 第一阶段：只读检查机械臂健康

不要先运动。先读状态：

```bash
conda activate piper_lerobot
python ~/piper_read_health.py
```

重点看：

```text
Arm Status: NORMAL
Error Code: 0
Enable status
motor_temp
driver_temp
motor_overheat
driver_overheat
overcurrent
stall
driver_error
```

若出现过温、过流、堵转、错误，先不要运动。

## 13. 第二阶段：测试 RH56F2 灵巧手

```bash
conda activate piper_lerobot
python ~/test_rh56f2_hand.py --port /dev/ttyUSB0 --id 1
```

默认参数：

- port：`/dev/ttyUSB0`
- baudrate：`115200`
- hand id：`1`

脚本会读取：

- `angleAct`
- `errCode`
- `status`
- `temp`

然后让你选择 actuator 小步测试。

## 14. 第三阶段：测试 Piper 小幅运动

只在空旷区域、手扶机械臂的情况下执行：

```bash
conda activate piper_lerobot
python ~/test_piper_enable.py
```

按提示输入 `MOVE` 后会做小幅动作。测试完成后注意是否提示 disable。机械臂带灵巧手时，不建议突然失能，失能前要托住。

## 15. LeRobot 工程安装检查

```bash
conda activate piper_lerobot
cd ~/robot_ws/lerobot_robot_piper
python - <<'PY'
from lerobot_robot_piper import PiperRH56F2Follower, PiperRH56F2FollowerConfig
print(PiperRH56F2Follower)
print(PiperRH56F2FollowerConfig)
PY
```

如果能打印类名，说明 LeRobot 插件导入正常。

## 16. 固定点位抓球放球

点位文件：

```text
~/piper_ball_waypoints.json
```

检查：

```bash
cat ~/piper_ball_waypoints.json
```

运行：

```bash
conda activate piper_lerobot
cd ~/robot_ws/lerobot_robot_piper
python scripts/pick_place_ball.py
```

可用功能：

```bash
python scripts/pick_place_ball.py --hand-test
python scripts/pick_place_ball.py --capture home
python scripts/pick_place_ball.py --capture pre_grasp
python scripts/pick_place_ball.py --capture grasp
python scripts/pick_place_ball.py --capture pre_place
python scripts/pick_place_ball.py --capture place
python scripts/pick_place_ball.py --sync-current-joint6
```

退出程序时必须托住机械臂，并按提示输入大写 `D` 才会失能电机。

## 17. 键盘遥控

完整调试模式：

```bash
conda activate piper_lerobot
cd ~/robot_ws/lerobot_robot_piper
python scripts/keyboard_teleop_piper_hand.py --line-mode
```

进入后输入：

```text
RUN
```

常用键：

```text
p       打印当前状态
m       打印 Piper 状态
1..6    选择关节
[ / ]   当前关节负/正方向小步移动
o / c   灵巧手张开/合拢
z / x   灵巧手小步张开/合拢
q       退出
```

实时按键模式：

```bash
python scripts/keyboard_teleop_piper_hand.py
```

## 18. 抓娃娃模式

保守模式：

```bash
conda activate piper_lerobot
cd ~/robot_ws/lerobot_robot_piper
python scripts/keyboard_teleop_piper_hand.py --claw-mode
```

按键：

```text
a / d   J1 左右
w / s   J2 + J3 + J5 联动前伸/后退
o / c   张手/合手
z / x   小步张手/合手
p       打印状态
m       打印 Piper 状态
q       退出
```

当前安全限制：

- 默认步长：`0.3 deg`
- 抓娃娃速度：`8%`
- J1 最大偏离启动点：`20 deg`
- J2 最大偏离启动点：`8 deg`
- J3 最大偏离启动点：`4 deg`
- J5 最大偏离启动点：`8 deg`
- J4/J6 不发命令

如果 `w/s` 方向反了：

```bash
python scripts/keyboard_teleop_piper_hand.py --claw-mode --invert-claw-reach
```

如果 J5 发热，先禁用 J5 联动：

```bash
python scripts/keyboard_teleop_piper_hand.py --claw-mode --claw-j5-gain 0
```

## 19. 摄像头与手势识别

先检查摄像头：

```bash
ls /dev/video*
```

如果没有摄像头设备，检查 BIOS、权限或 USB 摄像头。

视觉只读，不控制机器人：

```bash
conda activate piper_lerobot
cd ~/robot_ws/lerobot_robot_piper
python scripts/vision_gesture_pick_place_lerobot.py --follow --web-viewer
```

浏览器打开：

```text
http://127.0.0.1:8765
```

当前环境使用 `opencv-python-headless`，不要依赖 `cv2.imshow`，用 `--web-viewer` 看画面。

## 20. 视觉跟踪 + LeRobot 执行

先用终端触发，避免误识别直接运动：

```bash
conda activate piper_lerobot
cd ~/robot_ws/lerobot_robot_piper
python scripts/vision_gesture_pick_place_lerobot.py \
  --execute-follow \
  --web-viewer \
  --terminal-trigger
```

更保守的小范围：

```bash
python scripts/vision_gesture_pick_place_lerobot.py \
  --execute-follow \
  --web-viewer \
  --terminal-trigger \
  --follow-max-j1-offset 10 \
  --follow-max-j2-offset 8
```

## 21. 常见问题

### can0 不存在

检查 USB-CAN：

```bash
lsusb
ip link show
```

重新插拔 USB-CAN。

### can0 是 DOWN

```bash
sudo ip link set can0 down
sudo ip link set can0 up type can bitrate 1000000
```

### /dev/ttyUSB0 权限不足

```bash
sudo chmod a+rw /dev/ttyUSB0
sudo usermod -aG dialout $USER
```

永久生效需要注销重登或重启。

### OpenCV imshow 报错

不要用 GUI 窗口，使用：

```bash
--web-viewer
```

然后打开：

```text
http://127.0.0.1:8765
```

### 机械臂发热

立刻停止运动脚本：

```text
q
```

或：

```text
Ctrl+C
```

然后读健康状态：

```bash
python ~/piper_read_health.py
```

重点看过温、过流、堵转、驱动错误。不要在明显发烫时继续测试。

## 22. 推荐验证顺序

严格按这个顺序：

1. `lsusb`
2. `ip link show`
3. `sudo ip link set can0 up type can bitrate 1000000`
4. `python ~/piper_read_health.py`
5. `python ~/test_rh56f2_hand.py --port /dev/ttyUSB0 --id 1`
6. `python ~/test_piper_enable.py`
7. `python scripts/keyboard_teleop_piper_hand.py --line-mode`
8. `python scripts/keyboard_teleop_piper_hand.py --claw-mode`
9. `python scripts/pick_place_ball.py --hand-test`
10. `python scripts/pick_place_ball.py`
11. `python scripts/vision_gesture_pick_place_lerobot.py --follow --web-viewer`
12. `python scripts/vision_gesture_pick_place_lerobot.py --execute-follow --web-viewer --terminal-trigger`

## 23. 当前关键文件清单

```text
~/piper_sdk
~/robot_ws/lerobot_robot_piper
~/robot_ws/lerobot_robot_piper/scripts/pick_place_ball.py
~/robot_ws/lerobot_robot_piper/scripts/keyboard_teleop_piper_hand.py
~/robot_ws/lerobot_robot_piper/scripts/vision_gesture_pick_place_lerobot.py
~/robot_ws/lerobot_robot_piper/src/lerobot_robot_piper/piper_rh56f2_follower.py
~/robot_ws/lerobot_robot_piper/src/lerobot_robot_piper/config_piper_rh56f2_follower.py
~/robot_ws/lerobot_robot_piper/src/lerobot_robot_piper/rh56f2_hand.py
~/robot_ws/lerobot_robot_piper/assets/hand_landmarker.task
~/piper_ball_waypoints.json
~/test_piper_enable.py
~/test_rh56f2_hand.py
~/piper_recover_standby.py
~/piper_read_health.py
```

## 24. 最重要安全规则

- 不确认姿态时，不运行自动抓取
- 不确认温度时，不连续长时间抓娃娃模式
- 退出使能前托住机械臂和灵巧手
- 不要绕过退出确认；失能电机前必须先托住机械臂并输入大写 `D`
- 出现发热、异响、抖动、砸落趋势，立刻停脚本并断电冷却
- 每次迁移到新电脑，先只读健康检查，再运动
