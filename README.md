# reBot Arm B601-RS 手掌跟随 + 捏合抓取 一体项目

把原来的两个独立项目合并为一个：

| 原项目 | 功能 | 合并后 |
| --- | --- | --- |
| `rebot_gripper_control` | MediaPipe 捏合距离 → 夹爪开合 | `palm_follow_grasp.py` 的抓取通道 |
| `rebot_palm_follow` | 掌心 3D → 机械臂固定姿态跟随 | `palm_follow_grasp.py` 的跟随通道 |

**一台 Gemini 2 相机、一次 MediaPipe 检测、一条 CAN 总线**，同时完成：
手掌移动 → 机械臂跟随（保持固定距离与姿态）；拇指食指捏合 → 夹爪开合。

> 日常调参只改 `config/follow.yaml`（跟随）和 `config/grasp.yaml`（夹爪）两个文件；
> 命令行参数临时覆盖。完整的技术原理与踩坑记录见 [技术手册.md](技术手册.md)。

## 最终版升级：运动更跟手，静止不再被 lead 推动

本版保留 v6 已验证有效的帧序号、单调时间戳 KF、真实 dt、深度冻结修复和
陈旧目标看门狗，并根据两组实机 A/B 结果重做 lead 控制。主入口、相机、标定、
MIT 控制、单 CAN 主机、J7 第 7 通道及安全回零方式不变。

1. **自适应 lead 三重门控**：额外 lead 由“KF 速度、剩余误差、速度/误差方向”
   共同决定。明确运动时恢复预测；低速和近目标自动衰减；反向纠偏立即清零。
   撤权时间常数 15ms、恢复时间常数 25ms，避免静止边界 `0↔15ms` 拍打。
2. **年龄补偿与额外 lead 分离**：目标为 `KF位置 + 速度×age + 速度×lead×权重`。
   关闭额外 lead 不会误删观测年龄补偿；静止锁存、冻结、跳变剔除和丢手仍优先。
3. **默认采用实机稳定基线**：`kf_fast_start_enable: false`，同时恢复
   `kf_lead_ms: 8.0` 并启用门控，不再用永久关闭 lead 换取静止稳定。
4. **水平径向工作空间**：在三轴软墙之外保留 `hypot(X,Y) ≤ 0.42m` 安全约束，
   只削水平向外速度，保留向内、切向和全部 Z 运动；不像三维球墙那样在高姿态
   过度压缩空间。超出水平半径再 5cm 自动回 HOME。
5. **启动前配置校验**：非有限数、危险速度/增益、错误 lead 阈值、失效看门狗、
   非法工作空间或超过 0.42m 的半径，会在相机和机械臂初始化之前直接报错。
6. **诊断增强**：DBG 显示 `实际lead/原始lead`、权重 `w`、KF 速度 `kv`、
   门控误差 `e` 和锁存 `latch`，可直接判断静止时是谁在推目标。

实机确认静止稳定后，本修订版将 lead 全量介入距离由 55mm 提前到 45mm、恢复
时间常数由 40ms 缩短到 25ms；静止阈值、反向清零、增益和速度上限均不变。
离线验证结果为：**21 项单元测试通过**；完整假硬件主程序通过
“静止→移动→回中心停手”，末段连续 15 个控制周期关节指令为零，统一安全停止
被调用。离线测试不替代实机低速验收。

稳定回退组合（保留其他修复，仅关闭额外 lead 和快速解锁）：

```yaml
kf_fast_start_enable: false
kf_lead_ms: 0
```

`kf_lead_adaptive: false` 仅用于复现旧 lead 行为，不建议实机使用。
不要为排查晃动关闭深度保护、软墙或安全回零。

离线验证（不连接相机/CAN）：

```bash
python tools/test_servo_upgrades.py
python tools/offline_smoke_test.py
```

## 合并时的关键设计

1. **夹爪改走机械臂控制器的第 7 通道**（最重要）
   旧 gripper 项目自己开一个 motorbridge 连接直连电机 7；若与跟随程序同时跑，
   两个 CAN 主机会在 can0 上争抢电机 7。合并后夹爪目标角由
   `rebot_adapter.set_gripper_deg()` 设置，随 rebot_control 的 100Hz 关节指令
   一同下发，单一 CAN 主机，从根上避免总线冲突。
   - 夹爪角度单位统一为**度**（rebot_control 约定），不再做弧度换算。
   - 夹爪最大速度由适配层统一设为 500°/s，角度软限幅仍为 -100° ~ +400°。

2. **一次检测，两处复用**
   掌心（9 号关键点）用于跟随，拇指尖(4)—食指尖(8) 距离用于夹爪，
   取自同一个 HandLandmarker 结果，不增加检测开销。

3. **安全约束全部保留并叠加**
   跟随侧：虚拟软墙/硬墙、可操作度奇异保护、丢手保持、Esc 急停、
   退出时「缓慢回零→到位→失能」（防止自由下落）；
   夹爪侧：启动时以电机当前角为平滑起点（不猛动）、映射两端截断、
   One Euro 滤波防抖 + 死区、适配层角度硬限幅（-100° ~ +400°）兜底。

## 项目结构

```
rebot_palm_follow_grasp/
├── palm_follow_grasp.py           ★ 主程序（跟随 + 夹爪，一个脚本）
├── palm_follow_fixed_orientation.py   仅跟随版（原跟踪主程序，保留作对照调试）
│
├── gemini2_camera.py              相机层（Orbbec SDK，RGB+深度硬件对齐）
├── palm_detection.py              检测层（掌心跟踪 + 捏合距离，共用一次检测）
├── rebot_adapter.py               机械臂适配层（速度伺服 + FK/雅可比 + J7 夹爪通道）
├── servo_timing.py                时间戳/真实dt/自适应lead/水平径向软墙工具
│
├── config/                        所有配置与标定产物
│   ├── follow.yaml                ★ 跟随伺服参数（距离/增益/死区/前馈/虚拟墙，调参改这里）
│   ├── grasp.yaml                 捏合→夹爪映射与开合标定值
│   ├── hand_eye_calibration.json  手眼标定结果（本项目实测，跟随必需）
│   ├── joint_offset.json          关节零位修正（本项目实测，适配层自动加载）
│   ├── home_pose.json             示教 HOME 位姿（本项目实测，启动优先加载）
│   ├── viewpoint.json             联合标定示教点（重新标定时免重复示教）
│   ├── handeye_joint_samples.npz  联合标定原始样本（重求解免重新采样）
│   └── hand_landmarker.task       MediaPipe 手掌模型
│
├── tools/                         标定与诊断工具（按需运行，日常不用碰）
│   ├── gripper_j7_calibrate.py    夹爪开合标定（键盘点动 J7，结果写回 grasp.yaml）
│   ├── teach_home.py              HOME 位姿示教
│   ├── hand_eye_calibration.py    手眼标定（手动棋盘格 JOG 采样）
│   ├── hand_eye_calibration_auto.py   手眼标定（ArUco + 关节空间自动扰动）
│   ├── handeye_joint_calib.py     手眼+关节零位联合标定（精度最高，推荐）
│   ├── hand_eye_solver.py         手眼标定求解器（OpenCV 5.0 移除了 calibrateHandEye）
│   ├── calib_diagnose.py          标定体检（不标定，只定位问题）
│   ├── camera_intrinsic_check.py  相机内参核对
│   ├── test_servo_upgrades.py     升级逻辑离线单元测试
│   └── offline_smoke_test.py      假相机/假机械臂完整主程序冒烟测试
│
├── requirements.txt
├── 技术手册.md                     项目全部技术与技巧汇总（感知/标定/控制/稳定性/安全）
└── README.md
```

## 快速开始

### 1. 环境

```bash
cd rebot_palm_follow_grasp
pip install -r requirements.txt
git clone https://github.com/LAN-GER/rebot_control.git        # 运动下发（源码仓库）
git clone https://github.com/Seeed-Projects/reBotArm_control_py.git  # URDF
```

确认 `rebot_adapter.py` 顶部的 `REBOT_CONTROL_PATH` 与 `URDF_PATH` 指向本机实际路径。

### 2. CAN 总线（每次插拔 USB-CAN 后都要执行）

```bash
#1. 套件里是 PCAN-USB，通常应该直接出现 can0 或 can1
sudo modprobe peak_usb
ip -br link

#2. 如果出现 can0，再设置 bitrate
sudo ip link set can0 down 2>/dev/null
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up

#3. 把 ROS 路径从环境变量里剔除
export PYTHONPATH=$(echo $PYTHONPATH | tr ':' '\n' | grep -v /opt/ros | paste -sd:)
export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -v /opt/ros | paste -sd:)
#4. 验证加载的是 conda 环境里的 pinocchio
python -c "import pinocchio as pin; print(pin.__version__, pin.__file__)"
```

### 3. 分步验证（强烈建议按此顺序）

```bash
# 第 0 步：纯视觉演练——不碰硬件，验证掌心检测 + 捏合映射
python palm_follow_grasp.py --no-arm

# 第 1 步：仅夹爪——手臂保持不动，捏合验证夹爪开合方向与范围
python palm_follow_grasp.py --no-follow

# 第 2 步：仅跟随——低速假目标测试（先别上真人手）
python palm_follow_grasp.py --no-gripper --gain 1.5 --max-vel 0.10

# 第 3 步：完整模式
python palm_follow_grasp.py

# 如若机械臂故障，先关闭电源重启，等电源响声停止后，清除电机故障指令
motorbridge run --vendor robstride --channel can0 --motor-id 1 --feedback-id 0xFD --mode clear-error
```

常用启动参数（两个主程序都支持）：

```bash
python palm_follow_grasp.py --distance 0.35   # 手掌与相机的跟随距离（米），
                                              # 建议 0.20~0.50，默认 0.30
python palm_follow_grasp.py --fps 60          # 相机 60fps，感知延迟砍半
python palm_follow_grasp.py --gain-x 1.4      # 水平跟随增益（左右振荡时调低）
```

操作：`g`/空格 启停跟随，`q` 退出，`Esc` 急停回零。

**最终版首次实机验收**：先用上面的 `--no-gripper --gain 1.5 --max-vel 0.10`。
清空工作区并准备随时按 q；依次做静止 30 秒、慢速同向移动、匀速移动、
急停、换向、短时丢手和贴墙返回。确认低速全部正常后再用默认参数。
静止时 DBG 的 `lead` 应回到 0；若仍晃，直接采用上面的稳定回退组合，
不要继续提高增益或速度。任何改参一次只改一项并重新做静止与急停测试。

### 4. 标定（环境变动后才需要重做）

```bash
python tools/gripper_j7_calibrate.py     # 夹爪开合标定 → config/grasp.yaml
python tools/teach_home.py               # 示教顺手的工作 HOME → config/home_pose.json
python tools/handeye_joint_calib.py      # 手眼+零位联合标定（需先有 config/viewpoint.json，
                                         #   由 tools/hand_eye_calibration_auto.py 示教生成）
```

随附的 `config/hand_eye_calibration.json` / `config/joint_offset.json` /
`config/home_pose.json` 是本机实测结果；**换了相机安装位置或重设过电机零点后必须重标定**。

## 配置文件与参数优先级

项目有两份配置文件，**日常调参只改这两个文件，不用动代码**：

| 文件 | 管什么 |
| --- | --- |
| `config/follow.yaml` | 跟随伺服全部参数：距离、增益、速度上限、死区/迟滞、前馈、滤波、虚拟墙 |
| `config/grasp.yaml` | 捏合→夹爪映射：像素阈值、滤波、开合标定角度、冻结开关 |

**参数优先级：命令行参数 > yaml 配置 > 代码内置默认值。**

```bash
python palm_follow_grasp.py                    # 全部用 follow.yaml 的值
python palm_follow_grasp.py --distance 0.35    # 仅距离被覆盖，其余仍走 yaml
python palm_follow_grasp.py --follow-cfg my.yaml   # 换一份配置文件
```

## 调参：跟随（config/follow.yaml）

每个参数在 yaml 里都有行内注释，这里给"按症状调"的速查（新架构）：

| 症状 | 改哪个 | 方向 |
| --- | --- | --- |
| 停手后振荡（含伸展位） | 先确认 `kf_lead_adaptive: true`；仍晃用稳定回退 `kf_lead_ms: 0` | 再查 DBG 的 `w/kv/e/latch`，不要先提增益 |
| 停手后上下小幅振荡 | `position_deadband`（须 > BS 深度噪声 ±20mm） | 0.017→0.022→0.025 |
| 死区状态边界闪烁（停手瞬间一抽一抽） | `kf_ff_exit_frames` | 5→7（起步略钝可降回） |
| 跟随偏肉 | `gain` → `smooth_alpha` → `kf_q` | 2.2→2.35 ／ 0.45→0.48 ／ 1.5→1.8 |
| 跟随发冲、关节速度频繁冲 2~3 | `kf_q` → `smooth_alpha` | 2.0→1.5 ／ 0.52→0.45 |
| 静止时仍有微抖 | `depth_oe_min_cutoff` → `position_deadband` | 1.0→0.7→0.5 ／ →0.025 |
| 丢手偏多 | 先补光、确认 `color_exposure_ms: 6.0`；手别挥出画面 | 硬件手段优先于参数 |
| 老架构（`kf_enable: false`）回摆/振荡 | `ff_taper_dist` → `vel_feedforward` → `smooth_alpha` | 0.07→0.10 ／ 0.7→0.5 ／ 0.38→0.55 |

注意：`max_vel`（末端线速度）和适配层 `MAX_JOINT_SPEED`（关节限速）是两道串联的墙，
只提末端限速而关节限速不动，左右跟随不会有任何变化——先提 J1。

## 调参：夹爪（config/grasp.yaml）

| 参数 | 含义 | 建议 |
| --- | --- | --- |
| `pinch.closed_px` / `open_px` | 捏合/张开时两指尖像素距离 | 手离相机远则整体调小，按实测改 |
| `pinch.filter_type` | 夹爪目标滤波：`one_euro`（默认，推荐）/ `ema` | one_euro = 静止强滤波、快动自动放开，两全其美 |
| `pinch.min_cutoff_hz` / `beta` | One Euro 参数 | 跟随慢/肉：beta 0.010→0.015~0.020；静止仍抖：min_cutoff 2.0→1.5 |
| `pinch.smoothing` | 仅 `filter_type=ema` 时的平滑系数 | 0.35 是速度与平滑的甜点；更大更跟手但更抖 |
| `pinch.deadband_deg` | 夹爪目标死区 | 指尖是噪声最大的关键点，建议 ≥2；夹爪仍抖就加到 3 |
| `gripper.hold_while_moving` | 手臂运动时冻结夹爪 | **默认 true，跟随稳定性关键**，见下节 |
| `gripper.move_speed_threshold` | 「手臂在运动」判定阈值 (rad/s) | 一般不动；冻结解除太迟钝可调小到 0.03 |
| `gripper.closed_deg` / `open_deg` | 夹爪闭合/张开角度 | **用标定工具实测，勿手猜** |

### 关于跟随稳定性（为什么默认冻结夹爪）

跟随链路与旧单独项目逐字节一致，但合并后多了一个**物理干扰源**：
相机装在手腕上（eye-in-hand），而夹爪电机 J7 就在相机旁边。
跟随过程中手指的微动会让夹爪持续小动作，振动直接抖动画面的掌心
3D 估计——这是合并后跟随「不如以前稳」的最可能原因。

默认开启 `hold_while_moving`：手臂运动时夹爪冻结（HUD 显示
`[FROZEN]`），手停下后夹爪才跟随捏合——符合「移动到位 → 再抓取」的
自然节奏。若你确认需要「边跟随边捏合」，把它设为 `false` 即可。

自查方法：跑 `python palm_follow_grasp.py --no-gripper`，
此时 J7 全程锁死，与旧单独项目行为完全等价——若稳定性恢复，
即可确认是夹爪干扰而非跟随链路问题。

## 提升跟随速度与精度

左右跟随的物理上限 ≈ J1 偏航限速 × 臂展，高速滞后来自感知+控制延迟。
按见效程度排序的调优手段：

| 手段 | 怎么用 | 预期效果 |
| --- | --- | --- |
| 相机 60fps | `python palm_follow_grasp.py --fps 60` 或 yaml `follow.fps: 60` | 感知延迟砍半，高速跟随立竿见影（需 CPU 跑得动 MediaPipe@60fps） |
| 目标速度前馈 | 已内置（`follow.vel_feedforward: 0.7`） | 消除「手快臂慢」的高速滞后；若出现超前摆动调到 0.5 |
| 提 J1 限速 | `rebot_adapter.py` 的 `MAX_JOINT_SPEED`（已从 0.6 提到 0.8） | 左右跟随上限 ≈ 0.8×臂展 ≈ 0.28 m/s；可逐步试 1.0→1.2，观察电机温度 |
| 提水平增益 | `--gain-x 2.2`（配前馈后裕量更大） | 响应更快，出现振荡就退回 |
| 提末端限速 | `--max-vel 0.40` 或 yaml `follow.max_vel` | 仅在关节限速提过之后才有意义 |
| 提高定位精度 | `follow.position_deadband` 0.015 → 0.010 | 静止时贴得更近；太小会围绕目标微抖 |
| 检测滤波 | `palm_detection.py` 掌心检测已改用 One Euro Filter（`PALM_FILTER="one_euro"`，min_cutoff=1.0, beta=0.02） | 静止更稳、快速移动不拖尾；一般不用动 |

### 旧 gripper 项目调速建议在本项目的对应情况

| 旧建议 | 本项目状态 |
| --- | --- |
| smoothing 0.15→0.35 | 已应用（`pinch.smoothing: 0.35`，ema 模式生效） |
| One Euro Filter | **已应用**：掌心检测（palm_detection.py）+ 捏合目标（默认 `filter_type: one_euro`）两处都换上了 |
| 相机 60fps / MJPG | 已支持：`--fps 60`；Gemini 2 用 pyorbbecsdk 取流，帧率由 SDK 控制，无 CAP_PROP_BUFFERSIZE 概念 |
| `CAP_PROP_BUFFERSIZE=1` | 不适用（非 cv2.VideoCapture），SDK 内部已是最新帧语义 |
| MediaPipe `model_complexity=0` | 不适用：新版 HandLandmarker tasks API 无此参数，模型已是最快的手部模型 |
| MIT 模式 kp/kd + 速度前馈 | kp/kd 在 rebot_control 驱动侧配置；视觉侧已内置目标速度前馈（`VEL_FEEDFORWARD=0.7`） |
| CAN 1Mbps / 48V 供电 | **仍然有效**，属硬件前提，见下方安全 checklist |

注意：`--max-vel` / `follow.max_vel`（末端线速度）和 `MAX_JOINT_SPEED`（关节限速）是两道串联的墙，
只提末端限速而关节限速不动，左右跟随不会有任何变化——先提 J1。

跟随侧全部参数都在 `config/follow.yaml`（增益/死区/前馈/滤波/虚拟墙），
命令行可用 `--distance / --gain / --gain-x / --max-vel / --fps` 临时覆盖，
或 `--follow-cfg` 换整份配置。

## 故障排查

| 症状 | 处理 |
| --- | --- |
| **到位后振荡几下才停稳**（手已停，臂冲过目标再拉回反复几次） | 回环振荡：手停瞬间前馈估计带惯性，把臂推过目标。已内置**前馈近端渐弱**（`follow.ff_taper_dist: 0.07`，误差 <7cm 时前馈线性退出）。调参顺序：① `ff_taper_dist` 加大到 0.10（更早开始退出前馈）② `vel_feedforward` 0.7→0.5（巡航提速变弱但更稳）③ `smooth_alpha` 0.38→0.55（减小速度平滑滞后，响应更"脆"，过冲更小） |
| **手掌不动时机械臂持续小幅振荡** | 已内置三重修复：① 深度通道 One Euro 滤波（`follow.depth_oe_min_cutoff/beta`，静止时 z3d 抖动 6mm→0.5mm）② 死区迟滞（`follow.deadband_hysteresis: 1.5`，进入死区后误差要超过 1.5×死区才重启）③ 前馈静止门控（`follow.ff_min_speed: 0.05`，手不动时前馈置零，避免深度噪声被 ÷0.02s 放大成假速度注入）。若仍有轻微抖动：`depth_oe_min_cutoff` 1.0→0.7，或 `position_deadband` 0.015→0.020 |
| 左右快速移动时机械臂振荡、先反向后拉回 | 水平跟随靠底座 J1 偏航（惯量大、限速低），增益过高会饱和振荡。调低水平增益 `--gain-x 1.2`，或减小 `--max-vel 0.20`。程序已内置分方向增益 + tanh 软饱和 + 反转清惯性三重缓解 |
| 夹爪不动 | `--no-follow` 单独验证；确认 grasp.yaml 中 `gripper.enabled: true`；按 `tools/gripper_j7_calibrate.py` 重标定 |
| 夹爪开合方向反了 | 把 `closed_deg` / `open_deg` 对调，或检查是否重设过零点 |
| 夹爪响应慢/肉 | 适配层 `GRIPPER_SPEED_DEG_S` 适当加大；确认 rebot_control 的 J7 MIT 增益 |
| 机械臂朝反方向跑 | 先把 `SERVO_DIRECTION` 改为 -1.0 试；不行就重做手眼标定 |
| 跟随一直偏一个固定量 | 重做联合标定的不动点验证（分散度 < 2cm 合格） |
| 深度经常为 0 | 手离相机 < 15cm（Gemini2 近距盲区），保持 d ≥ 0.25m |
| 电机不响应 | CAN 没起来/波特率不对：重跑 `ip link` 命令，确认 1 Mbps |

## 安全 checklist（每次运行前）

- [ ] 底座固定、相机支架拧紧、线缆有余量
- [ ] CAN 已配置（1 Mbps）、电机温度正常
- [ ] 三轴虚拟墙已按实际桌面修改；`workspace.max_radius` 不超过 0.42m
- [ ] 首次/改参后：`--no-arm` → `--no-follow` → `--no-gripper` 低速 → 完整模式
- [ ] Esc 急停有效（rebot_control 内置，触发平滑回零）
- [ ] 工作半径内无人；**眼睛看机械臂，不看屏幕**
- [ ] ⚠️ 任何退出路径都必须走完「保持→缓慢回零→到位→失能」，
      严禁在非零位直接失能电机（屈肘姿态失能 = 自由下落，有真实事故教训）

## 致谢与许可

- 跟随控制数学移植自 [HighTorque-Robotics/Panthera_HT_SDK_Extensions](https://github.com/HighTorque-Robotics/Panthera_HT_SDK_Extensions)
- 夹爪手势映射思路移植自 [reBotArm_develop_hjx](https://github.com/HJX-exoskeleton/reBotArm_develop_hjx)（MIT License）
- 电机控制依赖 [motorbridge](https://github.com/tianrking/MotorBridge) 与 [rebot_control](https://github.com/LAN-GER/rebot_control)
- 硬件参数参考 [Seeed reBot Arm B601-RS 官方 Wiki](https://wiki.seeedstudio.com/rebot_arm_b601_rs_mit_control/)

MIT License.

## 新/老伺服架构切换

`config/follow.yaml` 中 `kf_enable` 一个键即可切换，重启脚本生效：

- `kf_enable: true` —— 新架构（卡尔曼恒速滤波）：速度估计更干净，可叠加
  lead 外推（`kf_lead_ms`）与静止目标锁存（`kf_hold_tol`），跟随更跟手；
- `kf_enable: false` —— 老架构（帧间差分+EMA）：无 lead、无锁存，
  运动风格更温润，`ff_smooth` / `ff_min_speed` / `ff_taper_dist` 生效。

两套架构共用全部安全机制（软墙 / 伸展衰减 / 加速度限制 / 丢失外推 /
深度保护链），切换不影响安全性。全部可调参数均在 follow.yaml 内；
yaml 末尾【老架构配套参数组】注释里保留了一套实测稳定的老架构参数。

## 本版升级技术要点（最终版）

按实机日志逐轮迭代出的新架构完整技术栈：

1. **卡尔曼恒速滤波 + 自适应 lead 外推**（`kf_q/kf_r/kf_lead_*`）
   目标是滤波位置 + 速度×观测年龄 + 速度×额外lead×权重。速度、距离和方向
   三重门控让巡航保留延迟补偿，停手、近目标和反向纠偏时收回额外提前量。

2. **静止判定三件套**（修停手振荡）
   - 速度死区带回差：`kf_ff_deadband_off/on`（0.06/0.13 m/s），
     残余假速度低于下沿才进静止，超过上沿才回运动；
   - 跳出确认帧：`kf_ff_exit_frames=7`，须连续 7 周期越过上沿才
     恢复运动，杜绝死区边界状态闪烁引起的停手一抽一抽；
   - 静止目标锁存：`kf_hold_tol=0.04`，手停时目标钉死在锁存点，
     观测噪声完全隔离；手慢拖时锁存点以 0.06 m/s 限速平滑跟随。

3. **伸展衰减 EMA 平滑**（`reach_att_tau`）
   伸展衰减增益不再逐帧跳变（曾是伸展位极限环振荡的根因），
   按 τ≈0.15s 平滑过渡。

4. **深度冻结帧跳过 KF 更新**
   深度无效、z 用占位值的帧只做 predict 不做 update，
   防止占位深度把滤波器状态拉飞。

5. **重获目标强化**
   连续丢手 ≥30 帧后重获目标须 5 帧一致（原 3 帧），且深度超过
   `distance + 0.25m` 的一律判为背景拒绝（修背景深度被当手、
   手臂冲向墙壁的问题）。

6. **检测+光流混合跟踪**（`hybrid_track_*`）
   MediaPipe 单帧漏检时，以上一帧掌心为种子做 LK 光流桥接
   （3×3 点阵，≤15 帧，单帧位移 ≤90px 防飞点），桥接帧自动冻结
   夹爪（landmarks 缺失），短丢手不再触发保持-重启的顿挫。

7. **相机帧率结论**
   `tools/enumerate_camera_modes.py` 枚举确认 Gemini 2 彩色/深度
   均 60fps 硬顶（彩色 1280x720@60、深度 640x400@60），
   提帧率这条路已封死，延迟优化只能走 lead 外推。

8. **自适应过程噪声 Q**（`kf_adapt_*`，**本机默认关闭**）
   按创新（观测-预测偏差）动态放大 Q：加速段更信观测、平稳段回基准。
   实测在暗曝光高噪声环境下测量噪声会误触发，反而变冲——
   低噪声环境（补光充足/自动曝光）可开 `kf_adapt_enable: true` 试，
   参考 `kf_adapt_ref 0.10 / max 2.5 / decay 0.85`。

9. **曝光进 yaml**（`color_exposure_ms`，命令行 `--color-exposure` 可覆盖）
   手动曝光压运动模糊：6ms 配合补光是实测最佳点；
   4ms 更暗、MediaPipe 丢失明显增多，不建议。

**最终调定参数**（实机 9 轮日志迭代收敛，适合：60fps BS 深度 +
6ms 手动曝光 + 补光的环境）：

```
distance 0.35 | gain 2.35 / gain_x 2.4 | max_vel 0.50 | max_vel_joint 1.50
kf_enable true | kf_q 1.5 | kf_r 0.004 | kf_lead_ms 8.0
kf_lead_adaptive true | lead near/full 0.022/0.045 | attack/release 0.025/0.015
kf_fast_start_enable false | workspace.max_radius 0.42（仅水平X/Y）
死区 off/on 0.06/0.13 | exit_frames 7 | hold_tol 0.04
position_deadband 0.022 | smooth_alpha 0.48 | color_exposure_ms 6.0
kf_adapt_enable false | hybrid_track_enable true
```
