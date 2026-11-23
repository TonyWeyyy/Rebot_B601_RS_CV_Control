"""
Seeed reBot Arm B601-RS MuJoCo 4.0+ 数字孪生场景构建器（MjSpec）
=================================================================
以官方 `ReBot_Arm_RS.urdf`（含全部 32 份精模 STL 与 Seeed 阳极银 / 电机黑 / Seeed 绿涂装）
为基础，通过 `mujoco.MjSpec` 在内存中装配完整的实时数字孪生 3D 场景：

  - 保留全部多材质视觉网格（`group=1`），将原始碰撞壳（`group=0`）移至 `group=3`
    （默认隐藏，避免遮挡精模，可在 Studio Geom Group 3 中按需打开检视）；
  - 按 `config/hand_eye_calibration.json` 的真实手眼外参 `T_tcp_camera`，
    在 `link6` 上精确挂载 Orbbec Gemini 2 相机视觉外壳、双目镜头与 0.40m 光轴瞄准线；
  - 挂载 3 个零物理开销的 mocap 叠加体：
      1. `palm_target`：实时 3D 手掌位置（翠绿发光球体）；
      2. `cmd_tcp_marker`：期望指令末端位姿 `q_cmd` 幽灵标记（青色半透明球）；
      3. `workspace_ring`：r = 0.42m 水平工作空间 70% 安全软墙指示环（琥珀色细环）。
  - 注册 `home` 关键帧（对应 `config/home_pose.json` + `config/joint_offset.json`）。
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import re
from typing import Sequence

import mujoco
import numpy as np

PROJECT_ROOT = pathlib.Path(os.path.abspath(__file__)).parent.parent
URDF_PATH = "/home/tony/wey/rebot_developer/reBotArm_control_py/urdf/RS/urdf/ReBot_Arm_RS.urdf"
DEFAULT_HOME_Q = np.array([-0.4313, 0.3445, 0.3691, 0.0938, 0.0018, -0.0015], dtype=np.float64)
ARM_JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
GRIPPER_JOINT_NAMES = ("gripper_joint1", "gripper_joint2")


def load_joint_offset_rad() -> np.ndarray:
    """加载 `config/joint_offset.json` 的 6 轴 URDF 零位修正（rad）。"""
    path = PROJECT_ROOT / "config" / "joint_offset.json"
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            off = np.asarray(data["joint_offset_rad"], dtype=np.float64)
            if off.shape == (6,) and np.all(np.isfinite(off)):
                return off
        except Exception:
            pass
    return np.zeros(6, dtype=np.float64)


def load_home_q_rad() -> np.ndarray:
    """加载 `config/home_pose.json` 的示教 HOME 关节角（SDK 弧度）。"""
    path = PROJECT_ROOT / "config" / "home_pose.json"
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            q = np.asarray(data["home_joint_pos"], dtype=np.float64)
            if q.shape == (6,) and np.all(np.isfinite(q)):
                return q
        except Exception:
            pass
    return DEFAULT_HOME_Q.copy()


def load_hand_eye_matrix() -> np.ndarray:
    """加载 `config/hand_eye_calibration.json` 的 4×4 `T_tcp_camera`。"""
    path = PROJECT_ROOT / "config" / "hand_eye_calibration.json"
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            T = np.asarray(data["T_tcp_camera"], dtype=np.float64)
            if T.shape == (4, 4) and np.all(np.isfinite(T)):
                return T
        except Exception:
            pass
    return np.eye(4, dtype=np.float64)


def _rotmat_to_quat_wxyz(R: np.ndarray) -> list[float]:
    """将 3×3 旋转矩阵安全转换为 MuJoCo `[w, x, y, z]` 单位四元数。"""
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, np.asarray(R, dtype=np.float64).reshape(9))
    norm = float(np.linalg.norm(quat))
    if norm < 1e-12:
        return [1.0, 0.0, 0.0, 0.0]
    return (quat / norm).tolist()


def gripper_deg_to_prismatic_m(
    gripper_deg: float,
    closed_deg: float = 5.2,
    open_deg: float = 285.2,
    max_stroke_m: float = 0.05,
) -> float:
    """将实机 J7 夹爪角度（deg）线性映射到 URDF 左右指滑移量（m，0=闭合，0.05=全开）。"""
    span = max(1.0, float(open_deg) - float(closed_deg))
    ratio = float(np.clip((float(gripper_deg) - float(closed_deg)) / span, 0.0, 1.0))
    return ratio * float(max_stroke_m)


def _prepare_visual_urdf_string(urdf_path: str) -> str:
    """读取 URDF、解析相对 STL 路径为绝对路径，并注入 `discardvisual=false`。"""
    urdf_dir = os.path.dirname(os.path.abspath(urdf_path))
    xml = pathlib.Path(urdf_path).read_text(encoding="utf-8")

    def _abs_mesh(match: re.Match[str]) -> str:
        rel = match.group(1)
        absp = os.path.abspath(os.path.join(urdf_dir, rel))
        return f'filename="{absp}"'

    xml = re.sub(r'filename="([^"]+)"', _abs_mesh, xml)
    compiler_block = (
        '<mujoco><compiler discardvisual="false" strippath="false" '
        'fusestatic="false"/></mujoco>'
    )
    xml_out, count = re.subn(
        r"(<robot\b[^>]*>)",
        r"\1" + compiler_block,
        xml,
        count=1,
    )
    if count == 0:
        raise ValueError(f"无法在 URDF 中找到 <robot> 根标签: {urdf_path}")
    return xml_out


def build_b601_spec(
    *,
    urdf_path: str = URDF_PATH,
    workspace_radius_m: float = 0.42,
    target_distance_m: float = 0.40,
    show_camera_ray: bool = True,
    home_q_sdk: Sequence[float] | None = None,
    joint_offset_rad: Sequence[float] | None = None,
    T_tcp_camera: np.ndarray | None = None,
) -> mujoco.MjSpec:
    """构建完整的 reBot Arm B601-RS 数字孪生 `MjSpec`（纯 Python，热重载友好）。"""
    xml_str = _prepare_visual_urdf_string(urdf_path)
    spec = mujoco.MjSpec.from_string(xml_str)
    spec.modelname = "rebot_b601_rs_digital_twin"

    # 运动学数字孪生：100Hz 步进、关闭重力下垂，由实机/回放/滑块状态精确驱动
    spec.option.timestep = 0.01
    spec.option.gravity = [0.0, 0.0, 0.0]
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

    # 工作室渲染视角与三点布光
    spec.visual.headlight.diffuse = [0.58, 0.60, 0.62]
    spec.visual.headlight.ambient = [0.28, 0.30, 0.32]
    spec.visual.headlight.specular = [0.15, 0.15, 0.15]
    spec.visual.rgba.haze = [0.12, 0.16, 0.22, 1.0]
    spec.visual.global_.azimuth = 135.0
    spec.visual.global_.elevation = -22.0
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720
    spec.stat.center = [0.16, 0.0, 0.26]
    spec.stat.extent = 0.72

    spec.add_texture(
        name="studio_skybox",
        type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=[0.18, 0.24, 0.32],
        rgb2=[0.04, 0.06, 0.09],
        width=512,
        height=3072,
    )
    spec.add_texture(
        name="workbench_checker",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        mark=mujoco.mjtMark.mjMARK_EDGE,
        rgb1=[0.16, 0.19, 0.23],
        rgb2=[0.12, 0.15, 0.19],
        markrgb=[0.28, 0.33, 0.39],
        width=300,
        height=300,
    )
    floor_mat = spec.add_material(
        name="workbench_mat",
        texrepeat=[8.0, 8.0],
        texuniform=True,
        reflectance=0.14,
        roughness=0.55,
    )
    floor_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "workbench_checker"

    # 分离视觉精模（group=1）与碰撞壳（移至 group=3 默认隐藏，彻底消除 Z-fighting）
    for geom in spec.geoms:
        if geom.group == 0:
            geom.group = 3
            geom.contype = 0
            geom.conaffinity = 0
        else:
            geom.contype = 0
            geom.conaffinity = 0

    world = spec.worldbody
    world.add_light(
        name="key_light",
        pos=[0.8, 0.9, 1.8],
        dir=[-0.4, -0.45, -1.0],
        diffuse=[0.85, 0.85, 0.82],
        specular=[0.35, 0.35, 0.35],
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
    )
    world.add_light(
        name="rim_light",
        pos=[-0.9, -0.8, 1.3],
        dir=[0.5, 0.45, -0.75],
        diffuse=[0.35, 0.42, 0.52],
        specular=[0.15, 0.18, 0.22],
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
    )
    world.add_geom(
        name="workbench_floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        pos=[0.0, 0.0, 0.0],
        size=[0.0, 0.0, 0.05],
        material="workbench_mat",
        group=1,
        contype=0,
        conaffinity=0,
    )

    # 在 link6 上按标定外参 T_tcp_camera 精确安装 Orbbec Gemini 2 相机视觉组件与光轴瞄准线
    T_cam = load_hand_eye_matrix() if T_tcp_camera is None else np.asarray(T_tcp_camera, dtype=np.float64)
    cam_pos = T_cam[:3, 3].tolist()
    cam_quat = _rotmat_to_quat_wxyz(T_cam[:3, :3])
    link6_body = spec.body("link6")
    if link6_body is not None:
        link6_body.add_site(
            name="tcp_site",
            pos=[0.0, 0.0, 0.0],
            size=[0.008, 0.008, 0.008],
            rgba=[0.95, 0.85, 0.15, 0.9],
            group=2,
        )
        cam_body = link6_body.add_body(
            name="gemini2_camera_body",
            pos=cam_pos,
            quat=cam_quat,
        )
        # 相机光学中心 site（+Z 为相机光轴，供回放模式把 p_cam 反投影到世界系）
        cam_body.add_site(
            name="gemini2_cam",
            pos=[0.0, 0.0, 0.0],
            size=[0.006, 0.006, 0.006],
            rgba=[0.1, 0.85, 1.0, 0.9],
            group=2,
        )
        # Gemini 2 铝壳机身（尺寸约 90×25×25 mm）
        cam_body.add_geom(
            name="gemini2_housing",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[0.0, 0.0, -0.014],
            size=[0.045, 0.0125, 0.0125],
            rgba=[0.11, 0.13, 0.15, 1.0],
            group=1,
            contype=0,
            conaffinity=0,
        )
        # 前置深色玻璃面板
        cam_body.add_geom(
            name="gemini2_visor",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[0.0, 0.0, -0.001],
            size=[0.042, 0.010, 0.002],
            rgba=[0.02, 0.03, 0.04, 1.0],
            group=1,
            contype=0,
            conaffinity=0,
        )
        # 0.40m 光轴期望跟随距离瞄准线与停靠目标点
        if show_camera_ray:
            ray_len = max(0.05, float(target_distance_m))
            cam_body.add_geom(
                name="gemini2_aim_ray",
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                pos=[0.0, 0.0, 0.5 * ray_len],
                size=[0.0018, 0.5 * ray_len, 0.0],
                rgba=[0.15, 0.85, 1.0, 0.30],
                group=2,
                contype=0,
                conaffinity=0,
            )
            cam_body.add_geom(
                name="gemini2_aim_tip",
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                pos=[0.0, 0.0, ray_len],
                size=[0.010, 0.0, 0.0],
                rgba=[0.15, 0.85, 1.0, 0.55],
                group=2,
                contype=0,
                conaffinity=0,
            )

    # Mocap 1: 实时 3D 人手掌心标记球（默认藏于地下，检测到手掌后实时同步）
    palm_body = world.add_body(
        name="palm_target",
        pos=[0.0, 0.0, -10.0],
        mocap=True,
    )
    palm_body.add_geom(
        name="palm_target_halo",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.038, 0.0, 0.0],
        rgba=[0.18, 0.96, 0.48, 0.45],
        group=2,
        contype=0,
        conaffinity=0,
    )
    palm_body.add_geom(
        name="palm_target_core",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.015, 0.0, 0.0],
        rgba=[0.30, 1.0, 0.55, 0.92],
        group=2,
        contype=0,
        conaffinity=0,
    )

    # Mocap 2: 指令 TCP 幽灵标记球（直观对比 q_cmd 与实测 q_act 的跟踪差）
    cmd_body = world.add_body(
        name="cmd_tcp_marker",
        pos=[0.0, 0.0, -10.0],
        mocap=True,
    )
    cmd_body.add_geom(
        name="cmd_tcp_sphere",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.015, 0.0, 0.0],
        rgba=[0.10, 0.78, 1.0, 0.65],
        group=2,
        contype=0,
        conaffinity=0,
    )

    # Mocap 3: r = 0.42m 水平工作空间安全软墙圆环（36 段细胶囊组成清晰全息边界）
    ring_body = world.add_body(
        name="workspace_ring",
        pos=[0.0, 0.0, 0.004],
        mocap=True,
    )
    n_seg = 36
    r = float(workspace_radius_m)
    for k in range(n_seg):
        a0 = 2.0 * math.pi * k / n_seg
        a1 = 2.0 * math.pi * (k + 1) / n_seg
        p0 = [r * math.cos(a0), r * math.sin(a0), 0.0]
        p1 = [r * math.cos(a1), r * math.sin(a1), 0.0]
        ring_body.add_geom(
            name=f"ws_ring_seg_{k:02d}",
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=p0 + p1,
            size=[0.0028, 0.0, 0.0],
            rgba=[1.0, 0.68, 0.16, 0.55],
            group=2,
            contype=0,
            conaffinity=0,
        )

    # Mocap 4 (v15.0): 可在 3D 视口直接鼠标拖拽的 IK 目标球（双击选中后 Ctrl+右键平移 / Ctrl+左键旋转）
    ik_body = world.add_body(
        name="ik_target",
        pos=[0.0, 0.0, -10.0],
        mocap=True,
    )
    ik_body.add_geom(
        name="ik_target_shell",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.028, 0.0, 0.0],
        rgba=[1.0, 0.52, 0.08, 0.65],
        group=1,
        contype=0,
        conaffinity=0,
    )
    ik_body.add_geom(
        name="ik_target_core",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.011, 0.0, 0.0],
        rgba=[1.0, 0.86, 0.18, 0.96],
        group=1,
        contype=0,
        conaffinity=0,
    )
    # RGB 局部三轴指示杆，便于 6-DoF 姿态 IK 时直观查看末端朝向
    for axis_name, end_pt, rgba in (
        ("ik_axis_x", [0.055, 0.0, 0.0], [0.95, 0.22, 0.22, 0.9]),
        ("ik_axis_y", [0.0, 0.055, 0.0], [0.22, 0.92, 0.32, 0.9]),
        ("ik_axis_z", [0.0, 0.0, 0.055], [0.22, 0.58, 1.00, 0.9]),
    ):
        ik_body.add_geom(
            name=axis_name,
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=[0.0, 0.0, 0.0] + end_pt,
            size=[0.0024, 0.0, 0.0],
            rgba=rgba,
            group=1,
            contype=0,
            conaffinity=0,
        )

    # 添加示教 HOME 关键帧（按编译顺序填充关节初值，Studio Reset 按钮一键归位）
    q_home = load_home_q_rad() if home_q_sdk is None else np.asarray(home_q_sdk, dtype=np.float64)
    q_off = load_joint_offset_rad() if joint_offset_rad is None else np.asarray(joint_offset_rad, dtype=np.float64)
    q_urdf_home = q_home + q_off
    g_open = gripper_deg_to_prismatic_m(285.2)
    joint_map = {
        "joint1": float(q_urdf_home[0]),
        "joint2": float(q_urdf_home[1]),
        "joint3": float(q_urdf_home[2]),
        "joint4": float(q_urdf_home[3]),
        "joint5": float(q_urdf_home[4]),
        "joint6": float(q_urdf_home[5]),
        "gripper_joint1": g_open,
        "gripper_joint2": g_open,
    }
    key_qpos = [joint_map.get(j.name, 0.0) for j in spec.joints]
    spec.add_key(name="home", qpos=key_qpos)

    return spec
