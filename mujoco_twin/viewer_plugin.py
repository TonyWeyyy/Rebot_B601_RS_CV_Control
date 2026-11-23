"""Dear ImGui v15.0 Bidirectional Digital Twin & Teleoperation Dashboard for MuJoCo Studio."""

from __future__ import annotations

import messages as user_messages
from mujoco.studio import messages
from mujoco.studio import plugin
from mujoco.studio.app import plugins as app_plugins

imgui = app_plugins.imgui

JOINT_LABELS = ("J1", "J2", "J3", "J4", "J5", "J6")
JOINT_LIMITS_DEG = (
    (-158.0, 158.0),
    (-6.0, 173.0),
    (-12.0, 173.0),
    (-98.0, 98.0),
    (-88.0, 88.0),
    (-158.0, 158.0),
)


class ViewerPlugin(plugin.ViewerPlugin):
    """reBot Arm B601-RS v15.0 Bidirectional Digital Twin Dashboard (Dear ImGui)."""

    window_open: bool = True
    telemetry: user_messages.TwinTelemetry = user_messages.TwinTelemetry()

    # Display & teleop UI state
    show_palm: bool = True
    show_cmd_marker: bool = True
    show_workspace_ring: bool = True
    show_camera_ray: bool = True

    teleop_mode: str = "MONITOR"  # MONITOR | JOINT | IK_3D
    teleop_max_speed_deg_s: float = 25.0
    ik_lock_orientation: bool = True
    manual_q_deg: list[float] | None = None
    manual_gripper_deg: float = 285.2
    selected_log_idx: int = 0
    _sync_sliders_from_sim: bool = True

    def _ensure_manual_buf(self) -> list[float]:
        if self.manual_q_deg is None or len(self.manual_q_deg) != 6:
            self.manual_q_deg = [-24.7, 19.7, 21.1, 5.4, 0.1, -0.1]
        return self.manual_q_deg

    @messages.handler
    def on_telemetry(self, snapshot: user_messages.TwinTelemetry) -> None:
        self.telemetry = snapshot
        if snapshot.available_logs and self.selected_log_idx >= len(snapshot.available_logs):
            self.selected_log_idx = 0
        # Keep UI sliders synchronized when not actively in JOINT drag mode or after Sync/Arm
        if self._sync_sliders_from_sim or snapshot.teleop_mode != "JOINT":
            if len(snapshot.q_des_deg) == 6:
                self.manual_q_deg = [float(x) for x in snapshot.q_des_deg]
            self.manual_gripper_deg = float(snapshot.des_gripper_deg)
            self._sync_sliders_from_sim = False

    def _send_display(self) -> None:
        buf = self._ensure_manual_buf()
        self.viewer.send_to_sim(
            user_messages.TwinDisplayConfig(
                show_palm=self.show_palm,
                show_cmd_marker=self.show_cmd_marker,
                show_workspace_ring=self.show_workspace_ring,
                show_camera_ray=self.show_camera_ray,
                teleop_mode=self.teleop_mode,
                teleop_max_speed_deg_s=float(self.teleop_max_speed_deg_s),
                ik_lock_orientation=self.ik_lock_orientation,
                manual_q_deg=tuple(float(x) for x in buf),
                manual_gripper_deg=float(self.manual_gripper_deg),
            )
        )

    def _send_action(
        self,
        action: str,
        *,
        log_name: str = "",
        seek_frac: float = 0.0,
        speed: float = 1.0,
        ik_xyz: tuple[float, float, float] = (0.135, 0.061, 0.314),
    ) -> None:
        self.viewer.send_to_sim(
            user_messages.TwinActionEvent(
                action=action,
                log_name=log_name,
                seek_frac=seek_frac,
                speed=speed,
                ik_xyz=ik_xyz,
            )
        )

    def build_status_gui(self) -> None:
        t = self.telemetry
        status = t.link_status
        if status == "LIVE_HW_TELEOP":
            badge = f"BIDIRECTIONAL HW TELEOP ({t.rx_hz:.0f} Hz)  :{t.udp_port}<->:{t.cmd_port}"
        elif status == "LIVE_SIM_TELEOP":
            badge = f"BIDIRECTIONAL SIM TELEOP ({t.rx_hz:.0f} Hz)  :{t.udp_port}<->:{t.cmd_port}"
        elif status == "LIVE_HW":
            badge = f"LIVE HARDWARE MONITOR ({t.rx_hz:.0f} Hz)  udp:{t.udp_port}"
        elif status == "LIVE_SIM":
            badge = f"LIVE SIM MONITOR ({t.rx_hz:.0f} Hz)  udp:{t.udp_port}"
        elif status == "REPLAY":
            state_txt = "PLAYING" if t.replay_playing else "PAUSED"
            badge = f"CSV LOG REPLAY [{state_txt} {t.replay_speed:.2f}x]"
        elif status == "TELEOP_PREVIEW":
            badge = f"OFFLINE PREVIEW ({t.teleop_mode}, no HW bridge)"
        else:
            badge = f"STANDBY (rx:{t.udp_port} / tx:{t.cmd_port})"

        imgui.TextUnformatted(f"Link: {badge}")
        if t.bind_error:
            imgui.TextUnformatted(f"UDP Bind Error: {t.bind_error}")

        fault_txt = "LATCHED!" if t.m6_fault else "OK"
        imgui.TextUnformatted(
            f"FSM: {t.fsm_mode} | Track Max: {t.track_err_deg:4.2f} deg | "
            f"MOS: {t.tmos_max:4.1f} C | M6: {fault_txt} | J7: {t.gripper_deg:5.1f} deg"
        )
        tx, ty, tz = t.tcp_xyz
        if t.palm_xyz is not None:
            px, py, pz = t.palm_xyz
            imgui.TextUnformatted(
                f"TCP: ({tx:+.3f}, {ty:+.3f}, {tz:+.3f})m | Palm: ({px:+.3f}, {py:+.3f}, {pz:+.3f})m"
            )
        else:
            imgui.TextUnformatted(f"TCP: ({tx:+.3f}, {ty:+.3f}, {tz:+.3f}) m")

    def build_teleop_gui(self) -> None:
        t = self.telemetry
        imgui.TextUnformatted("v15.0 MuJoCo -> Real Robot Teleoperation")

        # Mode switch buttons
        mode_changed = False
        for mode_id, label in (
            ("MONITOR", "Monitor Only"),
            ("JOINT", "Joint Sliders"),
            ("IK_3D", "3D IK Sphere"),
        ):
            if mode_id != "MONITOR":
                imgui.SameLine()
            prefix = "* " if self.teleop_mode == mode_id else ""
            if imgui.Button(f"{prefix}{label}##Mode_{mode_id}"):
                if self.teleop_mode != mode_id:
                    self.teleop_mode = mode_id
                    self._sync_sliders_from_sim = True
                    mode_changed = True
        if mode_changed:
            self._send_display()

        if self.teleop_mode == "MONITOR":
            imgui.TextUnformatted("Read-only mode: MuJoCo mirrors live robot or CSV replay.")
            return

        # Hardware bridge arming & safety bar
        if t.hw_teleop_connected:
            arm_badge = "ARMED (REAL ARM MOVING)" if t.hw_teleop_armed else "DISARMED (SAFE HOLD)"
            imgui.TextUnformatted(
                f"HW Bridge: {arm_badge} | Reason: {t.hw_teleop_reason} | Align: {t.align_err_deg:.1f} deg"
            )
        else:
            imgui.TextUnformatted(
                "HW Bridge: OFFLINE (preview only) - run 'python3 mujoco_teleop_v15.py [--sim]'"
            )

        if imgui.Button("Sync to Real Pose##TeleopSync"):
            self._sync_sliders_from_sim = True
            self._send_action("teleop_sync_to_real")
        imgui.SameLine()
        if t.teleop_armed_req or t.hw_teleop_armed:
            if imgui.Button("DISARM / HOLD NOW###TeleopArmBtn"):
                self._send_action("teleop_disarm")
        else:
            if imgui.Button("ARM Real Robot Follow###TeleopArmBtn"):
                self._sync_sliders_from_sim = True
                self._send_action("teleop_arm")
        imgui.SameLine()
        if imgui.Button("Reset HOME##TeleopHome"):
            self._sync_sliders_from_sim = True
            self._send_action("reset_home")
        if t.m6_fault:
            imgui.SameLine()
            if imgui.Button("Clear M6 Fault##TeleopClearFault"):
                self._send_action("teleop_clear_fault")

        spd_changed, new_spd = imgui.SliderFloat(
            "Max Speed (deg/s)##TeleopMaxSpeed",
            self.teleop_max_speed_deg_s,
            5.0,
            60.0,
        )
        if spd_changed:
            self.teleop_max_speed_deg_s = new_spd
            self._send_display()

        # Mode-specific controls
        if self.teleop_mode == "JOINT":
            buf = self._ensure_manual_buf()
            changed = False
            for i in range(6):
                lo, hi = JOINT_LIMITS_DEG[i]
                c, val = imgui.SliderFloat(
                    f"{JOINT_LABELS[i]} Target (deg)##TeleopJ{i+1}", buf[i], lo, hi
                )
                if c:
                    buf[i] = val
                    changed = True
            cg, gval = imgui.SliderFloat(
                "J7 Gripper (deg)##TeleopGripperJ", self.manual_gripper_deg, 5.2, 285.2
            )
            if cg:
                self.manual_gripper_deg = gval
                changed = True
            if changed:
                self._send_display()

        elif self.teleop_mode == "IK_3D":
            imgui.TextUnformatted(
                "3D Drag: Double-click orange sphere -> Ctrl+RightDrag (move) / Ctrl+LeftDrag (rotate)"
            )
            lock_changed, self.ik_lock_orientation = imgui.Checkbox(
                "Lock TCP Orientation (Position-only IK)##IkLockOri",
                self.ik_lock_orientation,
            )
            imgui.SameLine()
            imgui.TextUnformatted(f"IK Residual: {t.ik_residual_mm:.2f} mm")
            if lock_changed:
                self._send_display()

            ix, iy, iz = t.ik_target_xyz
            cx, nx = imgui.SliderFloat("Target X (m)##IkX", ix, -0.42, 0.42)
            cy, ny = imgui.SliderFloat("Target Y (m)##IkY", iy, -0.42, 0.42)
            cz, nz = imgui.SliderFloat("Target Z (m)##IkZ", iz, 0.06, 0.62)
            if cx or cy or cz:
                self._send_action("teleop_set_ik_pos", ik_xyz=(nx, ny, nz))

            cg, gval = imgui.SliderFloat(
                "J7 Gripper (deg)##TeleopGripperIK", self.manual_gripper_deg, 5.2, 285.2
            )
            if cg:
                self.manual_gripper_deg = gval
                self._send_display()

    def build_joint_table_gui(self) -> None:
        t = self.telemetry
        target_label = "q_des (deg)" if self.teleop_mode in ("JOINT", "IK_3D") else "q_cmd (deg)"
        target_vec = t.q_des_deg if self.teleop_mode in ("JOINT", "IK_3D") else t.q_cmd_deg
        if imgui.BeginTable("JointTelemetryTable", 6):
            imgui.TableSetupColumn("Joint")
            imgui.TableSetupColumn("q_act (deg)")
            imgui.TableSetupColumn(target_label)
            imgui.TableSetupColumn("Err (deg)")
            imgui.TableSetupColumn("dq (deg/s)")
            imgui.TableSetupColumn("Tau (Nm)")
            imgui.TableHeadersRow()
            for i in range(6):
                qa = t.q_act_deg[i] if i < len(t.q_act_deg) else 0.0
                qt = target_vec[i] if i < len(target_vec) else 0.0
                dq = t.dq_act_deg_s[i] if i < len(t.dq_act_deg_s) else 0.0
                tau = t.tau_act[i] if i < len(t.tau_act) else 0.0
                imgui.TableNextRow()
                imgui.TableNextColumn()
                imgui.TextUnformatted(JOINT_LABELS[i])
                imgui.TableNextColumn()
                imgui.TextUnformatted(f"{qa:+7.2f}")
                imgui.TableNextColumn()
                imgui.TextUnformatted(f"{qt:+7.2f}")
                imgui.TableNextColumn()
                imgui.TextUnformatted(f"{qt - qa:+6.2f}")
                imgui.TableNextColumn()
                imgui.TextUnformatted(f"{dq:+7.1f}")
                imgui.TableNextColumn()
                imgui.TextUnformatted(f"{tau:+6.2f}")
            imgui.EndTable()

    def build_overlays_gui(self) -> None:
        changed = False
        c1, self.show_palm = imgui.Checkbox("Palm 3D Target Sphere##ShowPalm", self.show_palm)
        changed |= c1
        imgui.SameLine()
        c2, self.show_cmd_marker = imgui.Checkbox("Cmd/Des TCP Marker##ShowCmd", self.show_cmd_marker)
        changed |= c2
        c3, self.show_workspace_ring = imgui.Checkbox(
            "0.42m Safety Ring##ShowRing", self.show_workspace_ring
        )
        changed |= c3
        imgui.SameLine()
        c4, self.show_camera_ray = imgui.Checkbox("Camera 0.40m Ray##ShowRay", self.show_camera_ray)
        changed |= c4
        if changed:
            self._send_display()

    def build_replay_gui(self) -> None:
        t = self.telemetry
        if not imgui.CollapsingHeader("Hardware Log Replay###ReplayHeader"):
            return

        logs = t.available_logs
        if not logs:
            imgui.TextUnformatted("No logs/servo_*.csv files found.")
            if imgui.Button("Refresh Logs##RefreshLogs"):
                self._send_action("refresh_logs")
            return

        cur_name = logs[self.selected_log_idx]
        imgui.TextUnformatted(f"Selected: {cur_name}")
        if imgui.Button("< Prev##PrevLog"):
            self.selected_log_idx = (self.selected_log_idx - 1) % len(logs)
        imgui.SameLine()
        if imgui.Button("Next >##NextLog"):
            self.selected_log_idx = (self.selected_log_idx + 1) % len(logs)
        imgui.SameLine()
        if imgui.Button("Load##LoadLog"):
            self.teleop_mode = "MONITOR"
            self._send_action("replay_load", log_name=logs[self.selected_log_idx])
        imgui.SameLine()
        if imgui.Button("Refresh##RefreshLogsBtn"):
            self._send_action("refresh_logs")

        if t.replay_playing:
            if imgui.Button("Pause##PauseReplay"):
                self._send_action("replay_pause")
        else:
            if imgui.Button("Play##PlayReplay"):
                self.teleop_mode = "MONITOR"
                if t.replay_file != cur_name or t.replay_total == 0:
                    self._send_action("replay_load", log_name=cur_name)
                self._send_action("replay_play")
        imgui.SameLine()
        if imgui.Button("Stop##StopReplay"):
            self._send_action("replay_stop")
        imgui.SameLine()
        if imgui.Button("Home##ResetHomeBtn"):
            self.teleop_mode = "MONITOR"
            self._send_action("reset_home")

        for spd in (0.25, 0.5, 1.0, 2.0):
            if spd > 0.25:
                imgui.SameLine()
            if imgui.Button(f"{spd:g}x##Speed_{spd}"):
                self._send_action("replay_speed", speed=spd)

        if t.replay_total > 1:
            frac = float(t.replay_idx) / float(t.replay_total - 1)
            imgui.TextUnformatted(
                f"Time: {t.replay_time_s:5.2f} / {t.replay_duration_s:5.2f} s "
                f"(frame {t.replay_idx}/{t.replay_total})"
            )
            seek_changed, new_frac = imgui.SliderFloat(
                "Timeline###ReplayTimeline", frac, 0.0, 1.0
            )
            if seek_changed:
                self._send_action("replay_seek", seek_frac=new_frac)

    @messages.handler
    def on_build_gui(self, event: messages.BuildGuiEvent) -> None:
        del event
        if not self.window_open:
            return
        expanded, self.window_open = imgui.Begin(
            "reBot B601-RS Digital Twin v15.0###B601DigitalTwinWindow", self.window_open
        )
        try:
            if not expanded:
                return
            self.build_status_gui()
            imgui.Separator()
            self.build_teleop_gui()
            imgui.Separator()
            self.build_joint_table_gui()
            imgui.Separator()
            self.build_overlays_gui()
            imgui.Separator()
            self.build_replay_gui()
        finally:
            imgui.End()
