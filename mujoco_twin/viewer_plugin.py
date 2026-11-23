"""Dear ImGui digital twin telemetry & replay panel for MuJoCo Studio."""

from __future__ import annotations

import messages as user_messages
from mujoco.studio import messages
from mujoco.studio import plugin
from mujoco.studio.app import plugins as app_plugins

imgui = app_plugins.imgui

JOINT_LABELS = ("J1", "J2", "J3", "J4", "J5", "J6")
JOINT_LIMITS_DEG = (
    (-160.0, 160.0),
    (-10.0, 175.0),
    (-15.0, 175.0),
    (-100.0, 100.0),
    (-90.0, 90.0),
    (-160.0, 160.0),
)


class ViewerPlugin(plugin.ViewerPlugin):
    """reBot Arm B601-RS Real-Time Digital Twin Dashboard (Dear ImGui)."""

    window_open: bool = True
    telemetry: user_messages.TwinTelemetry = user_messages.TwinTelemetry()

    # Display & manual test state
    show_palm: bool = True
    show_cmd_marker: bool = True
    show_workspace_ring: bool = True
    show_camera_ray: bool = True
    manual_mode: bool = False
    manual_q_deg: list[float] | None = None
    manual_gripper_deg: float = 285.2
    selected_log_idx: int = 0

    def _ensure_manual_buf(self) -> list[float]:
        if self.manual_q_deg is None or len(self.manual_q_deg) != 6:
            self.manual_q_deg = [-24.7, 19.7, 21.1, 5.4, 0.1, -0.1]
        return self.manual_q_deg

    @messages.handler
    def on_telemetry(self, snapshot: user_messages.TwinTelemetry) -> None:
        self.telemetry = snapshot
        if snapshot.available_logs and self.selected_log_idx >= len(snapshot.available_logs):
            self.selected_log_idx = 0

    def _send_display(self) -> None:
        buf = self._ensure_manual_buf()
        self.viewer.send_to_sim(
            user_messages.TwinDisplayConfig(
                show_palm=self.show_palm,
                show_cmd_marker=self.show_cmd_marker,
                show_workspace_ring=self.show_workspace_ring,
                show_camera_ray=self.show_camera_ray,
                manual_mode=self.manual_mode,
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
    ) -> None:
        self.viewer.send_to_sim(
            user_messages.TwinActionEvent(
                action=action,
                log_name=log_name,
                seek_frac=seek_frac,
                speed=speed,
            )
        )

    def build_status_gui(self) -> None:
        t = self.telemetry
        status = t.link_status
        if status == "LIVE_HW":
            badge = f"LIVE HARDWARE  ({t.rx_hz:.0f} Hz)  udp:{t.udp_port}"
        elif status == "LIVE_SIM":
            badge = f"LIVE SIM BRIDGE  ({t.rx_hz:.0f} Hz)  udp:{t.udp_port}"
        elif status == "REPLAY":
            state_txt = "PLAYING" if t.replay_playing else "PAUSED"
            badge = f"CSV LOG REPLAY  [{state_txt} {t.replay_speed:.2f}x]"
        elif status == "MANUAL":
            badge = "MANUAL JOINT TEST"
        else:
            badge = f"STANDBY  (waiting udp://127.0.0.1:{t.udp_port})"

        imgui.TextUnformatted(f"Link: {badge}")
        if t.bind_error:
            imgui.TextUnformatted(f"UDP Bind Error: {t.bind_error}")

        follow_txt = "ON" if t.following else "OFF"
        palm_txt = "VISIBLE" if t.visible else "NONE"
        fault_txt = "LATCHED!" if t.m6_fault else "OK"
        imgui.TextUnformatted(
            f"FSM: {t.fsm_mode} | Follow: {follow_txt} | Palm: {palm_txt} | M6: {fault_txt}"
        )
        imgui.TextUnformatted(
            f"3D Err: {t.err_norm_cm:5.1f} cm | Track Max: {t.track_err_deg:4.2f} deg | "
            f"MOS: {t.tmos_max:4.1f} C | J7: {t.gripper_deg:5.1f} deg"
        )
        tx, ty, tz = t.tcp_xyz
        if t.palm_xyz is not None:
            px, py, pz = t.palm_xyz
            imgui.TextUnformatted(
                f"TCP: ({tx:+.3f}, {ty:+.3f}, {tz:+.3f})m | Palm: ({px:+.3f}, {py:+.3f}, {pz:+.3f})m"
            )
        else:
            imgui.TextUnformatted(f"TCP: ({tx:+.3f}, {ty:+.3f}, {tz:+.3f}) m")

    def build_joint_table_gui(self) -> None:
        t = self.telemetry
        if imgui.BeginTable("JointTelemetryTable", 6):
            imgui.TableSetupColumn("Joint")
            imgui.TableSetupColumn("q_act (deg)")
            imgui.TableSetupColumn("q_cmd (deg)")
            imgui.TableSetupColumn("Err (deg)")
            imgui.TableSetupColumn("dq (deg/s)")
            imgui.TableSetupColumn("Tau (Nm)")
            imgui.TableHeadersRow()
            for i in range(6):
                qa = t.q_act_deg[i] if i < len(t.q_act_deg) else 0.0
                qc = t.q_cmd_deg[i] if i < len(t.q_cmd_deg) else 0.0
                dq = t.dq_act_deg_s[i] if i < len(t.dq_act_deg_s) else 0.0
                tau = t.tau_act[i] if i < len(t.tau_act) else 0.0
                imgui.TableNextRow()
                imgui.TableNextColumn()
                imgui.TextUnformatted(JOINT_LABELS[i])
                imgui.TableNextColumn()
                imgui.TextUnformatted(f"{qa:+7.2f}")
                imgui.TableNextColumn()
                imgui.TextUnformatted(f"{qc:+7.2f}")
                imgui.TableNextColumn()
                imgui.TextUnformatted(f"{qc - qa:+6.2f}")
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
        c2, self.show_cmd_marker = imgui.Checkbox("Cmd TCP Marker##ShowCmd", self.show_cmd_marker)
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
            self._send_action("replay_load", log_name=logs[self.selected_log_idx])
        imgui.SameLine()
        if imgui.Button("Refresh##RefreshLogsBtn"):
            self._send_action("refresh_logs")

        if t.replay_playing:
            if imgui.Button("Pause##PauseReplay"):
                self._send_action("replay_pause")
        else:
            if imgui.Button("Play##PlayReplay"):
                if t.replay_file != cur_name or t.replay_total == 0:
                    self._send_action("replay_load", log_name=cur_name)
                self._send_action("replay_play")
        imgui.SameLine()
        if imgui.Button("Stop##StopReplay"):
            self._send_action("replay_stop")
        imgui.SameLine()
        if imgui.Button("Home##ResetHomeBtn"):
            self.manual_mode = False
            self._send_action("reset_home")

        # Speed presets
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

    def build_manual_gui(self) -> None:
        if not imgui.CollapsingHeader("Manual Joint Test###ManualHeader"):
            return
        buf = self._ensure_manual_buf()
        changed, self.manual_mode = imgui.Checkbox(
            "Enable Manual Sliders (overridden by Live UDP)##ManualEnable",
            self.manual_mode,
        )
        if imgui.Button("Reset to Taught HOME##ManualResetHome"):
            self.manual_q_deg = [-24.7, 19.7, 21.1, 5.4, 0.1, -0.1]
            self.manual_gripper_deg = 285.2
            changed = True
        for i in range(6):
            lo, hi = JOINT_LIMITS_DEG[i]
            c, val = imgui.SliderFloat(
                f"{JOINT_LABELS[i]} (deg)##ManualJ{i+1}", buf[i], lo, hi
            )
            if c:
                buf[i] = val
                changed = True
        cg, gval = imgui.SliderFloat(
            "J7 Gripper (deg)##ManualGripper", self.manual_gripper_deg, 5.2, 285.2
        )
        if cg:
            self.manual_gripper_deg = gval
            changed = True
        if changed:
            self._send_display()

    @messages.handler
    def on_build_gui(self, event: messages.BuildGuiEvent) -> None:
        del event
        if not self.window_open:
            return
        expanded, self.window_open = imgui.Begin(
            "reBot B601-RS Digital Twin###B601DigitalTwinWindow", self.window_open
        )
        try:
            if not expanded:
                return
            self.build_status_gui()
            imgui.Separator()
            self.build_joint_table_gui()
            imgui.Separator()
            self.build_overlays_gui()
            imgui.Separator()
            self.build_replay_gui()
            self.build_manual_gui()
        finally:
            imgui.End()
