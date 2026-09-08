from __future__ import annotations
import csv
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict


# Keep project imports stable if Kit changes the process working directory.
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Importing the original harness launches Isaac Sim in the required order.
# Its main() is guarded by ``if __name__ == "__main__"`` and is therefore not
# run here.  Most names below are aliases to libraries/configuration it already
# imported after AppLauncher initialized Kit.
from source.test import locomotion_test as original_test

import numpy as np
import torch

from source.utils.CONTROLLER.themis_gait_prime import (
    AlternatingGaitPlanner,
    GaitMeasurement,
    GaitPlannerConfig,
)
from source.utils.CONTROLLER.themis_wbc_qp import (
    ControllerState,
    run_wbc_qp,
)
from source.utils.robot_model.forward_kinematics import compute_FK
from source.utils.robot_model.math_function import rot2euler


args_cli = original_test.args_cli
simulation_app = original_test.simulation_app


# The QP defaults are intentionally modest.  Values here are experiment-level
# overrides rather than hidden modifications to the legacy controller.
QP_PARAMETERS: Dict[str, Any] = {
    "robot_mass": 37.86976,
    "friction_coefficient": 0.60,
    "maximum_point_normal_force": 250.0,
    # A contact switch can require a noticeable torque redistribution.  This
    # is still far below the configured actuator limits and prevents a single
    # discontinuous command.
    "maximum_torque_step": 15.0,
    "minimum_stance_force": 35.0,
    "require_stance_force_guard": True,
    "maximum_hard_constraint_residual": 5.0e-3,
    # Track the planner's smooth contact weights strongly enough for measured
    # load-transfer gates to become meaningful.
    "force_tracking_weight": 2.0e-3,
    "force_rate_regularization": 2.0e-5,
    # Keep the whole sole level during swing.  The WBC default gives Cartesian
    # position five times more authority than orientation; on THEMIS that
    # permits several degrees of ankle roll/pitch while the leg reaches
    # forward, which makes the feet look crossed even though their lateral
    # lanes remain 22 cm apart.  Equal task authority and critically damped
    # angular gains keep each sole parallel to its measured initial pose.
    "swing_orientation_weight": 1000.0,
    "swing_orientation_kp": np.array([60.0, 60.0, 35.0]),
    "swing_orientation_kd": np.array([15.0, 15.0, 10.0]),
}


TELEMETRY_COLUMNS = (
    "time",
    "planner_time",
    "stage",
    "step_index",
    "support_side",
    "swing_side",
    "solver_status",
    "solver_failures",
    "dynamics_residual",
    "contact_residual",
    "com_x",
    "com_y",
    "com_z",
    "com_des_x",
    "com_des_y",
    "com_des_z",
    "com_vx",
    "com_vy",
    "com_vz",
    "dcm_x",
    "dcm_y",
    "dcm_des_x",
    "dcm_des_y",
    "right_ankle_x",
    "right_ankle_y",
    "right_ankle_z",
    "left_ankle_x",
    "left_ankle_y",
    "left_ankle_z",
    "right_roll",
    "right_pitch",
    "right_yaw",
    "left_roll",
    "left_pitch",
    "left_yaw",
    "base_roll",
    "base_pitch",
    "base_yaw",
    "force_measured_r",
    "force_measured_l",
    "force_commanded_r",
    "force_commanded_l",
    "maximum_joint_torque",
    "safety_reason",
)


def _read_contact_force(sensor: Any) -> float:
    """Return one contact sensor's world vertical force for environment zero."""

    return float(sensor.data.net_forces_w.view(-1, 3)[0, 2].item())


def _foot_forces(scene: Any) -> tuple[float, float]:
    """Sum heel and both toe sensors into right/left vertical foot loads."""

    force_l = (
        _read_contact_force(scene["contact_h_l"])
        + _read_contact_force(scene["contact_tl_l"])
        + _read_contact_force(scene["contact_tr_l"])
    )
    force_r = (
        _read_contact_force(scene["contact_h_r"])
        + _read_contact_force(scene["contact_tl_r"])
        + _read_contact_force(scene["contact_tr_r"])
    )
    return abs(force_l), abs(force_r)


def _make_measurement(
    planner_time: float,
    state: Dict[str, np.ndarray],
    fk: Dict[str, np.ndarray],
    force_l: float,
    force_r: float,
) -> GaitMeasurement:
    """Package the measured state used by the contact-aware gait supervisor."""

    return GaitMeasurement(
        time=planner_time,
        com_position=fk["p_wg"],
        com_velocity=fk["v_wg"],
        base_rotation=fk["R_wb"],
        base_angular_velocity=state["w_bb_np"][0],
        right_ankle_position=fk["p_wa_r"],
        right_ankle_velocity=fk["v_wa_r"],
        right_ankle_rotation=fk["R_wa_r"],
        left_ankle_position=fk["p_wa_l"],
        left_ankle_velocity=fk["v_wa_l"],
        left_ankle_rotation=fk["R_wa_l"],
        contact_force_r_z=force_r,
        contact_force_l_z=force_l,
    )


def _commanded_foot_loads(point_forces: np.ndarray) -> tuple[float, float]:
    """Sum the QP's three point-normal forces for each foot."""

    point_forces = np.asarray(point_forces, dtype=float).reshape(6, 3)
    return (
        float(np.sum(point_forces[0:3, 2])),
        float(np.sum(point_forces[3:6, 2])),
    )


def _telemetry_row(
    elapsed_time: float,
    planner_time: float,
    planner_output: Dict[str, Any],
    controller_state: ControllerState,
    fk: Dict[str, np.ndarray],
    force_l: float,
    force_r: float,
    point_forces: np.ndarray,
    torque_isaac: np.ndarray,
) -> Dict[str, Any]:
    """Build a human-readable/loggable snapshot of one controller sample."""

    com_position = np.asarray(fk["p_wg"], dtype=float)
    com_velocity = np.asarray(fk["v_wg"], dtype=float)
    omega = np.sqrt(
        9.81 / max(float(planner_output["com_pos_des"][2]), 0.10)
    )
    actual_dcm = com_position.copy()
    actual_dcm[:2] += com_velocity[:2] / omega
    commanded_r, commanded_l = _commanded_foot_loads(point_forces)
    right_rpy = rot2euler(fk["R_wa_r"])
    left_rpy = rot2euler(fk["R_wa_l"])
    base_rpy = rot2euler(fk["R_wb"])

    return {
        "time": elapsed_time,
        "planner_time": planner_time,
        "stage": planner_output["stage"],
        "step_index": planner_output["step_index"],
        "support_side": planner_output["support_side"],
        "swing_side": planner_output["swing_side"],
        "solver_status": controller_state.last_status,
        "solver_failures": controller_state.failure_count,
        "dynamics_residual": controller_state.last_dynamics_residual,
        "contact_residual": controller_state.last_contact_residual,
        "com_x": com_position[0],
        "com_y": com_position[1],
        "com_z": com_position[2],
        "com_des_x": planner_output["com_pos_des"][0],
        "com_des_y": planner_output["com_pos_des"][1],
        "com_des_z": planner_output["com_pos_des"][2],
        "com_vx": com_velocity[0],
        "com_vy": com_velocity[1],
        "com_vz": com_velocity[2],
        "dcm_x": actual_dcm[0],
        "dcm_y": actual_dcm[1],
        "dcm_des_x": planner_output["dcm_pos_des"][0],
        "dcm_des_y": planner_output["dcm_pos_des"][1],
        "right_ankle_x": fk["p_wa_r"][0],
        "right_ankle_y": fk["p_wa_r"][1],
        "right_ankle_z": fk["p_wa_r"][2],
        "left_ankle_x": fk["p_wa_l"][0],
        "left_ankle_y": fk["p_wa_l"][1],
        "left_ankle_z": fk["p_wa_l"][2],
        "right_roll": right_rpy[0],
        "right_pitch": right_rpy[1],
        "right_yaw": right_rpy[2],
        "left_roll": left_rpy[0],
        "left_pitch": left_rpy[1],
        "left_yaw": left_rpy[2],
        "base_roll": base_rpy[0],
        "base_pitch": base_rpy[1],
        "base_yaw": base_rpy[2],
        "force_measured_r": force_r,
        "force_measured_l": force_l,
        "force_commanded_r": commanded_r,
        "force_commanded_l": commanded_l,
        "maximum_joint_torque": float(np.max(np.abs(torque_isaac))),
        "safety_reason": planner_output["safety_reason"],
    }


def _show_final_plots(history: Dict[str, list[Any]]) -> None:
    """Display completed telemetry and wait until the user closes the window."""

    if not history["time"]:
        print("[PRIME] No telemetry samples were collected.", flush=True)
        return

    plt = original_test.plt
    plt.ioff()
    figure, axes = plt.subplots(2, 3, figsize=(15, 8))
    figure.canvas.manager.set_window_title("THEMIS final controller telemetry")
    time_axis = np.asarray(history["time"], dtype=float)

    axes[0, 0].plot(time_axis, history["dcm_x"], label="actual x")
    axes[0, 0].plot(time_axis, history["dcm_y"], label="actual y")
    axes[0, 0].plot(time_axis, history["dcm_des_x"], "--", label="desired x")
    axes[0, 0].plot(time_axis, history["dcm_des_y"], "--", label="desired y")
    axes[0, 0].set_title("DCM position")
    axes[0, 0].set_ylabel("m")

    axes[0, 1].plot(time_axis, history["com_x"], label="actual x")
    axes[0, 1].plot(time_axis, history["com_y"], label="actual y")
    axes[0, 1].plot(time_axis, history["com_z"], label="actual z")
    axes[0, 1].plot(time_axis, history["com_des_x"], "--", label="desired x")
    axes[0, 1].plot(time_axis, history["com_des_y"], "--", label="desired y")
    axes[0, 1].plot(time_axis, history["com_des_z"], "--", label="desired z")
    axes[0, 1].set_title("COM position")
    axes[0, 1].set_ylabel("m")

    axes[0, 2].plot(time_axis, history["right_ankle_x"], label="right x")
    axes[0, 2].plot(time_axis, history["left_ankle_x"], label="left x")
    axes[0, 2].plot(time_axis, history["right_ankle_z"], "--", label="right z")
    axes[0, 2].plot(time_axis, history["left_ankle_z"], "--", label="left z")
    axes[0, 2].set_title("Ankle trajectories")
    axes[0, 2].set_ylabel("m")

    axes[1, 0].plot(time_axis, history["force_measured_r"], label="measured R")
    axes[1, 0].plot(time_axis, history["force_measured_l"], label="measured L")
    axes[1, 0].plot(time_axis, history["force_commanded_r"], "--", label="QP R")
    axes[1, 0].plot(time_axis, history["force_commanded_l"], "--", label="QP L")
    axes[1, 0].set_title("Vertical foot forces")
    axes[1, 0].set_ylabel("N")

    axes[1, 1].plot(time_axis, history["right_roll"], label="right roll")
    axes[1, 1].plot(time_axis, history["right_pitch"], label="right pitch")
    axes[1, 1].plot(time_axis, history["right_yaw"], label="right yaw")
    axes[1, 1].plot(time_axis, history["left_roll"], "--", label="left roll")
    axes[1, 1].plot(time_axis, history["left_pitch"], "--", label="left pitch")
    axes[1, 1].plot(time_axis, history["left_yaw"], "--", label="left yaw")
    axes[1, 1].set_title("Foot orientation")
    axes[1, 1].set_ylabel("rad")

    axes[1, 2].plot(time_axis, history["base_roll"], label="roll")
    axes[1, 2].plot(time_axis, history["base_pitch"], label="pitch")
    axes[1, 2].plot(time_axis, history["base_yaw"], label="yaw")
    axes[1, 2].set_title("Base orientation")
    axes[1, 2].set_ylabel("rad")

    for axis in axes.flat:
        axis.set_xlabel("time [s]")
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best", fontsize=8)
    figure.suptitle("THEMIS contact-constrained walking telemetry")
    figure.tight_layout()
    print(
        "[PRIME] Final plots are open. Close the plot window to end the process.",
        flush=True,
    )
    plt.show(block=True)


def _print_summary(
    history: Dict[str, list[Any]],
    initial_right_ankle: np.ndarray | None,
    initial_left_ankle: np.ndarray | None,
    final_fk: Dict[str, np.ndarray] | None,
    planner: AlternatingGaitPlanner,
    controller_state: ControllerState,
) -> None:
    """Print the measured outcome rather than inferring success from a timer."""

    if final_fk is None or initial_right_ankle is None or initial_left_ankle is None:
        print("[PRIME] Simulation ended before a complete state was read.", flush=True)
        return

    right_displacement = final_fk["p_wa_r"] - initial_right_ankle
    left_displacement = final_fk["p_wa_l"] - initial_left_ankle
    maximum_tilt = 0.0
    if history["base_roll"]:
        maximum_tilt = float(
            np.max(
                np.abs(
                    np.column_stack(
                        (history["base_roll"], history["base_pitch"])
                    )
                )
            )
        )
    print(
        "\n[PRIME RESULT]\n"
        f"  completed load-confirmed steps : {planner.step_index}\n"
        f"  final gait stage               : {planner.stage.value}\n"
        "  right ankle displacement [m]  : "
        f"{np.array2string(right_displacement, precision=4)}\n"
        "  left ankle displacement [m]   : "
        f"{np.array2string(left_displacement, precision=4)}\n"
        f"  maximum recorded base tilt    : {np.rad2deg(maximum_tilt):.2f} deg\n"
        f"  final QP status                : {controller_state.last_status}\n"
        f"  consecutive QP failures       : {controller_state.failure_count}\n"
        f"  planner safety reason         : "
        f"{getattr(planner, '_safety_reason', '') or 'none'}",
        flush=True,
    )


def main() -> None:
    """Create the Isaac scene and run the prime planner/QP control loop."""

    if args_cli.num_envs != 1:
        raise ValueError(
            "locomotion_test_prime currently validates one robot at a time; "
            "run it with --num_envs 1"
        )

    # Build the same robot/ground/sensor scene as the untouched legacy test.
    sim_cfg = original_test.sim_utils.SimulationCfg(
        device=args_cli.device,
        use_fabric=False,
    )
    sim = original_test.sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([3.2, 3.2, 2.0], [0.0, 0.0, 0.75])
    scene = original_test.InteractiveScene(
        original_test.ThemisSceneCfg(args_cli.num_envs, env_spacing=3.0)
    )
    sim.reset()
    robot = scene["themis"]

    # Reset the articulation to the configured crouched, flat-foot pose.
    target_position = torch.tensor(
        [
            original_test.THEMIS_INIT_JOINTS.get(name, 0.0)
            for name in robot.joint_names
        ],
        dtype=torch.float32,
        device=sim.device,
    ).repeat(args_cli.num_envs, 1)
    zero_velocity = torch.zeros_like(target_position)
    root_state = robot.data.default_root_state.clone()
    scene.reset_to(
        state={
            "articulation": {
                "themis": {
                    "root_pose": root_state[:, :7],
                    "root_velocity": root_state[:, 7:],
                    "joint_position": target_position,
                    "joint_velocity": zero_velocity,
                }
            }
        },
        is_relative=True,
    )
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim.get_physics_dt())

    # Effort control must not be mixed with the USD actuator PD gains.
    zero_gain = torch.zeros_like(target_position)
    for actuator in robot.actuators.values():
        actuator.stiffness[:] = 0.0
        actuator.damping[:] = 0.0
    robot.write_joint_stiffness_to_sim(zero_gain)
    robot.write_joint_damping_to_sim(zero_gain)
    robot.reset()

    controller_state = ControllerState(environment_id=0)
    gait_config = GaitPlannerConfig(
        robot_mass=float(QP_PARAMETERS["robot_mass"]),
        com_height=0.85,
        step_progression=0.1,
        swing_height=0.03,
        # Isaac's unloaded contact sensors report only about 0.4--1.4 N when
        # the geometrically on-target sole first touches the plane.  The former
        # 3%-body-weight onset gate therefore created a circular wait: the QP
        # would not load the contact until the planner acknowledged it, while
        # the planner waited for 11 N before acknowledging it.  A sub-newton
        # onset is safe here because XY/Z/orientation gates must also pass.
        contact_probe_depth=0.004,
        touchdown_onset_ratio=0.001,
        touchdown_debounce=0.0,
        touchdown_tilt_degrees=3.0,
        touchdown_blend_duration=1.50,
        touchdown_timeout=2.50,
    )
    planner = AlternatingGaitPlanner(gait_config)
    controller_parameters = dict(QP_PARAMETERS)
    controller_parameters["dt"] = float(sim.get_physics_dt())

    # Model-order limits/nominal position keep the QP's predicted joint state
    # inside Isaac's configured range without changing the robot asset.
    model_indices = np.asarray(original_test.MODEL_ORDER_IND, dtype=int)
    controller_parameters["nominal_joint_position"] = (
        target_position[0, model_indices].cpu().numpy().astype(np.float64)
    )
    if hasattr(robot.data, "soft_joint_pos_limits"):
        soft_limits = (
            robot.data.soft_joint_pos_limits[0, model_indices]
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        controller_parameters["joint_position_min"] = soft_limits[:, 0]
        controller_parameters["joint_position_max"] = soft_limits[:, 1]

    telemetry_history: Dict[str, list[Any]] = defaultdict(list)
    telemetry_path = os.path.join(PROJECT_ROOT, "prime_telemetry.csv")
    telemetry_file = open(telemetry_path, "w", newline="")
    telemetry_writer = csv.DictWriter(
        telemetry_file,
        fieldnames=TELEMETRY_COLUMNS,
    )
    telemetry_writer.writeheader()

    elapsed_time = 0.0
    planner_time = 0.0
    next_telemetry_time = 0.0
    telemetry_period = 0.10
    previous_stage = None
    initial_right_ankle = None
    initial_left_ankle = None
    final_fk = None
    interrupted = False

    try:
        while simulation_app.is_running():
            sim_dt = float(sim.get_physics_dt())
            state = original_test._get_robot_state(robot)
            fk = compute_FK(
                state["q_raw_all"][0],
                state["dq_raw_all"][0],
                state["R_wb_np"][0],
                state["p_wb_np"][0],
                state["w_bb_np"][0],
                state["v_bb_np"][0],
            )
            final_fk = fk
            if initial_right_ankle is None:
                initial_right_ankle = fk["p_wa_r"].copy()
                initial_left_ankle = fk["p_wa_l"].copy()

            if args_cli.validate_model:
                original_test._validate_kinematic_model(robot, state, fk)
                print(
                    "[MODEL VALIDATION] H/CG/AG shapes: "
                    f"{fk['H'].shape}, {fk['CG'].shape}, {fk['AG'].shape}",
                    flush=True,
                )
                return

            force_l, force_r = _foot_forces(scene)

            # Freeze only the planner clock after a QP failure; physical time
            # and safety measurements continue.  A solved cycle releases it.
            if (
                not args_cli.freeze_reference
                and controller_state.failure_count == 0
            ):
                planner_time += sim_dt
            measurement = _make_measurement(
                planner_time,
                state,
                fk,
                force_l,
                force_r,
            )
            planner_output = planner.sample(measurement)

            torque_isaac, _, _, _, _, point_forces = run_wbc_qp(
                controller_state,
                fk,
                planner_output,
                controller_parameters,
                force_l,
                force_r,
            )
            torque_tensor = torch.tensor(
                torque_isaac,
                dtype=torch.float32,
                device=sim.device,
            ).unsqueeze(0)
            robot.set_joint_effort_target(torque_tensor)

            stage = planner_output["stage"]
            if stage != previous_stage:
                print(
                    f"\n[GAIT t={elapsed_time:6.2f}s] "
                    f"stage={stage}, support={planner_output['support_side']}, "
                    f"swing={planner_output['swing_side']}, "
                    f"step={planner_output['step_index']}",
                    flush=True,
                )
                if planner_output["safety_reason"]:
                    print(
                        "[GAIT SAFETY] "
                        f"{planner_output['safety_reason']}",
                        flush=True,
                    )
                previous_stage = stage

            if elapsed_time + 1.0e-9 >= next_telemetry_time:
                row = _telemetry_row(
                    elapsed_time,
                    planner_time,
                    planner_output,
                    controller_state,
                    fk,
                    force_l,
                    force_r,
                    point_forces,
                    torque_isaac,
                )
                telemetry_writer.writerow(row)
                telemetry_file.flush()
                for name in TELEMETRY_COLUMNS:
                    telemetry_history[name].append(row[name])
                print(
                    f"[LIVE t={elapsed_time:6.2f}s] "
                    f"{stage:>10s} step={planner_output['step_index']} | "
                    f"Fz L/R={force_l:6.1f}/{force_r:6.1f} N | "
                    f"COM=({row['com_x']:+.3f},{row['com_y']:+.3f},"
                    f"{row['com_z']:+.3f}) m | "
                    f"QP={controller_state.last_status} "
                    f"res={controller_state.last_dynamics_residual:.1e}",
                    flush=True,
                )
                if controller_state.last_message:
                    print(
                        f"[QP] {controller_state.last_message}",
                        flush=True,
                    )
                next_telemetry_time += telemetry_period

            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)
            elapsed_time += sim_dt

            # A small pause makes the GUI easier to inspect without changing
            # simulated time.  Headless validation runs as fast as possible.
            if not getattr(args_cli, "headless", False):
                time.sleep(0.002)

            if (
                args_cli.duration is not None
                and elapsed_time >= args_cli.duration
            ):
                break
    except KeyboardInterrupt:
        interrupted = True
        print(
            "\n[PRIME] Simulation interrupted; preparing final telemetry.",
            flush=True,
        )
    finally:
        telemetry_file.close()

    _print_summary(
        telemetry_history,
        initial_right_ankle,
        initial_left_ankle,
        final_fk,
        planner,
        controller_state,
    )
    print(f"[PRIME] Telemetry saved to {telemetry_path}", flush=True)

    # The requested behavior is an end-of-run plot that remains visible.
    # Headless/--no_live_plot runs intentionally skip the GUI.
    if not args_cli.no_live_plot and not getattr(args_cli, "headless", False):
        _show_final_plots(telemetry_history)
    elif interrupted:
        print("[PRIME] Final plot suppressed by command-line options.", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()