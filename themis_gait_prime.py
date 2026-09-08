"""Contact-aware slow alternating gait planner for THEMIS.

This module contains only reference generation and the gait state machine.  It
does not compute joint torques.  The whole-body controller is expected to track
the returned COM, DCM, swing-foot, and contact-transition references.

The planner intentionally starts with a slow quasi-static gait instead of the
full dynamic Xi-DES trajectory.  Each new foothold advances the robot by 5 cm:

* the right foot moves to ``x0 + 0.05``;
* the left foot moves to ``x0 + 0.10``;
* the right foot then moves to ``x0 + 0.15``; and so on.

Consequently, after the first short placement, each individual foot travels
10 cm while the body advances 5 cm per contact exchange.  The measured initial
right and left lateral coordinates are retained as fixed walking lanes, so the
feet are never intentionally crossed.

All positions are in metres, velocities in metres per second, accelerations in
metres per second squared, forces in newtons, and rotations are 3x3 matrices
mapping local-frame vectors into the world frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import numpy as np


class GaitStage(str, Enum):
    """Hybrid states used by :class:`AlternatingGaitPlanner`."""

    PREPARE_DS = "prepare_ds"
    SHIFT = "shift"
    LIFT = "lift"
    TRANSFER = "transfer"
    LOWER = "lower"
    TOUCHDOWN = "touchdown"
    RECOVER = "recover"


@dataclass(frozen=True)
class GaitMeasurement:
    """Measured robot state consumed once per planner update.

    ``base_rotation`` and both ankle rotations map vectors from their local
    frames into the world frame.  The two contact loads are total vertical
    forces summed over all contact sensors belonging to that foot.
    """

    time: float
    com_position: np.ndarray
    com_velocity: np.ndarray
    base_rotation: np.ndarray
    base_angular_velocity: np.ndarray
    right_ankle_position: np.ndarray
    right_ankle_velocity: np.ndarray
    right_ankle_rotation: np.ndarray
    left_ankle_position: np.ndarray
    left_ankle_velocity: np.ndarray
    left_ankle_rotation: np.ndarray
    contact_force_r_z: float
    contact_force_l_z: float


@dataclass(frozen=True)
class GaitPlannerConfig:
    """Timing, geometry, and safety settings for the slow gait.

    The default mass matches the dynamics model used elsewhere in this
    repository (37.86976 kg).  Force thresholds are expressed as fractions of
    body weight so that contact decisions remain meaningful if the mass is
    changed.
    """

    robot_mass: float = 40.86976
    gravity: float = 9.81
    com_height: float = 0.85

    # One new alternating contact every 10 cm; each same-side foot subsequently
    # travels 10 cm.
    step_progression: float = 0.10
    swing_height: float = 0.03
    contact_probe_depth: float = 0.002

    prepare_duration: float = 1.00
    shift_duration: float = 1.5
    lift_duration: float = 0.80
    transfer_duration: float = 1.20
    lower_duration: float = 0.70
    touchdown_blend_duration: float = 0.60

    prepare_timeout: float = 4.00
    shift_timeout: float = 5.00
    lift_timeout: float = 1.80
    transfer_timeout: float = 2.50
    lower_timeout: float = 2.00
    touchdown_timeout: float = 1.80
    recovery_timeout: float = 3.00

    # The ankle frame is behind the approximate sole centre in the nominal
    # model.  These dimensions are used only by the planner's DCM safety gate;
    # a QP controller should use the exact contact geometry.
    support_center_offset_x: float = 0.040
    support_half_length: float = 0.070
    support_half_width: float = 0.045
    support_polygon_margin: float = 0.005

    ready_total_load_ratio: float = 0.75
    ready_stance_share: float = 0.80
    ready_swing_share: float = 0.20
    ready_com_speed: float = 0.15
    ready_tilt_degrees: float = 7.0
    ready_com_error: float = 0.04

    liftoff_force_ratio: float = 0.05
    liftoff_clearance: float = 0.020
    touchdown_onset_ratio: float = 0.03
    touchdown_loaded_ratio: float = 0.20
    touchdown_old_stance_ratio: float = 0.20
    touchdown_total_ratio: float = 0.75

    transfer_position_tolerance: float = 0.015
    transfer_height_tolerance: float = 0.015
    touchdown_position_tolerance: float = 0.020
    touchdown_height_tolerance: float = 0.015
    touchdown_tilt_degrees: float = 8.0

    ready_debounce: float = 0.15
    liftoff_debounce: float = 0.08
    touchdown_debounce: float = 0.08
    loaded_debounce: float = 0.15
    unsafe_debounce: float = 0.05

    minimum_stance_load_ratio: float = 0.45
    maximum_com_speed: float = 0.50
    maximum_tilt_degrees: float = 10.0
    maximum_swing_tracking_error: float = 0.06
    recovery_descent_speed: float = 0.08
    initial_safety_grace: float = 0.50

    # A recovered gait remains in a safe double-support hold by default.  A
    # higher-level supervisor may reset the planner after inspecting the fault.
    resume_after_recovery: bool = False

    def __post_init__(self) -> None:
        positive = (
            self.robot_mass,
            self.gravity,
            self.com_height,
            self.step_progression,
            self.swing_height,
            self.prepare_duration,
            self.shift_duration,
            self.lift_duration,
            self.transfer_duration,
            self.lower_duration,
            self.touchdown_blend_duration,
        )
        if any(value <= 0.0 for value in positive):
            raise ValueError("mass, geometry, and nominal gait durations must be positive")
        if not 0.0 <= self.contact_probe_depth <= 0.01:
            raise ValueError("contact_probe_depth must lie between 0 and 1 cm")
        ratios = (
            self.ready_total_load_ratio,
            self.ready_stance_share,
            self.ready_swing_share,
            self.liftoff_force_ratio,
            self.touchdown_onset_ratio,
            self.touchdown_loaded_ratio,
            self.touchdown_old_stance_ratio,
            self.touchdown_total_ratio,
            self.minimum_stance_load_ratio,
        )
        if any(not 0.0 <= value <= 1.0 for value in ratios):
            raise ValueError("all force ratios must lie in [0, 1]")
        if self.ready_swing_share >= self.ready_stance_share:
            raise ValueError("ready swing share must be below ready stance share")


class AlternatingGaitPlanner:
    """Generate slow right-left stepping references with measured safety gates.

    Call :meth:`sample` once per outer control update.  The first call anchors
    the walking lanes, ankle ground heights, flat-foot orientations, and COM
    reference to the measured robot pose.

    The planner starts by moving the COM smoothly over the left foot and then
    swings the right foot.  After a load-confirmed right touchdown it moves the
    COM over the right foot and swings the left foot.  This sequence repeats
    until a safety condition places the planner in :class:`GaitStage.RECOVER`.
    """

    def __init__(self, config: Optional[GaitPlannerConfig] = None):
        self.config = config or GaitPlannerConfig()
        self.reset()

    @property
    def stage(self) -> GaitStage:
        """Current hybrid gait stage."""
        return self._stage

    @property
    def step_index(self) -> int:
        """Number of load-confirmed footsteps completed since reset."""
        return self._step_index

    @property
    def body_weight(self) -> float:
        """Nominal robot weight in newtons."""
        return self.config.robot_mass * self.config.gravity

    def reset(self) -> None:
        """Forget all anchors and restart with a right-foot-first gait."""
        self._initialized = False
        self._stage = GaitStage.PREPARE_DS
        self._stage_enter_time = 0.0
        self._initial_time = 0.0
        self._last_time = None
        self._reference_time = 0.0

        self._step_index = 0
        self._swing_side = "right"
        self._stance_side = "left"
        self._lane_y: Dict[str, float] = {}
        self._ground_ankle_z: Dict[str, float] = {}
        self._initial_mid_x = 0.0
        self._flat_foot_rotation: Dict[str, np.ndarray] = {}

        self._com_hold = np.zeros(3)
        self._shift_start = np.zeros(3)
        self._shift_target = np.zeros(3)
        self._touchdown_com_start = np.zeros(3)
        self._touchdown_com_target = np.zeros(3)
        self._swing_start = np.zeros(3)
        self._swing_target = np.zeros(3)
        self._swing_apex_z = 0.0

        self._ready_since = None
        self._liftoff_since = None
        self._touchdown_since = None
        self._loaded_since = None
        self._unsafe_since = None
        self._unsafe_reason = ""

        self._recovery_start = np.zeros(3)
        self._recovery_target = np.zeros(3)
        self._recovery_contact_since = None
        self._recovery_landed = False
        self._halt_requested = False
        self._safety_reason = ""

    # ------------------------------------------------------------------
    # Geometry and smooth-reference helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _vector(value: Any, size: int, name: str) -> np.ndarray:
        result = np.asarray(value, dtype=float).reshape(-1)
        if result.shape != (size,) or not np.all(np.isfinite(result)):
            raise ValueError(f"{name} must contain {size} finite values")
        return result.copy()

    @staticmethod
    def _rotation(value: Any, name: str) -> np.ndarray:
        result = np.asarray(value, dtype=float)
        if result.shape != (3, 3) or not np.all(np.isfinite(result)):
            raise ValueError(f"{name} must be a finite 3x3 rotation matrix")
        # Reject grossly malformed matrices while tolerating normal numerical
        # round-off from a simulator quaternion conversion.
        if np.linalg.norm(result.T @ result - np.eye(3), ord="fro") > 5e-2:
            raise ValueError(f"{name} is not approximately orthonormal")
        return result.copy()

    def _normalize_measurement(self, measured: GaitMeasurement) -> GaitMeasurement:
        time = float(measured.time)
        if not np.isfinite(time):
            raise ValueError("measurement time must be finite")
        force_r = abs(float(measured.contact_force_r_z))
        force_l = abs(float(measured.contact_force_l_z))
        if not np.isfinite(force_r) or not np.isfinite(force_l):
            raise ValueError("contact forces must be finite")
        return GaitMeasurement(
            time=time,
            com_position=self._vector(measured.com_position, 3, "com_position"),
            com_velocity=self._vector(measured.com_velocity, 3, "com_velocity"),
            base_rotation=self._rotation(measured.base_rotation, "base_rotation"),
            base_angular_velocity=self._vector(
                measured.base_angular_velocity, 3, "base_angular_velocity"
            ),
            right_ankle_position=self._vector(
                measured.right_ankle_position, 3, "right_ankle_position"
            ),
            right_ankle_velocity=self._vector(
                measured.right_ankle_velocity, 3, "right_ankle_velocity"
            ),
            right_ankle_rotation=self._rotation(
                measured.right_ankle_rotation, "right_ankle_rotation"
            ),
            left_ankle_position=self._vector(
                measured.left_ankle_position, 3, "left_ankle_position"
            ),
            left_ankle_velocity=self._vector(
                measured.left_ankle_velocity, 3, "left_ankle_velocity"
            ),
            left_ankle_rotation=self._rotation(
                measured.left_ankle_rotation, "left_ankle_rotation"
            ),
            contact_force_r_z=force_r,
            contact_force_l_z=force_l,
        )

    @staticmethod
    def _quintic(phase: float) -> Tuple[float, float, float]:
        """Return quintic blend and first/second phase derivatives."""
        u = float(np.clip(phase, 0.0, 1.0))
        blend = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
        derivative = 30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4
        second = 60.0 * u - 180.0 * u**2 + 120.0 * u**3
        return blend, derivative, second

    @classmethod
    def _segment(
        cls,
        start: np.ndarray,
        end: np.ndarray,
        elapsed: float,
        duration: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Sample a zero-end-velocity quintic Cartesian segment."""
        phase = float(np.clip(elapsed / duration, 0.0, 1.0))
        blend, derivative, second = cls._quintic(phase)
        displacement = np.asarray(end) - np.asarray(start)
        position = np.asarray(start) + blend * displacement
        velocity = derivative * displacement / duration
        acceleration = second * displacement / (duration * duration)
        return position, velocity, acceleration, phase

    @staticmethod
    def _flat_yaw_rotation(rotation: np.ndarray) -> np.ndarray:
        """Keep measured foot yaw while requesting zero roll and pitch."""
        yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        cosine, sine = np.cos(yaw), np.sin(yaw)
        return np.array(
            [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
        )

    @staticmethod
    def _roll_pitch(rotation: np.ndarray) -> Tuple[float, float]:
        roll = float(np.arctan2(rotation[2, 1], rotation[2, 2]))
        pitch = float(
            np.arctan2(
                -rotation[2, 0],
                np.hypot(rotation[2, 1], rotation[2, 2]),
            )
        )
        return roll, pitch

    @staticmethod
    def _rotation_error_angle(desired: np.ndarray, actual: np.ndarray) -> float:
        cosine = 0.5 * (np.trace(desired.T @ actual) - 1.0)
        return float(np.arccos(np.clip(cosine, -1.0, 1.0)))

    @staticmethod
    def _ankle_position(measured: GaitMeasurement, side: str) -> np.ndarray:
        return (
            measured.right_ankle_position
            if side == "right"
            else measured.left_ankle_position
        )

    @staticmethod
    def _ankle_velocity(measured: GaitMeasurement, side: str) -> np.ndarray:
        return (
            measured.right_ankle_velocity
            if side == "right"
            else measured.left_ankle_velocity
        )

    @staticmethod
    def _ankle_rotation(measured: GaitMeasurement, side: str) -> np.ndarray:
        return (
            measured.right_ankle_rotation
            if side == "right"
            else measured.left_ankle_rotation
        )

    @staticmethod
    def _foot_force(measured: GaitMeasurement, side: str) -> float:
        return (
            measured.contact_force_r_z
            if side == "right"
            else measured.contact_force_l_z
        )

    def _actual_dcm(self, measured: GaitMeasurement) -> np.ndarray:
        omega = np.sqrt(self.config.gravity / self.config.com_height)
        dcm = measured.com_position.copy()
        dcm[:2] += measured.com_velocity[:2] / omega
        dcm[2] = self.config.com_height
        return dcm

    def _dcm_inside_support(
        self,
        dcm: np.ndarray,
        measured: GaitMeasurement,
        side: str,
        margin: Optional[float] = None,
    ) -> bool:
        """Approximate whether horizontal DCM lies inside one sole polygon."""
        cfg = self.config
        ankle = self._ankle_position(measured, side)
        rotation = self._ankle_rotation(measured, side)
        centre = ankle + rotation @ np.array(
            [cfg.support_center_offset_x, 0.0, 0.0]
        )
        delta_world = np.array(
            [dcm[0] - centre[0], dcm[1] - centre[1], 0.0]
        )
        delta_foot = rotation.T @ delta_world
        inset = cfg.support_polygon_margin if margin is None else float(margin)
        half_length = max(cfg.support_half_length - inset, 0.0)
        half_width = max(cfg.support_half_width - inset, 0.0)
        return bool(
            abs(delta_foot[0]) <= half_length
            and abs(delta_foot[1]) <= half_width
        )

    # ------------------------------------------------------------------
    # State setup, gates, and safety supervision
    # ------------------------------------------------------------------
    def _initialize(self, measured: GaitMeasurement) -> None:
        cfg = self.config
        self._initial_time = measured.time
        self._last_time = measured.time
        self._stage_enter_time = measured.time

        self._lane_y = {
            "right": float(measured.right_ankle_position[1]),
            "left": float(measured.left_ankle_position[1]),
        }
        self._ground_ankle_z = {
            "right": float(measured.right_ankle_position[2]),
            "left": float(measured.left_ankle_position[2]),
        }
        self._initial_mid_x = 0.5 * float(
            measured.right_ankle_position[0]
            + measured.left_ankle_position[0]
        )
        self._flat_foot_rotation = {
            "right": self._flat_yaw_rotation(measured.right_ankle_rotation),
            "left": self._flat_yaw_rotation(measured.left_ankle_rotation),
        }

        self._com_hold = measured.com_position.copy()
        self._com_hold[2] = cfg.com_height
        self._shift_start = self._com_hold.copy()
        self._shift_target = self._stance_com_target(measured, self._stance_side)
        self._swing_start = measured.right_ankle_position.copy()
        self._swing_target = self._next_foot_target("right")
        self._swing_apex_z = max(
            self._swing_start[2], self._swing_target[2]
        ) + cfg.swing_height
        self._initialized = True

    def _next_foot_target(self, side: str) -> np.ndarray:
        # Event 0 lands at +5 cm, event 1 at +10 cm, etc.  Fixed measured lane
        # y values prevent accidental foot crossing.
        event_x = self._initial_mid_x + (
            self._step_index + 1
        ) * self.config.step_progression
        return np.array(
            [
                event_x,
                self._lane_y[side],
                self._ground_ankle_z[side]
                - self.config.contact_probe_depth,
            ],
            dtype=float,
        )

    def _stance_com_target(
        self, measured: GaitMeasurement, side: str
    ) -> np.ndarray:
        ankle = self._ankle_position(measured, side)
        rotation = self._ankle_rotation(measured, side)
        sole_centre = ankle + rotation @ np.array(
            [self.config.support_center_offset_x, 0.0, 0.0]
        )
        return np.array(
            [sole_centre[0], sole_centre[1], self.config.com_height],
            dtype=float,
        )

    def _double_support_com_target(
        self, measured: GaitMeasurement
    ) -> np.ndarray:
        """Return the midpoint of the two measured sole centres.

        A newly landed foot cannot accept significant vertical load while the
        COM remains directly above the old stance foot.  Moving to the support
        midpoint during touchdown creates a physically consistent load ramp;
        the following SHIFT stage then moves from this midpoint to the new
        single-support foot.
        """

        right = self._stance_com_target(measured, "right")
        left = self._stance_com_target(measured, "left")
        midpoint = 0.5 * (right + left)
        midpoint[2] = self.config.com_height
        return midpoint

    def _enter(self, stage: GaitStage, measured: GaitMeasurement) -> None:
        """Enter a stage and capture every reference needed by that stage."""
        self._stage = stage
        self._stage_enter_time = measured.time
        self._ready_since = None
        self._liftoff_since = None
        self._touchdown_since = None
        self._loaded_since = None

        if stage == GaitStage.SHIFT:
            self._shift_start = self._com_hold.copy()
            self._shift_target = self._stance_com_target(
                measured, self._stance_side
            )
        elif stage == GaitStage.LIFT:
            self._swing_start = self._ankle_position(
                measured, self._swing_side
            ).copy()
            self._swing_target = self._next_foot_target(self._swing_side)
            self._swing_apex_z = max(
                self._swing_start[2], self._swing_target[2]
            ) + self.config.swing_height
        elif stage == GaitStage.TOUCHDOWN:
            # LOWER permits a small Cartesian tolerance so the sole can make
            # physical contact instead of chasing a reference through the
            # floor.  Once contact is detected, hold that exact measured pose.
            # The QP captures the same pose as its rigid-contact anchor, which
            # prevents the soft swing task and hard contact constraint from
            # requesting two different accelerations during the load ramp.
            self._swing_target = self._ankle_position(
                measured, self._swing_side
            ).copy()
            self._touchdown_com_start = self._com_hold.copy()
            self._touchdown_com_target = self._double_support_com_target(
                measured
            )
        elif stage == GaitStage.RECOVER:
            self._recovery_start = self._ankle_position(
                measured, self._swing_side
            ).copy()
            self._recovery_target = self._recovery_start.copy()
            self._recovery_target[2] = (
                self._ground_ankle_z[self._swing_side]
                - self.config.contact_probe_depth
            )
            self._recovery_contact_since = None
            self._recovery_landed = False
            self._halt_requested = False

    def _debounced(
        self,
        condition: bool,
        measured_time: float,
        timer_name: str,
        duration: float,
    ) -> bool:
        since = getattr(self, timer_name)
        if condition:
            if since is None:
                since = measured_time
                setattr(self, timer_name, since)
            return measured_time - since >= duration
        setattr(self, timer_name, None)
        return False

    def _base_tilt(self, measured: GaitMeasurement) -> float:
        roll, pitch = self._roll_pitch(measured.base_rotation)
        return max(abs(roll), abs(pitch))

    def _double_support_stable(self, measured: GaitMeasurement) -> bool:
        cfg = self.config
        total = measured.contact_force_r_z + measured.contact_force_l_z
        return bool(
            total >= cfg.ready_total_load_ratio * self.body_weight
            and measured.contact_force_r_z >= 0.10 * self.body_weight
            and measured.contact_force_l_z >= 0.10 * self.body_weight
            and np.linalg.norm(measured.com_velocity[:2])
            <= cfg.ready_com_speed
            and self._base_tilt(measured)
            <= np.deg2rad(cfg.ready_tilt_degrees)
        )

    def _support_ready(self, measured: GaitMeasurement) -> bool:
        cfg = self.config
        stance_force = self._foot_force(measured, self._stance_side)
        swing_force = self._foot_force(measured, self._swing_side)
        total = stance_force + swing_force
        if total <= 1e-6:
            return False
        actual_dcm = self._actual_dcm(measured)
        return bool(
            total >= cfg.ready_total_load_ratio * self.body_weight
            and stance_force / total >= cfg.ready_stance_share
            and swing_force / total <= cfg.ready_swing_share
            # The active-contact mask is released at the next stage boundary.
            # Requiring the foot to be genuinely unloaded here prevents a
            # finite contact wrench from disappearing in one control sample.
            and swing_force
            <= cfg.liftoff_force_ratio * self.body_weight
            and np.linalg.norm(measured.com_velocity[:2])
            <= cfg.ready_com_speed
            and self._base_tilt(measured)
            <= np.deg2rad(cfg.ready_tilt_degrees)
            and np.linalg.norm(measured.com_position - self._shift_target)
            <= cfg.ready_com_error
            and self._dcm_inside_support(
                actual_dcm, measured, self._stance_side
            )
        )

    def _touchdown_pose_ok(self, measured: GaitMeasurement) -> bool:
        cfg = self.config
        position = self._ankle_position(measured, self._swing_side)
        rotation = self._ankle_rotation(measured, self._swing_side)
        return bool(
            np.linalg.norm(position[:2] - self._swing_target[:2])
            <= cfg.touchdown_position_tolerance
            and abs(position[2] - self._swing_target[2])
            <= cfg.touchdown_height_tolerance
            and self._rotation_error_angle(
                self._flat_foot_rotation[self._swing_side], rotation
            )
            <= np.deg2rad(cfg.touchdown_tilt_degrees)
        )

    def _unsafe_reason_now(
        self,
        measured: GaitMeasurement,
        commanded_swing_position: np.ndarray,
    ) -> str:
        """Return an active severe condition, or an empty string."""
        cfg = self.config
        if measured.time - self._initial_time < cfg.initial_safety_grace:
            return ""
        if self._base_tilt(measured) > np.deg2rad(cfg.maximum_tilt_degrees):
            return "base tilt exceeded limit"
        if np.linalg.norm(measured.com_velocity[:2]) > cfg.maximum_com_speed:
            return "horizontal COM speed exceeded limit"

        total = measured.contact_force_r_z + measured.contact_force_l_z
        if total < cfg.minimum_stance_load_ratio * self.body_weight:
            return "insufficient total support force"

        airborne_stage = self._stage in (
            GaitStage.LIFT,
            GaitStage.TRANSFER,
            GaitStage.LOWER,
        )
        if airborne_stage:
            stance_force = self._foot_force(measured, self._stance_side)
            if stance_force < cfg.minimum_stance_load_ratio * self.body_weight:
                return "stance-foot load was lost"
            # Use an expanded polygon for the emergency threshold.  The tighter
            # inset polygon is used by the pre-lift readiness gate.
            if not self._dcm_inside_support(
                self._actual_dcm(measured),
                measured,
                self._stance_side,
                margin=-0.010,
            ):
                return "actual DCM left the stance support polygon"
            tracking_error = np.linalg.norm(
                self._ankle_position(measured, self._swing_side)
                - commanded_swing_position
            )
            if (
                measured.time - self._stage_enter_time > 0.20
                and tracking_error > cfg.maximum_swing_tracking_error
            ):
                return "swing-foot tracking error exceeded limit"
        return ""

    def _request_recovery(
        self, measured: GaitMeasurement, reason: str
    ) -> None:
        self._safety_reason = reason
        # Capture the measured COM instead of jumping back to the previous
        # phase endpoint.  Recovery should arrest the current motion, not
        # introduce a second reference discontinuity.
        self._com_hold = measured.com_position.copy()
        self._com_hold[2] = self.config.com_height
        self._enter(GaitStage.RECOVER, measured)

    # ------------------------------------------------------------------
    # Stage references and transitions
    # ------------------------------------------------------------------
    def _nominal_duration(self) -> Optional[float]:
        cfg = self.config
        return {
            GaitStage.PREPARE_DS: cfg.prepare_duration,
            GaitStage.SHIFT: cfg.shift_duration,
            GaitStage.LIFT: cfg.lift_duration,
            GaitStage.TRANSFER: cfg.transfer_duration,
            GaitStage.LOWER: cfg.lower_duration,
            GaitStage.TOUCHDOWN: cfg.touchdown_blend_duration,
            GaitStage.RECOVER: None,
        }[self._stage]

    def _advance_reference_clock(self, dt: float, state_elapsed: float) -> None:
        duration = self._nominal_duration()
        if duration is None or state_elapsed >= duration:
            return
        self._reference_time += min(dt, duration - state_elapsed)

    def _com_reference(
        self, state_elapsed: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._stage == GaitStage.SHIFT:
            position, velocity, acceleration, _ = self._segment(
                self._shift_start,
                self._shift_target,
                state_elapsed,
                self.config.shift_duration,
            )
            return position, velocity, acceleration
        if self._stage == GaitStage.TOUCHDOWN:
            position, velocity, acceleration, _ = self._segment(
                self._touchdown_com_start,
                self._touchdown_com_target,
                state_elapsed,
                self.config.touchdown_blend_duration,
            )
            return position, velocity, acceleration
        return (
            self._com_hold.copy(),
            np.zeros(3),
            np.zeros(3),
        )

    def _swing_reference(
        self,
        measured: GaitMeasurement,
        state_elapsed: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        cfg = self.config
        if self._stage == GaitStage.LIFT:
            lift_end = self._swing_start.copy()
            lift_end[2] = self._swing_apex_z
            position, velocity, acceleration, phase = self._segment(
                self._swing_start,
                lift_end,
                state_elapsed,
                cfg.lift_duration,
            )
            return position, velocity, acceleration, 0.20 * phase

        if self._stage == GaitStage.TRANSFER:
            transfer_start = self._swing_start.copy()
            transfer_start[2] = self._swing_apex_z
            transfer_end = self._swing_target.copy()
            transfer_end[2] = self._swing_apex_z
            position, velocity, acceleration, phase = self._segment(
                transfer_start,
                transfer_end,
                state_elapsed,
                cfg.transfer_duration,
            )
            return position, velocity, acceleration, 0.20 + 0.60 * phase

        if self._stage == GaitStage.LOWER:
            lower_start = self._swing_target.copy()
            lower_start[2] = self._swing_apex_z
            position, velocity, acceleration, phase = self._segment(
                lower_start,
                self._swing_target,
                state_elapsed,
                cfg.lower_duration,
            )
            return position, velocity, acceleration, 0.80

        if self._stage == GaitStage.TOUCHDOWN:
            timed_blend, _, _ = self._quintic(
                state_elapsed / cfg.touchdown_blend_duration
            )
            measured_load_blend = np.clip(
                self._foot_force(measured, self._swing_side)
                / (cfg.touchdown_loaded_ratio * self.body_weight),
                0.0,
                1.0,
            )
            # Keep the legacy swing_phase interface contact-aware as well as
            # the explicit contact weights.  A controller that has not yet
            # adopted the new fields will therefore pause its load exchange
            # when touchdown force disappears.
            blend = min(timed_blend, measured_load_blend)
            return (
                self._swing_target.copy(),
                np.zeros(3),
                np.zeros(3),
                0.80 + 0.20 * blend,
            )

        if self._stage == GaitStage.RECOVER:
            distance = abs(self._recovery_start[2] - self._recovery_target[2])
            descent_duration = max(
                distance / cfg.recovery_descent_speed, 0.25
            )
            position, velocity, acceleration, phase = self._segment(
                self._recovery_start,
                self._recovery_target,
                state_elapsed,
                descent_duration,
            )
            contact = self._foot_force(measured, self._swing_side)
            load_blend = np.clip(
                contact / (cfg.touchdown_loaded_ratio * self.body_weight),
                0.0,
                1.0,
            )
            swing_phase = 0.80 + 0.20 * min(phase, load_blend)
            return position, velocity, acceleration, swing_phase

        # In both double-support states the next swing foot remains fixed.
        position = self._ankle_position(measured, self._swing_side).copy()
        return position, np.zeros(3), np.zeros(3), 0.0

    def _contact_schedule(
        self,
        measured: GaitMeasurement,
        state_elapsed: float,
        swing_phase: float,
    ) -> Tuple[float, float, float, Tuple[bool, bool]]:
        """Return continuous right/left load fractions and task authority.

        The two load weights sum to one in every nominal stage.  Their
        boundaries match the COM reference: balanced double support, complete
        unload before lift, single support during swing, then a balanced
        touchdown.  The booleans separately state which no-slip constraints
        are physically active.
        """
        if self._stage == GaitStage.PREPARE_DS:
            stance_contact = 0.5
            swing_contact = 0.5
            swing_task = 0.0
            active = (True, True)
        elif self._stage == GaitStage.SHIFT:
            blend, _, _ = self._quintic(
                state_elapsed / self.config.shift_duration
            )
            stance_contact = 0.5 * (1.0 + blend)
            swing_contact = 0.5 * (1.0 - blend)
            swing_task = 0.0
            active = (True, True)
        elif self._stage == GaitStage.LIFT:
            blend, _, _ = self._quintic(
                state_elapsed / self.config.lift_duration
            )
            stance_contact = 1.0
            swing_contact = 0.0
            swing_task = blend
            active = (
                self._stance_side == "right",
                self._stance_side == "left",
            )
        elif self._stage in (GaitStage.TRANSFER, GaitStage.LOWER):
            stance_contact = 1.0
            swing_contact = 0.0
            swing_task = 1.0
            active = (
                self._stance_side == "right",
                self._stance_side == "left",
            )
        elif self._stage == GaitStage.TOUCHDOWN:
            timed_blend, _, _ = self._quintic(
                state_elapsed / self.config.touchdown_blend_duration
            )
            # LOWER has already verified on-target physical contact before this
            # stage can begin.  Ramp the new support weight from that event;
            # do not multiply it by measured load.  Doing so creates a circular
            # deadlock: the QP will not command load until load is measured,
            # while the unloaded foot cannot generate the measurement.  The
            # loaded/contact gates in _advance_state still prevent committing
            # the support exchange if physical force does not follow the ramp.
            stance_contact = 1.0 - 0.5 * timed_blend
            swing_contact = 0.5 * timed_blend
            swing_task = 1.0 - timed_blend
            active = (True, True)
        else:  # RECOVER
            contact = self._foot_force(measured, self._swing_side)
            load_blend = np.clip(
                contact
                / (self.config.touchdown_loaded_ratio * self.body_weight),
                0.0,
                1.0,
            )
            stance_contact = 1.0 - 0.5 * load_blend
            swing_contact = 0.5 * load_blend
            swing_task = 1.0 - load_blend
            active = (
                self._stance_side == "right"
                or contact
                > self.config.touchdown_onset_ratio * self.body_weight,
                self._stance_side == "left"
                or contact
                > self.config.touchdown_onset_ratio * self.body_weight,
            )

        if self._swing_side == "right":
            return swing_contact, stance_contact, swing_task, active
        return stance_contact, swing_contact, swing_task, active

    def _advance_state(
        self,
        measured: GaitMeasurement,
        state_elapsed: float,
        swing_position_des: np.ndarray,
    ) -> None:
        cfg = self.config
        time = measured.time

        if self._stage == GaitStage.PREPARE_DS:
            stable = self._double_support_stable(measured)
            ready = self._debounced(
                stable,
                time,
                "_ready_since",
                cfg.ready_debounce,
            )
            if state_elapsed >= cfg.prepare_duration and ready:
                self._enter(GaitStage.SHIFT, measured)
            elif state_elapsed > cfg.prepare_timeout:
                self._request_recovery(
                    measured, "double-support preparation timed out"
                )
            return

        if self._stage == GaitStage.SHIFT:
            ready = self._debounced(
                self._support_ready(measured),
                time,
                "_ready_since",
                cfg.ready_debounce,
            )
            if state_elapsed >= cfg.shift_duration and ready:
                self._com_hold = self._shift_target.copy()
                self._enter(GaitStage.LIFT, measured)
            elif state_elapsed > cfg.shift_timeout:
                self._request_recovery(
                    measured, "stance-load transfer timed out"
                )
            return

        if self._stage == GaitStage.LIFT:
            swing_force = self._foot_force(measured, self._swing_side)
            measured_clearance = (
                self._ankle_position(measured, self._swing_side)[2]
                - self._swing_start[2]
            )
            airborne = (
                swing_force
                < cfg.liftoff_force_ratio * self.body_weight
                and measured_clearance >= cfg.liftoff_clearance
            )
            confirmed = self._debounced(
                airborne,
                time,
                "_liftoff_since",
                cfg.liftoff_debounce,
            )
            if state_elapsed >= cfg.lift_duration and confirmed:
                self._enter(GaitStage.TRANSFER, measured)
            elif state_elapsed > cfg.lift_timeout:
                self._request_recovery(
                    measured, "swing-foot liftoff timed out"
                )
            return

        if self._stage == GaitStage.TRANSFER:
            ankle = self._ankle_position(measured, self._swing_side)
            position_ok = (
                np.linalg.norm(ankle[:2] - self._swing_target[:2])
                <= cfg.transfer_position_tolerance
            )
            height_ok = (
                abs(ankle[2] - self._swing_apex_z)
                <= cfg.transfer_height_tolerance
            )
            if (
                state_elapsed >= cfg.transfer_duration
                and position_ok
                and height_ok
            ):
                self._enter(GaitStage.LOWER, measured)
            elif state_elapsed > cfg.transfer_timeout:
                self._request_recovery(
                    measured, "horizontal swing transfer timed out"
                )
            return

        if self._stage == GaitStage.LOWER:
            swing_force = self._foot_force(measured, self._swing_side)
            contact_on_target = (
                swing_force
                > cfg.touchdown_onset_ratio * self.body_weight
                and self._touchdown_pose_ok(measured)
            )
            confirmed = self._debounced(
                contact_on_target,
                time,
                "_touchdown_since",
                cfg.touchdown_debounce,
            )
            if confirmed:
                self._enter(GaitStage.TOUCHDOWN, measured)
            elif state_elapsed > cfg.lower_timeout:
                self._request_recovery(
                    measured, "target touchdown was not detected"
                )
            return

        if self._stage == GaitStage.TOUCHDOWN:
            new_force = self._foot_force(measured, self._swing_side)
            old_force = self._foot_force(measured, self._stance_side)
            total = new_force + old_force
            loaded = (
                new_force
                >= cfg.touchdown_loaded_ratio * self.body_weight
                and old_force
                >= cfg.touchdown_old_stance_ratio * self.body_weight
                and total
                >= cfg.touchdown_total_ratio * self.body_weight
                and self._touchdown_pose_ok(measured)
            )
            confirmed = self._debounced(
                loaded,
                time,
                "_loaded_since",
                cfg.loaded_debounce,
            )
            if (
                state_elapsed >= cfg.touchdown_blend_duration
                and confirmed
            ):
                landed_side = self._swing_side
                # Keep the initial lateral lane and flat-ground ankle height
                # fixed.  The next lift still starts at the measured landed
                # pose, but small placement/contact errors are not allowed to
                # drift the future lane or ground plane.
                self._step_index += 1
                self._stance_side = landed_side
                self._swing_side = (
                    "left" if landed_side == "right" else "right"
                )
                # PREPARE starts in balanced double support.  The next SHIFT
                # then transfers from this midpoint to the newly landed stance
                # foot before the opposite leg is allowed to lift.
                self._com_hold = self._touchdown_com_target.copy()
                self._enter(GaitStage.PREPARE_DS, measured)
            elif state_elapsed > cfg.touchdown_timeout:
                self._request_recovery(
                    measured, "new foothold did not accept body load"
                )
            return

        # RECOVER: vertically land the swing foot at its current horizontal
        # location, then latch a double-support hold.  Automatic restart is
        # optional because a failed support exchange deserves inspection.
        right_loaded = (
            measured.contact_force_r_z
            >= cfg.touchdown_old_stance_ratio * self.body_weight
        )
        left_loaded = (
            measured.contact_force_l_z
            >= cfg.touchdown_old_stance_ratio * self.body_weight
        )
        total_loaded = (
            measured.contact_force_r_z + measured.contact_force_l_z
            >= cfg.touchdown_total_ratio * self.body_weight
        )
        recovered = self._debounced(
            right_loaded and left_loaded and total_loaded,
            time,
            "_recovery_contact_since",
            cfg.loaded_debounce,
        )
        if recovered:
            self._recovery_landed = True
            self._halt_requested = not cfg.resume_after_recovery
            if cfg.resume_after_recovery:
                self._com_hold = measured.com_position.copy()
                self._com_hold[2] = cfg.com_height
                self._enter(GaitStage.PREPARE_DS, measured)
        elif state_elapsed > cfg.recovery_timeout:
            self._halt_requested = True

    # ------------------------------------------------------------------
    # Public sampling API
    # ------------------------------------------------------------------
    def sample(self, measured: GaitMeasurement) -> Dict[str, Any]:
        """Advance the hybrid planner and return controller references.

        The returned dictionary is backward-compatible with the keys consumed
        by ``themis_controller_prime.run_controller`` and adds explicit contact
        weights, active-contact flags, Cartesian accelerations, stage/step
        diagnostics, and recovery state for a future inverse-dynamics QP.
        """
        measured = self._normalize_measurement(measured)
        if not self._initialized:
            self._initialize(measured)
        if self._last_time is not None and measured.time < self._last_time:
            raise ValueError("measurement time must be monotonically non-decreasing")

        dt = (
            0.0
            if self._last_time is None
            else max(measured.time - self._last_time, 0.0)
        )
        state_elapsed = max(measured.time - self._stage_enter_time, 0.0)
        self._advance_reference_clock(dt, state_elapsed)

        com_pos, com_vel, com_acc = self._com_reference(state_elapsed)
        swing_pos, swing_vel, swing_acc, swing_phase = self._swing_reference(
            measured, state_elapsed
        )

        # Debounce severe faults so one noisy sensor sample cannot switch the
        # gait, but respond before an unsafe single-support state can develop.
        if self._stage != GaitStage.RECOVER:
            unsafe = self._unsafe_reason_now(measured, swing_pos)
            if unsafe:
                if unsafe != self._unsafe_reason:
                    self._unsafe_reason = unsafe
                    self._unsafe_since = measured.time
                elif (
                    self._unsafe_since is not None
                    and measured.time - self._unsafe_since
                    >= self.config.unsafe_debounce
                ):
                    self._request_recovery(measured, unsafe)
            else:
                self._unsafe_since = None
                self._unsafe_reason = ""

        # A safety transition above changes the commanded descent immediately.
        if self._stage == GaitStage.RECOVER:
            state_elapsed = max(measured.time - self._stage_enter_time, 0.0)
            swing_pos, swing_vel, swing_acc, swing_phase = self._swing_reference(
                measured, state_elapsed
            )
            com_pos, com_vel, com_acc = self._com_reference(state_elapsed)

        self._advance_state(measured, state_elapsed, swing_pos)

        # A nominal transition can occur in _advance_state; resample at phase
        # zero so the output is continuous and its stage label is current.
        new_state_elapsed = max(measured.time - self._stage_enter_time, 0.0)
        if self._stage != GaitStage.RECOVER and new_state_elapsed != state_elapsed:
            state_elapsed = new_state_elapsed
            com_pos, com_vel, com_acc = self._com_reference(state_elapsed)
            swing_pos, swing_vel, swing_acc, swing_phase = self._swing_reference(
                measured, state_elapsed
            )
        elif self._stage == GaitStage.RECOVER:
            state_elapsed = new_state_elapsed
            com_pos, com_vel, com_acc = self._com_reference(state_elapsed)
            swing_pos, swing_vel, swing_acc, swing_phase = self._swing_reference(
                measured, state_elapsed
            )

        alpha_r, alpha_l, swing_weight, active = self._contact_schedule(
            measured, state_elapsed, swing_phase
        )

        omega = np.sqrt(self.config.gravity / self.config.com_height)
        dcm_pos = com_pos.copy()
        dcm_pos[:2] += com_vel[:2] / omega
        dcm_vel = com_vel.copy()
        dcm_vel[:2] += com_acc[:2] / omega

        both_contacts = bool(active[0] and active[1])
        loco_state = 0 if both_contacts and swing_weight <= 1e-6 else 2
        leg_stance = -1.0 if self._stance_side == "left" else 1.0

        self._last_time = measured.time
        return {
            # Existing controller interface.
            "com_pos_des": com_pos.copy(),
            "com_vel_des": com_vel.copy(),
            "feet_ori_des": self._flat_foot_rotation[
                self._swing_side
            ].copy(),
            "swing_leg_pos_des": swing_pos.copy(),
            "swing_leg_vel_des": swing_vel.copy(),
            "swing_phase": float(swing_phase),
            "swing_stage": self._stage.value,
            "swing_contact_force": float(
                self._foot_force(measured, self._swing_side)
            ),
            "leg_stance": leg_stance,
            "loco_state": loco_state,
            "dcm_pos_des": dcm_pos.copy(),
            # Acceleration-level and explicit contact interface for a WBC QP.
            "com_acc_des": com_acc.copy(),
            "dcm_vel_des": dcm_vel.copy(),
            "swing_leg_acc_des": swing_acc.copy(),
            "contact_weight_r": float(alpha_r),
            "contact_weight_l": float(alpha_l),
            "swing_task_weight": float(swing_weight),
            "active_contacts": {
                "right": bool(active[0]),
                "left": bool(active[1]),
            },
            "contact_active_r": bool(active[0]),
            "contact_active_l": bool(active[1]),
            # Supervisor diagnostics.
            "stage": self._stage.value,
            "step_index": int(self._step_index),
            "reference_time": float(self._reference_time),
            "support_side": self._stance_side,
            "swing_side": self._swing_side,
            "recovery_mode": self._stage == GaitStage.RECOVER,
            "recovery_landed": bool(self._recovery_landed),
            "halt_requested": bool(self._halt_requested),
            "safety_reason": self._safety_reason,
        }


__all__ = [
    "AlternatingGaitPlanner",
    "GaitMeasurement",
    "GaitPlannerConfig",
    "GaitStage",
]
 