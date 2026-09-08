"""Contact-aware whole-body inverse-dynamics QP for THEMIS.

This module is intentionally independent of ``locomotion_test.py`` and of the
legacy projected-PD controller.  It exposes the same controller call shape:

``run_controller(state, fk, planner_state, gains, contact_l_z, contact_r_z)``

and returns:

``(tau_isaac, zmp, com_position, com_velocity, centroidal_wrench, point_forces)``.

The QP decision vector is

``x = [qdd(34), tau(28), f_contact(18)]``

where the generalized velocity order used by the compiled THEMIS model is

``[floating_base(6), right_leg(6), left_leg(6), right_arm(7),
  left_arm(7), head(2)]``.

The six contact forces are ordered as

``[right inner toe, right outer toe, right heel,
  left inner toe, left outer toe, left heel]``.

Hard constraints enforce floating-base dynamics, active-foot acceleration,
unilateral/friction limits, joint torque limits, torque-rate limits, and
one-step joint acceleration/velocity/position limits.  Soft objectives track
COM/DCM acceleration, base orientation, the airborne foot pose, centroidal
angular momentum, joint posture, and smooth contact forces.

The implementation uses OSQP directly.  Its workspace is reused whenever the
sparsity pattern remains unchanged.  A failed or invalid solve never advances
planner state here: the controller reuses one last valid torque sample and then
falls back to clipped gravity/bias compensation plus joint damping.

Notes on generalized Jacobians
------------------------------
The existing ``compute_FK`` returns correct 12-column *actuated-leg*
Jacobians, but deliberately removes the floating-base columns for the legacy
controller.  This module accepts full 34-column Jacobians when the caller adds
them under ``<legacy_key>_full`` or ``<legacy_key>_34``.  Until then, it embeds
the 12 leg columns and reconstructs the floating-base rigid-body columns from
the measured base pose.  Supplying the original full Jacobians from the
compiled kinematics (or validated PhysX Jacobians) is preferred for production.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import osqp
from scipy import sparse


# ---------------------------------------------------------------------------
# Model dimensions and index conventions
# ---------------------------------------------------------------------------

N_BASE = 6
N_ACTUATED = 28
N_GENERALIZED = N_BASE + N_ACTUATED
N_POINT_CONTACTS = 6
N_FORCE_VARIABLES = 3 * N_POINT_CONTACTS
N_VARIABLES = N_GENERALIZED + N_ACTUATED + N_FORCE_VARIABLES

QDD_SLICE = slice(0, N_GENERALIZED)
TAU_SLICE = slice(N_GENERALIZED, N_GENERALIZED + N_ACTUATED)
FORCE_SLICE = slice(N_GENERALIZED + N_ACTUATED, N_VARIABLES)

# Model-order torque slices.
RIGHT_LEG = slice(0, 6)
LEFT_LEG = slice(6, 12)
RIGHT_ARM = slice(12, 19)
LEFT_ARM = slice(19, 26)
HEAD = slice(26, 28)

# Model-order -> Isaac articulation-order mapping.  These are the same mappings
# used by the existing controller and by ``themis_config.MODEL_ORDER_IND``.
ISAAC_RIGHT_LEG = np.array([2, 7, 11, 15, 19, 23], dtype=int)
ISAAC_LEFT_LEG = np.array([1, 6, 10, 14, 18, 22], dtype=int)
ISAAC_RIGHT_ARM = np.array([4, 9, 13, 17, 21, 25, 27], dtype=int)
ISAAC_LEFT_ARM = np.array([3, 8, 12, 16, 20, 24, 26], dtype=int)
ISAAC_HEAD = np.array([0, 5], dtype=int)

# Contact-point order and corresponding legacy FK keys.
CONTACT_POINTS = (
    ("r", "inner", "p_wit_r", "Jv_wit_r"),
    ("r", "outer", "p_wot_r", "Jv_wot_r"),
    ("r", "heel", "p_whl_r", "Jv_whl_r"),
    ("l", "inner", "p_wit_l", "Jv_wit_l"),
    ("l", "outer", "p_wot_l", "Jv_wot_l"),
    ("l", "heel", "p_whl_l", "Jv_whl_l"),
)

ANKLE_KEYS = {
    "r": {
        "position": "p_wa_r",
        "rotation": "R_wa_r",
        "linear_velocity": "v_wa_r",
        "angular_velocity": "w_fa_r",
        "linear_jacobian": "Jv_wa_r",
        # ``Jw_fa_r`` from the compiled kinematics is ankle-local.  The
        # forward-kinematics adapter additionally exposes ``Jw_wa_r_full`` in
        # the world frame, matching the world SO(3) errors used by this QP.
        "angular_jacobian_world": "Jw_wa_r",
        "angular_jacobian_local": "Jw_fa_r",
        "linear_bias": "dJvdq_wa_r",
        "angular_bias": "dJwdq_fa_r",
    },
    "l": {
        "position": "p_wa_l",
        "rotation": "R_wa_l",
        "linear_velocity": "v_wa_l",
        "angular_velocity": "w_fa_l",
        "linear_jacobian": "Jv_wa_l",
        "angular_jacobian_world": "Jw_wa_l",
        "angular_jacobian_local": "Jw_fa_l",
        "linear_bias": "dJvdq_wa_l",
        "angular_bias": "dJwdq_fa_l",
    },
}


def _array(value: Any, shape: tuple[int, ...] | None = None) -> np.ndarray:
    """Convert a tensor/sequence to a finite float64 NumPy array."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    result = np.asarray(value, dtype=np.float64)
    if shape is not None:
        result = result.reshape(shape)
    if not np.all(np.isfinite(result)):
        raise ValueError("QP input contains NaN or infinity")
    return result


def _vector_parameter(
    parameters: Mapping[str, Any],
    name: str,
    default: Sequence[float] | np.ndarray,
    length: int,
) -> np.ndarray:
    """Read and validate one vector-valued controller parameter."""

    value = _array(parameters.get(name, default)).reshape(-1)
    if value.size == 1:
        value = np.full(length, float(value[0]), dtype=np.float64)
    if value.size != length:
        raise ValueError(f"{name} must contain {length} values, got {value.size}")
    return value


def _skew(vector: Sequence[float]) -> np.ndarray:
    """Return the cross-product matrix such that ``skew(a) @ b == a × b``."""

    x, y, z = _array(vector).reshape(3)
    return np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=np.float64,
    )


def _rotation_error_world(desired: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Return the shortest world-frame rotation vector from current to desired."""

    error_rotation = _array(desired, (3, 3)) @ _array(current, (3, 3)).T
    cosine = np.clip((np.trace(error_rotation) - 1.0) * 0.5, -1.0, 1.0)
    angle = float(np.arccos(cosine))
    vee = np.array(
        [
            error_rotation[2, 1] - error_rotation[1, 2],
            error_rotation[0, 2] - error_rotation[2, 0],
            error_rotation[1, 0] - error_rotation[0, 1],
        ],
        dtype=np.float64,
    )
    if angle < 1.0e-7:
        return 0.5 * vee
    sine = float(np.sin(angle))
    if abs(sine) > 1.0e-6:
        return 0.5 * angle / sine * vee

    # Near pi, the skew part is numerically small.  Recover a stable axis from
    # the diagonal and apply the sign indicated by the off-diagonal terms.
    diagonal = np.maximum((np.diag(error_rotation) + 1.0) * 0.5, 0.0)
    axis = np.sqrt(diagonal)
    largest = int(np.argmax(axis))
    if axis[largest] < 1.0e-7:
        axis = np.array([1.0, 0.0, 0.0])
    else:
        if largest == 0:
            axis[1] = np.copysign(axis[1], error_rotation[0, 1])
            axis[2] = np.copysign(axis[2], error_rotation[0, 2])
        elif largest == 1:
            axis[0] = np.copysign(axis[0], error_rotation[0, 1])
            axis[2] = np.copysign(axis[2], error_rotation[1, 2])
        else:
            axis[0] = np.copysign(axis[0], error_rotation[0, 2])
            axis[1] = np.copysign(axis[1], error_rotation[1, 2])
        axis /= max(np.linalg.norm(axis), 1.0e-12)
    return angle * axis


def _full_key_candidates(legacy_key: str) -> tuple[str, ...]:
    """Return supported names for a caller-provided 34-column Jacobian."""

    return (
        f"{legacy_key}_full",
        f"{legacy_key}_34",
        legacy_key.replace("Jv_", "Jv_full_").replace("Jw_", "Jw_full_"),
        legacy_key.replace("Jv_", "Jv34_").replace("Jw_", "Jw34_"),
    )


def _full_jacobian(
    fk: Mapping[str, Any],
    legacy_key: str,
    point_position: np.ndarray,
    *,
    angular: bool,
) -> np.ndarray:
    """Return a 3×34 generalized Jacobian in compiled-model order.

    A full caller-provided Jacobian is used when available.  Otherwise this
    function embeds the legacy 12 leg columns and reconstructs base columns.
    The compiled dynamics accepts body-frame base twist coordinates, so the
    measured base rotation maps those columns into the world-frame task.
    """

    for candidate in _full_key_candidates(legacy_key):
        if candidate in fk:
            candidate_value = _array(fk[candidate])
            if candidate_value.shape == (3, N_GENERALIZED):
                return candidate_value.copy()

    legacy = _array(fk[legacy_key])
    if legacy.shape == (3, N_GENERALIZED):
        return legacy.copy()

    result = np.zeros((3, N_GENERALIZED), dtype=np.float64)
    if legacy.shape == (3, 18):
        # Already [base(6) | right_leg(6) | left_leg(6)].
        result[:, :18] = legacy
        return result
    if legacy.shape != (3, 12):
        raise ValueError(
            f"{legacy_key} must be 3x12, 3x18, or 3x34; got {legacy.shape}"
        )

    base_rotation = _array(fk["R_wb"], (3, 3))
    if angular:
        result[:, 0:3] = base_rotation
    else:
        base_position = _array(fk["p_wb"]).reshape(3)
        lever_world = _array(point_position).reshape(3) - base_position
        result[:, 0:3] = -_skew(lever_world) @ base_rotation
        result[:, 3:6] = base_rotation
    result[:, 6:18] = legacy
    return result


def _spatial_ankle_data(
    fk: Mapping[str, Any],
    side: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ankle pose, twist, 6×34 Jacobian, and ``Jdot*qdot``."""

    keys = ANKLE_KEYS[side]
    position = _array(fk[keys["position"]]).reshape(3)
    rotation = _array(fk[keys["rotation"]], (3, 3))
    linear_velocity = _array(fk[keys["linear_velocity"]]).reshape(3)
    angular_velocity_local = _array(
        fk[keys["angular_velocity"]]
    ).reshape(3)
    linear_jacobian = _full_jacobian(
        fk,
        keys["linear_jacobian"],
        position,
        angular=False,
    )
    world_key = keys["angular_jacobian_world"]
    local_key = keys["angular_jacobian_local"]
    world_jacobian_available = (
        world_key in fk
        and _array(fk[world_key]).shape == (3, N_GENERALIZED)
    ) or any(
        candidate in fk
        and _array(fk[candidate]).shape == (3, N_GENERALIZED)
        for candidate in _full_key_candidates(world_key)
    )
    if world_jacobian_available:
        angular_jacobian = _full_jacobian(
            fk,
            world_key,
            position,
            angular=True,
        )
    else:
        local_full_jacobian_available = (
            local_key in fk
            and _array(fk[local_key]).shape == (3, N_GENERALIZED)
        ) or any(
            candidate in fk
            and _array(fk[candidate]).shape == (3, N_GENERALIZED)
            for candidate in _full_key_candidates(local_key)
        )
        angular_jacobian_local = _full_jacobian(
            fk,
            local_key,
            position,
            angular=True,
        )
        if not local_full_jacobian_available:
            # The legacy 12-column ankle angular Jacobian contains only leg
            # joints.  ``_full_jacobian`` can reconstruct a *world-frame* base
            # block, but this branch is explicitly handling a foot-local
            # Jacobian.  Reconstruct that block in the local frame before the
            # complete Jacobian is rotated to world below:
            #
            #   omega_ankle_local = R_wa.T @ R_wb @ omega_base_body + ...
            #
            # Without this correction the two rotations would be multiplied as
            # ``R_wa @ R_wb`` and a rotated robot would receive a false angular
            # acceleration constraint.
            angular_jacobian_local[:, 0:3] = (
                rotation.T @ _array(fk["R_wb"], (3, 3))
            )
            angular_jacobian_local[:, 3:6] = 0.0
        angular_jacobian = rotation @ angular_jacobian_local
    linear_bias = _array(fk.get(keys["linear_bias"], np.zeros(3))).reshape(3)
    angular_bias_local = _array(
        fk.get(keys["angular_bias"], np.zeros(3))
    ).reshape(3)
    # Keep the angular velocity, Jacobian, acceleration bias, and rotation
    # errors in the same world frame.  This is particularly important once a
    # swing foot rotates away from its nominal level pose.
    angular_velocity = rotation @ angular_velocity_local
    angular_bias = rotation @ angular_bias_local
    spatial_jacobian = np.vstack((linear_jacobian, angular_jacobian))
    spatial_bias = np.concatenate((linear_bias, angular_bias))
    spatial_velocity = np.concatenate((linear_velocity, angular_velocity))
    return position, rotation, spatial_velocity, spatial_jacobian, spatial_bias


def _point_contact_data(
    fk: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Return six point positions and their stacked 18×34 linear Jacobian."""

    positions: list[np.ndarray] = []
    jacobians: list[np.ndarray] = []
    for _, _, position_key, jacobian_key in CONTACT_POINTS:
        position = _array(fk[position_key]).reshape(3)
        positions.append(position)
        jacobians.append(
            _full_jacobian(
                fk,
                jacobian_key,
                position,
                angular=False,
            )
        )
    return np.vstack(positions), np.vstack(jacobians)


def _active_contact_sides(planner_state: Mapping[str, Any]) -> tuple[str, ...]:
    """Translate the planner's locomotion convention into active contacts."""

    # An explicit contact schedule takes precedence over the coarse locomotion
    # state.  It permits a touchdown phase to activate the newly landed foot
    # while retaining the old stance foot, without prematurely declaring a
    # generic double-support phase.
    if (
        "contact_active_r" in planner_state
        or "contact_active_l" in planner_state
    ):
        active: list[str] = []
        if bool(planner_state.get("contact_active_r", False)):
            active.append("r")
        if bool(planner_state.get("contact_active_l", False)):
            active.append("l")
        return tuple(active)

    if int(planner_state.get("loco_state", 0)) != 2:
        return ("r", "l")
    # leg_stance = -1 means left stance/right swing; +1 is the opposite.
    return ("l",) if float(planner_state.get("leg_stance", -1.0)) < 0.0 else ("r",)


def _model_to_isaac_torque(torque_model: np.ndarray) -> np.ndarray:
    """Convert model-order actuator torque to Isaac articulation order."""

    torque_model = _array(torque_model).reshape(N_ACTUATED)
    torque_isaac = np.zeros(N_ACTUATED, dtype=np.float64)
    torque_isaac[ISAAC_RIGHT_LEG] = torque_model[RIGHT_LEG]
    torque_isaac[ISAAC_LEFT_LEG] = torque_model[LEFT_LEG]
    torque_isaac[ISAAC_RIGHT_ARM] = torque_model[RIGHT_ARM]
    torque_isaac[ISAAC_LEFT_ARM] = torque_model[LEFT_ARM]
    torque_isaac[ISAAC_HEAD] = torque_model[HEAD]
    return torque_isaac


@dataclass
class WBCQPConfig:
    """Numerical and control defaults for the whole-body QP.

    Every field can be overridden by a same-named entry in ``ctrl_params``.
    Vector gains may be passed as a scalar or as a vector of matching length.
    """

    dt: float = 1.0 / 60.0
    gravity: float = 9.81
    friction_coefficient: float = 0.60
    maximum_point_normal_force: float = 250.0
    maximum_torque_step: float = 5.0
    minimum_stance_force: float = 40.0
    require_stance_force_guard: bool = True

    com_kp: np.ndarray = field(
        default_factory=lambda: np.array([40.0, 40.0, 80.0])
    )
    com_kd: np.ndarray = field(
        default_factory=lambda: np.array([12.0, 12.0, 18.0])
    )
    dcm_acceleration_gain: float = 8.0
    base_orientation_kp: np.ndarray = field(
        default_factory=lambda: np.array([60.0, 60.0, 20.0])
    )
    base_orientation_kd: np.ndarray = field(
        default_factory=lambda: np.array([14.0, 14.0, 8.0])
    )
    swing_position_kp: np.ndarray = field(
        default_factory=lambda: np.array([80.0, 80.0, 120.0])
    )
    swing_position_kd: np.ndarray = field(
        default_factory=lambda: np.array([18.0, 18.0, 22.0])
    )
    swing_orientation_kp: np.ndarray = field(
        default_factory=lambda: np.array([30.0, 30.0, 15.0])
    )
    swing_orientation_kd: np.ndarray = field(
        default_factory=lambda: np.array([8.0, 8.0, 5.0])
    )
    contact_position_kp: float = 100.0
    contact_position_kd: float = 20.0
    contact_orientation_kp: float = 80.0
    contact_orientation_kd: float = 18.0
    posture_kp: float = 5.0
    posture_kd: float = 3.0
    momentum_damping: float = 2.0

    com_weight: float = 1000.0
    base_orientation_weight: float = 500.0
    swing_position_weight: float = 1000.0
    swing_orientation_weight: float = 200.0
    momentum_weight: float = 50.0
    posture_weight: float = 5.0
    force_tracking_weight: float = 1.0e-4
    qdd_regularization: float = 1.0e-4
    torque_regularization: float = 1.0e-6
    torque_rate_regularization: float = 1.0e-5
    force_regularization: float = 1.0e-7
    force_rate_regularization: float = 1.0e-6

    solver_absolute_tolerance: float = 1.0e-5
    solver_relative_tolerance: float = 1.0e-5
    solver_maximum_iterations: int = 2000
    maximum_hard_constraint_residual: float = 2.0e-3
    fallback_joint_damping: float = 2.0

    @classmethod
    def from_parameters(cls, parameters: Mapping[str, Any] | None) -> "WBCQPConfig":
        """Build configuration from a possibly sparse runtime parameter map."""

        parameters = {} if parameters is None else parameters
        config = cls()
        scalar_fields = (
            "dt",
            "gravity",
            "friction_coefficient",
            "maximum_point_normal_force",
            "maximum_torque_step",
            "minimum_stance_force",
            "dcm_acceleration_gain",
            "contact_position_kp",
            "contact_position_kd",
            "contact_orientation_kp",
            "contact_orientation_kd",
            "posture_kp",
            "posture_kd",
            "momentum_damping",
            "com_weight",
            "base_orientation_weight",
            "swing_position_weight",
            "swing_orientation_weight",
            "momentum_weight",
            "posture_weight",
            "force_tracking_weight",
            "qdd_regularization",
            "torque_regularization",
            "torque_rate_regularization",
            "force_regularization",
            "force_rate_regularization",
            "solver_absolute_tolerance",
            "solver_relative_tolerance",
            "maximum_hard_constraint_residual",
            "fallback_joint_damping",
        )
        for field_name in scalar_fields:
            if field_name in parameters:
                setattr(config, field_name, float(parameters[field_name]))
        if "solver_maximum_iterations" in parameters:
            config.solver_maximum_iterations = int(
                parameters["solver_maximum_iterations"]
            )
        if "require_stance_force_guard" in parameters:
            config.require_stance_force_guard = bool(
                parameters["require_stance_force_guard"]
            )

        config.com_kp = _vector_parameter(
            parameters, "com_kp", config.com_kp, 3
        )
        config.com_kd = _vector_parameter(
            parameters, "com_kd", config.com_kd, 3
        )
        config.base_orientation_kp = _vector_parameter(
            parameters,
            "base_orientation_kp",
            config.base_orientation_kp,
            3,
        )
        config.base_orientation_kd = _vector_parameter(
            parameters,
            "base_orientation_kd",
            config.base_orientation_kd,
            3,
        )
        config.swing_position_kp = _vector_parameter(
            parameters,
            "swing_position_kp",
            config.swing_position_kp,
            3,
        )
        config.swing_position_kd = _vector_parameter(
            parameters,
            "swing_position_kd",
            config.swing_position_kd,
            3,
        )
        config.swing_orientation_kp = _vector_parameter(
            parameters,
            "swing_orientation_kp",
            config.swing_orientation_kp,
            3,
        )
        config.swing_orientation_kd = _vector_parameter(
            parameters,
            "swing_orientation_kd",
            config.swing_orientation_kd,
            3,
        )
        if config.dt <= 0.0:
            raise ValueError("dt must be positive")
        if not 0.0 < config.friction_coefficient:
            raise ValueError("friction_coefficient must be positive")
        return config


class ControllerState:
    """Persistent OSQP workspace, contact anchors, and safe fallback state."""

    def __init__(self, environment_id: int = 0):
        self.environment_id = int(environment_id)
        self.solver: osqp.OSQP | None = None
        self._p_pattern: tuple[Any, ...] | None = None
        self._a_pattern: tuple[Any, ...] | None = None
        self.previous_solution: np.ndarray | None = None
        self.previous_torque: np.ndarray | None = None
        self.last_valid_torque: np.ndarray | None = None
        self.previous_force = np.zeros(N_FORCE_VARIABLES, dtype=np.float64)
        self.nominal_joint_position: np.ndarray | None = None
        self.contact_position_anchor: dict[str, np.ndarray] = {}
        self.contact_rotation_anchor: dict[str, np.ndarray] = {}
        self.active_contacts: tuple[str, ...] = ()
        self.failure_count = 0

        # Public diagnostic fields; a new harness can log these without
        # changing the controller return tuple.
        self.last_status = "uninitialized"
        self.last_message = ""
        self.last_dynamics_residual = np.inf
        self.last_contact_residual = np.inf
        self.last_constraint_residual = np.inf
        self.last_measured_force_l = 0.0
        self.last_measured_force_r = 0.0
        self.last_stance_force_guard_ok = True
        self.last_qdd = np.zeros(N_GENERALIZED)
        self.last_tau_model = np.zeros(N_ACTUATED)
        self.last_point_forces = np.zeros(N_FORCE_VARIABLES)

    def reset(self) -> None:
        """Clear solver warm-start data and all phase-dependent state."""

        environment_id = self.environment_id
        self.__init__(environment_id=environment_id)

    def _update_contact_anchors(
        self,
        fk: Mapping[str, Any],
        active_contacts: Iterable[str],
    ) -> None:
        """Capture the measured foot pose when a contact becomes active."""

        active_tuple = tuple(active_contacts)
        for side in tuple(self.contact_position_anchor):
            if side not in active_tuple:
                self.contact_position_anchor.pop(side, None)
                self.contact_rotation_anchor.pop(side, None)
        for side in active_tuple:
            if side not in self.contact_position_anchor:
                keys = ANKLE_KEYS[side]
                self.contact_position_anchor[side] = _array(
                    fk[keys["position"]]
                ).reshape(3)
                self.contact_rotation_anchor[side] = _array(
                    fk[keys["rotation"]], (3, 3)
                )
        self.active_contacts = active_tuple


# Descriptive alias for new code while preserving the familiar harness name.
WBCQPState = ControllerState


def _sparse_pattern(matrix: sparse.csc_matrix) -> tuple[Any, ...]:
    """Create a compact identity for an OSQP matrix sparsity pattern."""

    return (
        matrix.shape,
        matrix.indptr.tobytes(),
        matrix.indices.tobytes(),
    )


def _add_task_cost(
    hessian: np.ndarray,
    gradient: np.ndarray,
    task_jacobian: np.ndarray,
    target: np.ndarray,
    weight: float | np.ndarray,
) -> None:
    """Add ``||J qdd - target||_W²`` to the OSQP objective in place."""

    task_jacobian = _array(task_jacobian)
    target = _array(target).reshape(task_jacobian.shape[0])
    if np.isscalar(weight):
        weighted_jacobian = float(weight) * task_jacobian
        weighted_target = float(weight) * target
    else:
        weight_vector = _array(weight).reshape(task_jacobian.shape[0])
        weighted_jacobian = weight_vector[:, None] * task_jacobian
        weighted_target = weight_vector * target
    hessian[QDD_SLICE, QDD_SLICE] += 2.0 * (
        task_jacobian.T @ weighted_jacobian
    )
    gradient[QDD_SLICE] -= 2.0 * (
        task_jacobian.T @ weighted_target
    )


def _estimate_mass(mass_matrix: np.ndarray, parameters: Mapping[str, Any]) -> float:
    """Return robot mass from an override or the floating-base inertia block."""

    if "robot_mass" in parameters:
        mass = float(parameters["robot_mass"])
    else:
        # Generalized base order is [angular(3), linear(3)].  Each diagonal of
        # the translational 3x3 block equals total mass in this model.
        mass = float(np.trace(mass_matrix[3:6, 3:6]) / 3.0)
    if not np.isfinite(mass) or mass <= 1.0:
        raise ValueError(f"invalid robot mass estimate: {mass}")
    return mass


def _force_reference(
    active_contacts: tuple[str, ...],
    planner_state: Mapping[str, Any],
    total_weight: float,
) -> np.ndarray:
    """Construct a conservative desired point-force distribution."""

    force_reference = np.zeros((N_POINT_CONTACTS, 3), dtype=np.float64)
    if active_contacts == ("r",):
        right_fraction = 1.0
    elif active_contacts == ("l",):
        right_fraction = 0.0
    else:
        # The contact planner already computes the smooth support weights used
        # during unload and touchdown.  Normalizing them makes the QP's force
        # reference follow that same transition instead of reverting to a
        # fixed 50/50 split whenever both contacts are active.
        if (
            "contact_weight_r" in planner_state
            or "contact_weight_l" in planner_state
        ):
            weight_r = max(
                float(planner_state.get("contact_weight_r", 0.0)),
                0.0,
            )
            weight_l = max(
                float(planner_state.get("contact_weight_l", 0.0)),
                0.0,
            )
            weight_sum = weight_r + weight_l
            right_fraction = (
                weight_r / weight_sum if weight_sum > 1.0e-9 else 0.5
            )
        else:
            right_fraction = float(
                np.clip(
                    planner_state.get("right_load_fraction", 0.5),
                    0.0,
                    1.0,
                )
            )
    force_reference[0:3, 2] = right_fraction * total_weight / 3.0
    force_reference[3:6, 2] = (1.0 - right_fraction) * total_weight / 3.0
    return force_reference.reshape(N_FORCE_VARIABLES)


def _joint_bounds(
    joint_position: np.ndarray,
    joint_velocity: np.ndarray,
    config: WBCQPConfig,
    parameters: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Combine acceleration, velocity, and one-step position limits."""

    default_acceleration_limit = np.array(
        [30.0, 30.0, 35.0, 35.0, 40.0, 40.0] * 2
        + [35.0] * 14
        + [50.0] * 2,
        dtype=np.float64,
    )
    default_velocity_limit = np.array(
        [4.17, 4.17, 2.24, 2.24, 1.87, 1.87] * 2
        + [3.52, 1.58, 1.58, 1.58, 1.58, 1.58, 1.58] * 2
        + [10.67, 10.67],
        dtype=np.float64,
    )
    acceleration_limit = _vector_parameter(
        parameters,
        "joint_acceleration_limits",
        default_acceleration_limit,
        N_ACTUATED,
    )
    velocity_limit = _vector_parameter(
        parameters,
        "joint_velocity_limits",
        default_velocity_limit,
        N_ACTUATED,
    )
    lower = -np.abs(acceleration_limit)
    upper = np.abs(acceleration_limit)

    lower = np.maximum(
        lower,
        (-np.abs(velocity_limit) - joint_velocity) / config.dt,
    )
    upper = np.minimum(
        upper,
        (np.abs(velocity_limit) - joint_velocity) / config.dt,
    )

    if "joint_position_min" in parameters and "joint_position_max" in parameters:
        position_min = _vector_parameter(
            parameters,
            "joint_position_min",
            parameters["joint_position_min"],
            N_ACTUATED,
        )
        position_max = _vector_parameter(
            parameters,
            "joint_position_max",
            parameters["joint_position_max"],
            N_ACTUATED,
        )
        dt_squared = config.dt * config.dt
        lower = np.maximum(
            lower,
            2.0
            * (
                position_min
                - joint_position
                - config.dt * joint_velocity
            )
            / dt_squared,
        )
        upper = np.minimum(
            upper,
            2.0
            * (
                position_max
                - joint_position
                - config.dt * joint_velocity
            )
            / dt_squared,
        )
    return lower, upper


def _torque_limits(parameters: Mapping[str, Any]) -> np.ndarray:
    """Return conservative default actuator torque limits in model order."""

    default = np.array(
        [60.0, 60.0, 120.0, 120.0, 80.0, 60.0] * 2
        + [40.0] * 14
        + [10.0] * 2,
        dtype=np.float64,
    )
    return np.abs(
        _vector_parameter(parameters, "torque_limits", default, N_ACTUATED)
    )

#-------------------------OPTIMIZATION PROBLEM ASSEMBLY------------------------------
def _build_problem(
    state: ControllerState,
    fk: Mapping[str, Any],
    planner_state: Mapping[str, Any],
    parameters: Mapping[str, Any],
    config: WBCQPConfig,
) -> tuple[
    sparse.csc_matrix,
    np.ndarray,
    sparse.csc_matrix,
    np.ndarray,
    np.ndarray,
    Dict[str, Any],
]:
    """Assemble one convex inverse-dynamics QP and diagnostic metadata."""
#--------------------------ASSIGN STATE VARIABLES AND MAKE SURE OF THEIR ACTUAL DIMENSIONS--------------------------
    mass_matrix = _array(fk["H"], (N_GENERALIZED, N_GENERALIZED))
    mass_matrix = 0.5 * (mass_matrix + mass_matrix.T)
    bias_force = _array(fk["CG"]).reshape(N_GENERALIZED)
    joint_position = _array(fk["q"]).reshape(N_ACTUATED)
    joint_velocity = _array(fk["dq"]).reshape(N_ACTUATED)
    com_position = _array(fk["p_wg"]).reshape(3)
    com_velocity = _array(fk["v_wg"]).reshape(3)
    base_rotation = _array(fk["R_wb"], (3, 3))
    base_angular_velocity_body = _array(fk["w_bb"]).reshape(3)
    centroidal_matrix = _array(
        fk.get("AG", np.zeros((6, N_GENERALIZED))),
        (6, N_GENERALIZED),
    )
    centroidal_bias = _array(fk.get("dAGdq", np.zeros(6))).reshape(6)
    centroidal_momentum = _array(fk.get("h", np.zeros(6))).reshape(6)
    robot_mass = _estimate_mass(mass_matrix, parameters)

    point_positions, point_jacobian = _point_contact_data(fk)
    active_contacts = _active_contact_sides(planner_state)
    state._update_contact_anchors(fk, active_contacts)

    if state.nominal_joint_position is None:
        state.nominal_joint_position = joint_position.copy()
    if "nominal_joint_position" in parameters:
        state.nominal_joint_position = _vector_parameter(
            parameters,
            "nominal_joint_position",
            state.nominal_joint_position,
            N_ACTUATED,
        )

    hessian = np.zeros((N_VARIABLES, N_VARIABLES), dtype=np.float64)
    gradient = np.zeros(N_VARIABLES, dtype=np.float64)

    # ------------------------------------------------------------------
    # Soft COM/DCM acceleration tracking
    # ------------------------------------------------------------------
    com_desired = _array(
        planner_state.get("com_pos_des", com_position)
    ).reshape(3)
    com_velocity_desired = _array(
        planner_state.get("com_vel_des", np.zeros(3))
    ).reshape(3)
    com_acceleration_feedforward = _array(
        planner_state.get("com_acc_des", np.zeros(3))
    ).reshape(3)
    com_acceleration_desired = (
        com_acceleration_feedforward
        + config.com_kp * (com_desired - com_position)
        + config.com_kd * (com_velocity_desired - com_velocity)
    )
    if "dcm_pos_des" in planner_state:
        desired_dcm = _array(planner_state["dcm_pos_des"]).reshape(3)
        omega = np.sqrt(
            config.gravity / max(float(com_desired[2]), 0.10)
        )
        actual_dcm = com_position.copy()
        actual_dcm[:2] += com_velocity[:2] / omega
        com_acceleration_desired[:2] += (
            config.dcm_acceleration_gain
            * (desired_dcm[:2] - actual_dcm[:2])
        )
    com_jacobian = centroidal_matrix[3:6, :] / robot_mass
    com_bias = centroidal_bias[3:6] / robot_mass
    _add_task_cost(
        hessian,
        gradient,
        com_jacobian,
        com_acceleration_desired - com_bias,
        config.com_weight,
    )

    # ------------------------------------------------------------------
    # Soft base angular-acceleration tracking
    # ------------------------------------------------------------------
    base_rotation_desired = _array(
        planner_state.get("base_ori_des", np.eye(3)),
        (3, 3),
    )
    base_error_world = _rotation_error_world(
        base_rotation_desired,
        base_rotation,
    )
    base_error_body = base_rotation.T @ base_error_world
    base_angular_acceleration_desired = (
        config.base_orientation_kp * base_error_body
        - config.base_orientation_kd * base_angular_velocity_body
    )
    base_selector = np.zeros((3, N_GENERALIZED), dtype=np.float64)
    base_selector[:, 0:3] = np.eye(3)
    _add_task_cost(
        hessian,
        gradient,
        base_selector,
        base_angular_acceleration_desired,
        config.base_orientation_weight,
    )

    # ------------------------------------------------------------------
    # Soft swing-foot pose acceleration
    # ------------------------------------------------------------------
    swing_side: str | None = None
    if int(planner_state.get("loco_state", 0)) == 2:
        # The contact planner ramps this authority from zero after releasing a
        # foot and back to zero while accepting touchdown.  Scaling the QP cost
        # avoids an abrupt full-strength swing objective at the hard contact
        # mask transition.
        swing_task_weight = float(
            np.clip(planner_state.get("swing_task_weight", 1.0), 0.0, 1.0)
        )
        swing_side = "r" if float(planner_state.get("leg_stance", -1.0)) < 0.0 else "l"
        (
            swing_position,
            swing_rotation,
            swing_velocity,
            swing_jacobian,
            swing_bias,
        ) = _spatial_ankle_data(fk, swing_side)
        swing_position_desired = _array(
            planner_state.get("swing_leg_pos_des", swing_position)
        ).reshape(3)
        swing_linear_velocity_desired = _array(
            planner_state.get("swing_leg_vel_des", np.zeros(3))
        ).reshape(3)
        swing_linear_acceleration_feedforward = _array(
            planner_state.get("swing_leg_acc_des", np.zeros(3))
        ).reshape(3)
        swing_rotation_desired = _array(
            planner_state.get("feet_ori_des", np.eye(3)),
            (3, 3),
        )
        swing_angular_velocity_desired = _array(
            planner_state.get("swing_leg_ang_vel_des", np.zeros(3))
        ).reshape(3)
        swing_angular_acceleration_feedforward = _array(
            planner_state.get("swing_leg_ang_acc_des", np.zeros(3))
        ).reshape(3)
        swing_linear_acceleration_desired = (
            swing_linear_acceleration_feedforward
            + config.swing_position_kp
            * (swing_position_desired - swing_position)
            + config.swing_position_kd
            * (swing_linear_velocity_desired - swing_velocity[:3])
        )
        swing_rotation_error = _rotation_error_world(
            swing_rotation_desired,
            swing_rotation,
        )
        swing_angular_acceleration_desired = (
            swing_angular_acceleration_feedforward
            + config.swing_orientation_kp * swing_rotation_error
            + config.swing_orientation_kd
            * (swing_angular_velocity_desired - swing_velocity[3:])
        )
        _add_task_cost(
            hessian,
            gradient,
            swing_jacobian[:3, :],
            swing_linear_acceleration_desired - swing_bias[:3],
            swing_task_weight * config.swing_position_weight,
        )
        _add_task_cost(
            hessian,
            gradient,
            swing_jacobian[3:, :],
            swing_angular_acceleration_desired - swing_bias[3:],
            swing_task_weight * config.swing_orientation_weight,
        )

    # ------------------------------------------------------------------
    # Soft centroidal angular-momentum and joint-posture objectives
    # ------------------------------------------------------------------
    angular_momentum_rate_desired = (
        -config.momentum_damping * centroidal_momentum[:3]
    )
    _add_task_cost(
        hessian,
        gradient,
        centroidal_matrix[:3, :],
        angular_momentum_rate_desired - centroidal_bias[:3],
        config.momentum_weight,
    )
    posture_jacobian = np.zeros(
        (N_ACTUATED, N_GENERALIZED),
        dtype=np.float64,
    )
    posture_jacobian[:, N_BASE:] = np.eye(N_ACTUATED)
    posture_acceleration_desired = (
        config.posture_kp
        * (state.nominal_joint_position - joint_position)
        - config.posture_kd * joint_velocity
    )
    _add_task_cost(
        hessian,
        gradient,
        posture_jacobian,
        posture_acceleration_desired,
        config.posture_weight,
    )

    # ------------------------------------------------------------------
    # Regularization and smooth force/torque references
    # ------------------------------------------------------------------
    hessian[QDD_SLICE, QDD_SLICE] += (
        2.0 * config.qdd_regularization * np.eye(N_GENERALIZED)
    )
    hessian[TAU_SLICE, TAU_SLICE] += (
        2.0 * config.torque_regularization * np.eye(N_ACTUATED)
    )
    hessian[FORCE_SLICE, FORCE_SLICE] += (
        2.0 * config.force_regularization * np.eye(N_FORCE_VARIABLES)
    )
    if state.previous_torque is not None:
        hessian[TAU_SLICE, TAU_SLICE] += (
            2.0
            * config.torque_rate_regularization
            * np.eye(N_ACTUATED)
        )
        gradient[TAU_SLICE] -= (
            2.0
            * config.torque_rate_regularization
            * state.previous_torque
        )
    hessian[FORCE_SLICE, FORCE_SLICE] += (
        2.0
        * config.force_rate_regularization
        * np.eye(N_FORCE_VARIABLES)
    )
    gradient[FORCE_SLICE] -= (
        2.0
        * config.force_rate_regularization
        * state.previous_force
    )
    force_reference = _force_reference(
        active_contacts,
        planner_state,
        robot_mass * config.gravity,
    )
    hessian[FORCE_SLICE, FORCE_SLICE] += (
        2.0
        * config.force_tracking_weight
        * np.eye(N_FORCE_VARIABLES)
    )
    gradient[FORCE_SLICE] -= (
        2.0 * config.force_tracking_weight * force_reference
    )

    # Small diagonal makes P strictly positive definite for both OSQP and a
    # dense active-set cross-check, without materially changing the tasks.
    hessian += 2.0e-10 * np.eye(N_VARIABLES)

    # ------------------------------------------------------------------
    # Hard floating-base inverse dynamics
    #
    # H qdd + CG = S^T tau + J_point^T f
    # ------------------------------------------------------------------
    selection_transpose = np.zeros(
        (N_GENERALIZED, N_ACTUATED),
        dtype=np.float64,
    )
    selection_transpose[N_BASE:, :] = np.eye(N_ACTUATED)
    dynamics_matrix = np.hstack(
        (
            mass_matrix,
            -selection_transpose,
            -point_jacobian.T,
        )
    )
    constraint_rows: list[np.ndarray] = [dynamics_matrix]
    lower_bounds: list[np.ndarray] = [-bias_force]
    upper_bounds: list[np.ndarray] = [-bias_force]

    # ------------------------------------------------------------------
    # Hard active-foot pose-acceleration constraints with Baumgarte drift
    # correction.  Six independent ankle constraints are better conditioned
    # than nine redundant point-position constraints on the same rigid foot.
    # ------------------------------------------------------------------
    active_contact_records: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for side in active_contacts:
        (
            contact_position,
            contact_rotation,
            contact_velocity,
            contact_jacobian,
            contact_bias,
        ) = _spatial_ankle_data(fk, side)
        position_error = np.clip(
            contact_position - state.contact_position_anchor[side],
            -0.02,
            0.02,
        )
        rotation_error = np.clip(
            _rotation_error_world(
                state.contact_rotation_anchor[side],
                contact_rotation,
            ),
            -0.15,
            0.15,
        )
        desired_contact_linear_acceleration = (
            -config.contact_position_kp * position_error
            - config.contact_position_kd * contact_velocity[:3]
        )
        desired_contact_angular_acceleration = (
            config.contact_orientation_kp * rotation_error
            - config.contact_orientation_kd * contact_velocity[3:]
        )
        desired_contact_acceleration = np.concatenate(
            (
                desired_contact_linear_acceleration,
                desired_contact_angular_acceleration,
            )
        )
        contact_target = desired_contact_acceleration - contact_bias
        contact_row = np.zeros(
            (6, N_VARIABLES),
            dtype=np.float64,
        )
        contact_row[:, QDD_SLICE] = contact_jacobian
        constraint_rows.append(contact_row)
        lower_bounds.append(contact_target)
        upper_bounds.append(contact_target)
        active_contact_records.append(
            (contact_jacobian, contact_bias, desired_contact_acceleration)
        )

    # ------------------------------------------------------------------
    # Friction pyramids: ±fx <= mu*fz and ±fy <= mu*fz.
    # ------------------------------------------------------------------
    friction_rows = np.zeros(
        (4 * N_POINT_CONTACTS, N_VARIABLES),
        dtype=np.float64,
    )
    friction_upper = np.zeros(4 * N_POINT_CONTACTS, dtype=np.float64)
    for contact_index in range(N_POINT_CONTACTS):
        force_start = FORCE_SLICE.start + 3 * contact_index
        row_start = 4 * contact_index
        friction_rows[row_start + 0, force_start + 0] = 1.0
        friction_rows[row_start + 0, force_start + 2] = (
            -config.friction_coefficient
        )
        friction_rows[row_start + 1, force_start + 0] = -1.0
        friction_rows[row_start + 1, force_start + 2] = (
            -config.friction_coefficient
        )
        friction_rows[row_start + 2, force_start + 1] = 1.0
        friction_rows[row_start + 2, force_start + 2] = (
            -config.friction_coefficient
        )
        friction_rows[row_start + 3, force_start + 1] = -1.0
        friction_rows[row_start + 3, force_start + 2] = (
            -config.friction_coefficient
        )
    constraint_rows.append(friction_rows)
    lower_bounds.append(np.full(friction_rows.shape[0], -np.inf))
    upper_bounds.append(friction_upper)

    # Force component bounds.  Inactive-foot forces are exactly zero.
    force_bound_rows = np.zeros(
        (N_FORCE_VARIABLES, N_VARIABLES),
        dtype=np.float64,
    )
    force_bound_rows[:, FORCE_SLICE] = np.eye(N_FORCE_VARIABLES)
    force_lower = np.full(N_FORCE_VARIABLES, -np.inf)
    force_upper = np.full(N_FORCE_VARIABLES, np.inf)
    for contact_index, (side, _, _, _) in enumerate(CONTACT_POINTS):
        component_start = 3 * contact_index
        if side in active_contacts:
            force_lower[component_start + 2] = 0.0
            force_upper[component_start + 2] = (
                config.maximum_point_normal_force
            )
        else:
            force_lower[component_start : component_start + 3] = 0.0
            force_upper[component_start : component_start + 3] = 0.0
    constraint_rows.append(force_bound_rows)
    lower_bounds.append(force_lower)
    upper_bounds.append(force_upper)

    # Torque magnitude and per-control-tick torque-rate bounds share one set
    # of identity rows by intersecting their lower/upper limits.
    torque_bound_rows = np.zeros(
        (N_ACTUATED, N_VARIABLES),
        dtype=np.float64,
    )
    torque_bound_rows[:, TAU_SLICE] = np.eye(N_ACTUATED)
    torque_limit = _torque_limits(parameters)
    torque_lower = -torque_limit
    torque_upper = torque_limit
    if state.previous_torque is not None:
        torque_lower = np.maximum(
            torque_lower,
            state.previous_torque - config.maximum_torque_step,
        )
        torque_upper = np.minimum(
            torque_upper,
            state.previous_torque + config.maximum_torque_step,
        )
    constraint_rows.append(torque_bound_rows)
    lower_bounds.append(torque_lower)
    upper_bounds.append(torque_upper)

    # Actuated-joint acceleration, velocity, and optional position bounds.
    qdd_bound_rows = np.zeros(
        (N_ACTUATED, N_VARIABLES),
        dtype=np.float64,
    )
    qdd_bound_rows[:, N_BASE:N_GENERALIZED] = np.eye(N_ACTUATED)
    qdd_lower, qdd_upper = _joint_bounds(
        joint_position,
        joint_velocity,
        config,
        parameters,
    )
    constraint_rows.append(qdd_bound_rows)
    lower_bounds.append(qdd_lower)
    upper_bounds.append(qdd_upper)

    constraint_matrix = np.vstack(constraint_rows)
    lower = np.concatenate(lower_bounds)
    upper = np.concatenate(upper_bounds)
    if np.any(lower > upper):
        raise ValueError("QP has an internally inconsistent bound")

    # OSQP consumes only the upper triangular part of the symmetric Hessian.
    p_sparse = sparse.csc_matrix(np.triu(hessian))
    a_sparse = sparse.csc_matrix(constraint_matrix)
    p_sparse.sort_indices()
    a_sparse.sort_indices()
    metadata: Dict[str, Any] = {
        "mass_matrix": mass_matrix,
        "bias_force": bias_force,
        "point_positions": point_positions,
        "point_jacobian": point_jacobian,
        "active_contacts": active_contacts,
        "active_contact_records": active_contact_records,
        "com_position": com_position,
        "com_velocity": com_velocity,
        "robot_mass": robot_mass,
        "swing_side": swing_side,
    }
    return (
        p_sparse,
        gradient,
        a_sparse,
        lower,
        upper,
        metadata,
    )


def _solve_osqp(
    state: ControllerState,
    p_matrix: sparse.csc_matrix,
    gradient: np.ndarray,
    constraint_matrix: sparse.csc_matrix,
    lower: np.ndarray,
    upper: np.ndarray,
    config: WBCQPConfig,
) -> Any:
    """Set up or numerically update the cached direct-OSQP workspace."""

    p_pattern = _sparse_pattern(p_matrix)
    a_pattern = _sparse_pattern(constraint_matrix)
    can_update = (
        state.solver is not None
        and state._p_pattern == p_pattern
        and state._a_pattern == a_pattern
    )
    if can_update:
        try:
            state.solver.update(
                Px=p_matrix.data,
                Ax=constraint_matrix.data,
                q=gradient,
                l=lower,
                u=upper,
            )
        except Exception:
            can_update = False

    if not can_update:
        state.solver = osqp.OSQP()
        state.solver.setup(
            P=p_matrix,
            q=gradient,
            A=constraint_matrix,
            l=lower,
            u=upper,
            verbose=False,
            warm_start=True,
            polish=False,
            adaptive_rho=True,
            eps_abs=config.solver_absolute_tolerance,
            eps_rel=config.solver_relative_tolerance,
            max_iter=config.solver_maximum_iterations,
        )
        state._p_pattern = p_pattern
        state._a_pattern = a_pattern

    if state.previous_solution is not None:
        state.solver.warm_start(x=state.previous_solution)
    return state.solver.solve()


def _constraint_residuals(
    solution: np.ndarray,
    metadata: Mapping[str, Any],
) -> tuple[float, float]:
    """Compute independent dynamics and active-contact infinity residuals."""

    generalized_acceleration = solution[QDD_SLICE]
    torque = solution[TAU_SLICE]
    point_force = solution[FORCE_SLICE]
    selection_transpose = np.zeros(
        (N_GENERALIZED, N_ACTUATED),
        dtype=np.float64,
    )
    selection_transpose[N_BASE:, :] = np.eye(N_ACTUATED)
    dynamics_residual = (
        metadata["mass_matrix"] @ generalized_acceleration
        + metadata["bias_force"]
        - selection_transpose @ torque
        - metadata["point_jacobian"].T @ point_force
    )
    dynamics_maximum = float(np.max(np.abs(dynamics_residual)))

    contact_maximum = 0.0
    for (
        contact_jacobian,
        contact_bias,
        desired_contact_acceleration,
    ) in metadata["active_contact_records"]:
        contact_residual = (
            contact_jacobian @ generalized_acceleration
            + contact_bias
            - desired_contact_acceleration
        )
        contact_maximum = max(
            contact_maximum,
            float(np.max(np.abs(contact_residual))),
        )
    return dynamics_maximum, contact_maximum


def _bound_residual(
    constraint_matrix: sparse.csc_matrix,
    solution: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Return the largest violation of any finite QP row bound.

    Dynamics and rigid-contact equalities are checked independently above so
    their diagnostics remain easy to interpret.  This complete check also
    covers friction pyramids, unilateral forces, inactive-force zeros, torque
    and torque-rate limits, and all acceleration/joint-limit inequalities.
    """

    value = np.asarray(constraint_matrix @ solution, dtype=np.float64).reshape(-1)
    # Infinite sides are the normal OSQP representation of one-sided or
    # unbounded rows, so do not pass these arrays through ``_array`` (which
    # intentionally rejects infinities in physical model inputs).
    lower = np.asarray(lower, dtype=np.float64).reshape(value.shape)
    upper = np.asarray(upper, dtype=np.float64).reshape(value.shape)
    if np.any(np.isnan(lower)) or np.any(np.isnan(upper)):
        raise ValueError("QP row bounds contain NaN")
    maximum = 0.0
    finite_lower = np.isfinite(lower)
    if np.any(finite_lower):
        maximum = max(
            maximum,
            float(np.max(np.maximum(lower[finite_lower] - value[finite_lower], 0.0))),
        )
    finite_upper = np.isfinite(upper)
    if np.any(finite_upper):
        maximum = max(
            maximum,
            float(np.max(np.maximum(value[finite_upper] - upper[finite_upper], 0.0))),
        )
    return maximum


def _fallback_torque(
    state: ControllerState,
    fk: Mapping[str, Any],
    parameters: Mapping[str, Any],
    config: WBCQPConfig,
) -> np.ndarray:
    """Return one-sample hold, then gravity/bias plus damping on repeated failure."""

    if state.failure_count == 1 and state.last_valid_torque is not None:
        fallback = state.last_valid_torque.copy()
    else:
        bias_force = _array(fk["CG"]).reshape(N_GENERALIZED)
        joint_velocity = _array(fk["dq"]).reshape(N_ACTUATED)
        fallback = (
            bias_force[N_BASE:]
            - config.fallback_joint_damping * joint_velocity
        )
    limit = _torque_limits(parameters)
    fallback = np.clip(fallback, -limit, limit)
    if state.previous_torque is not None:
        fallback = np.clip(
            fallback,
            state.previous_torque - config.maximum_torque_step,
            state.previous_torque + config.maximum_torque_step,
        )
    state.previous_torque = fallback.copy()
    state.last_tau_model = fallback.copy()
    state.last_qdd[:] = 0.0
    state.last_point_forces[:] = 0.0
    state.previous_force[:] = 0.0
    return fallback


def _contact_outputs(
    point_positions: np.ndarray,
    com_position: np.ndarray,
    point_force: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ZMP and net centroidal wrench from optimized point forces."""

    forces = _array(point_force).reshape(N_POINT_CONTACTS, 3)
    normal_force = forces[:, 2]
    normal_sum = float(np.sum(normal_force))
    if normal_sum > 1.0e-8:
        zmp = point_positions[:, :2].T @ normal_force / normal_sum
    else:
        zmp = np.zeros(2)
    net_force = np.sum(forces, axis=0)
    net_moment = np.sum(
        np.cross(point_positions - com_position[None, :], forces),
        axis=0,
    )
    return zmp, np.concatenate((net_force, net_moment))


def run_controller(
    ctrl_state: ControllerState,
    fk: Mapping[str, Any],
    planner_state: Mapping[str, Any],
    ctrl_params: Mapping[str, Any] | None,
    contact_l_z: float | None = None,
    contact_r_z: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Solve and apply one contact-aware whole-body inverse-dynamics QP.

    Args:
        ctrl_state:
            Persistent :class:`ControllerState`, one per simulated robot.
        fk:
            Current model snapshot.  Required keys are ``H``, ``CG``, ``AG``,
            ``dAGdq``, ``q``, ``dq``, base/COM state, ankle state/Jacobians,
            and the six point-contact positions/Jacobians produced by
            ``compute_FK``.  Full 34-column Jacobians are preferred as described
            in the module docstring.
        planner_state:
            Desired COM/DCM, base orientation, swing-foot pose, ``loco_state``,
            and ``leg_stance``.  Optional feed-forward acceleration keys are
            consumed when present.
        ctrl_params:
            Sparse configuration/gain overrides.  See :class:`WBCQPConfig`.
            Joint position limits should be supplied from Isaac's
            ``soft_joint_pos_limits`` in model order.
        contact_l_z/contact_r_z:
            Measured total vertical contact force for the left/right foot [N].
            During planned single support, a missing stance contact invokes the
            fallback instead of enforcing a fictitious contact constraint.

    Returns:
        ``tau_isaac``:
            28 actuator torques in Isaac articulation order [N m].
        ``zmp``:
            ZMP from optimized normal point forces [m], world x/y.
        ``com_position`` / ``com_velocity``:
            Measured whole-robot COM state [m], [m/s].
        ``centroidal_wrench``:
            Net optimized contact force and moment about COM [N, N m].
        ``point_forces``:
            Six optimized world-frame point forces, flattened [N].
    """

    if not isinstance(ctrl_state, ControllerState):
        raise TypeError("ctrl_state must be a themis_wbc_qp.ControllerState")
    parameters: Mapping[str, Any] = (
        {} if ctrl_params is None else ctrl_params
    )
    config = WBCQPConfig.from_parameters(parameters)
    com_position = _array(fk["p_wg"]).reshape(3)
    com_velocity = _array(fk["v_wg"]).reshape(3)
    point_positions, _ = _point_contact_data(fk)
    ctrl_state.last_measured_force_l = abs(float(contact_l_z or 0.0))
    ctrl_state.last_measured_force_r = abs(float(contact_r_z or 0.0))

    try:
        active_contacts = _active_contact_sides(planner_state)
        guard_ok = True
        if (
            config.require_stance_force_guard
            and len(active_contacts) == 1
            and contact_l_z is not None
            and contact_r_z is not None
        ):
            measured_stance_force = (
                ctrl_state.last_measured_force_l
                if active_contacts[0] == "l"
                else ctrl_state.last_measured_force_r
            )
            guard_ok = measured_stance_force >= config.minimum_stance_force
        elif (
            config.require_stance_force_guard
            and len(active_contacts) == 2
            and ctrl_state.last_valid_torque is not None
            and contact_l_z is not None
            and contact_r_z is not None
        ):
            # Permit the first DS solve while sensors initialize.  Once a valid
            # solution has been applied, however, do not keep enforcing two
            # fictitious no-slip contacts after the robot has lost the ground.
            guard_ok = (
                ctrl_state.last_measured_force_l
                + ctrl_state.last_measured_force_r
                >= config.minimum_stance_force
            )
        ctrl_state.last_stance_force_guard_ok = guard_ok
        if not guard_ok:
            raise RuntimeError("planned stance foot has insufficient measured load")

        (
            p_matrix,
            gradient,
            constraint_matrix,
            lower,
            upper,
            metadata,
        ) = _build_problem(
            ctrl_state,
            fk,
            planner_state,
            parameters,
            config,
        )
        result = _solve_osqp(
            ctrl_state,
            p_matrix,
            gradient,
            constraint_matrix,
            lower,
            upper,
            config,
        )
        status = str(result.info.status).strip().lower()
        solution = (
            None
            if result.x is None
            else np.asarray(result.x, dtype=np.float64)
        )
        if (
            status != "solved"
            or solution is None
            or solution.shape != (N_VARIABLES,)
            or not np.all(np.isfinite(solution))
        ):
            raise RuntimeError(f"OSQP did not return a valid solution: {status}")

        dynamics_residual, contact_residual = _constraint_residuals(
            solution,
            metadata,
        )
        ctrl_state.last_dynamics_residual = dynamics_residual
        ctrl_state.last_contact_residual = contact_residual
        constraint_residual = _bound_residual(
            constraint_matrix,
            solution,
            lower,
            upper,
        )
        ctrl_state.last_constraint_residual = constraint_residual
        hard_residual = max(
            dynamics_residual,
            contact_residual,
            constraint_residual,
        )
        if hard_residual > config.maximum_hard_constraint_residual:
            raise RuntimeError(
                f"hard-constraint residual {hard_residual:.3e} exceeds "
                f"{config.maximum_hard_constraint_residual:.3e}"
            )

        generalized_acceleration = solution[QDD_SLICE].copy()
        torque_model = solution[TAU_SLICE].copy()
        point_force = solution[FORCE_SLICE].copy()
        ctrl_state.failure_count = 0
        ctrl_state.last_status = status
        ctrl_state.last_message = ""
        ctrl_state.previous_solution = solution.copy()
        ctrl_state.previous_torque = torque_model.copy()
        ctrl_state.last_valid_torque = torque_model.copy()
        ctrl_state.previous_force = point_force.copy()
        ctrl_state.last_qdd = generalized_acceleration
        ctrl_state.last_tau_model = torque_model
        ctrl_state.last_point_forces = point_force
    except Exception as exception:
        ctrl_state.failure_count += 1
        ctrl_state.last_status = "fallback"
        ctrl_state.last_message = str(exception)
        torque_model = _fallback_torque(
            ctrl_state,
            fk,
            parameters,
            config,
        )
        point_force = np.zeros(N_FORCE_VARIABLES, dtype=np.float64)
        ctrl_state.last_dynamics_residual = np.inf
        ctrl_state.last_contact_residual = np.inf
        ctrl_state.last_constraint_residual = np.inf

    torque_isaac = _model_to_isaac_torque(torque_model)
    zmp, centroidal_wrench = _contact_outputs(
        point_positions,
        com_position,
        point_force,
    )
    return (
        torque_isaac,
        zmp,
        com_position,
        com_velocity,
        centroidal_wrench,
        point_force,
    )


# Explicit descriptive alias for callers that do not need legacy naming.
run_wbc_qp = run_controller


__all__ = [
    "ControllerState",
    "WBCQPState",
    "WBCQPConfig",
    "run_controller",
    "run_wbc_qp",
]
