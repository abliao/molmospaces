import base64
import logging
from pathlib import Path
from typing import Any

import cv2
import msgpack_numpy
import numpy as np
from scipy.spatial.transform import Rotation as R

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.policy.learned_policy.websocket_policy import WebsocketPolicy
from molmo_spaces.utils.save_utils import save_frames_to_mp4

log = logging.getLogger(__name__)


def _fmt_debug_vector(vec: np.ndarray, decimals: int = 4) -> list[float]:
    arr = np.asarray(vec, dtype=np.float64).reshape(-1)
    return np.round(arr, decimals=decimals).tolist()


def _summarize_image(image: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(image)
    summary: dict[str, Any] = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
    }
    if arr.ndim == 3 and arr.shape[-1] == 3 and arr.size > 0:
        mean_rgb = arr.reshape(-1, 3).mean(axis=0)
        summary["mean_rgb"] = _fmt_debug_vector(mean_rgb, decimals=2)
    return summary


def _strip_action_prefix(key: str) -> str:
    if key.startswith("follow_"):
        return key[len("follow_") :]
    if key.startswith("master_"):
        return key[len("master_") :]
    return key


def _normalize_wallx_arm_name(arm: str) -> str:
    normalized = str(arm).strip().lower()
    aliases = {
        "left": "left",
        "follow1": "left",
        "follow1_pos": "left",
        "right": "right",
        "follow2": "right",
        "follow2_pos": "right",
    }
    if normalized not in aliases:
        raise ValueError(
            "WallXServerAdapterPolicy active_arm must be one of "
            f"{sorted(aliases)}, got {arm!r}"
        )
    return aliases[normalized]


def _normalize_wallx_io_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    aliases = {
        "follow_pose": "follow_pose",
        "follow": "follow_pose",
        "pose": "follow_pose",
        "joint": "joint",
        "joint_pos": "joint",
        "joint_position": "joint",
    }
    if normalized not in aliases:
        raise ValueError(
            "WallXServerAdapterPolicy wallx_io_mode must be one of "
            f"{sorted(aliases)}, got {mode!r}"
        )
    return aliases[normalized]


def _normalize_joint_gripper_scalar_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    aliases = {
        "normalized_open": "normalized_open",
        "continuous": "normalized_open",
        "continuous_open": "normalized_open",
        "binary": "binary",
        "threshold": "binary",
    }
    if normalized not in aliases:
        raise ValueError(
            "WallXServerAdapterPolicy wallx_joint_gripper_scalar_mode must be one of "
            f"{sorted(aliases)}, got {mode!r}"
        )
    return aliases[normalized]


def _normalize_joint_action_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    aliases = {
        "absolute": "absolute",
        "abs": "absolute",
        "joint_pos": "absolute",
        "joint_position": "absolute",
        "target": "absolute",
        "target_pos": "absolute",
        "delta": "delta",
        "relative": "delta",
        "increment": "delta",
        "incremental": "delta",
        "auto": "auto",
    }
    if normalized not in aliases:
        raise ValueError(
            "WallXServerAdapterPolicy wallx_joint_action_mode must be one of "
            f"{sorted(aliases)}, got {mode!r}"
        )
    return aliases[normalized]


def _merge_candidate_keys(*groups: tuple[str, ...]) -> tuple[str, ...]:
    merged: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for key in group:
            if not key or key in seen:
                continue
            merged.append(key)
            seen.add(key)
    return tuple(merged)


def _joint_arm_action_candidate_keys(configured_key: str) -> tuple[str, ...]:
    configured_key = str(configured_key)
    candidates = [configured_key]

    replacements = (
        ("_arm_joint_pos", ("_arm_joint_delta", "_arm_joint_pos_relative")),
        ("_arm_joint_delta", ("_arm_joint_pos", "_arm_joint_pos_relative")),
        ("_arm_joint_pos_relative", ("_arm_joint_delta", "_arm_joint_pos")),
    )
    for suffix, alternates in replacements:
        if configured_key.endswith(suffix):
            stem = configured_key[: -len(suffix)]
            candidates.extend(f"{stem}{alternate}" for alternate in alternates)

    return _merge_candidate_keys(tuple(candidates))


def _normalize_quat_order(order: str) -> str:
    normalized = str(order).strip().lower()
    if normalized not in {"xyzw", "wxyz"}:
        raise ValueError(
            f"Unsupported tcp_pose_quat_order={order!r}, expected 'xyzw' or 'wxyz'."
        )
    return normalized


def _quat_to_matrix(quat: np.ndarray, order: str) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).reshape(4)
    if order == "wxyz":
        return R.from_quat(quat, scalar_first=True).as_matrix()
    return R.from_quat(quat, scalar_first=False).as_matrix()


def _rotation_matrix_to_wallx_euler_xyz(rot_matrix: np.ndarray) -> np.ndarray:
    return R.from_matrix(rot_matrix).as_euler("ZYX").astype(np.float32)[::-1]


def _wallx_euler_xyz_to_matrix(euler_xyz: np.ndarray) -> np.ndarray:
    return R.from_euler("ZYX", np.asarray(euler_xyz, dtype=np.float64)[::-1]).as_matrix()


def _wallx_rotation_6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    rot6d = np.asarray(rot6d, dtype=np.float64).reshape(6)

    # Wall-X stores the first two rows of the rotation matrix, not the columns.
    row1 = rot6d[:3]
    row2 = rot6d[3:6]

    row1_norm = np.linalg.norm(row1)
    if row1_norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    basis1 = row1 / row1_norm

    row2_orth = row2 - np.dot(row2, basis1) * basis1
    row2_norm = np.linalg.norm(row2_orth)
    if row2_norm < 1e-12:
        candidate = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(np.dot(candidate, basis1)) > 0.9:
            candidate = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        row2_orth = candidate - np.dot(candidate, basis1) * basis1
        row2_norm = np.linalg.norm(row2_orth)
    basis2 = row2_orth / max(row2_norm, 1e-12)

    basis3 = np.cross(basis1, basis2)
    basis3_norm = np.linalg.norm(basis3)
    if basis3_norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    basis3 = basis3 / basis3_norm
    basis2 = np.cross(basis3, basis1)
    basis2 = basis2 / max(np.linalg.norm(basis2), 1e-12)

    return np.stack([basis1, basis2, basis3], axis=0)


def _rotation_error_deg(target_rot: np.ndarray, actual_rot: np.ndarray) -> float:
    rel_rot = np.asarray(target_rot, dtype=np.float64) @ np.asarray(actual_rot, dtype=np.float64).T
    return float(np.rad2deg(R.from_matrix(rel_rot).magnitude()))


def _default_action_dim_for_key(key: str) -> int | None:
    normalized = _strip_action_prefix(key)
    if normalized.endswith("_ee_cartesian_pos_relative") or normalized.endswith(
        "_ee_cartesian_pos"
    ):
        return 3
    if (
        normalized.endswith("_ee_rotation_6D_relative")
        or normalized.endswith("_ee_rotation_6D")
        or normalized.endswith("_ee_rotation_relative")
        or normalized.endswith("_ee_rotation")
    ):
        return 6 if "6D" in normalized else 3
    if (
        normalized.endswith("_arm_joint_pos")
        or normalized.endswith("_arm_joint_pos_relative")
        or normalized.endswith("_arm_joint_delta")
    ):
        return 7
    if normalized.endswith("_gripper"):
        return 1
    if normalized == "velocity_decomposed":
        return 3
    if normalized == "height":
        return 1
    if normalized == "head_actions":
        return 2
    return None


def _apply_local_translation(pose: np.ndarray, local_translation: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    translation = np.asarray(local_translation, dtype=np.float64).reshape(3)
    shifted_pose = pose.copy()
    shifted_pose[:3, 3] = pose[:3, 3] + pose[:3, :3] @ translation
    return shifted_pose


def _resize_rgb_to_height(image_rgb: np.ndarray, target_height: int) -> np.ndarray:
    image = np.asarray(image_rgb, dtype=np.uint8)
    if image.shape[0] == target_height:
        return image.copy()
    target_width = max(1, int(round(image.shape[1] * target_height / image.shape[0])))
    resized = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
    return np.asarray(resized, dtype=np.uint8)


def _annotate_rgb_image(
    image_rgb: np.ndarray,
    text: str,
    anchor: str = "top_left",
) -> np.ndarray:
    image = np.asarray(image_rgb, dtype=np.uint8).copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    outline_thickness = 4
    margin = 8
    baseline_pad = 4
    text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
    text_width, text_height = text_size

    if anchor == "top_left":
        org = (margin, margin + text_height)
    elif anchor == "top_right":
        org = (max(margin, image.shape[1] - margin - text_width), margin + text_height)
    elif anchor == "bottom_left":
        org = (margin, max(margin + text_height, image.shape[0] - margin - baseline))
    elif anchor == "bottom_right":
        org = (
            max(margin, image.shape[1] - margin - text_width),
            max(margin + text_height, image.shape[0] - margin - baseline),
        )
    else:
        raise ValueError(
            f"Unsupported annotation anchor={anchor!r}, expected one of "
            "'top_left', 'top_right', 'bottom_left', 'bottom_right'."
        )

    top_left = (
        max(0, org[0] - margin // 2),
        max(0, org[1] - text_height - baseline_pad),
    )
    bottom_right = (
        min(image.shape[1], org[0] + text_width + margin // 2),
        min(image.shape[0], org[1] + baseline + baseline_pad),
    )
    cv2.rectangle(image, top_left, bottom_right, (0, 0, 0), thickness=-1)
    cv2.putText(
        image,
        text,
        org,
        font,
        font_scale,
        (0, 0, 0),
        outline_thickness,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        org,
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return image


class WallXServerAdapterPolicy(WebsocketPolicy):
    """Adapt Wall-X sim websocket outputs to single-arm MolmoSpaces actions."""

    def __init__(
        self,
        exp_config: MlSpacesExpConfig,
        task_type: str,
    ) -> None:
        policy_config = exp_config.policy_config
        remote_config = getattr(policy_config, "remote_config", {}) or {}
        super().__init__(
            exp_config,
            model_name=getattr(policy_config, "model_name", "wallx_server"),
            host=remote_config.get("host", "127.0.0.1"),
            port=remote_config.get("port"),
            connection_timeout=getattr(policy_config, "connection_timeout", None),
        )

        self.task_type = task_type
        self.remote_config = remote_config
        self.infer_mode = getattr(policy_config, "infer_mode", "flow")
        self.return_action_format = getattr(policy_config, "return_action_format", "native")
        self.wallx_io_mode = _normalize_wallx_io_mode(
            getattr(policy_config, "wallx_io_mode", "follow_pose")
        )
        self.image_passing_mode = str(
            getattr(policy_config, "image_passing_mode", "base64")
        ).strip().lower()
        self.save_request_video = bool(getattr(policy_config, "save_request_video", False))
        self.request_video_max_episodes = max(
            0, int(getattr(policy_config, "request_video_max_episodes", 0))
        )
        self.request_video_dir = (
            Path(exp_config.output_dir)
            / str(getattr(policy_config, "request_video_dirname", "wallx_request_videos"))
        )
        self.tcp_pose_quat_order = _normalize_quat_order(
            getattr(policy_config, "tcp_pose_quat_order", "wxyz")
        )
        self.follow_pose_from_tcp_offset_local = np.asarray(
            getattr(policy_config, "wallx_follow_pose_from_tcp_offset_local", (0.0, 0.0, 0.0)),
            dtype=np.float64,
        ).reshape(3)
        self.follow_rotation_from_tcp_euler_xyz_deg = np.asarray(
            getattr(
                policy_config,
                "wallx_follow_rotation_from_tcp_euler_xyz_deg",
                (0.0, 0.0, 0.0),
            ),
            dtype=np.float64,
        ).reshape(3)
        self.follow_rotation_from_tcp_matrix = _wallx_euler_xyz_to_matrix(
            np.deg2rad(self.follow_rotation_from_tcp_euler_xyz_deg)
        )
        log.info(
            "Wall-X follow/TCP config: io_mode=%s offset_local=%s rot_offset_deg=%s",
            self.wallx_io_mode,
            _fmt_debug_vector(self.follow_pose_from_tcp_offset_local),
            _fmt_debug_vector(self.follow_rotation_from_tcp_euler_xyz_deg),
        )
        self.response_timeout = float(getattr(policy_config, "response_timeout", 300.0))
        self.grasping_threshold = float(getattr(policy_config, "grasping_threshold", 0.5))
        self.open_gripper_ctrl = float(getattr(policy_config, "open_gripper_ctrl", 0.0))
        self.closed_gripper_ctrl = float(getattr(policy_config, "closed_gripper_ctrl", 255.0))
        self.wallx_open_is_high = bool(getattr(policy_config, "wallx_open_is_high", True))
        self.active_arm = _normalize_wallx_arm_name(
            getattr(policy_config, "active_arm", "left")
        )
        self.front_camera_payload_key = str(
            getattr(policy_config, "front_camera_payload_key", "camera_front")
        )
        self.left_wrist_payload_key = str(
            getattr(policy_config, "left_wrist_payload_key", "camera_left")
        )
        self.right_wrist_payload_key = str(
            getattr(policy_config, "right_wrist_payload_key", "camera_right")
        )
        self.joint_state_arm_key = str(
            getattr(policy_config, "wallx_joint_state_arm_key", "follow_left_arm_joint_pos")
        )
        self.joint_state_gripper_key = str(
            getattr(policy_config, "wallx_joint_state_gripper_key", "follow_left_gripper")
        )
        self.joint_action_arm_key = str(
            getattr(policy_config, "wallx_joint_action_arm_key", "master_left_arm_joint_pos")
        )
        self.joint_action_gripper_key = str(
            getattr(policy_config, "wallx_joint_action_gripper_key", "master_left_gripper")
        )
        self.joint_action_mode = _normalize_joint_action_mode(
            getattr(policy_config, "wallx_joint_action_mode", "absolute")
        )
        self.joint_gripper_scalar_mode = _normalize_joint_gripper_scalar_mode(
            getattr(policy_config, "wallx_joint_gripper_scalar_mode", "normalized_open")
        )
        self.active_follow_state_key = "follow1_pos" if self.active_arm == "left" else "follow2_pos"
        self.inactive_follow_state_key = (
            "follow2_pos" if self.active_arm == "left" else "follow1_pos"
        )
        self.active_wrist_payload_key = (
            self.left_wrist_payload_key if self.active_arm == "left" else self.right_wrist_payload_key
        )
        self.inactive_wrist_payload_key = (
            self.right_wrist_payload_key
            if self.active_arm == "left"
            else self.left_wrist_payload_key
        )
        self.open_gripper_state_value = float(
            getattr(policy_config, "open_gripper_state_value", 1.0)
        )
        self.closed_gripper_state_value = float(
            getattr(policy_config, "closed_gripper_state_value", 0.0)
        )
        self.include_static_base_state = bool(
            getattr(policy_config, "include_static_base_state", True)
        )
        self.include_inactive_arm_state = bool(
            getattr(policy_config, "include_inactive_arm_state", False)
        )
        if self.active_arm == "right":
            self.include_inactive_arm_state = self.include_inactive_arm_state or bool(
                getattr(policy_config, "include_dummy_left_arm_state", False)
            )
        self.inactive_arm_gripper_state = float(
            getattr(
                policy_config,
                "inactive_arm_gripper_state",
                getattr(policy_config, "dummy_left_gripper_state", 1.0),
            )
        )
        self.static_velocity_decomposed = np.asarray(
            getattr(policy_config, "static_velocity_decomposed", (0.0, 0.0, 0.0)),
            dtype=np.float32,
        ).reshape(3)
        self.static_head_actions = np.asarray(
            getattr(policy_config, "static_head_actions", (0.0, 0.0)),
            dtype=np.float32,
        ).reshape(2)
        self.static_height = float(getattr(policy_config, "static_height", 0.0))
        self.front_camera_keys = tuple(getattr(policy_config, "front_camera_keys", ()))
        self.shared_wrist_camera_keys = tuple(
            getattr(policy_config, "shared_wrist_camera_keys", ())
        )
        self.left_wrist_camera_keys = tuple(getattr(policy_config, "left_wrist_camera_keys", ()))
        self.right_wrist_camera_keys = tuple(
            getattr(policy_config, "right_wrist_camera_keys", ())
        )
        if self.active_arm == "left":
            self.active_wrist_camera_keys = _merge_candidate_keys(
                self.left_wrist_camera_keys,
                self.shared_wrist_camera_keys,
            )
            self.inactive_wrist_camera_keys = self.right_wrist_camera_keys
        else:
            self.active_wrist_camera_keys = _merge_candidate_keys(
                self.right_wrist_camera_keys,
                self.shared_wrist_camera_keys,
            )
            self.inactive_wrist_camera_keys = self.left_wrist_camera_keys
        self.ik_eps = float(getattr(policy_config, "ik_eps", 1e-4))
        self.ik_max_iter = int(getattr(policy_config, "ik_max_iter", 200))
        self.ik_damping = float(getattr(policy_config, "ik_damping", 1e-10))
        self.ik_dt = float(getattr(policy_config, "ik_dt", 1.0))
        self.max_open_loop_steps = max(
            1, int(getattr(policy_config, "max_open_loop_steps", 32))
        )
        self.fallback_to_current_rotation_on_ik_failure = bool(
            getattr(policy_config, "fallback_to_current_rotation_on_ik_failure", True)
        )
        self.ik_debug_log_limit = int(getattr(policy_config, "ik_debug_log_limit", 5))
        self.ik_solution_debug_log_limit = int(
            getattr(policy_config, "ik_solution_debug_log_limit", 3)
        )
        self.request_debug_log_limit = int(
            getattr(policy_config, "request_debug_log_limit", 5)
        )
        self.response_debug_log_limit = int(
            getattr(policy_config, "response_debug_log_limit", 5)
        )

        self._action_buffer: np.ndarray | None = None
        self._buffer_index = 0
        self._action_keys: tuple[str, ...] = ()
        self._action_dims: tuple[int, ...] = ()
        self._action_slices: dict[str, slice] = {}
        self._last_obs: dict[str, Any] | None = None
        self._last_server_timing: dict[str, Any] | None = None
        self._last_action_horizon: int | None = None
        self._last_action_dim: int | None = None
        self._last_pose_source: str | None = None
        self._last_state_payload: dict[str, np.ndarray] = {}
        self._last_view_sources: dict[str, str] = {}
        self._last_view_summaries: dict[str, dict[str, Any]] = {}
        self._ik_failures = 0
        self._ik_debug_logs_emitted = 0
        self._ik_solution_debug_logs_emitted = 0
        self._request_debug_logs_emitted = 0
        self._response_debug_logs_emitted = 0
        self._request_video_episode_index = -1
        self._request_video_saved_episodes = 0
        self._request_video_step_index = 0
        self._request_video_frames: list[np.ndarray] = []

    def reset(self) -> None:
        self._flush_request_video()
        self._request_video_episode_index += 1
        self._request_video_step_index = 0
        self._action_buffer = None
        self._buffer_index = 0
        self._last_obs = None
        self._last_server_timing = None
        self._last_action_horizon = None
        self._last_action_dim = None
        self._last_pose_source = None
        self._last_state_payload = {}
        self._last_view_sources = {}
        self._last_view_summaries = {}
        self._ik_failures = 0
        self._ik_debug_logs_emitted = 0
        self._ik_solution_debug_logs_emitted = 0
        self._request_debug_logs_emitted = 0
        self._response_debug_logs_emitted = 0

    def close(self):
        self._flush_request_video()
        super().close()

    def _normalize_obs(self, obs: dict | list[dict]) -> dict:
        if isinstance(obs, list):
            if len(obs) > 1:
                log.warning(
                    "WallXServerAdapterPolicy received %d observations, only using the first one.",
                    len(obs),
                )
            return obs[0]
        return obs

    def _current_instruction(self) -> str:
        if self.task is not None:
            return self.task.get_task_description()
        return str(self.task_type)

    def _choose_camera_entry(
        self,
        obs: dict,
        candidate_keys: tuple[str, ...],
        label: str,
        required: bool,
    ) -> tuple[str, np.ndarray] | None:
        for key in candidate_keys:
            value = obs.get(key)
            if value is None:
                continue
            image = np.asarray(value)
            if image.ndim == 4 and image.shape[0] == 1:
                image = image[0]
            if image.ndim != 3:
                raise ValueError(f"Camera '{key}' for {label} must be HWC, got {image.shape}")
            return key, image.astype(np.uint8, copy=False)

        if required:
            raise KeyError(
                f"Missing required {label} image. Tried keys={candidate_keys}, "
                f"available_obs_keys={sorted(obs.keys())}"
            )
        return None

    def _current_robot(self):
        if self.task is None:
            raise RuntimeError("Policy is not attached to a task yet; cannot access robot state.")
        env = self.task.env
        if hasattr(env, "current_robot"):
            return env.current_robot
        return env.robots[0]

    def _current_gripper_state(self, obs: dict) -> float:
        try:
            robot = self._current_robot()
            gripper_group = robot.robot_view.get_move_group("gripper")
            return (
                self.open_gripper_state_value
                if gripper_group.is_open
                else self.closed_gripper_state_value
            )
        except Exception:
            pass

        qpos = obs.get("qpos", {})
        gripper_qpos = np.asarray(qpos.get("gripper", [0.0]), dtype=np.float32).reshape(-1)
        if gripper_qpos.size == 0:
            return self.open_gripper_state_value
        return (
            self.open_gripper_state_value
            if float(np.max(gripper_qpos)) > 1e-3
            else self.closed_gripper_state_value
        )

    def _current_arm_joint_pos(self, obs: dict) -> np.ndarray:
        try:
            robot = self._current_robot()
            return np.asarray(robot.robot_view.get_qpos_dict()["arm"], dtype=np.float32).copy()
        except Exception:
            pass

        qpos = obs.get("qpos", {})
        if isinstance(qpos, dict) and "arm" in qpos:
            return np.asarray(qpos["arm"], dtype=np.float32).reshape(-1).copy()

        raise KeyError(
            "WallX adapter joint mode expected current arm qpos in robot_view.get_qpos_dict()['arm'] "
            f"or obs['qpos']['arm']. Available observation keys: {sorted(obs.keys())}"
        )

    def _current_gripper_open_fraction(self, obs: dict) -> float:
        try:
            robot = self._current_robot()
            gripper_group = robot.robot_view.get_move_group("gripper")
            min_dist, max_dist = gripper_group.inter_finger_dist_range
            if max_dist > min_dist + 1e-8:
                return float(
                    np.clip(
                        (gripper_group.inter_finger_dist - min_dist) / (max_dist - min_dist),
                        0.0,
                        1.0,
                    )
                )
        except Exception:
            pass

        return float(np.clip(self._current_gripper_state(obs), 0.0, 1.0))

    def _current_tcp_pose_rel_base(self, obs: dict) -> np.ndarray:
        robot_state = obs.get("robot_state")
        if isinstance(robot_state, dict) and "ee_pose" in robot_state:
            pose = np.asarray(robot_state["ee_pose"], dtype=np.float64)
            if pose.shape != (4, 4):
                raise ValueError(
                    f"Expected robot_state['ee_pose'] shape (4, 4), got {pose.shape}"
                )
            self._last_pose_source = "robot_state.ee_pose"
            return pose

        if "ee_pose" in obs:
            pose = np.asarray(obs["ee_pose"], dtype=np.float64)
            if pose.shape != (4, 4):
                raise ValueError(f"Expected ee_pose shape (4, 4), got {pose.shape}")
            self._last_pose_source = "ee_pose"
            return pose

        if "tcp_pose" in obs:
            tcp_pose = np.asarray(obs["tcp_pose"], dtype=np.float64).reshape(-1)
            if tcp_pose.shape[0] != 7:
                raise ValueError(f"Expected tcp_pose shape (7,), got {tcp_pose.shape}")

            pose = np.eye(4, dtype=np.float64)
            pose[:3, 3] = tcp_pose[:3]
            pose[:3, :3] = _quat_to_matrix(tcp_pose[3:7], self.tcp_pose_quat_order)
            self._last_pose_source = f"tcp_pose[{self.tcp_pose_quat_order}]"
            return pose

        raise KeyError(
            "WallX adapter expected either robot_state['ee_pose'] (4x4), 'ee_pose' (4x4), "
            "or 'tcp_pose' (7D) in the observation. "
            f"Available observation keys: {sorted(obs.keys())}"
        )

    def _tcp_pose_to_follow_pose(self, tcp_pose_rel_base: np.ndarray) -> np.ndarray:
        tcp_pose_rel_base = np.asarray(tcp_pose_rel_base, dtype=np.float64)
        follow_pose_rel_base = np.eye(4, dtype=np.float64)
        follow_pose_rel_base[:3, :3] = (
            tcp_pose_rel_base[:3, :3] @ self.follow_rotation_from_tcp_matrix
        )
        follow_pose_rel_base[:3, 3] = (
            tcp_pose_rel_base[:3, 3]
            + follow_pose_rel_base[:3, :3] @ self.follow_pose_from_tcp_offset_local
        )
        return follow_pose_rel_base

    def _follow_pose_to_tcp_pose(self, follow_pose_rel_base: np.ndarray) -> np.ndarray:
        follow_pose_rel_base = np.asarray(follow_pose_rel_base, dtype=np.float64)
        tcp_pose_rel_base = np.eye(4, dtype=np.float64)
        tcp_pose_rel_base[:3, :3] = (
            follow_pose_rel_base[:3, :3] @ self.follow_rotation_from_tcp_matrix.T
        )
        tcp_pose_rel_base[:3, 3] = (
            follow_pose_rel_base[:3, 3]
            - follow_pose_rel_base[:3, :3] @ self.follow_pose_from_tcp_offset_local
        )
        return tcp_pose_rel_base

    def _build_state_payload(self, obs: dict) -> dict[str, Any]:
        if self.wallx_io_mode == "joint":
            self._last_pose_source = "robot_view.arm_joint_pos"
            return {
                self.joint_state_arm_key: self._current_arm_joint_pos(obs).astype(np.float32),
                self.joint_state_gripper_key: float(self._current_gripper_open_fraction(obs)),
            }

        tcp_pose = self._current_tcp_pose_rel_base(obs)
        follow_pose = self._tcp_pose_to_follow_pose(tcp_pose)
        follow_euler = _rotation_matrix_to_wallx_euler_xyz(follow_pose[:3, :3])
        gripper_state = self._current_gripper_state(obs)

        state = {
            self.active_follow_state_key: np.concatenate(
                [follow_pose[:3, 3], follow_euler, np.array([gripper_state], dtype=np.float64)],
                axis=0,
            ).astype(np.float32)
        }

        if self.include_inactive_arm_state:
            state[self.inactive_follow_state_key] = np.array(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, self.inactive_arm_gripper_state],
                dtype=np.float32,
            )

        return state

    def _build_views_payload(self, obs: dict) -> dict[str, Any]:
        if self.image_passing_mode not in {"base64", "numpy"}:
            raise ValueError(
                "WallXServerAdapterPolicy only supports image_passing_mode "
                f"'base64' or 'numpy', got {self.image_passing_mode!r}"
            )

        payload = {}
        view_sources: dict[str, str] = {}
        view_summaries: dict[str, dict[str, Any]] = {}

        front_entry = self._choose_camera_entry(
            obs,
            self.front_camera_keys,
            label="front camera",
            required=True,
        )
        assert front_entry is not None
        front_key, front_image = front_entry
        payload[self.front_camera_payload_key] = self._format_image_payload(front_image)
        view_sources[self.front_camera_payload_key] = front_key
        view_summaries[self.front_camera_payload_key] = _summarize_image(front_image)

        active_wrist_entry = self._choose_camera_entry(
            obs,
            self.active_wrist_camera_keys,
            label=f"{self.active_arm} wrist camera",
            required=True,
        )
        assert active_wrist_entry is not None
        active_wrist_key, active_wrist_image = active_wrist_entry
        payload[self.active_wrist_payload_key] = self._format_image_payload(active_wrist_image)
        view_sources[self.active_wrist_payload_key] = active_wrist_key
        view_summaries[self.active_wrist_payload_key] = _summarize_image(active_wrist_image)

        inactive_wrist_entry = self._choose_camera_entry(
            obs,
            self.inactive_wrist_camera_keys,
            label=f"inactive wrist camera ({self.inactive_wrist_payload_key})",
            required=False,
        )
        if inactive_wrist_entry is not None:
            inactive_wrist_key, inactive_wrist_image = inactive_wrist_entry
            payload[self.inactive_wrist_payload_key] = self._format_image_payload(
                inactive_wrist_image
            )
            view_sources[self.inactive_wrist_payload_key] = inactive_wrist_key
            view_summaries[self.inactive_wrist_payload_key] = _summarize_image(
                inactive_wrist_image
            )

        self._last_view_sources = view_sources
        self._last_view_summaries = view_summaries
        self._maybe_record_request_video_frame(front_image, active_wrist_image)

        return payload

    def _maybe_record_request_video_frame(
        self,
        front_image_rgb: np.ndarray,
        active_wrist_image_rgb: np.ndarray,
    ) -> None:
        if not self.save_request_video or self.request_video_max_episodes <= 0:
            return
        if self._request_video_episode_index < 0:
            return
        if self._request_video_saved_episodes >= self.request_video_max_episodes:
            return

        front_source = self._last_view_sources.get(
            self.front_camera_payload_key, self.front_camera_payload_key
        )
        wrist_source = self._last_view_sources.get(
            self.active_wrist_payload_key, self.active_wrist_payload_key
        )
        target_height = int(front_image_rgb.shape[0])
        front_frame = _annotate_rgb_image(
            _resize_rgb_to_height(front_image_rgb, target_height),
            f"{self.front_camera_payload_key}: {front_source}",
        )
        wrist_frame = _annotate_rgb_image(
            _resize_rgb_to_height(active_wrist_image_rgb, target_height),
            f"{self.active_wrist_payload_key}: {wrist_source}",
        )
        composite = np.concatenate([front_frame, wrist_frame], axis=1)
        composite = _annotate_rgb_image(
            composite,
            f"episode={self._request_video_episode_index:04d} step={self._request_video_step_index:04d}",
            anchor="top_right",
        )
        self._request_video_frames.append(np.asarray(composite, dtype=np.uint8))
        self._request_video_step_index += 1

    def _flush_request_video(self) -> None:
        if (
            not self.save_request_video
            or self.request_video_max_episodes <= 0
            or self._request_video_episode_index < 0
            or not self._request_video_frames
            or self._request_video_saved_episodes >= self.request_video_max_episodes
        ):
            self._request_video_frames = []
            return

        output_path = (
            self.request_video_dir
            / f"episode_{self._request_video_episode_index:04d}_wallx_request_views.mp4"
        )
        try:
            save_frames_to_mp4(
                self._request_video_frames,
                str(output_path),
                fps=float(self.config.fps),
            )
            self._request_video_saved_episodes += 1
            log.warning(
                "Saved Wall-X request video: %s (%d frames)",
                output_path,
                len(self._request_video_frames),
            )
        except Exception:
            log.warning("Failed to save Wall-X request video to %s", output_path, exc_info=True)
        finally:
            self._request_video_frames = []

    def _encode_image_base64(self, image_rgb: np.ndarray) -> str:
        image = np.asarray(image_rgb)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"Expected HWC RGB image, got {image.shape}")
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        ok, buffer = cv2.imencode(".png", image_bgr)
        if not ok:
            raise ValueError("Failed to encode Wall-X request image as PNG")
        return base64.b64encode(buffer.tobytes()).decode("utf-8")

    def _format_image_payload(self, image: np.ndarray) -> str | np.ndarray:
        if self.image_passing_mode == "base64":
            return self._encode_image_base64(image)
        return image[None]

    def obs_to_model_input(self, obs):
        obs = self._normalize_obs(obs)
        self._last_obs = obs
        model_input = {
            "state": self._build_state_payload(obs),
            "views": self._build_views_payload(obs),
            "instruction": self._current_instruction(),
            "infer_mode": self.infer_mode,
            "return_action_format": self.return_action_format,
        }
        self._last_state_payload = {
            key: np.asarray(value, dtype=np.float32).copy()
            for key, value in model_input["state"].items()
        }
        self._maybe_log_request_debug(model_input)
        return model_input

    def _maybe_log_request_debug(self, model_input: dict[str, Any]) -> None:
        if self._request_debug_logs_emitted >= self.request_debug_log_limit:
            return

        payload: dict[str, Any] = {
            "pose_source": self._last_pose_source,
            "instruction": model_input.get("instruction"),
            "state": {
                key: _fmt_debug_vector(value)
                for key, value in model_input.get("state", {}).items()
            },
            "view_sources": dict(self._last_view_sources),
            "view_summaries": dict(self._last_view_summaries),
        }
        log.warning("Wall-X request debug: %s", payload)
        self._request_debug_logs_emitted += 1

    def _maybe_log_response_debug(self, action_row: np.ndarray) -> None:
        if self._response_debug_logs_emitted >= self.response_debug_log_limit:
            return

        payload: dict[str, Any] = {
            "action_horizon": self._last_action_horizon,
            "action_dim": self._last_action_dim,
            "action_keys": list(self._action_keys),
        }

        if self.wallx_io_mode == "joint":
            try:
                arm_slice, arm_key = self._find_action_slice(
                    *_joint_arm_action_candidate_keys(self.joint_action_arm_key)
                )
                payload[arm_key] = _fmt_debug_vector(action_row[arm_slice])
            except Exception:
                pass

            try:
                grip_slice, grip_key = self._find_action_slice(self.joint_action_gripper_key)
                payload[grip_key] = _fmt_debug_vector(action_row[grip_slice])
            except Exception:
                pass

            log.warning("Wall-X response debug: %s", payload)
            self._response_debug_logs_emitted += 1
            return

        try:
            pos_slice, pos_key = self._find_action_slice(
                f"{self.active_arm}_ee_cartesian_pos_relative",
                f"{self.active_arm}_ee_cartesian_pos",
            )
            payload[pos_key] = _fmt_debug_vector(action_row[pos_slice])
        except Exception:
            pass

        try:
            rot_slice, rot_key = self._find_action_slice(
                f"{self.active_arm}_ee_rotation_6D_relative",
                f"{self.active_arm}_ee_rotation_6D",
                f"{self.active_arm}_ee_rotation_relative",
                f"{self.active_arm}_ee_rotation",
            )
            payload[rot_key] = _fmt_debug_vector(action_row[rot_slice])
        except Exception:
            pass

        try:
            grip_slice, grip_key = self._find_action_slice(f"{self.active_arm}_gripper")
            payload[grip_key] = _fmt_debug_vector(action_row[grip_slice])
        except Exception:
            pass

        log.warning("Wall-X response debug: %s", payload)
        self._response_debug_logs_emitted += 1

    def _update_action_layout(
        self,
        action_keys: list[str] | tuple[str, ...],
        action_dims: list[int] | tuple[int, ...],
        total_dim: int,
    ) -> None:
        if len(action_keys) != len(action_dims):
            raise ValueError(
                "Wall-X action layout mismatch: "
                f"{len(action_keys)} keys vs {len(action_dims)} dims"
            )
        if sum(int(dim) for dim in action_dims) != total_dim:
            raise ValueError(
                "Wall-X action dims do not sum to action_dim: "
                f"keys={action_keys}, dims={action_dims}, action_dim={total_dim}"
            )

        offset = 0
        slices: dict[str, slice] = {}
        for key, dim in zip(action_keys, action_dims, strict=True):
            dim = int(dim)
            current_slice = slice(offset, offset + dim)
            slices[key] = current_slice
            slices[_strip_action_prefix(key)] = current_slice
            offset += dim

        self._action_keys = tuple(action_keys)
        self._action_dims = tuple(int(dim) for dim in action_dims)
        self._action_slices = slices

    def _infer_action_dims(
        self,
        action_keys: list[str] | tuple[str, ...],
        total_dim: int,
    ) -> list[int]:
        dims: list[int] = []
        for key in action_keys:
            dim = _default_action_dim_for_key(key)
            if dim is None:
                raise ValueError(
                    "Wall-X response omitted predict_action_dims and the adapter could not infer "
                    f"the dim for action key {key!r}. Available keys={action_keys}"
                )
            dims.append(int(dim))

        if sum(dims) != total_dim:
            raise ValueError(
                "Wall-X inferred action dims do not sum to action_dim: "
                f"keys={action_keys}, inferred_dims={dims}, action_dim={total_dim}"
            )
        return dims

    def _request_remote_actions(self, model_input: dict[str, Any]) -> dict[str, Any]:
        self.prepare_model()
        packed = msgpack_numpy.packb(model_input)
        self._ws.send(packed)
        response = self._ws.recv(timeout=self.response_timeout)
        if isinstance(response, str):
            raise RuntimeError(f"Error in Wall-X inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    def inference_model(self, model_input):
        if self._action_buffer is None or self._buffer_index >= len(self._action_buffer):
            try:
                response = self._request_remote_actions(model_input)
            except Exception:
                log.warning("Wall-X websocket request failed, reconnecting and retrying once.", exc_info=True)
                self.close()
                self._prepared = False
                response = self._request_remote_actions(model_input)

            predict_action = response.get("predict_action")
            if predict_action is None:
                raise ValueError(f"Wall-X server returned no predict_action payload: {response.keys()}")

            predict_action_np = np.asarray(predict_action, dtype=np.float32)
            if predict_action_np.ndim == 3 and predict_action_np.shape[0] == 1:
                predict_action_np = predict_action_np[0]
            if predict_action_np.ndim != 2:
                raise ValueError(
                    f"Wall-X predict_action must be 2D [horizon, dim], got {predict_action_np.shape}"
                )

            action_keys = response.get("predict_action_keys")
            action_dims = response.get("predict_action_dims")
            if action_keys is None:
                raise ValueError("Wall-X native response must include predict_action_keys.")
            if action_dims is None:
                action_dims = self._infer_action_dims(
                    action_keys=action_keys,
                    total_dim=int(predict_action_np.shape[1]),
                )

            self._update_action_layout(
                action_keys=action_keys,
                action_dims=action_dims,
                total_dim=int(predict_action_np.shape[1]),
            )
            execute_horizon = min(self.max_open_loop_steps, int(predict_action_np.shape[0]))
            self._action_buffer = predict_action_np[:execute_horizon].copy()
            self._buffer_index = 0
            self._last_server_timing = response.get("server_timing")
            self._last_action_horizon = int(response.get("action_horizon", predict_action_np.shape[0]))
            self._last_action_dim = int(response.get("action_dim", predict_action_np.shape[1]))

        action_row = self._action_buffer[self._buffer_index].copy()
        self._buffer_index += 1
        self._maybe_log_response_debug(action_row)
        return {
            "action_row": action_row,
            "server_timing": self._last_server_timing,
        }

    def _find_action_slice(self, *candidate_keys: str) -> tuple[slice, str]:
        for key in candidate_keys:
            normalized = _strip_action_prefix(key)
            if normalized in self._action_slices:
                return self._action_slices[normalized], normalized
            if key in self._action_slices:
                return self._action_slices[key], key
        raise KeyError(
            "Could not find any of the expected Wall-X action keys "
            f"{candidate_keys}. Available keys={self._action_keys}"
        )

    def _decode_target_pose_rel_base(self, action_row: np.ndarray, current_pose: np.ndarray) -> np.ndarray:
        target_pose = np.eye(4, dtype=np.float64)
        current_pos = current_pose[:3, 3]
        current_rot = current_pose[:3, :3]

        pos_slice, pos_key = self._find_action_slice(
            f"{self.active_arm}_ee_cartesian_pos_relative",
            f"{self.active_arm}_ee_cartesian_pos",
        )
        pos_value = np.asarray(action_row[pos_slice], dtype=np.float64)
        if pos_key.endswith("_relative"):
            target_pose[:3, 3] = current_pos + pos_value[:3]
        else:
            target_pose[:3, 3] = pos_value[:3]

        try:
            rot_slice, rot_key = self._find_action_slice(
                f"{self.active_arm}_ee_rotation_6D_relative",
                f"{self.active_arm}_ee_rotation_6D",
                f"{self.active_arm}_ee_rotation_relative",
                f"{self.active_arm}_ee_rotation",
            )
        except KeyError as exc:
            raise KeyError(
                "Wall-X action row does not contain a supported "
                f"{self.active_arm}-arm rotation key."
            ) from exc

        rot_value = np.asarray(action_row[rot_slice], dtype=np.float64)
        if rot_value.shape[0] == 6:
            decoded_rot = _wallx_rotation_6d_to_matrix(rot_value)
        elif rot_value.shape[0] == 3:
            decoded_rot = _wallx_euler_xyz_to_matrix(rot_value)
        else:
            raise ValueError(
                f"Unsupported Wall-X rotation width {rot_value.shape[0]} for key {rot_key}."
            )

        if rot_key.endswith("_relative"):
            target_pose[:3, :3] = decoded_rot @ current_rot
        else:
            target_pose[:3, :3] = decoded_rot

        return target_pose

    def _decode_gripper_action(self, action_row: np.ndarray) -> np.ndarray:
        grip_slice, _ = self._find_action_slice(f"{self.active_arm}_gripper")
        grip_value = float(np.asarray(action_row[grip_slice], dtype=np.float64)[0])
        if self.wallx_open_is_high:
            is_open = grip_value >= self.grasping_threshold
        else:
            is_open = grip_value < self.grasping_threshold
        ctrl = self.open_gripper_ctrl if is_open else self.closed_gripper_ctrl
        return np.array([ctrl], dtype=np.float32)

    def _decode_joint_arm_action(self, action_row: np.ndarray) -> np.ndarray:
        arm_slice, arm_key = self._find_action_slice(
            *_joint_arm_action_candidate_keys(self.joint_action_arm_key)
        )
        arm_value = np.asarray(action_row[arm_slice], dtype=np.float32).reshape(-1)
        joint_action_mode = self.joint_action_mode
        if joint_action_mode == "auto":
            if "delta" in arm_key or "relative" in arm_key:
                joint_action_mode = "delta"
            else:
                joint_action_mode = "absolute"

        if joint_action_mode == "delta":
            current_arm_qpos = self._current_arm_joint_pos(self._last_obs).astype(np.float32)
            arm_value = current_arm_qpos + arm_value

        try:
            ctrl_limits = np.asarray(
                self._current_robot().robot_view.get_move_group("arm").ctrl_limits,
                dtype=np.float32,
            )
            if ctrl_limits.shape == (arm_value.shape[0], 2):
                arm_value = np.clip(arm_value, ctrl_limits[:, 0], ctrl_limits[:, 1])
        except Exception:
            pass
        return arm_value.copy()

    def _decode_joint_gripper_action(self, action_row: np.ndarray) -> np.ndarray:
        grip_slice, _ = self._find_action_slice(self.joint_action_gripper_key)
        grip_value = float(np.asarray(action_row[grip_slice], dtype=np.float64).reshape(-1)[0])

        if self.joint_gripper_scalar_mode == "normalized_open":
            open_fraction = float(np.clip(grip_value, 0.0, 1.0))
            ctrl = self.closed_gripper_ctrl + open_fraction * (
                self.open_gripper_ctrl - self.closed_gripper_ctrl
            )
            return np.array([ctrl], dtype=np.float32)

        if self.wallx_open_is_high:
            is_open = grip_value >= self.grasping_threshold
        else:
            is_open = grip_value < self.grasping_threshold
        ctrl = self.open_gripper_ctrl if is_open else self.closed_gripper_ctrl
        return np.array([ctrl], dtype=np.float32)

    def _maybe_log_ik_debug(
        self,
        current_pose_rel_base: np.ndarray,
        target_pose_rel_base: np.ndarray,
        action_row: np.ndarray,
        reason: str,
    ) -> None:
        if self._ik_debug_logs_emitted >= self.ik_debug_log_limit:
            return

        robot = self._current_robot()
        fk_pose_rel_base = np.asarray(
            robot.robot_view.get_move_group("arm").leaf_frame_to_robot,
            dtype=np.float64,
        )
        payload: dict[str, Any] = {
            "reason": reason,
            "pose_source": self._last_pose_source,
            "current_pos": _fmt_debug_vector(current_pose_rel_base[:3, 3]),
            "target_pos": _fmt_debug_vector(target_pose_rel_base[:3, 3]),
            "delta_pos": _fmt_debug_vector(target_pose_rel_base[:3, 3] - current_pose_rel_base[:3, 3]),
            "current_rpy_xyz": _fmt_debug_vector(
                _rotation_matrix_to_wallx_euler_xyz(current_pose_rel_base[:3, :3])
            ),
            "target_rpy_xyz": _fmt_debug_vector(
                _rotation_matrix_to_wallx_euler_xyz(target_pose_rel_base[:3, :3])
            ),
            "fk_pos": _fmt_debug_vector(fk_pose_rel_base[:3, 3]),
            "obs_minus_fk_pos": _fmt_debug_vector(
                current_pose_rel_base[:3, 3] - fk_pose_rel_base[:3, 3]
            ),
            "fk_rpy_xyz": _fmt_debug_vector(
                _rotation_matrix_to_wallx_euler_xyz(fk_pose_rel_base[:3, :3])
            ),
            "obs_self_ik_ok": self._run_arm_ik(current_pose_rel_base) is not None,
            "request_state": {
                key: _fmt_debug_vector(value)
                for key, value in self._last_state_payload.items()
            },
            "view_sources": dict(self._last_view_sources),
        }

        try:
            pos_slice, _ = self._find_action_slice(
                f"{self.active_arm}_ee_cartesian_pos_relative",
                f"{self.active_arm}_ee_cartesian_pos",
            )
            payload["action_pos_component"] = _fmt_debug_vector(action_row[pos_slice])
        except Exception:
            pass

        try:
            rot_slice, _ = self._find_action_slice(
                f"{self.active_arm}_ee_rotation_6D_relative",
                f"{self.active_arm}_ee_rotation_6D",
                f"{self.active_arm}_ee_rotation_relative",
                f"{self.active_arm}_ee_rotation",
            )
            payload["action_rot_component"] = _fmt_debug_vector(action_row[rot_slice])
        except Exception:
            pass

        log.warning("Wall-X IK debug: %s", payload)
        self._ik_debug_logs_emitted += 1

    def _fk_pose_for_arm_qpos(self, arm_qpos: np.ndarray) -> np.ndarray:
        robot = self._current_robot()
        qpos_dict = {
            mg_id: np.asarray(qpos, dtype=np.float64).copy()
            for mg_id, qpos in robot.robot_view.get_qpos_dict().items()
        }
        qpos_dict["arm"] = np.asarray(arm_qpos, dtype=np.float64).copy()
        fk_dict = robot.kinematics.fk(
            qpos_dict,
            robot.robot_view.base.pose,
            rel_to_base=True,
        )
        return np.asarray(fk_dict["arm"], dtype=np.float64)

    def _maybe_log_ik_solution_debug(
        self,
        *,
        stage: str,
        executed_target_pose_rel_base: np.ndarray,
        arm_qpos: np.ndarray,
        current_arm_qpos: np.ndarray,
        requested_target_pose_rel_base: np.ndarray | None = None,
    ) -> None:
        if self._ik_solution_debug_logs_emitted >= self.ik_solution_debug_log_limit:
            return

        solved_fk_pose_rel_base = self._fk_pose_for_arm_qpos(arm_qpos)
        payload: dict[str, Any] = {
            "stage": stage,
            "pose_source": self._last_pose_source,
            "executed_target_pos": _fmt_debug_vector(executed_target_pose_rel_base[:3, 3]),
            "executed_target_rpy_xyz": _fmt_debug_vector(
                _rotation_matrix_to_wallx_euler_xyz(executed_target_pose_rel_base[:3, :3])
            ),
            "solved_fk_pos": _fmt_debug_vector(solved_fk_pose_rel_base[:3, 3]),
            "solved_fk_rpy_xyz": _fmt_debug_vector(
                _rotation_matrix_to_wallx_euler_xyz(solved_fk_pose_rel_base[:3, :3])
            ),
            "solved_pos_error": _fmt_debug_vector(
                solved_fk_pose_rel_base[:3, 3] - executed_target_pose_rel_base[:3, 3]
            ),
            "solved_pos_error_norm": round(
                float(
                    np.linalg.norm(
                        solved_fk_pose_rel_base[:3, 3] - executed_target_pose_rel_base[:3, 3]
                    )
                ),
                6,
            ),
            "solved_rot_error_deg": round(
                _rotation_error_deg(
                    executed_target_pose_rel_base[:3, :3],
                    solved_fk_pose_rel_base[:3, :3],
                ),
                6,
            ),
            "arm_qpos_solution": _fmt_debug_vector(arm_qpos),
            "arm_qpos_delta_from_current": _fmt_debug_vector(
                np.asarray(arm_qpos, dtype=np.float64) - np.asarray(current_arm_qpos, dtype=np.float64)
            ),
            "arm_qpos_delta_norm": round(
                float(
                    np.linalg.norm(
                        np.asarray(arm_qpos, dtype=np.float64)
                        - np.asarray(current_arm_qpos, dtype=np.float64)
                    )
                ),
                6,
            ),
        }

        if requested_target_pose_rel_base is not None:
            payload["requested_target_pos"] = _fmt_debug_vector(
                requested_target_pose_rel_base[:3, 3]
            )
            payload["requested_target_rpy_xyz"] = _fmt_debug_vector(
                _rotation_matrix_to_wallx_euler_xyz(requested_target_pose_rel_base[:3, :3])
            )
            payload["requested_vs_executed_rot_error_deg"] = round(
                _rotation_error_deg(
                    requested_target_pose_rel_base[:3, :3],
                    executed_target_pose_rel_base[:3, :3],
                ),
                6,
            )

        log.warning("Wall-X IK solution debug: %s", payload)
        self._ik_solution_debug_logs_emitted += 1

    def _run_arm_ik(self, target_pose_rel_base: np.ndarray) -> np.ndarray | None:
        robot = self._current_robot()
        q0 = robot.robot_view.get_qpos_dict()
        return robot.kinematics.ik(
            "arm",
            target_pose_rel_base,
            ["arm"],
            q0,
            robot.robot_view.base.pose,
            rel_to_base=True,
            eps=self.ik_eps,
            max_iter=self.ik_max_iter,
            damping=self.ik_damping,
            dt=self.ik_dt,
        )

    def _solve_arm_ik(
        self,
        target_pose_rel_base: np.ndarray,
        current_pose_rel_base: np.ndarray,
        action_row: np.ndarray,
    ) -> np.ndarray:
        robot = self._current_robot()
        q0 = robot.robot_view.get_qpos_dict()
        current_arm_qpos = np.asarray(q0["arm"], dtype=np.float64).copy()
        ik_solution = self._run_arm_ik(target_pose_rel_base)
        if ik_solution is not None:
            arm_qpos = np.asarray(ik_solution["arm"], dtype=np.float32).copy()
            self._maybe_log_ik_solution_debug(
                stage="full_pose_success",
                executed_target_pose_rel_base=target_pose_rel_base,
                arm_qpos=arm_qpos,
                current_arm_qpos=current_arm_qpos,
            )
            return arm_qpos

        self._ik_failures += 1
        self._maybe_log_ik_debug(
            current_pose_rel_base=current_pose_rel_base,
            target_pose_rel_base=target_pose_rel_base,
            action_row=action_row,
            reason="full_pose_failed",
        )

        if self.fallback_to_current_rotation_on_ik_failure:
            fallback_pose_rel_base = target_pose_rel_base.copy()
            fallback_pose_rel_base[:3, :3] = current_pose_rel_base[:3, :3]
            fallback_solution = self._run_arm_ik(fallback_pose_rel_base)
            if fallback_solution is not None:
                arm_qpos = np.asarray(fallback_solution["arm"], dtype=np.float32).copy()
                self._maybe_log_ik_solution_debug(
                    stage="fallback_orientation_only_success",
                    executed_target_pose_rel_base=fallback_pose_rel_base,
                    requested_target_pose_rel_base=target_pose_rel_base,
                    arm_qpos=arm_qpos,
                    current_arm_qpos=current_arm_qpos,
                )
                log.warning(
                    "Wall-X adapter IK failed for target pose; recovered by keeping current orientation. failure_count=%d",
                    self._ik_failures,
                )
                return arm_qpos

        log.warning(
            "Wall-X adapter IK failed for target pose; keeping current arm qpos. failure_count=%d",
            self._ik_failures,
        )
        return np.asarray(q0["arm"], dtype=np.float32).copy()

    def model_output_to_action(self, model_output):
        if self._last_obs is None:
            raise RuntimeError("Wall-X adapter has no cached observation for action decoding.")

        action_row = np.asarray(model_output["action_row"], dtype=np.float32).reshape(-1)
        if self.wallx_io_mode == "joint":
            return {
                "arm": self._decode_joint_arm_action(action_row),
                "gripper": self._decode_joint_gripper_action(action_row),
            }

        current_tcp_pose = self._current_tcp_pose_rel_base(self._last_obs)
        current_follow_pose = self._tcp_pose_to_follow_pose(current_tcp_pose)
        target_follow_pose_rel_base = self._decode_target_pose_rel_base(
            action_row, current_follow_pose
        )
        target_tcp_pose_rel_base = self._follow_pose_to_tcp_pose(target_follow_pose_rel_base)
        arm_action = self._solve_arm_ik(target_tcp_pose_rel_base, current_tcp_pose, action_row)
        gripper_action = self._decode_gripper_action(action_row)

        return {
            "arm": arm_action,
            "gripper": gripper_action,
        }

    def get_info(self) -> dict:
        info = super().get_info()
        info["policy_name"] = "wallx_server_adapter"
        info["policy_model_name"] = self.model_name
        info["policy_uri"] = self._uri
        info["wallx_io_mode"] = self.wallx_io_mode
        info["wallx_active_arm"] = self.active_arm
        info["wallx_active_follow_state_key"] = self.active_follow_state_key
        info["wallx_joint_state_arm_key"] = self.joint_state_arm_key
        info["wallx_joint_state_gripper_key"] = self.joint_state_gripper_key
        info["wallx_joint_action_arm_key"] = self.joint_action_arm_key
        info["wallx_joint_action_gripper_key"] = self.joint_action_gripper_key
        info["wallx_joint_action_mode"] = self.joint_action_mode
        info["wallx_joint_gripper_scalar_mode"] = self.joint_gripper_scalar_mode
        info["wallx_front_camera_payload_key"] = self.front_camera_payload_key
        info["wallx_active_wrist_payload_key"] = self.active_wrist_payload_key
        info["policy_infer_mode"] = self.infer_mode
        info["policy_return_action_format"] = self.return_action_format
        info["wallx_tcp_pose_quat_order"] = self.tcp_pose_quat_order
        info["wallx_follow_pose_from_tcp_offset_local"] = _fmt_debug_vector(
            self.follow_pose_from_tcp_offset_local
        )
        info["wallx_follow_rotation_from_tcp_euler_xyz_deg"] = _fmt_debug_vector(
            self.follow_rotation_from_tcp_euler_xyz_deg
        )
        info["wallx_pose_source"] = self._last_pose_source
        info["wallx_view_sources"] = dict(self._last_view_sources)
        info["wallx_view_summaries"] = dict(self._last_view_summaries)
        info["wallx_action_horizon"] = self._last_action_horizon
        info["wallx_action_dim"] = self._last_action_dim
        info["wallx_max_open_loop_steps"] = self.max_open_loop_steps
        info["wallx_executed_action_horizon"] = (
            None if self._action_buffer is None else int(len(self._action_buffer))
        )
        info["wallx_save_request_video"] = self.save_request_video
        info["wallx_request_video_dir"] = str(self.request_video_dir)
        info["wallx_request_video_episode_index"] = self._request_video_episode_index
        info["wallx_request_video_saved_episodes"] = self._request_video_saved_episodes
        info["wallx_request_video_pending_frames"] = int(len(self._request_video_frames))
        info["wallx_action_keys"] = list(self._action_keys)
        info["wallx_server_timing"] = self._last_server_timing
        info["wallx_buffer_remaining"] = (
            None
            if self._action_buffer is None
            else int(len(self._action_buffer) - self._buffer_index)
        )
        info["ik_failures"] = self._ik_failures
        info["ik_debug_logs_emitted"] = self._ik_debug_logs_emitted
        info["ik_solution_debug_logs_emitted"] = self._ik_solution_debug_logs_emitted
        info["request_debug_logs_emitted"] = self._request_debug_logs_emitted
        info["response_debug_logs_emitted"] = self._response_debug_logs_emitted
        info["prompt"] = self._current_instruction()
        return info
