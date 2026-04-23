from molmo_spaces.configs.policy_configs import BasePolicyConfig


class PiPolicyConfig(BasePolicyConfig):
    checkpoint_path: str = "checkpoints/pi"
    # remote_config: None -> launch local server
    # or dict(host,port) -> attaches to remote server
    remote_config: dict | None = dict(host="39.101.65.229", port=32332)
    grasping_type: str = "binary"
    grasping_threshold: float = 0.5
    chunk_size: int = 8

    policy_cls: type = None
    policy_type: str = "learned"

    def model_post_init(self, __context) -> None:
        """Set policy_cls after initialization to avoid circular imports."""
        super().model_post_init(__context)
        if self.policy_cls is None:
            from molmo_spaces.policy.learned_policy.pi_policy import PI_Policy

            self.policy_cls = PI_Policy


class DreamZeroPolicyConfig(BasePolicyConfig):
    checkpoint_path: str = "checkpoints/dreamzero"
    remote_config: dict = dict(host="localhost", port=0000)
    grasping_type: str = "binary"
    grasping_threshold: float = 0.5
    chunk_size: int = 24

    policy_cls: type = None
    policy_type: str = "learned"

    def model_post_init(self, __context) -> None:
        """Set policy_cls after initialization to avoid circular imports."""
        super().model_post_init(__context)
        if self.policy_cls is None:
            from molmo_spaces.policy.learned_policy.dreamzero_policy import DreamZero_Policy

            self.policy_cls = DreamZero_Policy


class CAPPolicyConfig(BasePolicyConfig):
    remote_config: dict = dict(host="localhost", port=8765)
    grasping_type: str = "binary"
    grasping_threshold: float = 0.7
    policy_cls: type = None
    policy_type: str = "learned"
    use_vlm: bool = False  # required for non-pick tasks
    exo_vlm: bool = True  # not used if use_vlm is False

    def model_post_init(self, __context) -> None:
        """Set policy_cls after initialization to avoid circular imports."""
        super().model_post_init(__context)
        if self.policy_cls is None:
            from molmo_spaces.policy.learned_policy.cap_policy import CAP_Policy

            self.policy_cls = CAP_Policy


class WallXServerAdapterPolicyConfig(BasePolicyConfig):
    remote_config: dict = dict(host="39.101.65.229", port=32178)
    model_name: str = "wallx_server"
    infer_mode: str = "flow"
    return_action_format: str = "native"
    image_passing_mode: str = "base64"
    max_open_loop_steps: int = 16
    save_request_video: bool = True
    request_video_max_episodes: int = 1
    request_video_dirname: str = "wallx_request_videos"
    # MolmoSpaces pose_mat_to_7d emits scalar-first quaternions: (qw, qx, qy, qz).
    tcp_pose_quat_order: str = "wxyz"
    connection_timeout: float | None = 300.0
    response_timeout: float = 300.0

    # Wall-X training semantics for this benchmark are follow1/left-arm-centric.
    # The adapter also accepts "follow1"/"follow2" as aliases.
    active_arm: str = "left"
    # "follow_pose" keeps the legacy TCP/follow-pose + IK path.
    # "joint" sends/receives arm joints directly and bypasses IK.
    wallx_io_mode: str = "follow_pose"
    front_camera_payload_key: str = "camera_front"
    left_wrist_payload_key: str = "camera_left"
    right_wrist_payload_key: str = "camera_right"
    wallx_joint_state_arm_key: str = "follow_left_arm_joint_pos"
    wallx_joint_state_gripper_key: str = "follow_left_gripper"
    wallx_joint_action_arm_key: str = "master_left_arm_joint_pos"
    wallx_joint_action_gripper_key: str = "master_left_gripper"
    # "absolute": server arm output is interpreted as target joint positions.
    # "delta": server arm output is added to current joint positions.
    # "auto": infer from the action key name (e.g. "*delta*" or "*relative*").
    wallx_joint_action_mode: str = "absolute"
    # "normalized_open": gripper scalar is treated as openness in [0, 1].
    # "binary": gripper scalar is thresholded with grasping_threshold.
    wallx_joint_gripper_scalar_mode: str = "normalized_open"

    grasping_threshold: float = 0.5
    open_gripper_ctrl: float = 0.0
    closed_gripper_ctrl: float = 255.0
    wallx_open_is_high: bool = True
    open_gripper_state_value: float = 1.0
    closed_gripper_state_value: float = 0.0
    # DROID training uses follow_left_position rather than follow_left_position_tcp.
    # In the raw dataset, this reference point is a rigid offset of [-0.15, 0, 0]
    # in the follow/tool local frame relative to position_tcp.
    wallx_follow_pose_from_tcp_offset_local: tuple[float, float, float] = (-0.15, 0.0, 0.0)
    # Optional fixed orientation offset from MolmoSpaces TCP frame to Wall-X's
    # training follow frame, expressed as [roll, pitch, yaw] in degrees.
    wallx_follow_rotation_from_tcp_euler_xyz_deg: tuple[float, float, float] = (
        0.0,
        0.0,
        0.0,
    )

    include_static_base_state: bool = True
    include_inactive_arm_state: bool = False
    inactive_arm_gripper_state: float = 1.0

    # Legacy compatibility for older right-arm configs.
    include_dummy_left_arm_state: bool = False
    dummy_left_gripper_state: float = 1.0
    static_velocity_decomposed: tuple[float, float, float] = (0.0, 0.0, 0.0)
    static_head_actions: tuple[float, float] = (0.0, 0.0)
    static_height: float = 0.0

    front_camera_keys: tuple[str, ...] = (
        "exo_camera_1",
        "droid_shoulder_light_randomization",
        "face_view",
        "camera_front",
    )
    shared_wrist_camera_keys: tuple[str, ...] = (
        "wrist_camera",
        "wrist_camera_zed_mini",
    )
    left_wrist_camera_keys: tuple[str, ...] = (
        "left_wrist_view",
        "camera_left",
    )
    right_wrist_camera_keys: tuple[str, ...] = (
        "right_wrist_view",
        "camera_right",
    )

    ik_eps: float = 1e-4
    ik_max_iter: int = 200
    ik_damping: float = 1e-10
    ik_dt: float = 1.0
    fallback_to_current_rotation_on_ik_failure: bool = True
    ik_debug_log_limit: int = 5
    ik_solution_debug_log_limit: int = 3
    request_debug_log_limit: int = 5
    response_debug_log_limit: int = 5

    policy_cls: type = None
    policy_type: str = "learned"

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        if self.policy_cls is None:
            from molmo_spaces.policy.learned_policy.wallx_server_adapter import (
                WallXServerAdapterPolicy,
            )

            self.policy_cls = WallXServerAdapterPolicy


class TeleopPolicyConfig(BasePolicyConfig):
    device: str = "keyboard"  # "spacemouse", "keyboard", "phone"
    policy_cls: type = None
    policy_type: str = "teleop"
    # keyboard params
    step_size: float = 0.005
    rot_step: float = 0.02
    # spacemouse params
    pos_sensitivity: float = 0.005
    rot_sensitivity: float = 0.02
    product_id: int = 50741  # 50741=wireless, 50734=wired

    def model_post_init(self, __context) -> None:
        """Set policy_cls after initialization to avoid circular imports."""
        super().model_post_init(__context)
        if self.policy_cls is None:
            if self.device == "keyboard":
                from molmo_spaces.policy.learned_policy.keyboard_policy import Keyboard_Policy

                self.policy_cls = Keyboard_Policy
            elif self.device == "spacemouse":
                from molmo_spaces.policy.learned_policy.spacemouse_policy import SpaceMouse_Policy

                self.policy_cls = SpaceMouse_Policy
            elif self.device == "phone":
                from molmo_spaces.policy.learned_policy.phone_policy import Phone_Policy

                self.policy_cls = Phone_Policy


class BimanualYamPiPolicyConfig(BasePolicyConfig):
    """Configuration for BimanualYamPiPolicy using LeRobot gRPC server."""

    name: str = "bimanual_yam_pi"
    checkpoint_path: str = "Jiafei1224/ppack200k"  # HuggingFace model ID
    remote_config: dict = dict(
        host="triton-cs-aus-454.reviz.ai2.in",
        port=8060,
        policy_type="pi05",
        device="cuda",
    )
    grasping_type: str = "binary"  # "binary" or "continuous"
    buffer_length: int = 50  # Number of actions per inference call

    # Camera mapping: MuJoCo camera name -> LeRobot observation key
    camera_mapping: dict = dict(
        left_wrist_camera="observation.images.left",
        right_wrist_camera="observation.images.right",
        exo_camera="observation.images.top",
    )

    policy_cls: type = None
    policy_type: str = "learned"

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        if self.policy_cls is None:
            from molmo_spaces.policy.learned_policy.bimanual_yam_pi_policy import (
                BimanualYamPiPolicy,
            )

            self.policy_cls = BimanualYamPiPolicy
