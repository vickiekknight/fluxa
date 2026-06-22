from typing import Optional

from pydantic import BaseModel


class WorkspaceProbeResult(BaseModel):
    x: tuple[float, float]
    y: tuple[float, float]
    z: tuple[float, float]


class JointLimitsProbeResult(BaseModel):
    """Self-collision-free region of configuration space.

    Self-collision is multi-joint, so there is no per-joint box. We store
    summary stats plus a path to the validated collision-free configs (.npy,
    shape (n_safe, n_joints), robot joint order), which downstream stages
    sample from directly.
    """
    n_sampled: int
    n_safe: int
    collision_rate: float
    seed: int
    joint_lower: list[float]
    joint_upper: list[float]
    safe_config_path: str


class SuccessThresholdProbeResult(BaseModel):
    """
    Placeholder for future probe output.
    """
    pass


class ControllerGainProbeResult(BaseModel):
    """
    Placeholder for future probe output.
    """
    pass


class RobotConfig(BaseModel):
    name: str


class ProbeResults(BaseModel):
    workspace: Optional[WorkspaceProbeResult] = None
    joint_limits: Optional[JointLimitsProbeResult] = None
    success_threshold: Optional[SuccessThresholdProbeResult] = None
    controller_gains: Optional[ControllerGainProbeResult] = None

class DiscoveredConfig(BaseModel):
    robot: RobotConfig
    probes: ProbeResults