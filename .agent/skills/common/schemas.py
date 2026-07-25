from typing import Optional

from pydantic import BaseModel


class WorkspaceProbeResult(BaseModel):
    x: tuple[float, float]
    y: tuple[float, float]
    z: tuple[float, float]


class JointLimitsProbeResult(BaseModel):
    n_sampled: int
    n_safe: int
    collision_rate: float
    seed: int
    joint_lower: list[float]
    joint_upper: list[float]
    safe_config_path: str


class SuccessThresholdProbeResult(BaseModel):
    threshold_m: float
    statistic: str                                     
    position_error_percentiles_m: dict[str, float]    
    orientation_error_percentiles_deg: Optional[dict[str, float]] = None
 
    n_targets: int
    n_measured: int
    convergence_rate: float 
 
    ee_frame: str                                      
    command_type: str                                  
    n_steps: int                                       
    physics_dt: float
    gravity_z: Optional[float] = None
    target_orientation_rpy: tuple[float, float, float]
    units: str = "meters"
    seed: int


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