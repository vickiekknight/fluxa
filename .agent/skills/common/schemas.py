from typing import Optional

from pydantic import BaseModel, ConfigDict, model_validator

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
    ee_frame: str
    threshold_m: float
    statistic: str
    position_error_percentiles_m: dict[str, float]
    orientation_error_percentiles_deg: Optional[dict[str, float]] = None
    n_targets: int
    n_measured: int
    convergence_rate: float
    physics_dt: float
    gravity_z: Optional[float] = None
    # PD gains the threshold was measured under (normally from
    # controller_gains_probe). Optional so configs written before this field
    # existed still validate.
    arm_stiffness: Optional[float] = None
    arm_damping: Optional[float] = None
    target_orientation_rpy: Optional[tuple[float, float, float]] = None
    units: str = "meters"
    seed: int

    @model_validator(mode="after")
    def _threshold_matches_statistic(self) -> "SuccessThresholdProbeResult":
        pct = self.position_error_percentiles_m
        if self.statistic not in pct:
            raise ValueError(
                f"statistic={self.statistic!r} not in percentiles {sorted(pct)}"
            )
        if abs(self.threshold_m - pct[self.statistic]) > 1e-9:
            raise ValueError(
                f"threshold_m={self.threshold_m} disagrees with "
                f"{self.statistic}={pct[self.statistic]}"
            )
        return self




class ControllerGainsProbeResult(BaseModel):
    kp: float
    kd: float
    default_kp: float
    default_kd: float
    n_candidates: int
    n_feasible: int
    steady_state_error_rad: float
    settling_steps: int
    seed: int


class RobotConfig(BaseModel):
    name: str


class ProbeResults(BaseModel):
    workspace: Optional[WorkspaceProbeResult] = None
    joint_limits: Optional[JointLimitsProbeResult] = None
    success_threshold: Optional[SuccessThresholdProbeResult] = None
    controller_gains: Optional[ControllerGainsProbeResult] = None
    model_config = ConfigDict(validate_assignment=True)

class DiscoveredConfig(BaseModel):
    robot: RobotConfig
    probes: ProbeResults

