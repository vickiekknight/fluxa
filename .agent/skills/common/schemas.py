from typing import Optional

from pydantic import BaseModel


class WorkspaceProbeResult(BaseModel):
    x: tuple[float, float]
    y: tuple[float, float]
    z: tuple[float, float]


class JointLimitsProbeResult(BaseModel):
    """
    Placeholder for future probe output.
    Replace fields once output format is finalized.
    """
    pass


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