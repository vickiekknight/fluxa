"""Joint-limits probe: discovers the self-collision-free region of config space.

The URDF joint limits define only the *mechanical* range of each joint
independently. They say nothing about which *combinations* of angles drive
links into each other. This probe samples configs inside the URDF box (Sobol,
for low-discrepancy coverage), labels each collision-free or self-colliding,
and returns the collision-free set as an empirical sampling distribution.

Self-collision is inherently multi-joint, so the output is NOT a tightened
per-joint box -- it is the validated safe set that downstream stages draw from.

Labeling backend (default): PhysX contacts inside Isaac Lab. The scene must be
built with self-collisions enabled, contact reporting on, a ContactSensor over
the robot bodies, gravity off, and a fixed base (see run_probe.py / run_skill.py).
The labeler is swappable: pass any callable (configs)->bool_mask to use a
different backend (e.g. CuRobo in the validation test).
"""
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch

# configs (B, J) -> bool mask (B,), True == self-colliding
CollisionLabeler = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class JointLimitsProbeResult:
    safe_configs: np.ndarray   # (M, J) collision-free configs, robot joint order
    labels: np.ndarray         # (N,) bool, True = self-colliding
    all_configs: np.ndarray    # (N, J) every sampled config (for diagnostics)
    joint_lower: np.ndarray    # (J,)
    joint_upper: np.ndarray    # (J,)
    n_sampled: int
    n_safe: int
    collision_rate: float
    seed: int
    runtime_seconds: float


def sample_configs_sobol(lo: torch.Tensor, hi: torch.Tensor, n: int,
                         seed: int) -> torch.Tensor:
    """Low-discrepancy Sobol sample of n configs in [lo, hi]. Returns (n, J).

    Sobol's equidistribution is best at powers of two; n is taken as given here
    (the orchestrator pads to a multiple of num_envs). Round n up to 2**k if you
    want the strict low-discrepancy guarantee.
    """
    j = lo.shape[0]
    engine = torch.quasirandom.SobolEngine(dimension=j, scramble=True, seed=seed)
    u = engine.draw(n).to(device=lo.device, dtype=lo.dtype)   # (n, J) in [0,1]
    return lo + u * (hi - lo)


class PhysxSelfCollisionLabeler:
    """Label configs as self-colliding via PhysX contacts in Isaac Lab.

    One config per parallel env per call. Reads net contact force per body after
    a short step; with no ground and gravity off, the only contacts are link-link
    self-collisions (PhysX auto-excludes joint-connected pairs).
    """
    def __init__(self, sim, scene, robot, contact_sensor_key="contact_forces",
                 force_threshold=1e-3, n_settle_steps=1):
        self.sim = sim
        self.scene = scene
        self.robot = robot
        self.sensor = scene[contact_sensor_key]
        self.force_threshold = force_threshold
        self.n_settle_steps = n_settle_steps
        self.dt = sim.get_physics_dt()

    def __call__(self, configs: torch.Tensor) -> torch.Tensor:
        assert configs.shape[0] == self.scene.num_envs, \
            "labeler expects exactly one config per env"
        zero_vel = torch.zeros_like(configs)
        self.robot.write_joint_state_to_sim(configs, zero_vel)
        self.robot.set_joint_position_target(configs)   # hold pose under the drive
        self.scene.write_data_to_sim()
        for _ in range(self.n_settle_steps):
            self.sim.step(render=False)
            self.scene.update(self.dt)
        net = self.sensor.data.net_forces_w            # (num_envs, num_bodies, 3)
        mag = torch.linalg.norm(net, dim=-1)           # (num_envs, num_bodies)
        return (mag > self.force_threshold).any(dim=-1)  # (num_envs,) bool


def joint_limits_probe(sim, scene, robot, n_samples: int, seed: int = 0,
                       labeler: Optional[CollisionLabeler] = None
                       ) -> JointLimitsProbeResult:
    """Probe the self-collision-free region via Sobol-sampled FK + collision label."""
    start = time.time()
    device = robot.device
    num_envs = scene.num_envs

    n_batches = (n_samples + num_envs - 1) // num_envs
    n_actual = n_batches * num_envs

    jl = robot.data.soft_joint_pos_limits[0]
    lo, hi = jl[:, 0].contiguous(), jl[:, 1].contiguous()

    configs_all = sample_configs_sobol(lo, hi, n_actual, seed)   # (n_actual, J)

    if labeler is None:
        labeler = PhysxSelfCollisionLabeler(sim, scene, robot)

    labels = torch.empty(n_actual, dtype=torch.bool, device=device)
    for b in range(n_batches):
        sl = slice(b * num_envs, (b + 1) * num_envs)
        labels[sl] = labeler(configs_all[sl]).to(device)

    labels_np = labels.cpu().numpy()
    configs_np = configs_all.cpu().numpy()
    safe = configs_np[~labels_np]

    return JointLimitsProbeResult(
        safe_configs=safe,
        labels=labels_np,
        all_configs=configs_np,
        joint_lower=lo.cpu().numpy(),
        joint_upper=hi.cpu().numpy(),
        n_sampled=n_actual,
        n_safe=int((~labels_np).sum()),
        collision_rate=float(labels_np.mean()),
        seed=seed,
        runtime_seconds=time.time() - start,
    ) 