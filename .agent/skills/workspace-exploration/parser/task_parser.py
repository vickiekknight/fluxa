"""Deterministic parser: natural-language task description → TaskSpec.

v1 supports only reach tasks on Franka. Other inputs raise explicit errors.
"""
from dataclasses import dataclass, asdict, field


# Robot registry. URDF path is unused for v1 (we use Isaac Lab's built-in
# FRANKA_PANDA_CFG asset config), but kept for forward compatibility.
ROBOT_REGISTRY = {
    "franka": {
        "urdf_path": None,  # use FRANKA_PANDA_CFG asset config in v1
        "ee_body_name": "panda_hand",
    },
    # add so100 here once you have the URDF set up
}

SUPPORTED_TASK_TYPES = {"reach"}


@dataclass
class TaskSpec:
    task_type: str
    robot_name: str
    robot_urdf_path: str | None
    ee_body_name: str
    objects: list = field(default_factory=list)
    constraints: dict = field(default_factory=dict)
    raw_description: str = ""

    def to_dict(self):
        return asdict(self)


def parse_task_description(description: str) -> TaskSpec:
    """Parse a natural-language task description into a TaskSpec.
    
    Raises:
        NotImplementedError: if task type is not supported in v1.
        ValueError: if robot name cannot be identified or is not in registry.
    """
    desc = description.lower()

    # --- Task type ---
    if "reach" in desc:
        task_type = "reach"
    else:
        raise NotImplementedError(
            f"v1 only supports reach tasks. Got: {description!r}"
        )

    # --- Robot ---
    if "franka" in desc:
        robot_name = "franka"
    elif "so100" in desc or "so-100" in desc:
        robot_name = "so100"
    else:
        raise ValueError(
            f"Could not identify robot in description: {description!r}. "
            f"Supported robots: {list(ROBOT_REGISTRY.keys())}"
        )

    if robot_name not in ROBOT_REGISTRY:
        raise ValueError(
            f"Robot '{robot_name}' is not in registry. "
            f"Supported: {list(ROBOT_REGISTRY.keys())}"
        )
    info = ROBOT_REGISTRY[robot_name]

    # --- Constraints ---
    constraints = {}
    if "table" in desc:
        constraints["surface"] = "table"

    return TaskSpec(
        task_type=task_type,
        robot_name=robot_name,
        robot_urdf_path=info["urdf_path"],
        ee_body_name=info["ee_body_name"],
        objects=[],
        constraints=constraints,
        raw_description=description,
    )