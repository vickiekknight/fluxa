"""Task parser: natural-language task description -> TaskSpec.

Two backends:
  - **LLM (Gemini, structured output)** -- handles arbitrary phrasing and emits
    an OPEN `task_type`, so understanding isn't limited to the keywords this
    file happens to hardcode.
  - **Deterministic keyword matching** -- the fallback used when no API key is
    set, when the LLM call fails, or when a caller explicitly asks for it.

Design note: whether a parsed task is actually *supported* is deliberately NOT
decided here. Understanding and support are different things -- the LLM can
classify "stack the blocks" perfectly well, but workspace-exploration's probes
are reach-shaped (EE reachability, EE position error to a target point) and
have nothing to say about grasp feasibility. So the parser reports what the
user asked for, and the caller (run_skill.py) owns the capability gate. That
way an unsupported task produces a legible "parsed X, we support Y" message
rather than the parser refusing to understand the sentence at all.
"""
import os
import time
from typing import Literal, Optional

from pydantic import BaseModel, Field

# Support surfaces the pipeline understands. Constrained to an enum rather
# than a free string so the model normalizes synonyms at the source: run_skill
# clamps the workspace z-bound on `surface == "table"`, and a free-text field
# happily returns "desk" for the same physical situation, silently skipping
# the clamp.
SurfaceKind = Literal["table", "floor", "shelf", "other"]


# Robot registry. URDF path is unused for v1 (we use Isaac Lab's built-in
# FRANKA_PANDA_CFG asset config), but kept for forward compatibility.
ROBOT_REGISTRY = {
    "franka": {
        "urdf_path": None,  # use FRANKA_PANDA_CFG asset config in v1
        "ee_body_name": "panda_hand",
    },
    # add so100 here once you have the URDF set up
}

# What the *probes* in this skill can currently measure. The parser does not
# enforce this -- it's exported so callers can gate on it and say why.
SUPPORTED_TASK_TYPES = {"reach"}

# Matches the model reward-designer already uses (cfg/reach.yaml).
DEFAULT_LLM_MODEL = "gemini-2.5-flash-lite"

# Retry before falling back. Gemini returns transient 503s under load, and
# falling straight back to keyword parsing is expensive: on a description the
# keywords can't classify, the fallback yields task_type="unknown", which the
# caller's capability gate rejects -- so one blip kills a whole pipeline run.
# Retrying every exception (rather than only the ones that look transient)
# keeps this independent of google-genai's exception taxonomy; the cost of
# retrying a permanent error is a few seconds before the same fallback.
LLM_MAX_ATTEMPTS = 3
LLM_RETRY_DELAY_S = 1.5


class TaskSpec(BaseModel):
    """Parsed task description. Pydantic (not a dataclass) so it can be used
    directly as the LLM's structured-output schema and validated on the way in.

    Note for callers: `helpers.io.save_json` only unwraps dataclasses, so write
    this with `save_json(spec.model_dump(), ...)`, matching how DiscoveredConfig
    is already persisted.
    """
    task_type: str
    robot_name: str
    robot_urdf_path: Optional[str] = None
    ee_body_name: str
    objects: list[str] = Field(default_factory=list)
    constraints: dict[str, str] = Field(default_factory=dict)
    # Free-text observations. Kept out of `constraints` on purpose: that dict is
    # what run_skill gates on, and a note is not something the pipeline honors.
    notes: list[str] = Field(default_factory=list)
    raw_description: str = ""
    parsed_by: str = "deterministic"   # provenance: "gemini" | "deterministic"

    def to_dict(self):
        return self.model_dump()


class _LLMTaskParse(BaseModel):
    """The narrow schema the LLM is asked to fill in.

    Deliberately smaller than TaskSpec: `ee_body_name` and `robot_urdf_path`
    are looked up from ROBOT_REGISTRY by code afterward, because those are
    facts about the robot asset, not about the sentence -- an LLM asked for a
    body name will happily invent a plausible-sounding one.

    `surface` is a typed optional field rather than an open constraints dict
    for two reasons: open-ended string maps are the shakiest part of structured
    output support, and it's the only constraint any consumer actually reads
    today (run_skill.py clamps the workspace z-bound on `surface == "table"`).
    Extra observations go in `notes`, which nothing gates on.
    """
    task_type: str = Field(
        description="Task family, lowercase, e.g. 'reach', 'lift', 'push', "
                    "'stack', 'pick_place'. Use the closest single word."
    )
    robot_name: Optional[str] = Field(
        default=None,
        description="Robot named in the description, lowercase (e.g. 'franka'). "
                    "Null if the description does not name one."
    )
    objects: list[str] = Field(
        default_factory=list,
        description="Objects the task manipulates or targets, if any."
    )
    surface: Optional[SurfaceKind] = Field(
        default=None,
        description="Support surface the task happens on, normalized to one of "
                    "the allowed values. Map desk/bench/workbench/counter/"
                    "tabletop to 'table'. Null if no surface is mentioned."
    )
    notes: list[str] = Field(
        default_factory=list,
        description="Other constraints or details worth recording."
    )


def _resolve_robot(robot_name: Optional[str], description: str) -> str:
    """Map an LLM-proposed (or keyword-matched) robot name onto the registry.

    Falls back to the sole registered robot when the description doesn't name
    one -- with only one robot registered, refusing to proceed would just be
    pedantry, and 'reach targets on a table' failing for not saying 'franka'
    is exactly the papercut this parser is meant to remove.
    """
    if robot_name and robot_name.lower() in ROBOT_REGISTRY:
        return robot_name.lower()

    if len(ROBOT_REGISTRY) == 1:
        only = next(iter(ROBOT_REGISTRY))
        if robot_name:
            print(f"[task_parser] robot {robot_name!r} is not in the registry; "
                  f"falling back to the only registered robot: {only!r}")
        else:
            print(f"[task_parser] no robot named in the description; "
                  f"defaulting to the only registered robot: {only!r}")
        return only

    raise ValueError(
        f"Could not identify a registered robot in: {description!r}. "
        f"Got {robot_name!r}. Supported robots: {list(ROBOT_REGISTRY)}"
    )


def _build_spec(task_type: str, robot_name: str, objects: list,
                constraints: dict, description: str, parsed_by: str,
                notes: Optional[list] = None) -> TaskSpec:
    info = ROBOT_REGISTRY[robot_name]
    return TaskSpec(
        task_type=task_type,
        robot_name=robot_name,
        robot_urdf_path=info["urdf_path"],
        ee_body_name=info["ee_body_name"],
        objects=objects,
        constraints=constraints,
        notes=notes or [],
        raw_description=description,
        parsed_by=parsed_by,
    )


def _parse_deterministic(description: str) -> TaskSpec:
    """Keyword fallback. Classifies what it can and reports `task_type`
    verbatim -- including 'unknown' -- rather than raising, so the caller's
    capability gate is the single place that decides what's supported."""
    desc = description.lower()

    if "reach" in desc:
        task_type = "reach"
    # "pick up", not bare "pick" -- "pick a point in the workspace" is a reach.
    elif "lift" in desc or "pick up" in desc:
        task_type = "lift"
    elif "push" in desc or "slide" in desc:
        task_type = "push"
    elif "stack" in desc:
        task_type = "stack"
    else:
        task_type = "unknown"

    if "franka" in desc:
        robot_name = _resolve_robot("franka", description)
    elif "so100" in desc or "so-100" in desc:
        robot_name = _resolve_robot("so100", description)
    else:
        robot_name = _resolve_robot(None, description)

    constraints = {}
    if "table" in desc:
        constraints["surface"] = "table"

    return _build_spec(task_type, robot_name, [], constraints,
                       description, "deterministic")


def _parse_with_llm(description: str, model_name: str) -> TaskSpec:
    """Parse via Gemini structured output. Raises on any failure; the caller
    decides whether to fall back."""
    from google import genai

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")

    client = genai.Client(api_key=api_key)
    prompt = (
        "Extract a structured robotics task specification from the "
        "description below.\n\n"
        "Report what the user actually asked for. Do not restrict yourself to "
        "any particular task type, and do not substitute a task you think is "
        "more feasible -- a downstream capability check decides what is "
        "supported.\n\n"
        f"Description: {description}"
    )

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": _LLMTaskParse,
        },
    )

    parsed = response.parsed
    if parsed is None:
        raise ValueError(f"LLM returned no parseable output: {response.text!r}")

    robot_name = _resolve_robot(parsed.robot_name, description)

    constraints = {}
    if parsed.surface:
        constraints["surface"] = parsed.surface

    return _build_spec(
        task_type=parsed.task_type.lower().strip(),
        robot_name=robot_name,
        objects=parsed.objects,
        constraints=constraints,
        description=description,
        parsed_by="gemini",
        notes=parsed.notes,
    )


def parse_task_description(description: str, *, use_llm: bool = True,
                           model_name: str = DEFAULT_LLM_MODEL) -> TaskSpec:
    """Parse a natural-language task description into a TaskSpec.

    Tries Gemini first (open task_type, arbitrary phrasing), falls back to
    keyword matching if the LLM is unavailable or fails. Never raises on an
    unsupported task type -- see this module's docstring for why.

    Args:
        use_llm: set False to force the deterministic backend (offline runs,
            reproducible tests).
        model_name: Gemini model to parse with.

    Raises:
        ValueError: if no registered robot can be resolved (only possible once
            ROBOT_REGISTRY holds more than one robot).
    """
    if use_llm and os.environ.get("GEMINI_API_KEY"):
        for attempt in range(1, LLM_MAX_ATTEMPTS + 1):
            try:
                return _parse_with_llm(description, model_name)
            except Exception as e:
                detail = f"{type(e).__name__}: {e}"
                if attempt < LLM_MAX_ATTEMPTS:
                    print(f"[task_parser] LLM parse attempt "
                          f"{attempt}/{LLM_MAX_ATTEMPTS} failed ({detail}); "
                          f"retrying in {LLM_RETRY_DELAY_S}s")
                    time.sleep(LLM_RETRY_DELAY_S)
                else:
                    print(f"[task_parser] LLM parse failed after "
                          f"{LLM_MAX_ATTEMPTS} attempts ({detail}); "
                          f"falling back to keyword parsing.")
    elif use_llm:
        print("[task_parser] GEMINI_API_KEY not set; using keyword parsing.")

    return _parse_deterministic(description)
