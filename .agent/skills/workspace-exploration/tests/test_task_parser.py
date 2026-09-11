"""Task-parser smoke test. Exercises both backends without Isaac Sim.

Run inside the container:
    /isaac-sim/python.sh /isaac-sim/fluxa/.agent/skills/workspace-exploration/tests/test_task_parser.py

The deterministic half needs no API key and no google-genai; the Gemini half
is skipped (with a printed reason) if GEMINI_API_KEY is unset or the import
fails, which is the same fallback path run_skill.py takes.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parser.task_parser import (
    ROBOT_REGISTRY,
    SUPPORTED_TASK_TYPES,
    parse_task_description,
)

DESCRIPTIONS = [
    "train the franka to reach targets on a table",
    "reach targets on a table",            # no robot named -- registry fallback
    "move the panda arm to random points above the desk",  # no 'reach' keyword
    "stack the red block on the blue block",               # unsupported family
]


def _show(spec):
    print(f"    task_type   : {spec.task_type}"
          f"{'' if spec.task_type in SUPPORTED_TASK_TYPES else '   (not supported by this skill)'}")
    print(f"    robot       : {spec.robot_name}  (ee={spec.ee_body_name})")
    if spec.objects:
        print(f"    objects     : {spec.objects}")
    if spec.constraints:
        print(f"    constraints : {spec.constraints}")
    if spec.notes:
        print(f"    notes       : {spec.notes}")
    print(f"    parsed_by   : {spec.parsed_by}")


def main():
    print(f"registry: {list(ROBOT_REGISTRY)}   supported: {sorted(SUPPORTED_TASK_TYPES)}")

    print("\n=== deterministic backend (use_llm=False) ===")
    for d in DESCRIPTIONS:
        print(f"\n  {d!r}")
        spec = parse_task_description(d, use_llm=False)
        _show(spec)
        assert spec.robot_name in ROBOT_REGISTRY
        assert spec.raw_description == d
        assert spec.parsed_by == "deterministic"

    print("\n=== gemini backend (use_llm=True) ===")
    if not os.environ.get("GEMINI_API_KEY"):
        print("  GEMINI_API_KEY not set -- skipping.")
        return
    for d in DESCRIPTIONS:
        print(f"\n  {d!r}")
        spec = parse_task_description(d)
        _show(spec)
        assert spec.robot_name in ROBOT_REGISTRY
        assert spec.raw_description == d

    print("\nDone. Any 'parsed_by: deterministic' above means the LLM path "
          "fell back -- the reason is printed by the parser itself.")


if __name__ == "__main__":
    main()
