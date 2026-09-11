"""Top-level skill orchestrator: NL description -> discovered_config.json.

Usage:
    python scripts/run_skill.py "train the franka to reach random targets on a table"

This script runs **no simulation of its own**. It parses the description,
decides whether the task is something this skill can characterize, and then
sequences `run_probe.py` invocations as subprocesses.

Why subprocesses rather than one process: gravity is fixed when the
SimulationContext is constructed, and the probes disagree about it --
joint_limits_probe's PhysX contact labeling is only valid with gravity OFF
(see run_probe.py's setup), while success_threshold_probe and
controller_gains_probe measure settling under gravity and need it ON. One
process gets one gravity setting, so the passes have to be separate processes.
Each pass merges only its own section into discovered_config.json, so they
compose in any order without clobbering each other.

Keeping this script sim-free also means it must not hold a GPU while spawning
those passes (this repo already hits that constraint in reward-designer, which
stops the streaming Isaac Sim instance before running headless evaluations),
and that an unsupported task costs a fraction of a second instead of a full
Kit boot.
"""
import argparse
import os
import subprocess
import sys
import traceback

_SKILL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SKILL_ROOT)

from parser.task_parser import parse_task_description, SUPPORTED_TASK_TYPES
from helpers.io import save_json, load_json

_RUN_PROBE = os.path.join(_SKILL_ROOT, "scripts", "run_probe.py")
_TASK_SPEC_PATH = os.path.join(_SKILL_ROOT, "outputs", "task_spec.json")

# Exit codes. UNSUPPORTED is distinct from ERROR so a wrapper script can tell
# "understood the task, can't characterize it" from "something broke" -- both
# are non-zero because neither produced a complete discovered_config.json.
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_UNSUPPORTED = 2


def _build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("description", type=str,
                   help="Natural-language task description.")
    p.add_argument("--num_envs", type=int, default=1000)
    p.add_argument("--n_samples", type=int, default=2000)
    p.add_argument("--n_targets", type=int, default=1000,
                   help="Targets for the success-threshold probe.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-validation", action="store_true",
                   help="Skip FK validation (CuRobo check) in the probe passes.")
    p.add_argument("--n_validate", type=int, default=50)
    p.add_argument("--no-llm-parser", action="store_true",
                   help="Force the deterministic keyword parser instead of the "
                        "Gemini-backed one (offline runs, reproducibility).")
    p.add_argument("--skip-success-threshold", action="store_true",
                   help="Skip the gravity-on success-threshold pass.")
    p.add_argument("--skip-controller-gains", action="store_true",
                   help="Skip the gravity-on controller-gains pass.")
    p.add_argument("--python", type=str, default=sys.executable,
                   help="Interpreter used for the probe subprocesses. Defaults "
                        "to this one, which inherits Isaac Sim's environment "
                        "when launched via python.sh.")
    return p


def _report_unsupported(task_spec, user_description: str):
    """Explain a task we understood but can't characterize.

    This is an expected outcome, not a failure -- the parser deliberately
    classifies open task types so the refusal can name what it understood.
    Printed rather than raised so it doesn't read like a crash.
    """
    bar = "=" * 66
    print(f"\n{bar}")
    print("Task understood, but not supported by workspace-exploration")
    print(bar)
    print(f"  description : {user_description!r}")
    print(f"  parsed as   : task_type={task_spec.task_type!r} "
          f"(via {task_spec.parsed_by})")
    if task_spec.objects:
        print(f"  objects     : {task_spec.objects}")
    print(f"  supported   : {sorted(SUPPORTED_TASK_TYPES)}")
    print()
    print("  This skill's probes are reach-shaped -- workspace_probe measures")
    print("  EE reachability, success_threshold_probe measures EE position error")
    print("  to a target point. Neither says anything about grasping, contact,")
    print("  or object placement, so a family like 'lift' or 'stack' needs new")
    print("  probes before it can be characterized here.")
    print()
    print("  No probes ran; discovered_config.json was not modified.")
    print("  (outputs/task_spec.json was still written, so you can see the")
    print("   full parse.)")
    print(f"{bar}\n")


def _run_pass(label: str, argv: list, python_exe: str) -> int:
    """Run one probe pass as a subprocess. Returns its exit code."""
    bar = "-" * 66
    print(f"\n{bar}")
    print(f">>> {label}")
    print(f"{bar}")
    cmd = [python_exe, _RUN_PROBE] + argv
    print(f"    {' '.join(cmd)}\n", flush=True)
    # Inherit stdout/stderr so probe output streams live rather than being
    # buffered up and replayed after the pass finishes.
    result = subprocess.run(cmd, cwd=_SKILL_ROOT)
    if result.returncode != 0:
        print(f"\n!!! {label} exited {result.returncode}")
    return result.returncode


def main(args) -> int:
    """Run the skill. Returns a process exit code."""
    # === Stage 1: Parse ===
    task_spec = parse_task_description(args.description,
                                       use_llm=not args.no_llm_parser)
    # .model_dump(): TaskSpec is Pydantic, and save_json only unwraps
    # dataclasses -- passing the object directly would stringify it.
    save_json(task_spec.model_dump(), "outputs/task_spec.json")
    print(f"Parsed: {task_spec.task_type} on {task_spec.robot_name} "
          f"(via {task_spec.parsed_by})")
    if task_spec.objects:
        print(f"Objects: {task_spec.objects}")
    if task_spec.constraints:
        print(f"Constraints: {task_spec.constraints}")
    if task_spec.notes:
        print(f"Notes: {task_spec.notes}")

    # === Stage 2: Capability gate ===
    # Parsing understands open task types; this is the single place that
    # decides what the probes can actually measure. Checked before any
    # subprocess spawns, so an unsupported task costs no sim time at all.
    if task_spec.task_type not in SUPPORTED_TASK_TYPES:
        _report_unsupported(task_spec, args.description)
        return EXIT_UNSUPPORTED

    # === Stage 3: Sequence the probe passes ===
    common = [
        "--num_envs", str(args.num_envs),
        "--n_samples", str(args.n_samples),
        "--seed", str(args.seed),
        "--n_validate", str(args.n_validate),
        "--ee_body_name", task_spec.ee_body_name,
        "--task-spec", _TASK_SPEC_PATH,
    ]

    passes = [
        # Gravity OFF: workspace + joint-limits. --write-config makes this pass
        # record both sections (run_skill.py used to do that write itself).
        ("workspace + joint limits (gravity off)",
         common + ["--write-config"] +
         ([] if args.skip_validation else ["--validate-fk"])),
    ]
    # Controller gains BEFORE success threshold: the gains characterize the
    # low-level PD controller that success_threshold_probe then measures
    # settling error through, so the dependency runs first. (Note: as of now
    # success_threshold_probe still uses the hardcoded gains in run_probe.py's
    # _make_franka_cfg rather than reading the discovered ones -- this ordering
    # is what makes that retrofit possible, not a substitute for it.)
    if not args.skip_controller_gains:
        passes.append(
            ("controller gains (gravity on)", common + ["--controller-gains"])
        )
    if not args.skip_success_threshold:
        passes.append(
            ("success threshold (gravity on)",
             common + ["--success-threshold", "--n_targets", str(args.n_targets)])
        )

    # Number the passes after assembling them -- hardcoded "Pass 1/3" labels
    # lied whenever a --skip flag dropped one.
    total = len(passes)
    for i, (label, argv) in enumerate(passes, start=1):
        rc = _run_pass(f"Pass {i}/{total}: {label}", argv, args.python)
        if rc != 0:
            print(f"\nStopping: {label} failed. discovered_config.json may be "
                  f"partially populated -- inspect it before relying on it.")
            return EXIT_ERROR

    # === Stage 4: Report ===
    bar = "=" * 66
    print(f"\n{bar}")
    print("Discovered config")
    print(bar)
    discovered = load_json("outputs/discovered_config.json") or {}
    probes = discovered.get("probes", {})
    for name in ("workspace", "joint_limits", "success_threshold",
                 "controller_gains"):
        section = probes.get(name)
        print(f"  {name:<18} {'present' if section else 'MISSING'}")
    print()
    print("Outputs written:")
    print("  outputs/task_spec.json")
    print("  outputs/discovered_config.json")
    print("  outputs/diagnostics/")
    print(bar)
    return EXIT_OK


if __name__ == "__main__":
    _args = _build_parser().parse_args()
    try:
        sys.exit(main(_args))
    except Exception:
        traceback.print_exc()
        sys.exit(EXIT_ERROR)
