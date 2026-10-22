#!/usr/bin/env python3
"""
1_eureka.py — Stage 1: Iterative reward generation using Eureka method.

Each candidate evaluation launches a fresh headless subprocess inside the Docker container. 
Metrics are passed back via a JSON file on the shared filesystem.
"""

import ast
import os
import sys
import json
import yaml
import re
import shutil
import textwrap
import subprocess
import time
from pathlib import Path

from google import genai
from google.genai import types
import wandb

_SKILLS_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, _SKILLS_DIR)
from common.env import load_env_file


# A candidate that never ran gets this score, so it can never win selection and
# is distinguishable from one that ran and legitimately scored 0.0. Matches the
# original's DUMMY_FAILURE.
DUMMY_FAILURE = -10000.0


def candidate_executed(result):
    """True only if eval_headless actually trained and scored this candidate.

    Anything else -- a reward that would not build, a crash, a timeout, a
    metrics file that never appeared -- is a failure. Without this check a
    candidate that died during env creation is indistinguishable from one that
    trained and scored 0.0, which reported execute_rate=1.0 for an iteration in
    which nothing executed at all.
    """
    return result is not None and result["metrics"].get("status") == "success"


class EurekaManager:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)

        # Gemini client. The key lives in the repo-root .env (see README);
        # nothing host-side loaded that file before, so a fresh shell left this
        # None and the failure surfaced later as an opaque API error.
        load_env_file()
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set and no .env above this script "
                "provided one. Add it to the repo-root .env (see README) "
                "or export it before running."
            )
        self.client = genai.Client(api_key=api_key)
        self.model_name = self.cfg['eureka'].get('model', 'gemini-2.5-flash-lite')
        self.run_index = 0
        self.seed = self.cfg['eureka'].get('seed', 42)
        # Eureka samples K i.i.d. candidates per iteration and relies on them
        # differing; at the API default they come back near-identical.
        self.temperature = self.cfg['eureka'].get('temperature', 1.0)

        self.designer_root = Path(self.cfg['designer_root']).expanduser()

        # W&B
        wandb.init(
            project="Fluxa-Reward-Designer",
            name=f"eureka-{self.cfg['task_name']}",
            config=self.cfg,
        )

        # Paths
        self.candidates_dir = self.designer_root / "outputs" / "candidates"
        self.candidates_dir.mkdir(parents=True, exist_ok=True)

        # Docker config
        self.docker_container = self.cfg.get('docker', {}).get('container', 'fluxa-isaacsim')
        self.docker_python = self.cfg.get('docker', {}).get('python', '/isaac-sim/python.sh')
        # Path to eval_headless.py INSIDE the container
        self.eval_script = self.cfg.get('docker', {}).get(
            'eval_script', '/isaac-sim/fluxa-ws/reward-designer/scripts/eval_headless.py'
        )
        # Shared directory visible to both host and container
        self.shared_dir = self.cfg.get('docker', {}).get(
            'shared_dir', '/isaac-sim/fluxa-ws/reward-designer'
        )

        # This script runs on either side of the container boundary. On the host
        # it reaches the simulator through `docker exec`; inside the container it
        # calls python.sh directly, because there is no `docker` binary in there.
        # In-container is the common case: requirements.txt installs google-genai
        # into Isaac Sim's Python env, not the host's, and 2_rapp.py already runs
        # that way.
        self.in_container = Path(self.docker_python).exists()
        print(f"[eureka] launching evals "
              f"{'directly (inside container)' if self.in_container else 'via docker exec (host)'}")

    # --- Prompt Building ---

    def _read(self, relative_path):
        """Read a file relative to the skill root."""
        with open(self.designer_root / relative_path, 'r') as f:
            return f.read()

    def load_env_context(self):
        """The environment source the LLM reasons over (Eureka 3.1).

        The paper feeds the raw environment source with the reward code removed:
        any reward is a function of the environment's state and action variables,
        so the only requirement on the source is that it exposes them. For a
        manager-based Isaac Lab env that means ObservationsCfg, CommandsCfg,
        ActionsCfg and the scene; RewardsCfg and CurriculumCfg are stripped,
        since handing over the existing hand-written reward just invites copying
        it. This replaces hand-maintained prose about which variables exist --
        which is how the model ended up inventing quat_conjugate and guessing
        the scene entity was named "franka" rather than "robot".

        Returns "" when the source is unreachable (it lives inside the
        container), so a host-side run degrades rather than failing.
        """
        settings = self.cfg.get('env_context') or {}
        source_path = settings.get('source_file')
        if not source_path:
            return ""

        path = Path(source_path)
        if not path.is_file():
            print(f"[eureka] env context unavailable at {path}; "
                  f"prompting without environment source")
            return ""

        source = path.read_text()
        strip = set(settings.get('strip_classes') or [])
        if not strip:
            return source

        drop_lines = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.ClassDef) and node.name in strip:
                start = node.lineno
                for decorator in node.decorator_list:
                    start = min(start, decorator.lineno)
                drop_lines.update(range(start, node.end_lineno + 1))

        kept = [line for n, line in enumerate(source.splitlines(), 1)
                if n not in drop_lines]
        print(f"[eureka] env context: {path.name}, {len(kept)} lines "
              f"(stripped {', '.join(sorted(strip))})")
        return "\n".join(kept)

    def build_system_prompt(self):
        """System turn: role, reward signature, generic formatting tips.

        code_output_tip is appended here AND to every feedback turn, as in the
        original -- the rules are what keep the model emitting a reward_dict
        rather than prose.
        """
        signature = self._read(self.cfg['reward_signature_file'])
        system = self._read("prompts/inital_system.txt").format(
            task_reward_signature_string=signature)
        return system + "\n" + self._read("prompts/code_output.tip.txt")

    def build_initial_user_prompt(self, task_name):
        """User turn: the environment source plus the task description.

        Names the SCENE ENTITY, not the task key. Saying "the robot is
        'franka-reach'" made the model write
        SceneEntityCfg("franka-reach", ...), which Isaac Lab rejects with "the
        scene entity 'franka-reach' does not exist" -- a contradiction between
        this sentence and formatting rule (4), and the more specific-sounding
        sentence won.
        """
        robot_cfg = self.cfg['robots'][task_name]
        asset_name = robot_cfg.get('asset_name', 'robot')
        ee_body_name = robot_cfg['ee_body_name']
        description = (
            f"{self.cfg['task_description'].strip()}\n"
            f"The robot's scene entity name is '{asset_name}' and its "
            f"end-effector body is named '{ee_body_name}'. Refer to the "
            f"end-effector as "
            f"SceneEntityCfg('{asset_name}', body_names=['{ee_body_name}'])."
        )
        return self._read("prompts/initial_user.txt").format(
            task_obs_code_string=self.env_context or "(environment source unavailable)",
            task_description=description,
        )

    def build_feedback_prompt(self, metrics, stride):
        """Reward reflection turn (Eureka 3.3).

        policy_feedback frames the numbers, build_reward_reflection supplies
        them, code_feedback asks for a per-component analysis, and
        code_output_tip repeats the formatting rules.
        """
        return "\n".join([
            self._read("prompts/policy_feedback.txt").format(epoch_freq=stride),
            self.build_reward_reflection(metrics),
            self._read("prompts/code_feedback.txt"),
            self._read("prompts/code_output.tip.txt"),
        ])

    @staticmethod
    def _subsample_stride(values, n_points=10):
        """Stride used to thin a series -- the original's epoch_freq."""
        return max(len(values) // n_points, 1) if values else 1

    @staticmethod
    def _subsample(values, n_points=10):
        """Thin a per-iteration series down to ~n_points for the prompt.

        The original computes epoch_freq = max(train_iterations // 10, 1) and
        slices [::epoch_freq]. Without this a 300-iteration run would put 300
        floats per reward term into the prompt.
        """
        if not values:
            return []
        stride = EurekaManager._subsample_stride(values, n_points)
        thinned = values[::stride]
        # Always keep the final value -- the LLM reasons about where a term
        # ended up, and striding can drop the last sample.
        if thinned[-1] != values[-1]:
            thinned.append(values[-1])
        return thinned

    @staticmethod
    def _error_line(metrics):
        """One-line summary of why a candidate failed, for the console.

        Takes the last UNINDENTED line of the traceback, not simply the last
        line: a multi-line exception message (Isaac Lab's "Missing values
        detected ..." lists each field on its own indented line) otherwise
        reported as "- rewards.deviation_penalty.weight", which says nothing
        about what went wrong.
        """
        traceback_text = (metrics.get("traceback") or "").rstrip()
        if traceback_text:
            lines = [ln for ln in traceback_text.splitlines() if ln.strip()]
            for line in reversed(lines):
                if not line[:1].isspace():
                    return line.strip()
            if lines:
                return lines[-1].strip()
        return metrics.get("error", "")

    def _build_all_failed_feedback(self, candidate_results):
        """Feedback for an iteration in which nothing executed.

        Names the actual errors instead of a generic "please write valid code".
        Without them the LLM has no way to know what it got wrong and simply
        repeats the same mistake next iteration.

        Matches the original, which builds per-candidate feedback but only ever
        sends contents[best_sample_idx] -- and since failures score
        DUMMY_FAILURE, the best is a successful candidate whenever any succeeds.
        So execution errors reach the model exactly when the best candidate is
        itself a failure, which is this path.
        """
        errors = []
        for result in candidate_results:
            if result is None:
                continue
            # llm_error means no code was ever produced (a 503, a dropped
            # connection). Telling the model to "fix the bug" in code it never
            # wrote is noise, so those are left out.
            if result["metrics"].get("status") == "llm_error":
                continue
            line = self._error_line(result["metrics"])
            if line and line not in errors:
                errors.append(line)

        traceback_msg = ("\n" + "\n".join(f"  - {e}" for e in errors)) if errors \
            else "the reward code could not be executed"
        return "\n".join([
            self._read("prompts/execution_error_feedback.txt").format(
                traceback_msg=traceback_msg),
            self._read("prompts/code_output.tip.txt"),
        ])

    def build_reward_reflection(self, metrics):
        """Per-component reward reflection (Eureka 3.3).

        Mirrors the original's construct_run_log feedback: every reward
        component's trajectory with Max/Mean/Min, plus the task score. The
        original supplies NUMBERS and lets the model draw the conclusions --
        3.3's argument is that fine-grained per-component data is what enables
        targeted reward editing, and it ships no automated diagnosis at all.

        Two things the previous version added are gone from the default output:

        - "The reward function needs improvement." was unconditional, so a
          candidate at 0.93 success with position error converging to 0.025m was
          still told it had failed.
        - mean_reward was printed as a co-equal headline next to success_rate.
          Over a 5-iteration run the summed positive weights went 6.0 -> 11.0
          and mean_reward 29.9 -> 56.1 while task success fell 0.932 -> 0.781:
          the model optimised the number it was shown. The original reports the
          reward only when no task score exists, so it is omitted here.

        Automated diagnostics live behind eureka.reflection_diagnostics and are
        OFF by default; see _reflection_diagnostics for why.
        """
        lines = []

        # Per-component reward trajectories -- the substance of 3.3.
        reward_components = metrics.get("reward_components", {})
        if reward_components:
            for term_name in sorted(reward_components.keys()):
                values = reward_components[term_name]
                if not values:
                    continue
                trajectory = [f"{v:.4f}" for v in self._subsample(values)]
                lines.append(
                    f"{term_name}: {trajectory}, "
                    f"Max: {max(values):.4f}, Mean: {sum(values) / len(values):.4f}, "
                    f"Min: {min(values):.4f}"
                )
        else:
            lines.append("No per-component reward data available.")

        # The task metric, reported the same way as a reward component.
        position_errors = metrics.get("position_error", [])
        if position_errors:
            trajectory = [f"{v:.4f}" for v in self._subsample(position_errors)]
            lines.append(
                f"position_error: {trajectory}, "
                f"Max: {max(position_errors):.4f}, "
                f"Mean: {sum(position_errors) / len(position_errors):.4f}, "
                f"Min: {min(position_errors):.4f}"
            )

        # The fitness function selection actually ranks on. The original reports
        # consecutive_successes under the name "task score".
        lines.append(f"task score: {metrics.get('success_rate', 0.0):.4f}")

        if self.cfg['eureka'].get('reflection_diagnostics', False):
            lines.extend(self._reflection_diagnostics(metrics))

        return "\n".join(lines)

    def _reflection_diagnostics(self, metrics):
        """Automated failure-mode notes. OFF by default -- see below.

        The original Eureka has nothing like this: it hands over the raw
        trajectories and lets the model diagnose. Kept behind
        eureka.reflection_diagnostics so the two can be compared directly,
        because in a full 5x16 run these were a net negative.

        SATURATED is the clearest offender. It fired on 'reach_position_coarse'
        in three of five iterations and advised "tighten its threshold or
        increase its weight" -- but a coarse distance term plateaus near its
        maximum precisely BECAUSE the policy has solved the coarse objective.
        The model duly tightened the shaping temperature from 0.15 to 0.06, at
        which point exp(-d/T) at the 0.24m start distance falls from 0.20 to
        0.018 and is essentially flat. With no position gradient left, PPO
        maximised orientation instead (cheap for a 7-DOF wrist, reachable
        without travelling) and stopped moving: position error 0.42m, task
        success 0.0001.

        FLIPPED SIGN and PENALTIES OUTWEIGH REWARDS each caught real bugs
        earlier in development, so they are preserved rather than deleted.
        """
        lines = []
        reward_components = metrics.get("reward_components", {})

        lines.append("\nThe reward function needs improvement.")

        if metrics.get("success_rate", 0) == 0:
            lines.append("The success rate is 0 — the robot is not reaching targets at all.")

        # Detect common failure modes from the component trajectories.
        # 10x was the original threshold and it missed a live 2.5x imbalance that
        # was still enough to drive total reward negative, so this is tunable.
        dominating_ratio = self.cfg['eureka'].get('dominating_ratio', 3.0)

        for term_name, values in reward_components.items():
            if len(values) < 3:
                continue

            # Saturating: value barely changes in the last 70% of training
            late_values = values[len(values) // 3:]
            if len(late_values) > 1:
                late_range = max(late_values) - min(late_values)
                full_range = max(values) - min(values)
                if full_range > 0 and late_range / full_range < 0.1:
                    lines.append(
                        f"  Note: '{term_name}' appears SATURATED — it stopped changing "
                        f"early in training (final value: {values[-1]:.4f}). "
                        f"Consider tightening its threshold or increasing its weight."
                    )

            # Dead: value is essentially zero throughout
            if all(abs(v) < 1e-6 for v in values):
                lines.append(
                    f"  Note: '{term_name}' is DEAD — it was zero throughout training. "
                    f"The condition may be too hard to trigger. "
                    f"Consider relaxing its threshold or adding intermediate shaping."
                )
            else:
                # Not firing: it triggered at some point but has been zero for
                # the whole back half. A sparse bonus gated behind a threshold
                # the policy cannot yet reach contributes nothing and cannot
                # bootstrap -- both success bonuses in a live run looked like
                # this while carrying weight 5.0.
                # Mostly-zero with nothing at the end. A "fraction of samples
                # that are zero" test rather than "the whole back half", because
                # a single lucky spike late in training would otherwise mask a
                # bonus that fires essentially never.
                zero_fraction = sum(1 for v in values if abs(v) < 1e-9) / len(values)
                if zero_fraction >= 0.8 and abs(values[-1]) < 1e-9:
                    lines.append(
                        f"  Note: '{term_name}' is NOT FIRING — it was zero for "
                        f"{zero_fraction:.0%} of training and is zero at the end "
                        f"(peak was {max(abs(v) for v in values):.4f}). Its condition "
                        f"is probably gated behind a threshold the policy cannot "
                        f"reach yet, so it contributes nothing and cannot bootstrap. "
                        f"Relax the threshold or replace the bonus with dense shaping "
                        f"that rewards progress toward it."
                    )

            # Sign flipped: a term whose NAME says penalty but whose logged
            # contribution is positive. Isaac Lab logs the WEIGHTED value, so a
            # function returning a negative magnitude combined with a negative
            # weight shows up here as positive -- the double negative turns the
            # penalty into a reward. Seen live: an "action_rate_penalty" worth
            # +9.99/s taught the policy to thrash its joints, scoring 82.67 mean
            # reward while task success collapsed.
            # Only checked in this direction: a term named for a reward may
            # legitimately be negative (Isaac Lab's own position tracking is).
            if (any(word in term_name.lower()
                    for word in ("penalty", "penalize", "cost", "regular"))
                    and values[-1] > 1e-9):
                lines.append(
                    f"  Note: '{term_name}' has a FLIPPED SIGN — it is named as a "
                    f"penalty but its weighted contribution is POSITIVE "
                    f"({values[-1]:+.4f}), so it REWARDS the behaviour it should "
                    f"suppress. You have most likely returned a negative value AND "
                    f"given the term a negative weight. Return the positive "
                    f"magnitude from the function and keep the weight negative."
                )

            # Dominating: compared on FINAL values, not peaks. Peak-vs-peak let
            # a single early spike in an unrelated term hide a penalty that was
            # 7x everything else by the time training ended -- which is the
            # balance the policy is actually optimising against.
            term_final = abs(values[-1])
            other_finals = [
                abs(vals[-1]) for name, vals in reward_components.items()
                if name != term_name and vals
            ]
            if other_finals:
                max_other = max(other_finals)
                if max_other > 1e-12 and term_final / max_other > dominating_ratio:
                    lines.append(
                        f"  Note: '{term_name}' is DOMINATING — at the end of "
                        f"training its magnitude is {term_final / max_other:.1f}x the "
                        f"largest other term. Consider reducing its weight."
                    )

        # Overall balance. The per-term checks above all passed on a reward whose
        # total was falling monotonically: a -0.138 action-rate penalty against
        # +0.018 of position tracking is only 2.5x by peak magnitude, but it
        # means the best thing the policy can do is stop moving.
        finals = {name: vals[-1] for name, vals in reward_components.items() if vals}
        positive = sum(v for v in finals.values() if v > 0)
        negative = -sum(v for v in finals.values() if v < 0)
        if negative > positive and negative > 0:
            penalties = sorted(
                ((name, v) for name, v in finals.items() if v < 0),
                key=lambda kv: kv[1],
            )
            worst = penalties[0][0] if penalties else "the penalty terms"
            lines.append(
                f"\n  Note: PENALTIES OUTWEIGH REWARDS — at the end of training the "
                f"penalty terms total {negative:.4f} against {positive:.4f} of "
                f"positive reward, so the total reward DECREASES as training "
                f"progresses and the best available strategy is to stop moving. "
                f"'{worst}' is the largest penalty. Reduce the penalty weights, or "
                f"increase the weight of the terms that reward task progress, so "
                f"that making progress always beats doing nothing."
            )

        # A list, not a joined string: build_reward_reflection extends with it.
        return lines

    # --- Conversation state (Eureka 3.2, in-context reward mutation) ---

    @staticmethod
    def _turn(role, text):
        """One conversation turn. Gemini calls the assistant role 'model'."""
        return {"role": role, "parts": [{"text": text}]}

    def reset_conversation(self, task_name):
        """Start the conversation: the original's [system, initial_user].

        The system turn rides in GenerateContentConfig.system_instruction rather
        than in the list, so self.messages holds only the user/model exchange
        and is one shorter than the original's `messages` throughout.
        """
        self.system_prompt = self.build_system_prompt()
        self.messages = [self._turn("user", self.build_initial_user_prompt(task_name))]

    def record_exchange(self, assistant_text, feedback_text):
        """Append on the first iteration, REPLACE on every one after.

        The original appends the best response and its feedback when
        len(messages) == 2, and thereafter overwrites messages[-2] and
        messages[-1] (`assert len(messages) == 4`). So the model always sees
        exactly: the task and environment, the single best reward found so far,
        and the reflection on that reward -- never a growing transcript.
        Accumulating would blow the context window within a few iterations and
        dilute the signal the reflection is supposed to carry.

        Keeping the initial turn fixed is what makes this in-context mutation
        rather than a fresh sample with hints attached: the model is editing
        code it can actually see.
        """
        exchange = [
            self._turn("model", assistant_text),
            self._turn("user", feedback_text),
        ]
        if len(self.messages) == 1:
            self.messages += exchange
        else:
            assert len(self.messages) == 3, f"unexpected history: {len(self.messages)}"
            self.messages[-2:] = exchange

    def save_conversation(self):
        """Mirror the original's messages.json dump, for inspecting a run.

        One file per restart, so independent searches do not overwrite each
        other's transcript.
        """
        suffix = f"_run{self.run_index}" if getattr(self, "run_index", 0) else ""
        path = self.designer_root / "outputs" / f"messages{suffix}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [{"role": "system", "parts": [{"text": self.system_prompt}]}]
        payload.extend(self.messages)
        with open(path, 'w') as f:
            json.dump(payload, f, indent=2)

    # --- LLM sampling ---

    @staticmethod
    def _short_error(exc):
        """Compact one-line rendering of an SDK exception."""
        code = getattr(exc, "code", None)
        message = getattr(exc, "message", None) or str(exc)
        message = " ".join(str(message).split())[:160]
        return f"{code} {message}" if code else f"{type(exc).__name__}: {message}"

    @staticmethod
    def _is_retryable(exc):
        """Whether another attempt could plausibly succeed.

        google-genai raises APIError subclasses carrying .code. 429 (rate limit)
        and any 5xx are transient -- a 503 "model is currently experiencing high
        demand" is the canonical case, and it took out a whole run because a
        single failure was treated as permanent. 400/401/403 are not transient:
        retrying a bad key or a malformed request only delays a clear message.
        Exceptions with no code are network-level (timeouts, dropped
        connections) and are worth retrying.
        """
        code = getattr(exc, "code", None)
        if code is None:
            return True
        return code == 429 or code >= 500

    def _generate_candidate(self, contents):
        """Query Gemini for one candidate. Returns (text, error_message).

        `contents` is the full conversation so far, so all K samples in an
        iteration are drawn i.i.d. from the same message checkpoint -- as in the
        original, which passes the same `messages` list with n=chunk_size.

        Eureka relies on sampling K candidates per iteration; without retries a
        transient endpoint failure silently shrinks K, and at K=2 it ends the
        run. The original retries each sample up to 3 times.
        """
        attempts = self.cfg['eureka'].get('llm_max_attempts', 4)
        backoff = self.cfg['eureka'].get('llm_backoff_seconds', 5)

        last_error = None
        for attempt in range(1, attempts + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=self.system_prompt,
                        temperature=self.temperature,
                    ),
                )
                if attempt > 1:
                    print(f"    succeeded on attempt {attempt}/{attempts}")
                return response.text, None

            except Exception as exc:
                last_error = exc
                if not self._is_retryable(exc):
                    return None, self._short_error(exc)
                if attempt < attempts:
                    delay = backoff * attempt  # linear backoff
                    print(f"    attempt {attempt}/{attempts} failed "
                          f"({self._short_error(exc)}); retrying in {delay}s")
                    time.sleep(delay)

        return None, (f"{attempts} attempts failed; last error: "
                      f"{self._short_error(last_error)}")

    # --- Code Extraction & Injection ---

    def _extract_code_block(self, raw_llm_output):
        """Pull the Python out of a model response.

        The original tries five patterns in order before falling back to the raw
        response; a single strict regex burns a candidate whenever the model uses
        a bare fence or omits the language tag.
        """
        patterns = [
            r"```python\s*\n(.*?)```",
            r"```py\s*\n(.*?)```",
            r"```\s*\n(.*?)```",
            r'"""(.*?)"""',
        ]
        for pattern in patterns:
            match = re.search(pattern, raw_llm_output, re.DOTALL)
            if match and match.group(1).strip():
                return match.group(1)

        # No fence at all: if it looks like code, take it as-is.
        if re.search(r"^\s*def\s+\w+", raw_llm_output, re.MULTILINE):
            return raw_llm_output
        return None

    def _validate_reward_code(self, tree):
        """Check the generated module defines a usable reward_dict.

        Returns an error string, or None when the code is acceptable. Every check
        here is one that would otherwise cost a full Isaac Sim boot to discover
        -- and the duplicate-key one would not be discovered at all, since it
        fails silently.
        """
        # The dict must be a module-level literal. Observed live: one candidate
        # wrapped its terms in `def franka_reach_reward(...) -> dict` building the
        # dict by item assignment, another named that factory `reward_dict`
        # itself. Neither leaves a dictionary for get_reward_cfg to return.
        assignment = None
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "reward_dict" for t in node.targets
            ):
                assignment = node

        if assignment is None:
            defines_factory = any(
                isinstance(n, ast.FunctionDef) and n.name == "reward_dict"
                for n in tree.body
            )
            if defines_factory:
                return ("reward_dict is defined as a function. It must be a "
                        "module-level dictionary literal: reward_dict = { ... }")
            return ("the code defines no module-level `reward_dict = { ... }`. "
                    "Do not wrap the terms in a factory function that returns a "
                    "dict; assign the dictionary at module level.")

        if not isinstance(assignment.value, ast.Dict):
            return ("reward_dict must be assigned a dictionary literal "
                    "{ \"term_name\": RewTerm(...), ... }")

        # Unique keys: Python keeps only the last value for a repeated key, so
        # the other terms vanish silently and the run still reports success.
        seen, duplicates = set(), []
        for key in assignment.value.keys:
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                if key.value in seen and key.value not in duplicates:
                    duplicates.append(key.value)
                seen.add(key.value)
        if duplicates:
            return (f"reward_dict reuses the key(s) "
                    f"{', '.join(repr(d) for d in duplicates)}. Python keeps only "
                    f"the last value for a repeated key, so the other reward "
                    f"terms would be silently discarded. Give every term a "
                    f"unique, descriptive name such as \"reaching_position\".")

        if not assignment.value.keys:
            return "reward_dict is empty; it must contain at least one RewTerm"

        # Each RewTerm's params keys must match its function's named parameters.
        # Isaac Lab unpacks params as **kwargs, so a function declared as
        # `def f(env, params: dict)` fails with "expects mandatory parameters:
        # ['params'] ... but received: ['asset_cfg', ...]".
        functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        for key, value in zip(assignment.value.keys, assignment.value.values):
            if not isinstance(value, ast.Call):
                continue
            term_name = key.value if isinstance(key, ast.Constant) else "<term>"

            func_name, param_keys = None, None
            for kw in value.keywords:
                if kw.arg == "func" and isinstance(kw.value, ast.Name):
                    func_name = kw.value.id
                elif kw.arg == "params" and isinstance(kw.value, ast.Dict):
                    param_keys = {
                        k.value for k in kw.value.keys
                        if isinstance(k, ast.Constant) and isinstance(k.value, str)
                    }

            if func_name is None or param_keys is None:
                continue
            target = functions.get(func_name)
            if target is None:
                return (f"term {term_name!r} points at func={func_name}, which is "
                        f"not defined at module level in your code.")

            # Skip `env`; args with defaults are optional.
            args = target.args.args[1:]
            n_defaults = len(target.args.defaults)
            required = {a.arg for a in (args[:-n_defaults] if n_defaults else args)}
            available = {a.arg for a in args}

            if "params" in available and param_keys - available:
                return (f"{func_name}() takes a single `params` dict. Isaac Lab "
                        f"unpacks params as keyword arguments, so declare one "
                        f"named parameter per key instead: "
                        f"def {func_name}(env, {', '.join(sorted(param_keys))}).")
            missing = required - param_keys
            extra = param_keys - available
            if missing or extra:
                return (f"term {term_name!r}: {func_name}() expects "
                        f"{sorted(required)} but params supplies "
                        f"{sorted(param_keys)}. They must match exactly.")

        return None

    def _validate_scene_entities(self, tree, allowed, robot_name):
        """Reject SceneEntityCfg names that are not real scene entities.

        Isaac Lab only discovers this once the scene exists, a full Isaac Sim
        boot later: "The scene entity 'franka-reach' does not exist. Available
        entities: [...]". Checking the string literals up front turns that into
        an instant, specific rejection.
        """
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "SceneEntityCfg"):
                continue

            name_node = None
            if node.args:
                name_node = node.args[0]
            else:
                for kw in node.keywords:
                    if kw.arg == "name":
                        name_node = kw.value

            if not (isinstance(name_node, ast.Constant)
                    and isinstance(name_node.value, str)):
                continue

            if name_node.value not in allowed:
                return (f"SceneEntityCfg({name_node.value!r}) names an entity that "
                        f"does not exist in the scene. The robot's scene entity is "
                        f"{robot_name!r} -- that is the asset's name in the scene, "
                        f"not the robot model, task or config name. Available: "
                        f"{sorted(allowed)}.")
        return None

    def inject_code(self, raw_llm_output, candidate_id=""):
        """Extract code from Gemini's response, inject into template, validate, save."""
        full_code = self._extract_code_block(raw_llm_output)
        if full_code is None:
            return None, "the response contained no Python code block"

        # Sanitize: fix common Gemini mistakes
        full_code = re.sub(r'(def\s+\w+)\s+(\w+)', r'\1\2', full_code)  # space in func names
        full_code = full_code.replace("asset_name=", "name=")  # wrong SceneEntityCfg kwarg
        full_code = textwrap.dedent(full_code)

        # Load template
        template_path = self.designer_root / "templates" / "reward_template.py"
        with open(template_path, 'r') as f:
            template = f.read()

        # Single injection point at module level -- no text splitting.
        final_code = template.replace("# INSERT_REWARD_CODE_HERE", full_code)

        # Each candidate gets its own file and is evaluated from there. The
        # shared reward_output_file is written ONCE, at the end of the run, from
        # the best candidate. Writing it per candidate had two consequences:
        # every candidate's path looked identical (so the best one could not be
        # recovered), and a run where everything failed still left broken code at
        # the exact path Stage 2 and manipulation-tasks read.
        try:
            tree = ast.parse(final_code)
        except SyntaxError as exc:
            return None, f"the generated file is not valid Python: {exc}"

        error = self._validate_reward_code(tree)
        if error:
            return None, error

        error = self._validate_scene_entities(
            tree, self.scene_entities, self.robot_asset_name)
        if error:
            return None, error

        if candidate_id:
            candidate_path = self.candidates_dir / f"{candidate_id}_reward.py"
            candidate_path.parent.mkdir(parents=True, exist_ok=True)
            with open(candidate_path, 'w') as f:
                f.write(final_code)
            return candidate_path, None

        output_path = self.designer_root / self.cfg['reward_output_file']
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(final_code)
        return output_path, None

    def _eval_timeout(self, train_iterations):
        """Wall-clock budget for one headless evaluation, in seconds.

        The old formula was `rollout_duration + 120`, and rollout_duration is not
        in reach.yaml, so it defaulted to 30 -- a 150s ceiling covering Isaac Sim
        boot (40-90s on its own), env creation, N PPO iterations and the eval
        rollout. Most real candidates were being killed and scored as failures.
        """
        startup = self.cfg['eureka'].get('eval_timeout_startup_s', 300)
        per_iter = self.cfg['eureka'].get('eval_timeout_per_iteration_s', 2.0)
        return int(startup + per_iter * train_iterations)

    # --- Evaluation via Subprocess ---

    def run_evaluation(self, task_name, candidate_id=""):
        """
        Launch eval_headless.py as a subprocess inside the Docker container.
        Read metrics from a shared JSON file when it finishes.
        """
        num_envs = self.cfg['eureka'].get('eval_num_envs', 8)

        # Paths as seen from INSIDE the container. The candidate's own file is
        # evaluated, not the shared output path, so a failed candidate never
        # contaminates what downstream stages read.
        reward_file_container = (
            f"{self.shared_dir}/outputs/candidates/{candidate_id}_reward.py"
        )
        metrics_file_container = f"{self.shared_dir}/outputs/candidates/{candidate_id}_metrics.json"

        # Path as seen from the HOST (for reading results)
        metrics_file_host = self.candidates_dir / f"{candidate_id}_metrics.json"

        # Log file on host
        log_file = self.candidates_dir / f"{candidate_id}.log"

        # Clear any metrics file left by an earlier run of this same candidate
        # id, so a salvaged read can never return stale results.
        if metrics_file_host.exists():
            metrics_file_host.unlink()

        cmd = self._eval_command([
            "--reward-file", reward_file_container,
            "--num-envs", str(num_envs),
            "--train-iterations", str(self.cfg['eureka'].get('train_iterations', 300)),
            "--seed", str(self.seed),
            "--output", metrics_file_container,
        ])

        print(f"Launching headless eval: {candidate_id}")
        print(f"Command: {' '.join(cmd[-6:])}")  # show just the script args

        try:
            with open(log_file, 'w') as f:
                proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)

            # Wait for completion with timeout
            timeout = self._eval_timeout(
                self.cfg['eureka'].get('train_iterations', 300)
            )
            returncode = proc.wait(timeout=timeout)

            if returncode != 0:
                print(f"Eval process exited with code {returncode}")
                # Print last 20 lines of log for debugging
                self._print_log_tail(log_file, lines=20)
                return self._recover_metrics(metrics_file_host, "process_error")

            # Read metrics from JSON file
            if metrics_file_host.exists():
                with open(metrics_file_host, 'r') as f:
                    metrics = json.load(f)
                correlation = metrics.get("reward_correlation")
                correlation_text = (f", corr={correlation:+.3f}"
                                    if correlation is not None else "")
                print(f"{candidate_id}: reward={metrics['mean_reward']:.4f}, "
                      f"success={metrics['success_rate']:.4f}{correlation_text}")
                return metrics
            else:
                print(f"Metrics file not found: {metrics_file_host}")
                self._print_log_tail(log_file, lines=20)
                return {"mean_reward": 0.0, "success_rate": 0.0, "status": "no_metrics"}

        except subprocess.TimeoutExpired:
            print(f"Eval timed out after {timeout}s, killing process...")
            proc.kill()
            proc.wait()
            return self._recover_metrics(metrics_file_host, "timeout")

        except Exception as e:
            print(f"Eval error: {e}")
            return {
                "mean_reward": 0.0,
                "success_rate": 0.0,
                "reward_components": {},
                "position_error": [],
                "error": str(e),
                "status": "error",
            }

    def _eval_command(self, script_args):
        """Build the command line that launches eval_headless.py.

        Prefixes `docker exec` only when we are outside the container; inside it
        that binary does not exist, which surfaced as a bare
        `FileNotFoundError: 'docker'` for every candidate.
        """
        if self.in_container:
            return [self.docker_python, self.eval_script] + script_args

        if shutil.which("docker") is None:
            raise RuntimeError(
                f"Cannot launch eval_headless.py: not inside the Isaac Sim "
                f"container ({self.docker_python} not found) and 'docker' is not "
                f"on PATH. Run this script on the host with docker available, or "
                f"inside the container with /isaac-sim/python.sh."
            )

        return ["docker", "exec", self.docker_container,
                self.docker_python, self.eval_script] + script_args

    def _recover_metrics(self, metrics_file_host, fallback_status):
        """Read metrics eval_headless.py wrote before its process died.

        eval_headless.py writes its metrics file -- including the traceback of a
        failed candidate -- and only then calls simulation_app.close(), which can
        hang after an exception. Without this the host times out, kills the
        process, and throws away a perfectly good result already on disk. That
        costs twice: the traceback is exactly what execution-error feedback needs,
        and a broken candidate otherwise burns the entire timeout for nothing.
        """
        if metrics_file_host.exists():
            try:
                with open(metrics_file_host, 'r') as f:
                    metrics = json.load(f)
                print(f"  Recovered metrics written before the process died "
                      f"(status={metrics.get('status')})")
                return metrics
            except Exception as exc:
                print(f"  Metrics file present but unreadable: {exc}")

        return {
            "mean_reward": 0.0,
            "success_rate": 0.0,
            "reward_components": {},
            "position_error": [],
            "status": fallback_status,
        }

    def _print_log_tail(self, log_file, lines=20):
        """Print the last N lines of a log file for debugging."""
        try:
            with open(log_file, 'r') as f:
                all_lines = f.readlines()
            tail = all_lines[-lines:]
            print(f"--- Last {lines} lines of {log_file.name} ---")
            for line in tail:
                print(f"  {line.rstrip()}")
            print("---")
        except Exception:
            pass

    # --- Main Eureka Loop -------------------------------------

    def run_search(self, task, run_index=0):
        """One independent Eureka search (Alg 1). Returns this run's best.

        Factored out of main_loop so it can be restarted: the paper performs
        "multiple random restarts to find better maxima ... 5 independent runs
        per environment", which the reference implementation does by launching
        the script five times. Each call starts from a fresh conversation, so
        the only thing carried across restarts is the best result.
        """
        # K = samples per iteration (Alg 1 line 5). The paper uses 16; the
        # probability that every candidate in an iteration is broken falls off
        # exponentially in K, which is what lets Eureka tolerate bad samples.
        K = self.cfg['eureka'].get('num_samples', 4)

        self.run_index = run_index

        # Every candidate in a run shares one seed, so differences in the
        # fitness function reflect the reward and not training noise -- that
        # comparison is the whole point of F. Across restarts the seed may vary
        # (eureka.seed_per_run), which guards against selecting a reward that
        # merely happens to suit one seed.
        base_seed = self.cfg['eureka'].get('seed', 42)
        if self.cfg['eureka'].get('seed_per_run', False):
            self.seed = base_seed + run_index
        else:
            self.seed = base_seed
        print(f"  seed: {self.seed}")

        self.reset_conversation(task)

        best_metrics = None
        best_reward_path = None

        # Global best across ALL iterations (Alg 1 line 11). Distinct from the
        # per-iteration best above: an iteration can regress, and the run must
        # still finish holding the best reward it ever saw.
        best_metrics_overall = None
        best_reward_path_overall = None
        best_correlation_overall = None

        for i in range(self.cfg['eureka']['iterations']):
            print(f"\n{'='*60}")
            print(f"  Iteration {i+1}/{self.cfg['eureka']['iterations']}  (K={K} candidates)")
            print(f"  conversation: system + {len(self.messages)} turn(s)")
            print(f"{'='*60}")

            # All K samples are drawn i.i.d. from the SAME message checkpoint,
            # as in the original (one `messages` list, n=chunk_size). Snapshot
            # the list rather than handing over the live one: record_exchange
            # replaces slots in self.messages, so anything holding a reference
            # would silently observe a later iteration's history. The turn dicts
            # are never mutated in place, so a shallow copy is enough.
            prompt = list(self.messages)

            # --- Sample K candidates from LLM ---
            raw_responses = []
            llm_errors = []
            for k in range(K):
                print(f"  Querying Gemini for candidate {k+1}/{K}...")
                time.sleep(2)  # rate limit buffer
                text, error = self._generate_candidate(prompt)
                if error:
                    print(f"  Gemini call failed for candidate {k}: {error}")
                raw_responses.append(text)
                llm_errors.append(error)

            # --- Inject and evaluate each candidate ---
            candidate_results = []  # list of (metrics, reward_path, raw_text)

            for k, raw_text in enumerate(raw_responses):
                candidate_id = f"iter{i}_k{k}"

                if raw_text is None:
                    # Distinct from a code failure: no code was produced at all,
                    # so there is nothing for reward reflection to talk about.
                    error = llm_errors[k] or "the LLM returned no text"
                    print(f"  {candidate_id}: skipped ({error})")
                    candidate_results.append({
                        "metrics": {
                            "mean_reward": 0.0,
                            "success_rate": 0.0,
                            "reward_components": {},
                            "position_error": [],
                            "error": error,
                            "status": "llm_error",
                        },
                        "reward_path": None,
                        "raw_text": None,
                        "candidate_id": candidate_id,
                    })
                    continue

                # Save raw LLM output
                raw_output_path = self.candidates_dir / f"{candidate_id}_raw_llm.txt"
                with open(raw_output_path, 'w') as f:
                    f.write(raw_text)

                # Inject into template
                output_file, inject_error = self.inject_code(
                    raw_text, candidate_id=candidate_id)
                if not output_file:
                    # Recorded as a failed candidate rather than dropped, so the
                    # reason reaches the LLM as execution-error feedback instead
                    # of vanishing into the console.
                    print(f"  {candidate_id}: code injection failed -- {inject_error}")
                    candidate_results.append({
                        "metrics": {
                            "mean_reward": 0.0,
                            "success_rate": 0.0,
                            "reward_components": {},
                            "position_error": [],
                            "error": inject_error,
                            "status": "injection_error",
                        },
                        "reward_path": None,
                        "raw_text": raw_text,
                        "candidate_id": candidate_id,
                    })
                    continue

                # Evaluate
                metrics = self.run_evaluation(task, candidate_id=candidate_id)
                candidate_results.append({
                    "metrics": metrics,
                    "reward_path": output_file,
                    "raw_text": raw_text,
                    "candidate_id": candidate_id,
                })

            # --- Select best candidate by task success ---
            executed = [r for r in candidate_results if candidate_executed(r)]

            # Per-candidate verdict, so a run's console output says what actually
            # happened rather than only naming a winner.
            for k, result in enumerate(candidate_results):
                cid = f"iter{i}_k{k}"
                if result is None:
                    print(f"    {cid}: FAILED (no code generated or injected)")
                elif not candidate_executed(result):
                    status = result["metrics"].get("status", "unknown")
                    print(f"    {cid}: FAILED ({status}) "
                          f"{self._error_line(result['metrics'])}")
                else:
                    print(f"    {cid}: ok  success_rate="
                          f"{result['metrics'].get('success_rate', 0):.4f}")

            if not executed:
                statuses = {r["metrics"].get("status")
                            for r in candidate_results if r is not None}
                if statuses == {"llm_error"}:
                    # No code was generated at all, so there is nothing to
                    # reflect on. Leave the checkpoint untouched and say plainly
                    # that this was a transport failure, not a bad reward.
                    print(f"  All {K} candidates failed: could not reach the LLM. "
                          f"This is not a reward-code failure; check the API "
                          f"status or raise eureka.llm_max_attempts.")
                else:
                    print(f"  All {K} candidates failed this iteration, retrying "
                          f"from the current message checkpoint...")
                    # The original repeats the iteration WITHOUT touching
                    # messages when K > 1, so the model learns nothing from a
                    # wholly failed iteration -- but it does record the error
                    # exchange when K == 1 (the `cfg.sample != 1` guard). We
                    # generalise the K == 1 behaviour to any K by default,
                    # attaching the first candidate that produced code; set
                    # eureka.feedback_on_all_failed: false for the original's
                    # exact K > 1 semantics.
                    if K == 1 or self.cfg['eureka'].get('feedback_on_all_failed', True):
                        failed = next((r for r in candidate_results
                                       if r is not None and r.get("raw_text")), None)
                        if failed is not None:
                            self.record_exchange(
                                failed["raw_text"],
                                self._build_all_failed_feedback(candidate_results),
                            )
                            self.save_conversation()

                wandb.log({
                    "iteration": i,
                    "execute_rate": 0.0,
                    "num_valid_candidates": 0,
                })
                continue

            # Sort by success_rate (task success), not mean_reward
            best = max(executed, key=lambda r: r["metrics"].get("success_rate", 0.0))

            best_metrics = best["metrics"]
            best_reward_path = best["reward_path"]
            best_candidate_id = best["candidate_id"]

            print(f"\n  Best candidate: {best_candidate_id}")
            print(f"    success_rate={best_metrics.get('success_rate', 0):.4f}, "
                  f"mean_reward={best_metrics.get('mean_reward', 0):.4f}")
            best_correlation = best_metrics.get("reward_correlation")
            if best_correlation is not None:
                print(f"    reward_correlation={best_correlation:+.3f} "
                      f"(agreement with Isaac Lab's hand-written reward)")

            # --- Update global best ---
            if (best_metrics_overall is None or
                    best_metrics.get("success_rate", 0) > best_metrics_overall.get("success_rate", 0)):
                best_metrics_overall = best_metrics
                best_reward_path_overall = best_reward_path
                # Tracked alongside the global best, as in the original: the
                # correlation reported for a run is the one belonging to the
                # reward that run actually selected.
                best_correlation_overall = best_metrics.get("reward_correlation")

            # --- Reward reflection, recorded against the code it describes ---
            # Pairing the best candidate's own response with its reflection is
            # what makes the next iteration a mutation of that reward rather
            # than a fresh sample: the model can see the code being critiqued.
            stride = self._subsample_stride(
                best["metrics"].get("position_error") or [])
            feedback = self.build_feedback_prompt(best["metrics"], stride)
            self.record_exchange(best["raw_text"], feedback)
            self.save_conversation()

            # --- W&B logging ---
            exec_rate = len(executed) / K
            wandb.log({
                "run": run_index,
                "iteration": i,
                "best_success_rate": best_metrics["success_rate"],
                "best_mean_reward": best_metrics["mean_reward"],
                "best_reward_correlation": best_metrics.get("reward_correlation"),
                "execute_rate": exec_rate,
                "num_valid_candidates": len(executed),
                "generated_code": wandb.Html(f"<pre><code>{best['raw_text']}</code></pre>"),
            })

        return best_metrics_overall, best_reward_path_overall, best_correlation_overall

    def main_loop(self, task):
        print(f"Starting Fluxa Stage 1 (Eureka) for {task}...")

        self.env_context = self.load_env_context()

        # Scene entities a generated reward may legitimately reference. The robot
        # asset first (its name is what the error message suggests), then the
        # other props the reach scene defines.
        robot_cfg = self.cfg['robots'][task]
        self.robot_asset_name = robot_cfg.get('asset_name', 'robot')
        self.scene_entities = {self.robot_asset_name}
        self.scene_entities.update(
            self.cfg.get('env_context', {}).get('scene_entities', []) or [])

        # Random restarts (Alg 1 / 3.2). The paper runs 5 independent searches
        # per environment and keeps the best; the reference implementation does
        # this by launching the script five times. The restarts differ only in
        # LLM sampling -- the PPO seed stays fixed so that differences in the
        # fitness function reflect the reward, not training noise.
        num_runs = self.cfg['eureka'].get('num_runs', 1)
        iterations = self.cfg['eureka']['iterations']
        K = self.cfg['eureka'].get('num_samples', 4)
        total_evals = num_runs * iterations * K
        seconds_each = self.cfg['eureka'].get('eval_seconds_estimate')
        estimate = ""
        if seconds_each:
            hours = total_evals * seconds_each / 3600.0
            estimate = (f", ~{hours:.1f}h at {seconds_each:.0f}s each"
                        if hours >= 1 else
                        f", ~{total_evals * seconds_each / 60.0:.0f}min "
                        f"at {seconds_each:.0f}s each")
        print(f"  {num_runs} run(s) x {iterations} iteration(s) x K={K} "
              f"= {total_evals} candidate evaluations{estimate}")

        best_metrics_overall = None
        best_reward_path_overall = None
        best_correlation_overall = None

        for run_index in range(num_runs):
            if num_runs > 1:
                print(f"\n{'#'*60}")
                print(f"  RESTART {run_index + 1}/{num_runs}")
                print(f"{'#'*60}")

            metrics, reward_path, correlation = self.run_search(task, run_index)

            if reward_path is None:
                print(f"  Run {run_index + 1} produced no working reward.")
                continue

            print(f"\n  Run {run_index + 1} best: "
                  f"success_rate={metrics.get('success_rate', 0):.4f}")

            if (best_metrics_overall is None
                    or metrics.get("success_rate", 0)
                    > best_metrics_overall.get("success_rate", 0)):
                best_metrics_overall = metrics
                best_reward_path_overall = reward_path
                best_correlation_overall = correlation
                if num_runs > 1:
                    print(f"  -> new best across restarts")

        if num_runs > 1 and best_metrics_overall is not None:
            print(f"\n{'#'*60}")
            print(f"  Best across {num_runs} restarts: "
                  f"success_rate={best_metrics_overall.get('success_rate', 0):.4f}")
            print(f"{'#'*60}")

        # --- Final long training with best reward ---
        if best_reward_path_overall:
            import shutil
            final_output = self.designer_root / self.cfg['reward_output_file']
            final_output.parent.mkdir(parents=True, exist_ok=True)
            if best_reward_path_overall != final_output:
                shutil.copy2(best_reward_path_overall, final_output)
            print(f"\n Best reward saved to: {final_output}")
            if best_correlation_overall is not None:
                print(f"   Reward correlation with the hand-written reward: "
                      f"{best_correlation_overall:+.3f}")
            print(f"   Best metrics: {best_metrics_overall}")

            # Long training run to produce checkpoint for Stage 2
            print(f"\n{'='*60}")
            print(f"   Final training with best reward")
            print(f"{'='*60}")

            final_train_iters = self.cfg['eureka'].get('final_train_iterations', 2000)
            # Output paths come from config so a reduced smoke config can write
            # somewhere harmless instead of overwriting the checkpoint Stage 2
            # depends on.
            policy_rel = self.cfg.get('policy_output_file', './outputs/eureka_policy.pt')
            final_metrics_rel = self.cfg.get('final_metrics_file', './outputs/final_metrics.json')

            reward_file_container = f"{self.shared_dir}/{self.cfg['reward_output_file']}"
            policy_path = f"{self.shared_dir}/{policy_rel}"
            final_metrics_path = f"{self.shared_dir}/{final_metrics_rel}"

            cmd = self._eval_command([
                "--reward-file", reward_file_container,
                "--num-envs", str(self.cfg['eureka'].get('eval_num_envs', 16)),
                "--train-iterations", str(final_train_iters),
                "--seed", str(self.seed),
                "--save-policy", policy_path,
                "--output", final_metrics_path,
            ])

            print(f"Training for {final_train_iters} iterations...")
            log_file = self.candidates_dir / "final_train.log"

            try:
                with open(log_file, 'w') as f:
                    proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)

                timeout = self._eval_timeout(final_train_iters)
                returncode = proc.wait(timeout=timeout)

                if returncode == 0:
                    final_metrics_host = self.designer_root / final_metrics_rel
                    if final_metrics_host.exists():
                        with open(final_metrics_host, 'r') as f:
                            final_metrics = json.load(f)

                        # eval_headless.py exits 0 even when it bails out early
                        # (a reward that will not build, say), so a zero exit code
                        # alone is not success. Reporting "Policy saved" here when
                        # no policy exists sends Stage 2 looking for a checkpoint
                        # that was never written.
                        if final_metrics.get("status") != "success":
                            print(f"  Final training did NOT produce a policy "
                                  f"(status={final_metrics.get('status')})")
                            print(f"  {self._error_line(final_metrics)}")
                            self._print_log_tail(log_file)
                        else:
                            print(f"  Final training complete!")
                            print(f"  mean_reward={final_metrics['mean_reward']:.4f}")
                            print(f"  Policy saved to: {policy_path}")
                            wandb.log({"final_mean_reward": final_metrics["mean_reward"]})
                    else:
                        print(f"  Training finished but metrics file not found")
                        self._print_log_tail(log_file)
                else:
                    print(f"  Final training failed with code {returncode}")
                    self._print_log_tail(log_file)

            except subprocess.TimeoutExpired:
                print(f"  Final training timed out, killing...")
                proc.kill()
                proc.wait()

            except Exception as e:
                # The search itself already succeeded at this point; a failure to
                # launch the final training run must not throw that away.
                print(f"  Final training could not run: {e}")
                print(f"  The best reward is still saved at {final_output}")
        else:
            print("\n All iterations failed to produce valid reward code.")

        wandb.finish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="franka-reach")
    parser.add_argument("--config", default="cfg/reach.yaml",
                        help="Path to the stage config (default: cfg/reach.yaml).")

    # Scope overrides, so reach.yaml can stay at the paper's settings while a
    # shorter pass is run from the command line. The full setting is 5 runs x 5
    # iterations x K=16 = 400 evaluations; at ~90s each that is ~10 hours.
    parser.add_argument("--runs", type=int, default=None,
                        help="Override eureka.num_runs (paper: 5). Use --runs 1 "
                             "for a single search, roughly a fifth of the time.")
    parser.add_argument("--iterations", type=int, default=None,
                        help="Override eureka.iterations (paper: 5).")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Override eureka.num_samples, i.e. K (paper: 16).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override eureka.seed. Shared by every candidate in "
                             "a run so the comparison isolates the reward.")
    args = parser.parse_args()

    manager = EurekaManager(args.config)

    for flag, key in (("runs", "num_runs"),
                      ("iterations", "iterations"),
                      ("num_samples", "num_samples"),
                      ("seed", "seed")):
        value = getattr(args, flag)
        if value is not None:
            print(f"[eureka] override: {key} = {value} "
                  f"(config has {manager.cfg['eureka'].get(key)})")
            manager.cfg['eureka'][key] = value

    manager.main_loop(args.task)