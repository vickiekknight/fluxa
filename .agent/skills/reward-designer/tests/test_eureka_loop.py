"""Eureka main_loop logic test. No GPU, no Isaac Sim, no API key, ~1 second.

Run from anywhere:
    python3 /home/vickie/fluxa/.agent/skills/reward-designer/tests/test_eureka_loop.py

Builds a throwaway copy of the skill under /tmp and stubs out Gemini, W&B, the
Isaac Sim subprocess and run_evaluation, so main_loop's bookkeeping can be
checked on its own. Nothing in the real skill directory is touched.

Scenario A -- iteration 1 REGRESSES (0.50 -> 0.20). Separates "keeps the best
reward it ever saw" from "keeps whichever candidate ran last"; the latter
silently shipped the wrong reward to Stage 2.

Scenario B -- every candidate fails to execute. Separates "nothing ran" from
"ran and scored zero"; conflating them reported execute_rate=1.0 for an
iteration in which nothing executed.

Covers:
  1. K is read from cfg.eureka.num_samples           (was an unbound NameError)
  2. the global best survives a regressing iteration (was read-before-assignment)
  3. reward_fn.py ends up holding the BEST candidate (inject_code returned a
     shared path, so every candidate looked identical and the last one won)
  4. eval timeouts scale with the workload            (was a flat 150s)
  5. metrics are salvaged when the eval process dies after writing them
  6. failed candidates cannot win selection, and are not counted as executed
  7. all-failed feedback names the real errors rather than being generic
"""
import ast
import copy
import subprocess
import importlib.util
import json
import os
import re
import shutil
import sys
import types
from pathlib import Path

REAL_SKILL = Path(__file__).resolve().parent.parent
REAL_SKILLS_DIR = REAL_SKILL.parent
SANDBOX_ROOT = Path("/tmp/fluxa_eureka_loop_test")

ORDER = ["iter0_k0", "iter0_k1", "iter1_k0", "iter1_k1"]

FAILURES = []


def check(label, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}"
          f"{('  -- ' + detail) if detail and not ok else ''}")
    if not ok:
        FAILURES.append(label)


def build_sandbox(name):
    """Mirror <skills>/{common,reward-designer} so relative imports resolve."""
    sandbox = SANDBOX_ROOT / name
    if sandbox.exists():
        shutil.rmtree(sandbox)
    skills = sandbox / "skills"
    ignore = shutil.ignore_patterns("__pycache__", "wandb", "outputs")
    shutil.copytree(REAL_SKILLS_DIR / "common", skills / "common", ignore=ignore)
    designer = skills / "reward-designer"
    for sub in ("cfg", "templates", "prompts", "scripts"):
        shutil.copytree(REAL_SKILL / sub, designer / sub, ignore=ignore)
    (designer / "outputs").mkdir()

    # Match on the key, not on surrounding comment text: pinning the comment
    # meant a reword of reach.yaml silently stopped shrinking the sandbox, and
    # the tests then ran at the full K.
    cfg_path = designer / "cfg/reach.yaml"
    cfg = cfg_path.read_text()
    for key, value in (("iterations", 2), ("num_samples", 2), ("num_runs", 1)):
        cfg, n = re.subn(rf"^(\s+){key}: \d+.*$", rf"\g<1>{key}: {value}",
                         cfg, count=1, flags=re.MULTILINE)
        assert n == 1, f"sandbox override for {key!r} matched nothing"
    cfg_path.write_text(cfg)
    return designer


def stub_external_modules():
    genai_mod = types.ModuleType("google.genai")
    genai_mod.Client = lambda **kw: types.SimpleNamespace(models=None)

    # 1_eureka.py does `from google.genai import types` for
    # GenerateContentConfig, so the stub has to carry it too. The real package
    # provides this fine inside the container; only the stub was missing it.
    genai_types = types.ModuleType("google.genai.types")
    genai_types.GenerateContentConfig = lambda **kw: dict(kw)
    genai_mod.types = genai_types

    google_mod = types.ModuleType("google")
    google_mod.genai = genai_mod
    sys.modules["google"] = google_mod
    sys.modules["google.genai"] = genai_mod
    sys.modules["google.genai.types"] = genai_types

    wandb = types.ModuleType("wandb")
    wandb.init = lambda **kw: None
    wandb.log = lambda *a, **k: None
    wandb.finish = lambda *a, **k: None
    wandb.Html = lambda x: x
    sys.modules["wandb"] = wandb


def candidate_code(candidate_id):
    """A minimal reward that inject_code can split and inject."""
    return (
        "```python\n"
        f"def r_{candidate_id}(env):\n"
        "    return torch.zeros(env.num_envs)\n"
        "\n"
        f'reward_dict = {{"t": RewTerm(func=r_{candidate_id}, weight=1.0, params={{}})}}\n'
        "```"
    )


def load_module(designer):
    os.chdir(designer)
    spec = importlib.util.spec_from_file_location(
        f"eureka_{designer.parent.parent.name}", designer / "scripts/1_eureka.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.time.sleep = lambda *_: None
    return mod


def make_manager(mod, designer, eval_fn):
    calls = []

    class FakeModels:
        # Mirrors the real signature: 1_eureka.py passes `config` to carry
        # system_instruction and temperature. Recording the calls lets the tests
        # assert on what was actually sent, not just that something was.
        def generate_content(self, model=None, contents=None, config=None):
            cid = ORDER[len(calls)]
            # Deep-copy: the conversation is a live list whose slots get
            # replaced each iteration, so recording it by reference would make
            # every call appear to carry the FINAL history.
            calls.append({
                "candidate": cid,
                "contents": copy.deepcopy(contents),
                "config": config,
            })
            return types.SimpleNamespace(text=candidate_code(cid))

    class FakeProc:
        def wait(self, timeout=None):
            (designer / "outputs/final_metrics.json").write_text(
                json.dumps({"mean_reward": 9.99, "status": "success"}))
            return 0

    # Rebind the module's `subprocess` name rather than assigning to
    # subprocess.Popen: the latter mutates the shared stdlib module and leaks
    # into every later test (a real subprocess.run() then got the fake).
    mod.subprocess = types.SimpleNamespace(
        Popen=lambda *a, **k: FakeProc(),
        TimeoutExpired=subprocess.TimeoutExpired,
        STDOUT=subprocess.STDOUT,
    )
    mod.EurekaManager.run_evaluation = eval_fn

    manager = mod.EurekaManager("cfg/reach.yaml")
    manager.client = types.SimpleNamespace(models=FakeModels())
    return manager, calls


# --------------------------------------------------------------- scenario A
def scenario_regression():
    print("\n--- Scenario A: iteration 1 regresses -----------------------------")
    designer = build_sandbox("regression")
    mod = load_module(designer)

    scores = {"iter0_k0": 0.30, "iter0_k1": 0.50, "iter1_k0": 0.10, "iter1_k1": 0.20}
    best, last = "iter0_k1", "iter1_k1"

    def fake_eval(self, task_name, candidate_id=""):
        score = scores[candidate_id]
        return {
            "mean_reward": score * 2,
            "success_rate": score,
            "reward_components": {"t": [0.1, 0.2, 0.3]},
            "position_error": [0.3, 0.2, 0.1],
            "status": "success",
        }

    manager, calls = make_manager(mod, designer, fake_eval)
    manager.main_loop("franka-reach")

    print()
    served = [c["candidate"] for c in calls]
    check("K is read from cfg.eureka.num_samples (4 calls over 2 iterations)",
          served == ORDER, f"got {served}")

    # What was actually sent to the model (items 8 and 9).
    first = calls[0] if calls else {}
    sent_config = first.get("config") or {}
    check("the system prompt is sent as system_instruction, not inlined",
          bool(sent_config.get("system_instruction")), str(sent_config)[:120])
    check("system_instruction carries the formatting tips",
          "reward_dict" in str(sent_config.get("system_instruction", "")))
    check("temperature is set so K samples actually differ",
          sent_config.get("temperature") == 1.0, str(sent_config.get("temperature")))
    check("the user turn carries the task description",
          "end-effector" in str(first.get("contents", "")).lower())

    # --- conversation state (Eureka 3.2) ---
    first_contents = first.get("contents") or []
    check("the first iteration sends one user turn",
          isinstance(first_contents, list) and len(first_contents) == 1,
          f"len={len(first_contents) if isinstance(first_contents, list) else 'n/a'}")

    # By the second iteration the model must be shown its own best code plus the
    # reflection on it -- that is what makes this mutation, not resampling.
    second_iter = calls[2]["contents"] if len(calls) > 2 else []
    roles = [t.get("role") for t in second_iter] if isinstance(second_iter, list) else []
    check("the second iteration sends user/model/user",
          roles == ["user", "model", "user"], str(roles))
    check("the model turn carries the best code from iteration 1",
          "r_iter0_k1" in str(second_iter), str(second_iter)[:160])
    check("the trailing user turn carries the reward reflection",
          "We trained a RL policy" in str(second_iter))

    # It must REPLACE, never accumulate: the losing candidate's code is gone.
    check("a losing candidate's code is not kept in the history",
          "r_iter0_k0" not in str(second_iter))
    check("history stays at 3 turns, it does not grow",
          len(manager.messages) == 3, str(len(manager.messages)))
    check("the initial task/env turn is never overwritten",
          "end-effector" in str(manager.messages[0]).lower())

    final_metrics = designer / "outputs/final_metrics.json"
    check("the final training stage actually ran",
          final_metrics.is_file(),
          "nothing wrote final_metrics.json -- the stage was skipped or threw")

    messages_json = designer / "outputs/messages.json"
    check("the conversation is dumped for inspection", messages_json.exists())
    if messages_json.exists():
        dumped = json.loads(messages_json.read_text())
        check("the dump leads with the system turn",
              dumped and dumped[0]["role"] == "system", str(dumped[0]["role"]) if dumped else "empty")

    # Read defensively: when a scenario regresses this file may be absent, and a
    # missing file should surface as a FAIL, not abort the whole suite.
    reward_path = designer / "outputs/reward_fn.py"
    final_reward = reward_path.read_text() if reward_path.exists() else ""
    check(f"reward_fn.py holds the best candidate ({best})",
          f"r_{best}" in final_reward)
    check(f"reward_fn.py does NOT hold the last candidate ({last})",
          f"r_{last}" not in final_reward)
    check("every candidate got its own reward file",
          all((designer / f"outputs/candidates/{c}_reward.py").exists() for c in ORDER))

    check("eval timeout scales with train_iterations (300 -> 900s)",
          manager._eval_timeout(300) == 900, str(manager._eval_timeout(300)))
    check("eval timeout scales with train_iterations (1000 -> 2300s)",
          manager._eval_timeout(1000) == 2300, str(manager._eval_timeout(1000)))
    check("eval timeout is no longer the flat 150s", manager._eval_timeout(300) > 150)

    salvage = designer / "outputs/candidates/salvage_metrics.json"
    salvage.write_text(json.dumps({
        "mean_reward": 0.0, "success_rate": 0.0,
        "status": "exception", "traceback": "NameError: quat_conjugate",
    }))
    recovered = manager._recover_metrics(salvage, "timeout")
    check("metrics written before a hang are recovered, not discarded",
          recovered.get("status") == "exception"
          and "quat_conjugate" in recovered.get("traceback", ""), str(recovered))

    missing = manager._recover_metrics(
        designer / "outputs/candidates/nope.json", "timeout")
    check("a genuinely missing metrics file still reports the fallback status",
          missing.get("status") == "timeout", str(missing))

    return mod


# --------------------------------------------------------------- scenario B
def scenario_all_fail(mod_a):
    print("\n--- Scenario B: every candidate fails to execute ------------------")
    designer = build_sandbox("all_fail")
    mod = load_module(designer)

    real_error = "TypeError: RewardTermCfg.__init__() got an unexpected keyword argument 'scene_entities'"

    def fake_eval(self, task_name, candidate_id=""):
        return {
            "mean_reward": 0.0,
            "success_rate": 0.0,
            "reward_components": {},
            "position_error": [],
            "traceback": f"Traceback (most recent call last):\n  ...\n{real_error}\n",
            "status": "reward_resolve_error",
        }

    manager, served = make_manager(mod, designer, fake_eval)
    manager.main_loop("franka-reach")

    print()
    check("a candidate that never executed is not counted as executed",
          not mod.candidate_executed({"metrics": {"status": "reward_resolve_error"}}))
    check("a candidate that executed is counted as executed",
          mod.candidate_executed({"metrics": {"status": "success"}}))
    check("a missing candidate is not counted as executed",
          not mod.candidate_executed(None))

    # eval_headless now refuses to fall back to Isaac Lab's hand-written reward
    # when --reward-file yields no terms. It reports status "no_reward_terms",
    # which must count as a failure -- the earlier fallback trained the default
    # reward and reported a healthy success_rate for a candidate that had
    # contributed nothing.
    no_terms = {"metrics": {"status": "no_reward_terms",
                            "error": "defines no usable reward terms"}}
    check("a candidate with no reward terms is not counted as executed",
          not mod.candidate_executed(no_terms))
    check("its reason reaches the console and the feedback",
          "no usable reward terms" in manager._error_line(no_terms["metrics"]),
          manager._error_line(no_terms["metrics"]))
    check("no_reward_terms is fed back as a code error, unlike llm_error",
          "no usable reward terms"
          in manager._build_all_failed_feedback([no_terms]))

    check("no reward_fn.py is produced when nothing ever executed",
          not (designer / "outputs/reward_fn.py").exists())

    feedback = manager._build_all_failed_feedback([
        {"metrics": {"traceback": f"...\n{real_error}\n", "status": "reward_resolve_error"}},
        None,
    ])
    check("all-failed feedback names the actual error",
          "scene_entities" in feedback, feedback)
    check("all-failed feedback is not the old generic message",
          "all candidates failed to produce valid code" not in feedback)

    check("DUMMY_FAILURE sorts below any real score",
          mod.DUMMY_FAILURE < 0.0)

    # Final-training honesty: a zero exit code with a non-success status must not
    # be reported as a saved policy.
    check("_error_line surfaces the last traceback line",
          manager._error_line({"traceback": f"x\n{real_error}\n"}) == real_error,
          manager._error_line({"traceback": f"x\n{real_error}\n"}))


# --------------------------------------------------------------- scenario C
def scenario_injection():
    """inject_code robustness, driven by real failures seen in live runs."""
    print("\n--- Scenario C: code injection ------------------------------------")
    designer = build_sandbox("injection")
    mod = load_module(designer)
    manager = mod.EurekaManager("cfg/reach.yaml")
    manager.scene_entities = {"robot", "table", "ground", "terrain", "light"}
    manager.robot_asset_name = "robot"

    # A uniformly indented response (the model wrapped everything one level in)
    # must still inject: dedent handles it.
    indented = (
        "```python\n"
        "    def r_indented(env):\n"
        "        return torch.zeros(env.num_envs)\n"
        "\n"
        '    reward_dict = {"t": RewTerm(func=r_indented, weight=1.0, params={})}\n'
        "```"
    )
    path, error = manager.inject_code(indented, candidate_id="indent_case")
    check("a uniformly indented response still injects", path is not None, str(error))

    if path:
        try:
            ast.parse(Path(path).read_text())
            parsed, detail = True, ""
        except SyntaxError as exc:
            parsed, detail = False, f"{type(exc).__name__}: {exc}"
        check("the injected file parses", parsed, detail)

    # Observed live: the model wrapped its terms in a factory returning a dict,
    # so nothing was left for get_reward_cfg to return.
    factory = (
        "```python\n"
        "def build_rewards(env, asset_cfg, command_name):\n"
        "    reward_dict = {}\n"
        '    reward_dict["reaching"] = RewTerm(func=build_rewards, weight=1.0, params={})\n'
        "    return reward_dict\n"
        "```"
    )
    path_f, error_f = manager.inject_code(factory, candidate_id="factory_case")
    check("a factory function that returns the dict is rejected", path_f is None)
    check("the factory error says to assign at module level",
          error_f and "module-level" in error_f, str(error_f))

    # Observed live: the model named its factory `reward_dict` itself.
    named = (
        "```python\n"
        "def reward_dict(env, asset_cfg, command_name):\n"
        "    return {}\n"
        "```"
    )
    path_n, error_n = manager.inject_code(named, candidate_id="named_case")
    check("a function named reward_dict is rejected", path_n is None)
    check("that error says reward_dict is defined as a function",
          error_n and "as a function" in error_n, str(error_n))

    # Observed live: `def f(env, params)` -- Isaac Lab unpacks params as kwargs.
    params_dict = (
        "```python\n"
        "def reaching_reward(env, params):\n"
        "    return torch.zeros(env.num_envs)\n"
        "\n"
        'reward_dict = {"reaching": RewTerm(func=reaching_reward, weight=1.0,\n'
        '    params={"asset_cfg": 1, "command_name": "ee_pose"})}\n'
        "```"
    )
    path_p, error_p = manager.inject_code(params_dict, candidate_id="params_case")
    check("a function taking a single params dict is rejected", path_p is None)
    check("that error explains params is unpacked as kwargs",
          error_p and "keyword arguments" in error_p, str(error_p))

    # A signature that simply disagrees with its params keys.
    mismatch = (
        "```python\n"
        "def r(env, asset_cfg, command_name):\n"
        "    return torch.zeros(env.num_envs)\n"
        "\n"
        'reward_dict = {"t": RewTerm(func=r, weight=1.0, params={"asset_cfg": 1})}\n'
        "```"
    )
    path_m, error_m = manager.inject_code(mismatch, candidate_id="mismatch_case")
    check("params keys that do not match the signature are rejected",
          path_m is None, str(error_m))

    # Observed live: both candidates used the TASK key as the scene entity,
    # because the user prompt said "the robot is 'franka-reach'".
    bad_entity = (
        "```python\n"
        "def r(env, asset_cfg):\n"
        "    return torch.zeros(env.num_envs)\n"
        "\n"
        'reward_dict = {"t": RewTerm(func=r, weight=1.0,\n'
        '    params={"asset_cfg": SceneEntityCfg("franka-reach", body_names=["panda_hand"])})}\n'
        "```"
    )
    path_e, error_e = manager.inject_code(bad_entity, candidate_id="entity_case")
    check("a bogus scene entity name is rejected", path_e is None)
    check("the scene-entity error points at the real asset name",
          error_e and "'robot'" in error_e, str(error_e))

    good_entity = (
        "```python\n"
        "def r(env, asset_cfg):\n"
        "    return torch.zeros(env.num_envs)\n"
        "\n"
        'reward_dict = {"t": RewTerm(func=r, weight=1.0,\n'
        '    params={"asset_cfg": SceneEntityCfg("robot", body_names=["panda_hand"])})}\n'
        "```"
    )
    path_g, error_g = manager.inject_code(good_entity, candidate_id="good_entity")
    check("the correct scene entity name is accepted", path_g is not None, str(error_g))

    # Extraction fallbacks (item 15): a bare fence must still work.
    bare = (
        "```\n"
        "def r_bare(env):\n"
        "    return torch.zeros(env.num_envs)\n"
        "\n"
        'reward_dict = {"t": RewTerm(func=r_bare, weight=1.0, params={})}\n'
        "```"
    )
    path_b, error_b = manager.inject_code(bare, candidate_id="bare_case")
    check("a bare ``` fence is still extracted", path_b is not None, str(error_b))

    check("inject_code does not write the shared reward_fn.py",
          not (designer / "outputs/reward_fn.py").exists())

    # Observed live: Gemini used the literal "RewTerm" as the key for all three
    # terms. Python keeps the last, so two rewards vanished and the run still
    # reported success -- the policy trained on an action-rate penalty alone.
    duplicate_keys = (
        "```python\n"
        "def a(env, asset_cfg, command_name):\n"
        "    return torch.zeros(env.num_envs)\n"
        "\n"
        "def b(env, asset_cfg, command_name):\n"
        "    return torch.zeros(env.num_envs)\n"
        "\n"
        "reward_dict = {\n"
        '    "RewTerm": RewTerm(func=a, weight=1.0, params={}),\n'
        '    "RewTerm": RewTerm(func=b, weight=1.0, params={}),\n'
        "}\n"
        "```"
    )
    path_dup, error_dup = manager.inject_code(duplicate_keys, candidate_id="dup_case")
    check("a reward_dict with duplicate keys is rejected", path_dup is None)
    check("the duplicate-key error names the offending key",
          error_dup is not None and "RewTerm" in error_dup, str(error_dup))
    check("no file is written for a rejected candidate",
          not (designer / "outputs/candidates/dup_case_reward.py").exists())

    # Unparseable output should fail at injection, not cost an Isaac Sim boot.
    broken = (
        "```python\n"
        "def a(env):\n"
        "    return torch.zeros(\n"
        "\n"
        'reward_dict = {"x": RewTerm(func=a, weight=1.0, params={})}\n'
        "```"
    )
    path_bad, error_bad = manager.inject_code(broken, candidate_id="broken_case")
    check("unparseable generated code is rejected at injection time",
          path_bad is None and error_bad is not None, str(error_bad))

    path_none, error_none = manager.inject_code("no code fence here", candidate_id="x")
    check("a response with no python block reports why",
          path_none is None and "code block" in (error_none or ""), str(error_none))

    # Isaac Lab's config validator raises a multi-line message; the last line is
    # a field name, which on its own says nothing about what failed.
    multiline = {"traceback": (
        "Traceback (most recent call last):\n"
        '  File "eval_headless.py", line 476, in run_evaluation\n'
        "    env = ManagerBasedRLEnv(cfg=env_cfg)\n"
        "TypeError: Missing values detected in object EvalFrankaReachEnvCfg "
        "for the following fields:\n"
        "  - rewards.ee_position.weight\n"
        "  - rewards.deviation_penalty.weight\n"
    )}
    line = manager._error_line(multiline)
    check("_error_line reports the exception, not the last continuation line",
          line.startswith("TypeError: Missing values detected"), line)


# --------------------------------------------------------------- scenario D
def scenario_prompts():
    """Environment-as-context (Eureka 3.1) and the prompt files (item 8)."""
    print("\n--- Scenario D: prompt construction -------------------------------")
    designer = build_sandbox("prompts")
    mod = load_module(designer)
    manager = mod.EurekaManager("cfg/reach.yaml")

    # --- env as context ---
    fake_env = designer / "outputs/fake_env_cfg.py"
    fake_env.write_text(
        "@configclass\n"
        "class ObservationsCfg:\n"
        "    joint_pos = ObsTerm(func=mdp.joint_pos_rel)\n"
        "\n"
        "@configclass\n"
        "class RewardsCfg:\n"
        "    secret_reward = RewTerm(func=mdp.hand_written, weight=1.0)\n"
        "\n"
        "@configclass\n"
        "class CurriculumCfg:\n"
        "    action_rate = CurrTerm(func=mdp.modify_reward_weight)\n"
    )
    manager.cfg["env_context"] = {
        "source_file": str(fake_env),
        "strip_classes": ["RewardsCfg", "CurriculumCfg"],
    }
    context = manager.load_env_context()
    check("env context keeps the observation spec", "joint_pos_rel" in context)
    check("env context strips the hand-written RewardsCfg",
          "secret_reward" not in context and "class RewardsCfg" not in context)
    check("env context strips CurriculumCfg", "class CurriculumCfg" not in context)

    # A host-side run cannot see the container path; degrade, do not crash.
    manager.cfg["env_context"] = {"source_file": "/nonexistent/reach_env_cfg.py"}
    check("a missing env source degrades to empty context",
          manager.load_env_context() == "")

    # --- system prompt ---
    system = manager.build_system_prompt()
    check("system prompt includes the reward signature example",
          "reward_dict = {" in system and "RewTerm(" in system)
    check("the signature example uses the real scene entity name",
          'SceneEntityCfg("robot"' in system)
    check("system prompt includes the code_output_tip rules", "(1)" in system)
    check("system prompt carries the weight= rule learned from a live failure",
          "MUST set a weight" in system)
    check("system prompt warns against the scene_entities kwarg",
          "scene_entities" in system)

    # --- initial user prompt ---
    manager.env_context = context
    user = manager.build_initial_user_prompt("franka-reach")
    check("user prompt embeds the environment source", "joint_pos_rel" in user)
    check("user prompt names the end-effector body", "panda_hand" in user)
    check("user prompt names the scene entity, not the task key",
          "'robot'" in user and "robot is 'franka-reach'" not in user)
    check("user prompt shows the SceneEntityCfg call to use",
          "SceneEntityCfg('robot'" in user)
    check("user prompt does not leak the hand-written reward",
          "secret_reward" not in user)

    # --- feedback prompt ---
    metrics = {
        "success_rate": 0.25,
        "mean_reward": 1.5,
        "reward_components": {"reach_position": [0.1, 0.2, 0.3]},
        "position_error": [0.3, 0.25, 0.2],
    }
    feedback = manager.build_feedback_prompt(metrics, stride=3)
    check("feedback uses the policy_feedback header",
          "We trained a RL policy" in feedback)
    check("feedback includes the per-component reflection",
          "reach_position" in feedback)
    check("feedback appends code_feedback",
          "carefully analyze" in feedback.lower())
    check("feedback repeats the formatting tips", "reward_dict" in feedback)

    # --- execution-error feedback ---
    err = manager._build_all_failed_feedback([
        {"metrics": {"traceback": "x\nTypeError: bad kwarg 'scene_entities'\n"}},
    ])
    check("all-failed feedback uses execution_error_feedback.txt",
          "has the following error" in err)
    check("all-failed feedback still names the real error",
          "scene_entities" in err)

    # An unreachable endpoint produced no code, so it must not be dressed up as
    # a code error and fed back as "fix the bug".
    err_llm = manager._build_all_failed_feedback([
        {"metrics": {"status": "llm_error", "error": "503 high demand"}},
    ])
    check("transport failures are not reported to the model as code errors",
          "503" not in err_llm, err_llm[:120])


# --------------------------------------------------------------- scenario E
def scenario_llm_retry():
    """Transient LLM failures must be retried, not treated as permanent."""
    print("\n--- Scenario E: LLM retry -----------------------------------------")
    designer = build_sandbox("retry")
    mod = load_module(designer)
    manager = mod.EurekaManager("cfg/reach.yaml")
    manager.system_prompt = "sys"
    mod.time.sleep = lambda *_: None

    class FakeAPIError(Exception):
        def __init__(self, code, message):
            super().__init__(f"{code} {message}")
            self.code = code
            self.message = message

    # Retry classification
    check("a 503 is retryable", mod.EurekaManager._is_retryable(FakeAPIError(503, "high demand")))
    check("a 429 is retryable", mod.EurekaManager._is_retryable(FakeAPIError(429, "rate limit")))
    check("a 500 is retryable", mod.EurekaManager._is_retryable(FakeAPIError(500, "internal")))
    check("a 403 is NOT retryable", not mod.EurekaManager._is_retryable(FakeAPIError(403, "denied")))
    check("a 400 is NOT retryable", not mod.EurekaManager._is_retryable(FakeAPIError(400, "bad request")))
    check("a codeless network error is retryable",
          mod.EurekaManager._is_retryable(TimeoutError("connection reset")))

    # Recovers when a transient failure clears: the exact case that killed a run.
    attempts = {"n": 0}

    class FlakyModels:
        def generate_content(self, model=None, contents=None, config=None):
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise FakeAPIError(503, "This model is currently experiencing high demand.")
            return types.SimpleNamespace(text="recovered")

    manager.client = types.SimpleNamespace(models=FlakyModels())
    text, error = manager._generate_candidate("prompt")
    check("a transient 503 is retried until it succeeds",
          text == "recovered" and error is None, f"text={text!r} error={error!r}")
    check("it took the expected number of attempts", attempts["n"] == 3, str(attempts["n"]))

    # A permanent error fails immediately rather than burning the backoff budget.
    tries = {"n": 0}

    class DeniedModels:
        def generate_content(self, model=None, contents=None, config=None):
            tries["n"] += 1
            raise FakeAPIError(403, "API key not valid")

    manager.client = types.SimpleNamespace(models=DeniedModels())
    text, error = manager._generate_candidate("prompt")
    check("an auth error is not retried", tries["n"] == 1, str(tries["n"]))
    check("an auth error reports the reason", error and "403" in error, str(error))

    # Exhausting the budget reports how many attempts were made.
    manager.cfg["eureka"]["llm_max_attempts"] = 2
    down = {"n": 0}

    class DownModels:
        def generate_content(self, model=None, contents=None, config=None):
            down["n"] += 1
            raise FakeAPIError(503, "high demand")

    manager.client = types.SimpleNamespace(models=DownModels())
    text, error = manager._generate_candidate("prompt")
    check("a persistent outage exhausts the configured attempts", down["n"] == 2, str(down["n"]))
    check("the exhaustion message says how many attempts were made",
          error and "2 attempts failed" in error, str(error))


# --------------------------------------------------------------- scenario F
def scenario_reflection():
    """Reward-reflection diagnostics, calibrated on a real failing run."""
    print("\n--- Scenario F: reward reflection ---------------------------------")
    designer = build_sandbox("reflection")
    mod = load_module(designer)
    manager = mod.EurekaManager("cfg/reach.yaml")
    manager.cfg["eureka"]["reflection_diagnostics"] = True

    # Verbatim final values from a live run whose total reward fell
    # monotonically while every old heuristic stayed silent.
    metrics = {
        "success_rate": 0.0205,
        "mean_reward": -0.7415,
        "reward_components": {
            "reaching_position": [0.0000001, 0.0005, 0.0031, 0.0086, 0.0172, 0.0184],
            "reaching_orientation": [0.0, 0.00001, 0.0016, 0.0087, 0.0236, 0.0002],
            "action_rate_penalty": [-0.0012, -0.0132, -0.0766, -0.1121, -0.1380, -0.1381],
            "position_success": [0.0, 0.0, 0.0, 0.0144, 0.0, 0.0],
            "orientation_success": [0.0, 0.0, 0.0, 0.0556, 0.0, 0.0],
        },
        "position_error": [0.227, 0.251, 0.281, 0.252, 0.224, 0.236],
    }
    reflection = manager.build_reward_reflection(metrics)

    check("the penalty/reward imbalance is reported",
          "PENALTIES OUTWEIGH REWARDS" in reflection, reflection[-400:])
    check("it names the largest penalty",
          "action_rate_penalty" in reflection)
    check("a 2.5x imbalance now trips DOMINATING (10x missed it)",
          "DOMINATING" in reflection)
    check("a bonus that stopped firing is reported",
          "NOT FIRING" in reflection, reflection[-400:])

    # A healthy reward must stay quiet, or the notes become noise.
    healthy = {
        "success_rate": 0.8,
        "mean_reward": 2.0,
        "reward_components": {
            "reaching_position": [0.5, 0.9, 1.4, 1.9, 2.3, 2.6],
            "action_rate_penalty": [-0.01, -0.02, -0.03, -0.04, -0.05, -0.06],
        },
        "position_error": [0.30, 0.20, 0.12, 0.07, 0.04, 0.02],
    }
    quiet = manager.build_reward_reflection(healthy)
    check("a healthy reward raises no imbalance note",
          "PENALTIES OUTWEIGH REWARDS" not in quiet)
    check("a healthy reward raises no NOT FIRING note",
          "NOT FIRING" not in quiet, quiet[-300:])


# --------------------------------------------------------------- scenario G
def scenario_faithful_reflection():
    """The DEFAULT reflection must match the original's format.

    The original supplies per-component trajectories and the task score, then
    lets the model diagnose. It has no automated notes, no unconditional
    "needs improvement", and does not report the reward when a task score
    exists -- a 5x16 run showed the model optimising mean_reward when it was
    displayed, inflating summed weights 6.0 -> 11.0 while success fell.
    """
    print("\n--- Scenario J: faithful reflection (diagnostics off) -------------")
    designer = build_sandbox("faithful")
    mod = load_module(designer)
    manager = mod.EurekaManager("cfg/reach.yaml")

    check("diagnostics default to off",
          manager.cfg["eureka"].get("reflection_diagnostics") is False,
          str(manager.cfg["eureka"].get("reflection_diagnostics")))

    metrics = {
        "success_rate": 0.7814,
        "mean_reward": 56.1474,
        "reward_components": {
            "reach_position_coarse": [0.005, 0.25, 0.45, 0.58, 0.68, 0.71],
            "action_rate": [-0.003, -0.06, -0.04, -0.03, -0.025, -0.024],
            "target_success": [0.0, 0.0005, 0.18, 0.63, 1.52, 1.58],
        },
        "position_error": [0.2419, 0.1409, 0.1019, 0.0691, 0.0538, 0.0488],
    }
    reflection = manager.build_reward_reflection(metrics)

    check("per-component trajectories are present",
          "reach_position_coarse:" in reflection and "Max:" in reflection)
    check("the task metric is reported", "position_error:" in reflection)
    check("the fitness function is reported as the task score",
          "task score: 0.7814" in reflection, reflection[-120:])

    check("mean_reward is NOT displayed as a headline",
          "56.1474" not in reflection and "mean_reward" not in reflection,
          reflection[:200])
    check("there is no unconditional 'needs improvement'",
          "needs improvement" not in reflection)
    for note in ("SATURATED", "DOMINATING", "NOT FIRING", "FLIPPED SIGN",
                 "PENALTIES OUTWEIGH"):
        check(f"no {note} note by default", note not in reflection)

    # Flipping the switch restores them, so the A/B is one config change.
    manager.cfg["eureka"]["reflection_diagnostics"] = True
    with_notes = manager.build_reward_reflection(metrics)
    check("the flag restores the diagnostics",
          "needs improvement" in with_notes)
    check("the trajectories survive with the flag on",
          "reach_position_coarse:" in with_notes)


def scenario_config_drift():
    """reach_smoke.yaml must stay a pure overlay on reach.yaml.

    It is generated by sed from reach.yaml, and it silently fell behind: the
    whole env_context block was missing, so smoke runs prompted with no
    environment source and the scene-entity allowlist was empty. Nothing failed
    loudly -- the feature was just absent.
    """
    print("\n--- Scenario G: config drift --------------------------------------")
    import yaml as _yaml

    full = _yaml.safe_load((REAL_SKILL / "cfg/reach.yaml").read_text())
    smoke = _yaml.safe_load((REAL_SKILL / "cfg/reach_smoke.yaml").read_text())

    check("the smoke config has every top-level key",
          set(full) == set(smoke),
          f"missing={sorted(set(full) - set(smoke))} extra={sorted(set(smoke) - set(full))}")

    check("the smoke config has every eureka key",
          set(full["eureka"]) == set(smoke["eureka"]),
          f"missing={sorted(set(full['eureka']) - set(smoke['eureka']))}")

    check("the smoke config keeps env_context", "env_context" in smoke)
    if "env_context" in smoke and "env_context" in full:
        check("env_context is complete",
              set(full["env_context"]) == set(smoke["env_context"]),
              f"missing={sorted(set(full['env_context']) - set(smoke['env_context']))}")

    # The overrides that are supposed to differ, and nothing else.
    expected_overrides = {
        "num_runs", "iterations", "num_samples", "eval_num_envs",
        "train_iterations", "final_train_iterations",
    }
    differing = {k for k in full["eureka"]
                 if full["eureka"].get(k) != smoke["eureka"].get(k)}
    check("only the intended eureka values are overridden",
          differing <= expected_overrides, f"unexpected={sorted(differing - expected_overrides)}")

    # Every other nested block must match exactly. A sed for `num_samples`
    # silently hit dr_eureka as well as eureka, which this would have caught.
    for block in sorted(set(full) & set(smoke)):
        if block in ("eureka", "dr_eureka") or not isinstance(full[block], dict):
            continue
        check(f"block {block!r} is untouched", full[block] == smoke[block],
              f"differs: {block}")

    check("dr_eureka keeps every key",
          set(full.get("dr_eureka", {})) == set(smoke.get("dr_eureka", {})))
    dr_differing = {k for k in full.get("dr_eureka", {})
                    if full["dr_eureka"].get(k) != smoke["dr_eureka"].get(k)}
    check("dr_eureka differs only in num_samples",
          dr_differing <= {"num_samples"}, f"unexpected={sorted(dr_differing)}")

    # The generator must reproduce the committed file byte for byte, so a stale
    # checked-in smoke config is a test failure rather than a silent surprise.
    import subprocess
    gen = REAL_SKILL / "cfg/make_smoke_config.py"
    check("the smoke config generator exists", gen.is_file())
    if gen.is_file():
        before = (REAL_SKILL / "cfg/reach_smoke.yaml").read_text()
        subprocess.run([sys.executable, str(gen)], capture_output=True, cwd=REAL_SKILL)
        after = (REAL_SKILL / "cfg/reach_smoke.yaml").read_text()
        check("the committed smoke config matches what the generator produces",
              before == after, "regenerate with: python3 cfg/make_smoke_config.py")

    check("the smoke run writes to outputs/smoke/",
          all("outputs/smoke/" in str(smoke[k])
              for k in ("reward_output_file", "policy_output_file", "final_metrics_file")))
    check("the smoke config cannot clobber the real policy",
          smoke["policy_output_file"] != full["policy_output_file"])


# --------------------------------------------------------------- scenario H
def scenario_sign_and_correlation():
    """Sign-convention diagnostic, and the paper's reward-correlation metric."""
    print("\n--- Scenario H: sign convention + correlation ---------------------")
    designer = build_sandbox("sign")
    mod = load_module(designer)
    manager = mod.EurekaManager("cfg/reach.yaml")
    manager.cfg["eureka"]["reflection_diagnostics"] = True

    # Observed live: a term returning a negative magnitude AND carrying a
    # negative weight. Isaac Lab logs the WEIGHTED value, so the double negative
    # shows up as a large POSITIVE contribution -- the policy was paid to thrash
    # its joints, scoring 82.67 mean reward while task success collapsed.
    flipped = {
        "success_rate": 0.004,
        "mean_reward": 82.67,
        "reward_components": {
            "reaching_position": [0.01, 0.05, 0.10, 0.13, 0.15, 0.16],
            "action_rate_penalty": [0.5, 2.0, 5.0, 8.0, 9.5, 9.99],
        },
        "position_error": [0.24, 0.28, 0.31, 0.33, 0.35, 0.357],
    }
    note = manager.build_reward_reflection(flipped)
    check("a penalty with a positive contribution is flagged",
          "FLIPPED SIGN" in note, note[-400:])
    check("the sign note names the offending term",
          "action_rate_penalty" in note)

    # A correctly signed penalty must stay quiet.
    correct = {
        "success_rate": 0.4,
        "mean_reward": 2.0,
        "reward_components": {
            "reaching_position": [0.5, 1.0, 1.5, 2.0, 2.4, 2.7],
            "action_rate_penalty": [-0.01, -0.02, -0.03, -0.04, -0.05, -0.06],
        },
        "position_error": [0.30, 0.22, 0.15, 0.10, 0.06, 0.03],
    }
    quiet = manager.build_reward_reflection(correct)
    check("a correctly signed penalty is not flagged",
          "FLIPPED SIGN" not in quiet, quiet[-300:])

    # A reward-named term may legitimately be negative (Isaac Lab's own position
    # tracking is), so the check must only fire in the penalty direction.
    negative_reward_term = {
        "success_rate": 0.4,
        "mean_reward": 1.0,
        "reward_components": {
            "position_tracking": [-0.5, -0.4, -0.3, -0.2, -0.15, -0.1],
            "fine_grained": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        },
        "position_error": [0.30, 0.22, 0.15, 0.10, 0.06, 0.03],
    }
    quiet2 = manager.build_reward_reflection(negative_reward_term)
    check("a negative reward-named term is not flagged as sign-flipped",
          "FLIPPED SIGN" not in quiet2)

    # --- correlation (paper's headline evaluation metric) ---
    spec = importlib.util.spec_from_file_location(
        "eval_headless_corr", designer / "scripts/eval_headless.py")
    source = (designer / "scripts/eval_headless.py").read_text()
    namespace = {"math": __import__("math")}
    start = source.index("def reward_correlation(")
    end = source.index("def mean_episode_infos(")
    exec(compile(source[start:end], "reward_correlation", "exec"), namespace)
    corr = namespace["reward_correlation"]

    check("perfectly aligned rewards correlate at +1",
          abs(corr([1, 2, 3, 4], [2, 4, 6, 8]) - 1.0) < 1e-9)
    check("opposed rewards correlate at -1",
          abs(corr([1, 2, 3, 4], [8, 6, 4, 2]) + 1.0) < 1e-9)
    check("a constant series yields no correlation",
          corr([1, 1, 1, 1], [1, 2, 3, 4]) is None)
    check("too few samples yields no correlation", corr([1.0], [2.0]) is None)
    check("mismatched lengths yield no correlation",
          corr([1, 2, 3], [1, 2]) is None)
    check("non-finite values yield no correlation",
          corr([1, 2, float("nan")], [1, 2, 3]) is None)

    # The ground-truth baseline silently collapsed to its two penalty terms
    # because RewardsCfg leaves body_names as dataclasses.MISSING, and
    # copy.deepcopy makes a NEW _MISSING_TYPE so an `is MISSING` check fails.
    # Correlation was then being measured against the wrong reward entirely.
    import copy as _copy
    from dataclasses import MISSING as _MISSING
    ns = {}
    src = (designer / "scripts/eval_headless.py").read_text()
    start = src.index("def needs_body_names(")
    end = src.index("def make_ground_truth_reward_fn(")
    exec(compile(src[start:end], "needs_body_names", "exec"),
         {"MISSING": _MISSING}, ns)
    needs_body_names = ns["needs_body_names"]

    check("an unset body_names is detected", needs_body_names(_MISSING))
    check("a DEEPCOPIED unset body_names is still detected",
          needs_body_names(_copy.deepcopy(_MISSING)),
          "identity check would fail here; must be by type")
    check("a real body_names list is left alone",
          not needs_body_names(["panda_hand"]))
    check("None body_names is left alone", not needs_body_names(None))


# --------------------------------------------------------------- scenario I
def scenario_restarts():
    """Random restarts (3.2): independent searches, best kept."""
    print("\n--- Scenario I: random restarts -----------------------------------")
    designer = build_sandbox("restarts")

    # build_sandbox has already pinned num_runs to 1; raise it by key so this
    # does not depend on whatever value reach.yaml happens to carry.
    cfg_path = designer / "cfg/reach.yaml"
    cfg, n = re.subn(r"^(\s+)num_runs: \d+.*$", r"\g<1>num_runs: 3",
                     cfg_path.read_text(), count=1, flags=re.MULTILINE)
    assert n == 1, "num_runs override matched nothing"
    cfg_path.write_text(cfg)

    mod = load_module(designer)

    # Run 2 is the strongest; runs 1 and 3 are worse. The best must survive both
    # the later run and the restart boundary.
    per_run = {0: 0.20, 1: 0.60, 2: 0.10}
    conversations = []

    def fake_search(self, task, run_index=0):
        conversations.append(run_index)
        score = per_run[run_index]
        return (
            {"success_rate": score, "mean_reward": score,
             "reward_correlation": 0.5, "status": "success"},
            designer / f"outputs/candidates/run{run_index}_reward.py",
            0.5,
        )

    mod.EurekaManager.run_search = fake_search

    class FakeProc:
        def wait(self, timeout=None):
            (designer / "outputs/final_metrics.json").write_text(
                json.dumps({"mean_reward": 1.0, "status": "success"}))
            return 0
    mod.subprocess = types.SimpleNamespace(
        Popen=lambda *a, **k: FakeProc(),
        TimeoutExpired=subprocess.TimeoutExpired,
        STDOUT=subprocess.STDOUT,
    )

    manager = mod.EurekaManager("cfg/reach.yaml")
    for i in range(3):
        path = designer / f"outputs/candidates/run{i}_reward.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# run {i}\n")

    manager.main_loop("franka-reach")

    check("every restart runs", len(conversations) == 3, str(len(conversations)))

    final = (designer / "outputs/reward_fn.py")
    check("the best restart's reward is the one kept",
          final.is_file() and "run 1" in final.read_text(),
          final.read_text() if final.is_file() else "missing")

    # A restart must begin from a fresh conversation, not inherit the previous
    # search's history -- otherwise it is a continuation, not a restart.
    designer2 = build_sandbox("restart_isolation")
    mod2 = load_module(designer2)
    manager2 = mod2.EurekaManager("cfg/reach.yaml")
    manager2.env_context = ""
    manager2.reset_conversation("franka-reach")
    manager2.record_exchange("some code", "some feedback")
    check("a search leaves 3 turns behind", len(manager2.messages) == 3)
    manager2.reset_conversation("franka-reach")
    check("the next restart starts from 1 turn again",
          len(manager2.messages) == 1, str(len(manager2.messages)))


def main():
    stub_external_modules()
    os.environ.setdefault("GEMINI_API_KEY", "test-key-not-used")

    print("=" * 70)
    mod_a = None
    for name, fn in [
        ("A/regression", scenario_regression),
        ("B/all-fail", lambda: scenario_all_fail(mod_a)),
        ("C/injection", scenario_injection),
        ("D/prompts", scenario_prompts),
        ("E/llm-retry", scenario_llm_retry),
        ("F/reflection", scenario_reflection),
        ("J/faithful-reflection", scenario_faithful_reflection),
        ("G/config-drift", scenario_config_drift),
        ("H/sign+correlation", scenario_sign_and_correlation),
        ("I/restarts", scenario_restarts),
    ]:
        try:
            result = fn()
            if name.startswith("A/"):
                mod_a = result
        except Exception as exc:
            import traceback as _tb
            _tb.print_exc()
            FAILURES.append(f"{name} raised {type(exc).__name__}: {exc}")
    print("=" * 70)

    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
