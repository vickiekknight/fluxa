# fluxa
Despite their promise for scalable robot skill acquisition, sim-to-real approaches remain slow and expert-dependent, requiring manual workspace boundary specification, manipulation task set up, reward function design, and deep familiarity with complex simulation platforms. In this paper, we present FLUXA, an agentic system that takes a natural language prompt and independently orchestrates the full sim-to-real pipeline in Isaac Lab. FLUXA uses Agent Skills that encapsulate Isaac Lab API knowledge, task-specific examples, and each pipeline component to dynamically generate reward functions and simulation configurations. We first demonstrate that FLUXA produces a policy that outperforms standard simulation-based training approaches. Then, we showcase that FLUXA is capable of solving novel robot tasks and handling new embodiments not included in the agent skill examples.

Branch version 3 is the most recent version.
Youtube video is from one of the initial versions.

[![Watch the demo](https://img.youtube.com/vi/JnMf_Rfo858/0.jpg)](https://www.youtube.com/watch?v=JnMf_Rfo858)


