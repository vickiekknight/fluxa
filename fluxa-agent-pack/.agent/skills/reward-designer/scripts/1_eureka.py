import os
import json
import yaml
import asyncio
import websockets
import re
from google import genai
from pathlib import Path
import wandb

class EurekaManager:
    def __init__(self, config_path):
        # Load Config
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
        
        # Initialize Gemini
        self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        self.model_name = self.cfg['eureka'].get('model', 'gemini-2.5-flash-lite')

        self.designer_root = Path(self.cfg['designer_root']).expanduser()

        # Initialize W&B Project
        wandb.init(
            project="Fluxa-Reward-Designer",
            name=f"eureka-{self.cfg['task_name']}",
            config=self.cfg  # Logs your reach.yaml settings automatically
        )

        # Load eval template
        eval_template_path = self.designer_root / "templates" / "eval_template.py"
        with open(eval_template_path, 'r') as f:
            self.eval_code_template = f.read()
        
    def build_prompt(self, task_name, iteration=0, feedback=""):
        """Combines YAML info and Signature into a prompt for Gemini."""
        robot_cfg = self.cfg['robots'][task_name]
        
        with open(self.designer_root / self.cfg['reward_signature_file'], 'r') as f:
            signature = f.read()
            
        prompt = f"""
        Task Context: {self.cfg['task_description']}
        Target Robot: {task_name}
        End-Effector Body Name: {robot_cfg['ee_body_name']}
        
        Formatting Rules:
        {signature}
        
        Instructions:
        Provide your response as a single Python code block.
        1. Write the reward functions (one per term).
        2. Define a dictionary called 'reward_dict' that contains 'RewTerm' objects 
           pointing to those functions.
        
        IMPORTANT: Do not include imports. Use the standard Isaac Lab MDP naming conventions.
        """
        
        if iteration > 0:
            prompt += f"\n\nOptimization Feedback from Previous Run:\n{feedback}\nModify the weights or logic to improve performance."
        
        return prompt

    def inject_code(self, raw_llm_output):
        """Extracts code from Gemini's response and updates the template."""

        # 1. Find the python block
        match = re.search(r"```python\n(.*?)\n```", raw_llm_output, re.DOTALL)
        if not match:
            print("❌ Error: LLM did not return a valid Python block.")
            return None

        full_code = match.group(1)

        # Sanitize function names — remove spaces that Gemini sometimes inserts
        # e.g. "reward_lin_vel_ Z_align" → "reward_lin_vel_Z_align"
        full_code = re.sub(r'(def\s+\w+)\s+(\w+)', r'\1\2', full_code)

        # 2. Split functions from reward_dict block
        lines = full_code.split('\n')
        dict_start_idx = next(
            (i for i, l in enumerate(lines) if l.strip().startswith('reward_dict')),
            None
        )
        if dict_start_idx is None:
            print("❌ Error: 'reward_dict' not found in LLM output.")
            return None

        functions = "\n".join(lines[:dict_start_idx])
        dictionary_block = "\n".join(lines[dict_start_idx:])

        # 3. Load template (must happen before inject)
        template_path = self.designer_root / "templates" / "reward_template.py"
        with open(template_path, 'r') as f:
            template = f.read()

        # 4. Inject into template
        final_code = template.replace("# INSERT_REWARD_FUNCTIONS_HERE", functions)
        final_code = final_code.replace("# INSERT_REWARD_DICTIONARY_HERE", dictionary_block)

        # 5. Save to output
        output_path = self.designer_root / self.cfg['reward_output_file']
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(final_code)

        return output_path

    async def run_evaluation(self, task_name):
        """Send generated reward fn to Isaac Sim, run short rollout, collect metrics."""

        # Read generated reward function
        reward_fn_path = self.designer_root / self.cfg['reward_output_file']
        with open(reward_fn_path, 'r') as f:
            reward_code = f.read()

        # Config
        isaac_host    = self.cfg.get('isaac_sim', {}).get('host', 'localhost')
        isaac_port    = self.cfg.get('isaac_sim', {}).get('port', 8765)
        results_port  = self.cfg.get('isaac_sim', {}).get('results_port', 8766)
        num_envs      = self.cfg['eureka'].get('eval_num_envs', 8)
        rollout_dur   = self.cfg['eureka'].get('rollout_duration', 30)
        host_ip = self.cfg.get('host_ip', '172.17.0.1')  # default Docker bridge gateway

        # Build eval code with reward injected
        eval_code = self.eval_code_template
        eval_code = eval_code.replace("FLUXA_REWARD_CODE", reward_code)
        eval_code = eval_code.replace("FLUXA_NUM_ENVS", str(num_envs))
        eval_code = eval_code.replace("FLUXA_ROLLOUT_DURATION", str(rollout_dur))
        eval_code = eval_code.replace("FLUXA_HOST_IP", host_ip)
        eval_code = eval_code.replace("FLUXA_RESULTS_PORT", str(results_port))

        # Plain TCP server to receive metrics (avoids WebSocket/asyncio conflicts in Isaac Sim)
        metrics_future = asyncio.get_event_loop().create_future()

        async def tcp_metrics_server():
            server = await asyncio.start_server(
                lambda r, w: handle_metrics(r, w, metrics_future),
                "0.0.0.0", results_port
            )
            return server

        async def handle_metrics(reader, writer, future):
            data = await reader.read(4096)
            writer.close()
            if not future.done():
                future.set_result(json.loads(data.decode("utf-8")))

        server = await tcp_metrics_server()
        print(f"📡 Listening for metrics on port {results_port}...")

        try:
            uri = f"ws://{isaac_host}:{isaac_port}"
            async with websockets.connect(uri) as ws:
                await ws.send(json.dumps({"type": "execute_python", "code": eval_code}))
                response = json.loads(await ws.recv())
                print(f"Isaac Sim: {response.get('message', response)}")
            break
        except (ConnectionRefusedError, websockets.exceptions.WebSocketException) as e:
            if attempt < 2:
                print(f"⚠️  Connection attempt {attempt+1} failed, retrying in 5s...")
                await asyncio.sleep(5)
            else:
                raise
                
            timeout = rollout_dur + 60
            metrics = await asyncio.wait_for(metrics_future, timeout=timeout)
            return metrics

        except asyncio.TimeoutError:
            print("⚠️  Evaluation timed out — Isaac Sim did not send metrics back.")
            return {"mean_reward": 0.0, "success_rate": 0.0}
        except Exception as e:
            print(f"❌ Evaluation error: {e}")
            return {"mean_reward": 0.0, "success_rate": 0.0}
        finally:
            server.close()
            await server.wait_closed()
        
    async def main_loop(self, task):
        print(f"Starting Fluxa Stage 1 (Eureka) for {task}...")
        
        for i in range(self.cfg['eureka']['iterations']):
            print(f"\n--- [Iteration {i+1}/{self.cfg['eureka']['iterations']}] ---")
            
            # 1. Generate Reward Code
            prompt = self.build_prompt(task, iteration=i)
            print("Querying Gemini...")
            await asyncio.sleep(4)
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            
            # 2. Inject into Template
            output_file = self.inject_code(response.text)
            if output_file:
                print(f"Saved generated reward to: {output_file}")
            
            # 3. Evaluate in Isaac Sim
            metrics = await self.run_evaluation(task)
            print(f"Results: {metrics}")

            # --- W&B LOGGING ---
            wandb.log({
                "iteration": i,
                "success_rate": metrics["success_rate"],
                "mean_reward": metrics["mean_reward"],
                # Log the actual code as a text artifact for comparison
                "generated_code": wandb.Html(f"<pre><code>{response.text}</code></pre>"),
            })
            # -------------------
        
        wandb.finish()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="franka-reach")
    args = parser.parse_args()
    
    # Ensure you are in the correct directory when running
    manager = EurekaManager("cfg/reach.yaml")
    asyncio.run(manager.main_loop(args.task))