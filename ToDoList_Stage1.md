You are an expert reinforcement learning and generative modeling engineer.

GOAL
-----
Implement a small, clean PyTorch codebase to run TOY experiments that compare the following classes of algorithms on simple continuous-control / low-dimensional tasks (e.g., Pendulum, CartPole-Continuous, 2D point-mass navigation, simple manipulation-like 2D tasks):

1. Diffusion Policy (imitation / offline RL style)
2. Flow Matching Policy
   - Including variants inspired by Reflected Flow and Consistency Flow
3. Diffusion Q-Learning with Double-Q style critics ("Diffusion Double Q Learning" in this project)
4. Consistency Policy Q-Learning (CPQL)
5. DPPO – Diffusion Policy Policy Optimization
6. ReinFlow – Online RL finetuning for flow-matching policies

The goal is NOT to perfectly reproduce large robot systems, but to:
- Implement **minimal, faithful toy versions** of these algorithms
- Reuse ideas, loss functions, and training recipes from the official repos and papers listed below
- Run them on simple environment(s) under a UNIFIED interface for fair comparison


REFERENCE REPOSITORIES & PAPERS (MUST READ)
-------------------------------------------
Use these as the primary references for algorithm design, losses, and training tricks.

[Diffusion Policy]
- Paper/project: Diffusion Policy: Visuomotor Policy Learning via Action Diffusion
- Official code:
  https://github.com/real-stanford/diffusion_policy

[Flow Matching Policies for Robotics]
- Affordance-based Robot Manipulation with Flow Matching (FM policy)
  Code:  https://github.com/HRI-EU/flow_matching
  Helper wrapper repo: https://github.com/HRI-EU/flow-matching-policy
- Generative Predictive Control: Flow Matching Policies for Dynamic Tasks
  (Good reference for flow policies in control/RL)
  https://github.com/vincekurtz/gpc

[Reflected Flow & Consistency Flow (for the ODE/velocity-field side)]
- Generic rectified / consistency flow implementation:
  https://github.com/lucidrains/rectified-flow-pytorch
- Consistency-Flow-based manipulation:
  ManiFlow – A General Robot Manipulation Policy via Consistency Flow Training
  https://github.com/allenai/maniflow

[Diffusion Q-Learning & Double-Q Style Critic]
- Diffusion Q-Learning (Diffusion-QL, offline RL with diffusion actor):
  https://github.com/Zhendong-Wang/Diffusion-Policies-for-Offline-RL
- Implicit Diffusion Q-Learning (IDQL, diffusion actor + IQL critic):
  https://github.com/philippe-eecs/IDQL

For this project, "Diffusion Double Q Learning" should be implemented as:
- A Diffusion-QL style actor
- Combined with **two Q networks** (Double-Q / TD3-style) for target computation

[Consistency Policy Q-Learning (CPQL)]
- Official code:
  https://github.com/cccedric/cpql

[DPPO: Diffusion Policy Policy Optimization]
- Project page:
  https://diffusion-ppo.github.io/
- Official implementation:
  https://github.com/irom-princeton/dppo

[ReinFlow: Fine-tuning Flow Matching Policies with Online RL]
- Website:
  https://reinflow.github.io/
- Code:
  https://github.com/ReinFlow/ReinFlow

[Extra (optional) reference for online diffusion RL]
- Efficient Online Reinforcement Learning for Diffusion Policy
  (Diffusion Policy Online RL, DPMD & SDAC)
  https://github.com/mahaitongdae/diffusion_policy_online_rl


GLOBAL REQUIREMENTS
-------------------
1. Framework & Environment
   - Python 3.10+ 
   - PyTorch (no JAX)
   - gymnasium or classic OpenAI gym-like API
   - Use CPU by default; allow easy switch to CUDA if available

2. Code Structure
   Create a small but well-organized package, for example:

   toy_diffusion_rl/
     envs/
       __init__.py
       point_mass_2d.py
       pendulum_continuous_wrapper.py
     common/
       replay_buffer.py
       networks.py      # MLP backbones, time embeddings, etc.
       utils.py
     algorithms/
       diffusion_policy/
         agent.py
         trainer.py
       flow_matching/
         base_flow.py
         fm_policy.py           # vanilla flow matching policy
         reflected_flow.py      # variant with boundary reflection (if feasible)
         consistency_flow.py    # consistency-flow-style update
       diffusion_double_q/
         agent.py               # diffusion actor + double Q critics
       cpql/
         agent.py
       dppo/
         agent.py
       reinflow/
         agent.py
     configs/
       base.yaml
       diffusion_policy_pendulum.yaml
       flow_matching_pendulum.yaml
       diffusion_double_q_pendulum.yaml
       cpql_pendulum.yaml
       dppo_pendulum.yaml
       reinflow_pendulum.yaml
     train.py
     eval.py
     README.md

3. Toy Environments
   - Implement at least ONE simple continuous-control toy env with **low-dimensional state and action**, e.g.:

     (A) 2D Point Mass:
         - State: (x, y, vx, vy)
         - Action: (ax, ay), bounded in [-1, 1]
         - Reward: negative distance to goal, small action penalty

     (B) (Optional) Gymnasium Pendulum-v1 continuous environment, wrapped for convenience.

   - Make sure all algorithms can be run on the same env:
     - Same observation and action dimensions
     - Same reward definition and episode length

4. Shared Components
   - A replay buffer that stores (s, a, r, s', done)
   - Common MLP modules:
     - For policies: MLP(s, t, noise-level) -> action or noise prediction
     - For Q networks: MLP(s, a) -> scalar Q
   - Common training loop logic (seed, logging, evaluation every N steps)


ALGORITHM-SPECIFIC IMPLEMENTATION DETAILS
-----------------------------------------

[1] Diffusion Policy (offline / imitation-style on toy env)
----------------------------------------------------------
Task:
- Implement a minimal 1D/2D action Diffusion Policy for behavior cloning / offline RL on the toy env.

Requirements:
- Follow the main design of real-stanford/diffusion_policy but simplify:
  - 1D temporal horizon (a_t only, or small action sequence length)
  - Use a simple MLP-based UNet-like noise predictor with time embedding
- Training:
  - Offline dataset collected from a simple expert (e.g., LQR-like controller or hand-coded policy)
  - Denoising diffusion loss in action space
- Inference:
  - Iterative denoising to produce an action given current state

Deliverables:
- algorithms/diffusion_policy/agent.py with:
  - class DiffusionPolicyAgent:
      - sample_action(state) -> action
      - train_step(batch) -> dict of losses
- A config file for training on the toy env.


[2] Flow Matching Policy (including reflected & consistency-inspired variants)
-------------------------------------------------------------------------------
Task:
- Implement flow-matching policies that learn a velocity field v_theta(x, t | s) over actions,
  inspired by:
  - HRI-EU flow_matching (robot manipulation)
  - rectified-flow-pytorch (rectified / consistency flows)
  - ManiFlow (consistency flow training)

Requirements:
- Base formulation:
  - Sample initial noisy action a_0 ~ N(0, I)
  - Integrate ODE da/dt = v_theta(a_t, t, s) from t=0 to t=1
  - Use supervised flow matching loss to match expert actions
- Implement:

  (A) Vanilla Flow Matching Policy:
      - v_theta: MLP(s, a_t, t)
      - Train with standard flow matching loss

  (B) Reflected Flow variant (simplified):
      - Implement a simple "reflection" at action bounds:
        if action goes out of [-1, 1], reflect it back (or clamp with gradient)
      - This is just a toy version for bounded actions

  (C) Consistency Flow variant:
      - Implement a single-step or few-step consistency-style update:
        - Parameterize a velocity field and enforce consistency between
          different time steps (e.g., t and t', as in consistency flow ideas)

Deliverables:
- algorithms/flow_matching/fm_policy.py
- algorithms/flow_matching/reflected_flow.py
- algorithms/flow_matching/consistency_flow.py
- Unified interface:
  - class FlowMatchingPolicyBase with sample_action(state) and train_step(batch)


[3] "Diffusion Double Q Learning" (Diffusion-QL + Double Q)
------------------------------------------------------------
Interpretation for this project:
- Use Diffusion-QL style actor (diffusion policy) and combine with **two Q networks** (Double-Q)
  for target estimation and policy improvement.

Base references:
- Diffusion Q-Learning (Diffusion-QL):
  https://github.com/Zhendong-Wang/Diffusion-Policies-for-Offline-RL
- IDQL:
  https://github.com/philippe-eecs/IDQL

Requirements:
- Actor:
  - A diffusion policy over actions a given state s
  - Can use the same diffusion backbone as in [1] but trained with Q-enhanced objective
- Critic:
  - Two Q networks Q1(s, a), Q2(s, a)
  - Target Q = min(Q1', Q2') similar to TD3-style clipped double Q-learning
- Training:
  - Critic trained via standard Bellman backup using replay buffer
  - Actor loss:
    - Diffusion-QL style: denoising loss + term that encourages higher Q-values of generated actions
- For toy experiments, keep the number of diffusion steps small (e.g., 5-10) to keep it fast.

Deliverables:
- algorithms/diffusion_double_q/agent.py
  - class DiffusionDoubleQAgent with:
    - sample_action(state)
    - train_step(batch) returning critic_loss, actor_loss, etc.


[4] Consistency Policy Q-Learning (CPQL)
----------------------------------------
Base repo:
- https://github.com/cccedric/cpql

Requirements:
- Implement a simplified version of CPQL on the toy env:
  - Policy: consistency model over actions (single-step denoising / consistency update)
  - Critic: standard Q(s, a) network
  - Use CPQL-style policy objective that enforces consistency while maximizing Q.

Deliverables:
- algorithms/cpql/agent.py
  - class CPQLAgent:
      - sample_action(state)
      - train_step(batch)


[5] DPPO – Diffusion Policy Policy Optimization
-----------------------------------------------
References:
- Project: https://diffusion-ppo.github.io/
- Code:   https://github.com/irom-princeton/dppo

Task:
- Implement a toy version of DPPO on the simple env, reusing our diffusion policy backbone from [1].

Requirements:
- Two-layer MDP interpretation is conceptual; for toy code:
  - Treat each denoising step as part of the policy and compute log-probabilities of the final action
  - Use PPO-style clipped policy gradient objective on the diffusion policy
- Needed components:
  - Old policy and new policy (for PPO ratios)
  - Value function V(s) network
  - Advantage estimator (e.g., GAE)

Deliverables:
- algorithms/dppo/agent.py
  - class DPPOAgent:
      - collect_rollout(env) -> trajectories
      - update(trajectories) -> dict of losses


[6] ReinFlow – Flow Matching + Online RL Fine-tuning
----------------------------------------------------
References:
- Website: https://reinflow.github.io/
- Code:   https://github.com/ReinFlow/ReinFlow

Task:
- On the toy env, implement a minimal ReinFlow-style algorithm:
  - Start from a flow-matching policy trained offline (as in [2])
  - Add learnable noise / stochasticity to the ODE path so that:
    - The resulting policy becomes a discrete-time Markov process with tractable likelihood
  - Apply a policy-gradient-style RL update to fine-tune the flow policy online.

Simplifications allowed:
- Use a simple Gaussian noise injection to the velocity field or actions
- Compute log-probabilities using the approximated flow/ODE discretization
- Use a basic actor-critic objective instead of the full generality of the paper,
  as long as it respects the ReinFlow idea: **fine-tuning a flow policy with RL**.

Deliverables:
- algorithms/reinflow/agent.py
  - class ReinFlowAgent:
      - offline_pretrain(dataset)
      - online_update(rollouts)


TRAINING SCRIPTS & EXPERIMENTS
------------------------------
Implement:

1) train.py
   - CLI arguments: algorithm name, env name, config file path, seed, total steps, etc.
   - Loads config, creates env and agent, runs training loop:
     - Collect experience
     - Update agent
     - Periodically evaluate average return over several episodes
     - Log to stdout and optionally to TensorBoard

2) eval.py
   - Loads a trained checkpoint and runs evaluation episodes
   - Outputs average return and optionally saves a simple plot of rewards vs. time

3) Example configs
   - configs/diffusion_policy_pendulum.yaml
   - configs/flow_matching_pendulum.yaml
   - configs/diffusion_double_q_pendulum.yaml
   - configs/cpql_pendulum.yaml
   - configs/dppo_pendulum.yaml
   - configs/reinflow_pendulum.yaml

Each config should at least specify:
   - env name and parameters
   - network hidden sizes
   - learning rates
   - batch size
   - discount factor
   - diffusion / flow specific hyperparameters (num steps, noise schedule, etc.)

Please write idiomatic, well-structured PyTorch code with clear comments and docstrings.
Focus on correctness, clarity, and making it easy to modify hyperparameters for further research.
