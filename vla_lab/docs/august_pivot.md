# Project Proposal: Carryover-Aware Supervisory Control 
**A VLA-Driven Approach to De-biasing Human Intent in Shared Autonomy**

**Author:** Kye Farhadi

## 1. Core Scientific Paradigm
The fundamental scientific question adapts the original carryover problem[cite: 1] to supervisory shared autonomy: *Can a Vision-Language-Action (VLA) model maintain a latent semantic memory of its own strategic interventions (carryover) to dynamically de-bias human instructions, thereby extracting the user's true, unprompted intent during robotic manipulation?*

In this setup, the Kinova Gen2 faces a tabletop workspace populated with objects requiring distinct manipulation strategies (e.g., top-down grasps vs. lateral grasps, or clearing obstacles vs. navigating around them). The human sits behind the robot, acting as a supervisor who issues sparse linguistic commands or waypoint corrections. The VLA performs end-to-end optimization of belief and policy learning, continuously estimating whether the human's commands represent their genuine strategy or merely cognitive compliance with the robot's recent behavior.

---

## 2. The Physical Task: Supervisory Fetch & Grasp
The human has no physical contact with the robot. The interaction loop is entirely visual and linguistic.

*   **The Setup:** The robot faces a cluttered manipulation workspace. The user observes the workspace from behind the robot.
*   **The Intervention (COACH):** The VLA intentionally biases the user's spatial reasoning. For example, over three consecutive trials, the VLA autonomously executes a highly specific strategy (e.g., clearing all obstacles before fetching a target) and narrates its action: *"Clearing path to ensure safe grasp."*
*   **The Measurement (ASSESS):** The robot encounters a new, ambiguous scene where multiple strategies are equally valid. It pauses and asks the human: *"How should I approach the target?"*
*   **The Carryover:** The user's linguistic instruction is mathematically likely to mimic the robot's recent strategy (compliance). The VLA must use its memory of the COACH trials to de-bias the user's response, determining if they actually prefer the obstacle-clearing strategy or were just primed to say it.

---

## 3. System Architecture: The Carryover-Aware VLA
The model operates as a unified VLA architecture that closes the loop between visual observation, contextual memory, and low-level kinematic control.

### 3.1 Multimodal Inputs
*   **Visual Stream (Observation):** Continuous RGB-D feeds from the Kinova's arm-mounted camera and an external workspace camera, capturing the object geometry and clutter distribution.
*   **Proprioception (State):** Current Kinova joint states, end-effector poses, and gripper status streamed via ROS 2.
*   **Human Instruction (Linguistic):** Sparse text or transcribed voice commands from the user (e.g., *"Move the blue block first"*).
*   **Carryover Memory Token (The Novelty):** A continuous, latent historical context encoding injected into the VLA prompt. 
    *   *Example Prompt:* `[User Command: "Approach from the top"] [Context: "Previous 3 trials: VLA executed and narrated top-down grasps. High probability of cognitive compliance bias. Decay parameter $\lambda = 0.8$"]`

### 3.2 The Policy (Latent Reasoning & De-biasing)
The VLA processes the multimodal inputs to calculate a posterior belief over the human's true intent. If the semantic context indicates recent heavy coaching, the VLA assigns a lower confidence score to user commands that perfectly match the prompted strategy. 

Let the user's instruction be $I_t$, the carryover state be $\kappa_t$, and the true intent be $\pi^*$. The VLA estimates $Pr[\pi^* \mid I_t, \kappa_t]$, dynamically weighting the user's command against the known prompt residue.

### 3.3 Dynamic Execution & Action Tokens
*   **Action Generation:** The VLA predicts 6-DoF end-effector poses to execute the fetch.
*   **Arbitration:** If the VLA detects that the user's command is heavily contaminated by recent robotic coaching, it can autonomously alter its behavior to extract true intent—either by physically shifting its approach trajectory to offer a visual alternative, or by generating a linguistic counter-proposal (e.g., *"I can also approach laterally to save time. Confirm top-down?"*).

---

## 4. Simulation & Training Pipeline
The training pipeline requires a robust simulation environment to generate synthetic compliance data before moving to hardware.

### 4.1 Digital Twin & Isaac Lab Integration
*   **Environment:** Build a high-fidelity digital twin of the Kinova Gen2 and the tabletop workspace using Isaac Sim and Isaac Lab. 
*   **Synthetic Interaction Generation:** Program simulated "human supervisors" with varying levels of behavioral sensitivity to robotic prompts. Generate thousands of episodes where the robot executes a strategy, and the simulated human subsequently issues commands with mathematically injected compliance bias.

### 4.2 Model Fine-Tuning (PyTorch)
*   **Base Model:** Initialize with a pre-trained VLA capable of standard visual grounding and pick-and-place tasks.
*   **Carryover-Aware Fine-Tuning:** Use PyTorch to fine-tune the VLA on the synthetic Isaac Lab dataset. The loss function is modified to penalize the VLA if it blindly executes a user command that perfectly matches the $\kappa_t$ memory token without verifying the intent. The VLA learns to actively disambiguate instructions when historical bias is high.
*   **Sim-to-Real Transfer:** Deploy the trained model to the physical Kinova using ROS 2, evaluating its ability to de-bias actual human subjects.

---

## 5. Required Manuscript Changes (Phase 0 Redesign)
To accommodate this supervisory setup, the Phase 0 validation study[cite: 1] must reflect cognitive rather than physical carryover.

*   **The Estimand:** The spatial unprompted arm-choice map[cite: 1] is replaced by the user's baseline strategy preference distribution (e.g., safety-first vs. speed-first object retrieval).
*   **The Measurement:** The ASSESS probe is a neutral linguistic query about how to solve an ambiguous spatial puzzle, rather than a physical reach target[cite: 1].
*   **The Baselines:** The fixed washout baseline[cite: 1] is replaced by a "Memoryless VLA" (a model that blindly executes whatever the user says, falling victim to automation bias) compared against the proposed "Carryover-Aware VLA."