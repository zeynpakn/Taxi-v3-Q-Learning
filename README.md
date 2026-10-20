<div align="center">

# 🚕 Taxi-v3 Q-Learning

### Teaching a taxi to pick up the right passenger and find the shortest route

<br>

![ALGORITHM](https://img.shields.io/badge/ALGORITHM-Q--LEARNING-F37626?style=for-the-badge&labelColor=555555)
![LIBRARY](https://img.shields.io/badge/LIBRARY-GYMNASIUM-4c1?style=for-the-badge&labelColor=555555)
![LANGUAGE](https://img.shields.io/badge/LANGUAGE-PYTHON-3776AB?style=for-the-badge&logo=python&logoColor=white&labelColor=555555)
![DOMAIN](https://img.shields.io/badge/DOMAIN-VALUE--BASED%20RL-E8742F?style=for-the-badge&labelColor=555555)

<br>

*No route planner. No hand-written policy. Just rewards, penalties, and a Q-Table that improves with experience.*

<br>

[![RUN ON COLAB](https://img.shields.io/badge/%E2%96%B6%20%E2%96%B6%20RUN%20ON%20COLAB-LIVE%20NOTEBOOK-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white&labelColor=555555)](https://colab.research.google.com/github/zeynpakn/Taxi-v3-Q-Learning-Reinforcement-Learning/blob/main/Taxi_v3.ipynb)

<br>

<a href="https://colab.research.google.com/github/zeynpakn/Taxi-v3-Q-Learning-Reinforcement-Learning/blob/main/Taxi_v3.ipynb">
  <img src="assets/demo.gif" alt="Taxi-v3 Q-Learning gameplay" width="480">
</a>

<sub>👆 Click the demo to open the notebook in Google Colab</sub>

</div>

---

## 📖 Overview

This project implements a **tabular Q-Learning agent from scratch** on Gymnasium's `Taxi-v3` environment.

At the beginning of each episode, the taxi can start from a different position. The passenger is placed at one of the designated pickup locations, and the destination is another designated colored location. The agent must:

1. navigate to the passenger,
2. perform a valid pickup,
3. drive to the correct destination,
4. perform a valid dropoff,
5. do all of this with as few unnecessary actions as possible.

The project was built while studying the [Hugging Face Deep RL Course — Unit 2](https://huggingface.co/learn/deep-rl-course/unit2/introduction), with the goal of understanding **value-based reinforcement learning** by implementing the learning process directly instead of relying on a prebuilt RL algorithm.

## 🌆 Environment

`Taxi-v3` has a discrete state and action space, which makes it suitable for a **tabular method**.

A state describes:

- the taxi's row,
- the taxi's column,
- the passenger's location,
- the destination.

In the notebook, Gymnasium represents these combinations with a single **state ID**. The environment contains:

| Component | Value |
| --------- | ----: |
| States | 500 |
| Actions | 6 |
| Q-Table size | 500 × 6 |

The six available actions are:

| Action | Meaning |
| :----: | ------- |
| 0 | Move South |
| 1 | Move North |
| 2 | Move East |
| 3 | Move West |
| 4 | Pick up passenger |
| 5 | Drop off passenger |

Each interaction follows the standard RL loop:

```text
State → Action → Reward → Next State
```

The agent is rewarded for successfully completing the trip, while unnecessary movements and invalid pickup/dropoff actions make the return worse. This gives the agent an incentive to learn efficient behavior.

## 🧠 Why Q-Learning?

Q-Learning is a **value-based reinforcement learning algorithm**.

It does not learn a policy directly. Instead, it learns a value for every state-action pair:

```text
Q(state, action)
```

That value estimates how useful an action is in a given state when future rewards are also considered.

Because Taxi-v3 has a manageable number of discrete states and actions, these values can be stored directly in a table:

```text
             South  North  East  West  Pickup  Dropoff
State 0        ...    ...   ...   ...    ...      ...
State 1        ...    ...   ...   ...    ...      ...
...
State 499      ...    ...   ...   ...    ...      ...
```

Once training is complete, the policy is simple: for the current state, choose the action with the highest Q-value.

```text
Policy(state) = argmax Q(state, action)
```

## 🔄 Q-Learning Process

The notebook builds the algorithm step by step.

### 1. Initialize the Q-Table

The Q-Table starts with zeros:

```text
500 states × 6 actions
```

At this point, the agent has no knowledge about which actions are useful.

### 2. Choose Actions with ε-Greedy

During training, the agent must balance:

- **Exploration** — try a random action and discover new possibilities.
- **Exploitation** — choose the action currently believed to be best.

This is handled with an **epsilon-greedy policy**:

```text
With probability ε     → random action
With probability 1 - ε → action with highest Q-value
```

Epsilon starts high and gradually decreases, so the agent explores more early in training and relies more on learned knowledge later.

### 3. Observe the Transition

After choosing an action, the environment returns:

```text
Next State + Reward + Episode Status
```

This experience tells the agent whether its decision was useful.

### 4. Update the Q-Value

The learned value is updated using the Bellman-style Q-Learning rule:

```text
Q(s,a) ← Q(s,a) + α [ r + γ max Q(s',a') - Q(s,a) ]
```

Where:

- `α` is the learning rate,
- `γ` is the discount factor,
- `r` is the immediate reward,
- `max Q(s',a')` is the estimated value of the best next action.

This update is repeated across thousands of episodes until the Q-Table represents a strong policy.

### 5. Evaluate with a Greedy Policy

After training, exploration is removed.

The agent always selects:

```text
argmax Q(state, action)
```

This lets the learned policy be evaluated without random exploration affecting the results.

## 🧪 What Was Implemented

The notebook follows the full workflow rather than only training a final agent:

```text
Create Taxi-v3 environment
        ↓
Inspect state representation
        ↓
Inspect the six available actions
        ↓
Initialize the 500 × 6 Q-Table
        ↓
Implement greedy policy
        ↓
Implement ε-greedy policy
        ↓
Implement Q-Learning training loop
        ↓
Train for 10,000 episodes
        ↓
Evaluate over 1,000 episodes
        ↓
Plot training progress
        ↓
Record a gameplay animation
        ↓
Compare learned routes with shortest paths
```

This makes the notebook useful not only as a trained example, but also as a step-by-step implementation of tabular Q-Learning.

## 🛣️ Route Optimality Analysis

Success alone does not tell us whether the taxi learned an efficient route.

To check this, the project also computes the **shortest possible path** for each tested starting state using **Breadth-First Search (BFS)** and compares it with the route selected by the learned Q-Learning policy.

This provides a second evaluation question:

> Did the agent merely learn to finish the task, or did it also learn routes close to the shortest possible ones?

The learned agent matched the optimal route in **98.9%** of the evaluated episodes and required only **0.02 additional steps on average**.

## 📁 Project Structure

```text
Taxi-v3-Q-Learning-Reinforcement-Learning/
├── Taxi_v3.ipynb
├── assets/
│   └── demo.gif
└── README.md
```

The notebook contains the complete implementation, evaluation, training-progress analysis, gameplay generation, and BFS route comparison.

## ▶️ Run It

Open the notebook using the **RUN ON COLAB** button above.

Running the notebook from top to bottom will:

1. install the required libraries,
2. create and inspect Taxi-v3,
3. initialize the Q-Table,
4. train the Q-Learning agent,
5. evaluate the learned policy,
6. visualize training progress,
7. generate a gameplay GIF,
8. compare learned routes with BFS shortest paths.

## 🎓 What I Learned

- Why Q-Learning is considered a **value-based** method rather than a policy-based method.
- How a Q-Table represents the value of every state-action pair.
- How epsilon-greedy exploration balances trying new actions with using learned knowledge.
- How the Bellman update gradually propagates useful reward information through the state space.
- Why evaluating over many episodes is more meaningful than looking at one successful run.
- Why task completion and route optimality are different evaluation questions.
- How BFS can be used as an external reference to check how close a learned policy is to the shortest possible route.

## ⚙️ Training

| Configuration | Value |
| ------------- | ----: |
| Environment | Taxi-v3 |
| Algorithm | Tabular Q-Learning |
| Method | Value-Based RL |
| Training Episodes | 10,000 |
| Learning Rate (α) | 0.7 |
| Discount Factor (γ) | 0.95 |
| Initial Epsilon | 1.0 |
| Minimum Epsilon | 0.01 |
| Epsilon Decay | 0.999 |
| Framework | Gymnasium |
| Random Seed | 42 |

## 📊 Results

The trained agent was evaluated over **1,000 episodes** using greedy action selection.

| Metric | Result |
| ------ | -----: |
| Mean Reward | **7.84** |
| Standard Deviation | **2.62** |
| Evaluation Score (Mean − Std) | **5.22** |
| Success Rate | **100%** |
| Average Steps | **13.16** |
| Optimal Average Steps | **13.14** |
| Optimal Route Rate | **98.9%** |
| Average Extra Steps | **0.02** |

The agent completed every evaluation episode and produced routes that were almost identical to the BFS shortest-path baseline.

---

<div align="center">

**Project by [zeynpakn](https://github.com/zeynpakn)**

*Started with random rides. Ended with almost optimal routes. 🚕💨*

</div>
