# RL Smart Energy Agent

RL Smart Energy Agent is a localhost university final project for an AI and Prompt Engineering course.  
The project simulates a smart room where different agents learn how to reduce electricity cost while keeping the room comfortable for a human.

## Project Idea

The main idea is simple:

- A virtual room has temperature, electricity price, battery level, device states, and human presence.
- An agent watches the room state.
- The agent chooses actions such as using a heater, cooler, light, or battery.
- The system gives a reward based on comfort, energy cost, and smart energy usage.
- Over time, reinforcement learning agents learn better behavior.

This makes the project both AI-related and useful in real life, because smart homes and energy optimization are important topics.

## Why This Is an AI Project

This project is about decision-making with reinforcement learning.

- The agent does not only follow fixed code.
- It interacts with an environment.
- It gets rewards and penalties.
- It improves its policy from experience.

The project also includes an AI explanation layer:

- `prompts.py` stores the explanation prompt structure.
- `ai_explainer.py` generates a simple English explanation.
- If no real API key is available, the project uses a mock explanation.
- The code is prepared so a real Groq API integration can be used later.

## Reinforcement Learning in Simple Words

Reinforcement learning means:

1. The agent sees the current state.
2. It picks an action.
3. The environment responds.
4. The agent gets a reward or penalty.
5. The agent tries to maximize future total reward.

In this project, reward is based on:

- keeping the room comfortable when a person is home
- lowering electricity cost
- avoiding waste
- using the battery during expensive price periods

## Smart Room Environment

The simulated room includes:

- current temperature
- outside temperature
- electricity price
- human presence: home or away
- battery level
- device states:
  heater
  cooler
  light
  battery charging
  battery usage

## RL State

The RL state includes:

- room temperature category
- electricity price category
- human presence
- battery level category
- time of day

These categories make the environment easier to understand and easier to explain in a university defense.

## Actions

The agent can choose:

- do nothing
- turn heater on
- turn cooler on
- turn light on or off
- charge battery
- use battery energy

## Reward Function

The reward function is designed to be easy to explain:

- positive reward if the room is comfortable when a human is home
- negative reward if electricity cost is high
- negative reward if temperature is too hot or too cold
- negative reward for wasting energy
- bonus reward for using battery during high-price hours

## Algorithms Used

This project compares six algorithms:

- Random Agent
- Rule-Based Agent
- Q-Learning Agent
- SARSA Agent
- Simplified DQN Agent
- Simplified PPO Agent

Notes:

- `Random Agent` is a baseline.
- `Rule-Based Agent` is a human-designed baseline.
- `Q-Learning` and `SARSA` are tabular RL methods.
- `DQN` is implemented in a simplified lightweight form for fast local use.
- `PPO` is also simplified to keep the project understandable and fast on a laptop.

## Tech Stack

- Backend: FastAPI
- Frontend: HTML, CSS, JavaScript
- Charts: Chart.js
- RL logic: Python
- Run mode: localhost only

## Project Structure

```text
smart-energy-rl/
│── main.py
│── environment.py
│── models.py
│── pydantic_models.py
│── prompts.py
│── ai_explainer.py
│── README.md
│── defense_script.md
│── requirements.txt
│
├── agents/
│   ├── base_agent.py
│   ├── random_agent.py
│   ├── rule_based_agent.py
│   ├── q_learning_agent.py
│   ├── sarsa_agent.py
│   ├── dqn_agent.py
│   ├── ppo_agent.py
│   └── rl_utils.py
│
└── static/
    ├── index.html
    ├── style.css
    └── app.js
```

## How to Run

### 1. Open the project folder

```bash
cd "/Users/zangaruskempir/project ai/smart-energy-rl"
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
```

### 3. Activate it

macOS / Linux:

```bash
source .venv/bin/activate
```

### 4. Install requirements

```bash
pip install -r requirements.txt
```

### 5. Run the app

```bash
uvicorn main:app --reload
```

### 6. Open in browser

```text
http://127.0.0.1:8000
```

## AI Explanation Logic

The explanation system works like this:

- `POST /explain` checks the latest comparison or simulation result.
- `prompts.py` builds a clean prompt for an LLM.
- `ai_explainer.py` checks if `GROQ_API_KEY` or `OPENAI_API_KEY` exists.
- If a key exists and the request succeeds, it can return a real AI-generated explanation.
- If no key exists, the app safely uses a mock explanation.

This makes the project easy to run for a teacher without extra setup.

### Optional future API setup

Example for Groq:

```bash
export GROQ_API_KEY="your_key_here"
```

If the key is missing, the mock explanation still works.

## FastAPI Endpoints

- `GET /` -> main dashboard
- `POST /train` -> train selected algorithm
- `POST /simulate` -> run simulation
- `GET /results` -> latest stored results
- `POST /compare` -> compare all algorithms
- `POST /explain` -> generate simple explanation

## What the Dashboard Shows

- project title and short explanation
- algorithm selector
- training and simulation controls
- total reward
- energy cost
- comfort score
- battery usage
- temperature chart
- cost chart
- reward-over-episodes chart
- algorithm comparison chart
- comparison leaderboard
- action log table
- AI explanation panel

## Sample Results

Results change by seed and training episodes, but a common pattern is:

- Random Agent performs worst.
- Rule-Based Agent performs better than random.
- RL agents often learn stronger reward over time.
- One of Q-Learning, SARSA, DQN, or PPO usually becomes the best algorithm in a comparison run.

Example observation:

- best algorithm: SARSA Agent or Q-Learning Agent
- reward: higher than rule-based
- cost: controlled better than random
- comfort: still maintained at a useful level


## Why This Project Is Useful

- It connects AI theory with a practical real-life problem.
- It is visual and easy to demonstrate.
- It compares classical and learning-based approaches.
- It includes backend, frontend, RL, and prompt engineering in one project.
- It is impressive, but still understandable for a university defense.
