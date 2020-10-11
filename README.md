# dqn-navigation
This repository contains my submission for Project 1: Navigation of the Udacity [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893).

## Project Details

The assignment is to train an agent that solves the [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents) "Banana" environment. 

The solution implements a Deep Q-Network based on [[1]](#dqn_paper) to solve the environment. For implementation details, please see [Report.md](Report.md).

#### Environment
_(The below description is replicated from the [udacity/deep-reinforcement-learning](https://github.com/udacity/deep-reinforcement-learning/blob/master/p1_navigation/README.md) repository.)_

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

## Getting started

#### Prerequisites
- Python >= 3.6
- A GPU is NOT required; the agent is simple enough to be trained on a standard CPU :tada:

#### Installation
1. Clone the repository.
```bash
git clone https://github.com/JunShern/dqn-navigation.git
```

2. Create a virtual environment to manage your dependencies.
```bash
cd dqn-navigation/
python3 -m venv .venv
source .venv/bin/activate # Activate the virtualenv
```

3. Install python dependencies
```bash
cd dqn-navigation/python
pip install .
```

4. Add your virtual environment to jupyter's kernels.
```bash
python -m ipykernel install --user --name .venv --display-name "deeprl"
```

5. Download the Banana environment from one of the links below:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
  
    Place the file in the root of this repository, and unzip (or decompress) the file.

## Instructions

The project is intended to be run from `Navigation.ipynb`.

1. Run jupyter-notebook
```bash
cd dqn-navigation/
source .venv/bin/activate # Activate the virtualenv
jupyter-notebook
```
2. Open `Navigation.ipynb`. 
3. Follow the instructions in the notebook.

## References

- <a name="dqn_paper">[1]</a> Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning." nature 518.7540 (2015): 529-533.