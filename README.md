# The source code for project for the PhD level course on Reinforcement Learning

This is a self-contained repository for source codes for the project for [the PhD level course on Reinforcement Learning given by Farnaz Adib Yaghmaie at the automatic control devision , ISY, Linköping University.](https://farad59.gitlab-pages.liu.se/reinforcement_learning_phd_course/)

## Running on local computer
* Go to [https://github.com/FarnazAdib/projects_for_RL_course.git] and clone the project.
* Open the project in your preferred editor or IDE.
* Create the project virtual environment and install all required libraries by running

```
python3 utilities.py
```
This creates a local `.venv` folder and installs the packages listed in [requirements.txt](requirements.txt).
* Run the scripts with the virtual-environment interpreter, for example

```
.venv/bin/python q_blackjack.py
```

If your machine has multiple Python versions installed, `utilities.py` prefers Python 3.10, 3.11, or 3.12 so that the pinned package versions install correctly.


## Project 1: Implementing basic tabular approaches on Blackjack
In the first project, we implement a couple of basic tabular algorithms from the following _book_

* R. Sutton and A. Barto, [“Reinforcement learning: An introduction”](http://incompleteideas.net/book/the-book-2nd.html), MIT press, 2018.

 on [the Blackjack environment provided by OpenAI Gymnasium.](https://gymnasium.farama.org/environments/toy_text/blackjack/) Complete the code by replacing ``None`` with code in the blocks denoted by ``start your code here`` and ``end your code here``.

*  [Q-leaning algorithm on page 131 of the book](q_blackjack.py). 
* [sarsa algorithm on page 130 of the book](sarsa_blackjack.py). 
* [expected sarsa on page 133 of the book](expected_sarsa_blackjack.py). 
* [Double Q-learning on page 136 of the book](doubleq_blackjack.py). 
* [on-policy first-visit MC control on page 101 of the book](mc_blackjack.py). 

## Project 2: Implementing Normalized Advantage Functions (NAF) on inverted pendulum
The purpose of this project is to implement Normalized Advantage Functions (NAF) on inverted pendulum given in the following paper:

* Gu, Shixiang, Timothy Lillicrap, Ilya Sutskever, and Sergey Levine. ["Continuous Deep Q-Learning with Model-based Acceleration,"](http://proceedings.mlr.press/v48/gu16.pdf) In International Conference on Machine Learning, pp. 2829-2838. 2016.

 on [the inverted pendulum provided by OpenAI Gymnasium.](https://gymnasium.farama.org/environments/classic_control/pendulum/) Complete the code by replacing ``None`` with code in the blocks denoted by ``start your code here`` and ``end your code here``. The file for this project is
 
 * [NAF algorithm](NAF.py)


## Project 3: Implementing Deep Deterministic Policy Gradient (DDPG) and Soft Actor-Critic (SAC) on inverted pendulum
The purpose of this project is to implement Deep Deterministic Policy Gradient (DDPG) and Soft Actor-Critic (SAC) on inverted pendulum given in the following papers:


* Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver,  Daan Wierstra, ["Continuous Control with Deep Reinforcement Learning",](https://arxiv.org/pdf/1509.02971.pdf) ICLR, 2016.

* Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. ["Soft Actor-Critic: Off- policy maximum entropy deep reinforcement learning with a stochastic actor",](http://proceedings.mlr.press/v80/haarnoja18b/haarnoja18b.pdf) In International Conference on Machine Learning, pages 1861–1870, July 2018. 

 on [the inverted pendulum provided by OpenAI Gymnasium.](https://gymnasium.farama.org/environments/classic_control/pendulum/) Complete the code by replacing ``None`` with code in the blocks denoted by ``start your code here`` and ``end your code here``. The files for this project are
 
 * [DDPG algorithm](DDPG.py)
 * [SAC algorithm](SAC.py)

## Project 4: Implementing on-policy actor-critic methods (TRPO and PPO) on inverted pendulum
The purpose of this project is to implement the on-policy actor-critic methods Trust Region Policy Optimization (TRPO) and Proximal Policy Optimization (PPO) on inverted pendulum given in the following papers:

* John Schulman, Sergey Levine, Philipp Moritz, Michael I. Jordan, Pieter Abbeel, ["Trust Region Policy Optimization",](https://arxiv.org/pdf/1502.05477.pdf) In International Conference on Machine Learning, pp. 1889-1897, 2015.

* John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov, ["Proximal Policy Optimization Algorithms",](https://arxiv.org/pdf/1707.06347.pdf) arXiv preprint arXiv:1707.06347, 2017.

 on [the inverted pendulum provided by OpenAI Gymnasium.](https://gymnasium.farama.org/environments/classic_control/pendulum/) Complete the code by replacing ``None`` with code in the blocks denoted by ``start your code here`` and ``end your code here``. Both algorithms share the on-policy rollout buffer with Generalized Advantage Estimation (GAE) and the running observation normalizer provided in [utilities.py](utilities.py). The files for this project are

 * [TRPO algorithm](TRPO.py)
 * [PPO algorithm](PPO.py)

## Project 5: Implementing Soft Q-Learning (SQL) on inverted pendulum
The purpose of this project is to implement Soft Q-Learning, a maximum-entropy Q-learning method with energy-based policies, given in the following paper:

* Tuomas Haarnoja, Haoran Tang, Pieter Abbeel, and Sergey Levine. ["Reinforcement Learning with Deep Energy-Based Policies",](http://proceedings.mlr.press/v70/haarnoja17a/haarnoja17a.pdf) In International Conference on Machine Learning, pp. 1352-1361, 2017.

 on [the inverted pendulum provided by OpenAI Gymnasium.](https://gymnasium.farama.org/environments/classic_control/pendulum/) The soft value function is estimated by importance sampling with a uniform proposal, and the policy is an amortized sampling network trained with Stein Variational Gradient Descent (SVGD). Complete the code by replacing ``None`` with code in the blocks denoted by ``start your code here`` and ``end your code here``. The file for this project is

 * [SQL algorithm](SQL.py)

## Repository layout
Besides the project files listed above, the repository contains a few shared helper modules:

* [utilities.py](utilities.py) — shared components (`RunningNormalizer`, `RolloutBuffer`, `OUNoise`, `OUActionNoise`) and the requirements installer entry point.
* [plotting.py](plotting.py) — helper functions for plotting scores and losses.
* [requirements.txt](requirements.txt) — list of required Python libraries installed into the local virtual environment.
