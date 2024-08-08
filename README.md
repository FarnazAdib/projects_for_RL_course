# The source code for project for the PhD level course on Reinforcement Learning

This is a self-contained repository for source codes for the project for [the PhD level course on Reinforcement Learning given by Farnaz Adib Yaghmaie at the automatic control devision , ISY, Linköping University.](https://farad59.gitlab-pages.liu.se/reinforcement_learning_phd_course/)

## Running on local computer
* Go to [https://github.com/FarnazAdib/projects_for_RL_course.git] and clone the project.
* Open PyCharm. From PyCharm. Click File and open project. Then, navigate to the project folder.
* Build a virtual environment and import required libraries.


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




