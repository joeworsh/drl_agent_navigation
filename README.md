# Deep Reinforcement Learning Navigation

This repository contains a solution for the
[Udacity DRL Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893)
[Project 1 - Navigation.](https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation)
This project trains a Deep Q-Network to solve the [Unity](https://unity.com/) Banana Navigation Task
(screen shot below).

<img src="docs/screenshot.png" alt="Unity3D Banana Explorer Screenshot" width="800px">

## The Environment

This environment, like most that are used in Reinformcent Learning (RL), is structured as a
[Markov Decision Process](https://en.wikipedia.org/wiki/Markov_decision_process) (MDP). This
means that it has a defined <b>state space</b>, <math>S</math>, a defined <b>action set</b>,
<math>A</math>, a probabilistic transition function that defines the transition from one state
to the next and a reward function to compute the reward, <math>R</math> at every time step
<math>t</math>. MDPs traditionally operate in fixed, discrete time intervals.

The state space,
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>S</mi><mo>&#x02208;</mo><msup><mi>&#x0211D;</mi><mrow><mn>37</mn></mrow></msup></mrow></math>,
has 37 different features which include velocity
and a representation of all the bananas surrounding the agent. This vector gives the agent
a perception of its surroundings in terms of what it really cares about - bananas. The
action space,
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo>&#x02208;</mo><mrow><mn>0</mn><mi>,</mi><mn>1</mn><mi>,</mi><mn>2</mn><mi>,</mi><mn>3</mn></mrow></mrow></math>
is a discrete integer space from 0 to 3 inclusive. Each
of these integers represent a different action that the agent can perform:
* <code>0</code>: move forward
* <code>1</code>: move backward
* <code>2</code>: turn left
* <code>3</code>: turn right

The transition function for this problem is quite simple. Driven by the [Unity](https://unity.com/)
game engine, the agent will be moved or rotated a small amount according to the requested action.
These transitions are deterministic (i.e. there is no chance that it will do something other
than what has been requested). Finally, the reward function will return a score of <math>+1</math>
when a yellow banana is collected (simply by colliding with it) and a score of <math>-1</math> when
collecting a blue banana. The goal, of course, is to maximize the total reward received over
time, which is a sum of all the scores at every timestep.

The environment is considered scored when an agent can receive an average score of 13
(or greater) over 100 episodes.

## Background

The most simple agent could do nothing as it would never receive a negative reward, but it
would never receive a positive award as well. The goal here is to train an agent that can
maximize its reward over time, even if that means we have to collect a few blue bananas in
the process. This exemplifies one of the fundamental problems solved with reinforcement
learning - how can an agent learn to make decisions that maximize future reward instead
of just immediate gratification? The answer is typically addressed with the
[Bellman Equation](https://en.wikipedia.org/wiki/Bellman_equation) from the field of
[Dynammic Programming](https://en.wikipedia.org/wiki/Dynamic_programming)
(which RL is built upon). In RL and Dynammic Programming the states of an MDP are
described by their <b>value</b> indicating the goodness of any state by the potential
accumulated reward that can be received from that state. According to the Bellman Equation
this is written as:

<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>v</mi><mi>&#x003C0;</mi></msub><mo stretchy="false">&#x00028;</mo><mi>s</mi><mo stretchy="false">&#x00029;</mo><mo>&#x0003D;</mo><msub><mi>&#x1D53C;</mi><mi>&#x003C0;</mi></msub><mi>\[</mi><msub><mi>R</mi><mrow><mi>t</mi><mo>&#x0002B;</mo><mn>1</mn></mrow></msub><mo>&#x0002B;</mo><mi>&#x003B3;</mi><msub><mi>v</mi><mi>&#x003C0;</mi></msub><mo stretchy="false">&#x00028;</mo><msub><mi>s</mi><mrow><mi>t</mi><mo>&#x0002B;</mo><mn>1</mn></mrow></msub><mo stretchy="false">&#x00029;</mo><mi>|</mi><msub><mi>s</mi><mi>t</mi></msub><mo>&#x0003D;</mo><mi>s</mi><mi>\]</mi></mrow></math>


Importantly this is showing that the value of any state can be represented by the
score at the given state,
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>R</mi><mrow><mi>t</mi><mo>&#x0002B;</mo><mn>1</mn></mrow></msub></mrow></math>,
plus the value of the following state,
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>v</mi><mi>&#x003C0;</mi></msub><mo stretchy="false">&#x00028;</mo><msub><mi>s</mi><mrow><mi>t</mi><mo>&#x0002B;</mo><mn>1</mn></mrow></msub><mo stretchy="false">&#x00029;</mo></mrow></math>.
This shows an important recursive nature of the MDP that Dynammic Programming paradigms
capitalize on. Additionally, the
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>&#x003B3;</mi></mrow></math>
value is used to <b>discount</b> the future
return when considering the value of the state. If
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>&#x003B3;</mi></mrow></math>
were set to <math>0</math> then the value of each state would be equal to the reward returned at that state only. If
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>&#x003B3;</mi></mrow></math>
were set to <math>1</math> that value would fully emcompass all future rewards as well. In practice
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>0</mn><mo>&lt;</mo><mi>&#x003B3;</mi><mo>&lt;</mo><mn>1</mn></mrow></math>
which can help to account for uncertainty in future estimation but still consider future reward when making decisions.
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>&#x003B3;</mi></mrow></math>
is one of the important hyperparameters that is explored within this project.

Another important aspect of RL is <b>exploration vs exploitation</b>. While learning
there is an important balance between <b>exploring</b>, taking an action with unknown
consequences, and <b>exploiting</b>, doing what you currently <i>think</i> is best. The
only way to learn is to try new strategies and see what works, but simultaneously the
agent needs to be refining strategies that it knows work. This is the mathematical equivalent
of our own learning by trial and error processes. During training this is modeled with
an
<b><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>&#x003F5;</mi></mrow></math>-greedy policy</b>
that will sometimes act randomly and sometimes act optimally depending at a rate defined by the hyperparameter
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>&#x003F5;</mi></mrow></math>.
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>&#x003F5;</mi></mrow></math> is
decayed during training and a well tuned
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>&#x003F5;</mi></mrow></math>
can make the difference between an agent that learns nothing and an expert agent. This field is another important value
that is explored within this project.

## Installation and Configuration

In order to run this code please follow these steps:

1) Checkout the Udacity DRL Gitlab [repository](https://github.com/udacity/deep-reinforcement-learning)
1) In the Udacity repository navigate to <code>deep-reinforcement-learning/python</code> and run <code>pip install -e .</code> (note you may want to do this in a python <code>venv</code>)
1) Download and install the Unity Banana executable [here](https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation)
1) Checkout this repository
1) Run <code>pip install -e .</code> to install this code and its dependencies (use the same <code>venv</code> if using one)

## File Structure

This repository contains a Deep Q-Network [1] implementation in the package `joe_agents`, Jupyter Notebooks
that contain training and evaluation experiments and saved models that can be used for inference and
validation.

The following notebooks are made available:

* [Experiment Grid Search.ipynb](https://github.com/joeworsh/drl_agent_navigation/blob/main/Experiment%20Grid%20Search.ipynb): A vast grid search
over hyperparameters to find the best model configurations for this environment
* [Analyze Results.ipynb](https://github.com/joeworsh/drl_agent_navigation/blob/main/Analyze%20Results.ipynb): A notebook
that explores all the results produced by the grid search
* [DQN Train.ipynb](https://github.com/joeworsh/drl_agent_navigation/blob/main/DQN%20Train.ipynb): Train a new model with the best
parameters found from the grid search
* [DQN Evaluate.ipynb](https://github.com/joeworsh/drl_agent_navigation/blob/main/DQN%20Evaluate.ipynb): Evaluate the model
trained by `DQN Train.ipynb`

## Instructions for Training and Evaluating

The easiest way to replicate the results from this project is to start with the notebook `DQN Train.ipynb`. One change is
required to configure this environment to run on your computer (this change must be made to all notebooks you wish to run).
Locate the block:
```python
# create the environment
exe = "../../deep-reinforcement-learning/p1_navigation/Banana_Windows_x86_64/Banana.exe"
evn_config = {"executable": exe, "train_mode": True}
env = BananaEnv(evn_config)
```
The value of `exe` is pointing to the local instance of the Unity Banana executable installed in step 3 above. This value
needs to be updated to point to the true location of the executable on your machine. With this change you should
be able to train and evaluate DQN models on the banana environment.
