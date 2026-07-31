---
title: "Week 5: Q-learning on a frozen lake"
date: 2026-07-19
---

This week, reinforcement learning officially began. I spent it on the first five lectures of David Silver's course, which cover most of the foundations. For a month my working words had been loss, gradient, and parameter. Now they are return, value, and policy. The game underneath is the same, though: define an objective and optimize it. To make sure the ideas actually landed, I built one of the environments from the lectures myself.
 
FrozenLake is a small grid where an agent has to reach a goal without falling through a hole. I used the slippery version, where the direction I choose only happens about a third of the time. Dynamic programming reads the transition function, so it can evaluate any state without ever visiting it. Q-learning only gets to step and see where it lands. That is the difference between planning and learning. And it means the agent learns nothing until it stumbles onto the goal by accident. The reward is 1 at the goal and 0 everywhere else. A random policy gets there about 1.4 percent of the time.
 
The hardest part of the week was a bug that never raises an error. I initialized the Q table to zeros and chose actions with `np.argmax`. When all four values are equal, `np.argmax` returns index 0. In FrozenLake that means LEFT. And from the start state, LEFT is a wall. So once epsilon decayed, the agent spent whole episodes pressed against the left edge. With seed 0 the agent never reached the goal in 2,000 episodes. With seed 1 it learned fine. The same code cannot be right and wrong at once. So the problem had to be in what the agent tried, not in how it updated. That sent me straight to the tie in the argmax. Breaking ties at random fixed every seed I ran. Over an uninformative Q table, a greedy policy is not a policy at all. It is just whatever `np.argmax` happens to do.
 
Once it was learning properly, the next question was how well. The training curve leveled off around 0.63. But that average is taken while the agent is still exploring and still updating. So I froze the Q table and evaluated it greedily over 10,000 episodes instead. It scored 0.743. Then I drew the policy out to see what it had settled on.
 
![Learned policy and value function](/assets/images/policy_4x4.png)
 
One square gave me pause. At the start state the learned action is to push left, into a wall. That looked wrong until I compared it with pushing down. Both reach the square below a third of the time, since a slip is always perpendicular to the direction I aim. The difference is the other two thirds. Aiming left, those slips leave me exactly where I am, because left and up are both walls. Aiming down, one of them pushes me right, away from the route. Nothing in this environment charges me for standing still. So waiting is cheaper than drifting somewhere worse.
 
Everything I built earlier this summer was shown the right answer during training. Q-learning never is. That is why most of this week went into checking whether I could trust what came back. Not knowing the answer in advance is also what makes it more satisfying. As usual, my notes on the lectures are on GitHub, with the reasoning filled in where I had to work it out myself. If you are working through the same lectures, I hope they save you some time:)
