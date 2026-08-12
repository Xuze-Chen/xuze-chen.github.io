---
title: "Week 7: Learning to read a training curve"
date: 2026-08-02
math: true
---

CartPole was enough to get VPG working, but it was still a simple environment. So this week I moved to HalfCheetah-v5, where each action is a continuous vector with six dimensions.

The code change was small. The policy head now outputs the mean of a Gaussian, while its log standard deviation is learned. I sum the log probabilities across the six action dimensions. The policy gradient estimator itself did not change.

Once it was running, I did a smoke test. The agent learned, but the run also showed me four things I did not yet understand. So I decided to use the next runs to find their causes, rather than change settings until the curve looked better. After all, I want to learn how to do research, not just how to get an agent to train.

The smoke test left me with four questions. Why did the return drop early? Why did the actor loss stay near zero? Why did the value loss fall and then rise? And why did the return swing so much between updates?

The curves showed me the symptoms, but not what was changing inside each update. So I added five diagnostics to the code: entropy, approximate KL between the old and new policies, value loss, explained variance, and gradient norm. From there, I began working through the four questions with controlled runs. I changed one setting at a time, kept the seed fixed, and wrote down my prediction before pressing enter. The complete run log is in the README.

The early drop was the first problem to go. At a learning rate of 1e-2, the approximate KL between consecutive policies stayed near 1. An update that large is less like adjusting a policy than replacing it. Lowering the learning rate to 3e-3 brought the KL down to about 0.010, and the early drop did not return.

After that, I started using KL to judge whether the learning rate was safe. A large KL meant the new policy had moved too far from the old one. So instead of asking whether the learning rate looked small, I checked whether the policy change was actually small.

The swinging return was harder because a point on the curve can move for two different reasons. The policy itself may have changed after an update. Or the same policy may simply have collected a better or worse set of episodes. On the curve, the two cases look identical.

A batch contains 5,000 steps, while a HalfCheetah episode lasts 1,000. So each point averages only a handful of episodes. Even a frozen policy would produce a different mean return from one batch to the next.

I wanted to separate that sampling noise from actual policy movement. So I estimated how much a frozen policy would fluctuate at the same sample size and compared it with what I saw during training. Early on, the sampling estimate was only 23.8 percent of the observed swing. That suggested the policy itself was moving too much, and the high KL supported the same conclusion. Late in training, sampling noise was large enough to account for the swings. By then, the curve was noisy mostly because each point contained too few episodes.

That distinction mattered because the two problems needed different fixes. Early instability called for smaller policy updates. Late sampling noise called for more episodes, or simply less concern about a jagged curve. Doubling the batch made the curve smoother without raising the return, which was consistent with that explanation.

The second question had a simpler answer. Last week I learned that policy loss could rise while the agent improved. This time it hovered around zero while return climbed. But I had forgotten that each update used a new batch and that I standardized its advantages separately. So the raw loss was not directly comparable across updates, and a value near zero did not mean learning had stopped.

The value loss was sneakier. It fell at first, then rose for the rest of training. Read on its own, that looked like a critic slowly failing. But explained variance stayed near 0.92, while the error relative to the scale of its targets kept falling.

The critic was keeping pace. Its targets were getting larger because the cheetah was getting faster. When the targets grow, the same relative prediction error produces a larger squared error. One way to understand it is that the student was improving, but the test was getting harder even faster.

Once I had answers to the four questions, I turned to improving the return. Along with the best hyperparameters from the earlier runs, I tried one more change. The 17 observation dimensions lived on very different scales. Joint angles stayed near 1, while angular velocities could reach 10. I suspected that bringing them to a similar scale would make the network easier to optimize.

So I normalized the observations using running statistics. I saved those statistics with the policy and reused them during evaluation. Below are the reward curves for the final configuration across three seeds.

![Final configuration on HalfCheetah-v5, three seeds](/assets/images/hc_final_3seeds.png)

The result was really encouraging. Here is what the final policy looked like:

![The learned gait](/assets/images/halfcheetah.gif)

This was my most rewarding week so far. The final policy worked better than I expected, but the best part was being able to explain the failures that came before it. Next week I plan to study PPO. After this week, I am excited to see what comes next.

As usual, my debugging notes and the complete **Training Diagnostics** log are in my GitHub repository, `vpg-from-mujoco`. The training curves are collected in a short W&B report, [What actually drives return?](https://api.wandb.ai/links/xuzechen-university-of-california-berkeley/chx0b11p). See you next week :)
