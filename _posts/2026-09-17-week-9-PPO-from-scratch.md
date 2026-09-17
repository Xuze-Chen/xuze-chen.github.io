---
title: "Week 9: PPO from scratch"
date: 2026-09-17
math: true
---

Last week I ended with two questions. What does PPO gain from reusing data, and what does clipping add? And why did my first PPO run on HalfCheetah fall behind my tuned VPG? This week I worked through both. The second question changed once I looked more closely at the comparison.

## Before comparing scores

My first PPO implementation reached a median return of about 2700 on HalfCheetah across five seeds. That was above the 1443 reported by CleanRL, which looked encouraging.

Then I looked at the individual seeds: 1100, 1300, 2700, 2700, 3500. Two were below the reference. The code and settings were the same, but the returns differed by a factor of three.

Before asking whether my PPO scored higher, I needed to check whether it behaved like the reference. Return alone could not tell me that. So I checked the 22 relevant items in [The 37 Implementation Details of PPO](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) against my implementation and CleanRL's code.

Thirteen differed. Only a few were deliberate choices. The entropy coefficient of 0.01 was left over from CartPole. The initial log standard deviation of -0.5 was a number I had typed in week 7 and never revisited.

I changed all thirteen to match the reference, then ran eight seeds of each implementation on the same laptop. The median returns were 1420 for mine and 1400 for CleanRL. The KL per update matched too. Each implementation also had one seed that found a much faster gait near 3500. That result was not unique to my code.

Alignment did not raise the median. It brought the two implementations into agreement and gave me a baseline for the next experiment.

![My aligned PPO and CleanRL on HalfCheetah-v4, eight seeds each, with the mean and minimum to maximum range](/assets/images/halfcheetah_wandb.png)

This also changed how I understood the earlier comparison with VPG. I had been comparing experiments with different settings. With returns ranging from 1100 to 3500 across seeds, one run from each could not tell me which algorithm worked better.

## Reuse versus clipping

With that baseline in place, I could return to the first question. I kept everything from the aligned build, including ten epochs on each batch, and removed clipping.

If reuse explained the faster learning, the unclipped runs should still learn quickly. But if clipping kept repeated updates stable, removing it should eventually cause trouble.

Both happened. For the first 100k steps, the unclipped runs were ahead. At 63k steps, one had reached 546 while the aligned median was still below zero.

But the KL per update was already around 1, compared with 0.01 in the aligned build. By 250k steps, entropy had fallen from 8.5 to around zero. The policy had become much less exploratory. One seed stalled near 800. The other stopped at 270k steps when the ratio assertion failed.

![Removing clipping: return, KL per update, and entropy compared with the aligned build](/assets/images/noclip_wandb.png)

These runs helped separate the two roles. Reuse gave the policy a faster start. Clipping kept those repeated updates under control. Without it, the early gains did not last.

## One deliberate difference

In my VPG README, I wrote that termination and truncation need different bootstrap values. The pole falling and the clock running out are different events.

In HalfCheetah, episodes only end when they reach the time limit. The CleanRL baseline I used treats that limit as a terminal state, with a bootstrap value of zero. My code instead bootstraps from the critic's value estimate.

I added this change back to the aligned build and tested three paired seeds. It improved return by about 25 percent. The curve also kept climbing past 600k steps, where the aligned build flattened out. CleanRL has a variant with the same change, and its published seeds reach a similar range.

Here is a ninety second walkthrough of the project.


## If I did it again

I would run five seeds before trusting a number and compare with a reference before tuning. I would also log the gradient norm from the start.

In my first build, the norm was between 12 and 26 against a clipping threshold of 0.5. That meant the gradients were being scaled down by a factor of about 25 to 50. Without logging the norm, I had no way to see how much clipping was doing.

The code and complete logs are on GitHub as [ppo-from-scratch](https://github.com/Xuze-Chen/ppo-from-scratch). The training curves are collected in a [W&B report](WANDB_REPORT_URL).
