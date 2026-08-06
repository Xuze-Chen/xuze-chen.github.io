---
title: "Week 6: Why you can differentiate an expectation"
date: 2026-07-26
---

Last week my agent learned a table of values and read its policy off with an argmax. This week I dropped the table and optimized the policy directly. The objective is easy to state. J(θ) is the expected return of the policy. But that expectation runs over whole trajectories, and trajectories come from sampling. You cannot backpropagate through a dice roll. So the whole week hung on the question in the title.

The answer is the log derivative trick. The identity ∇P = P ∇log P turns the gradient of an expectation into the expectation of a gradient, and an expectation is something I can estimate by sampling. My favorite step comes right after. Expand log P(τ) and you get three kinds of terms: the initial state distribution, the environment transitions, and the policy's own log probabilities. Only the last kind contains θ. So the moment the gradient arrives, everything about the environment vanishes.

![My derivation of the policy gradient theorem](/assets/images/policy_gradient.png)

That page is my second pass from memory. Last week I wrote that Q-learning learns without ever reading the transition function. Now that fact has a proof.

The simplest algorithm built on that gradient is REINFORCE: collect a batch of episodes, weight each log π(a|s) by the return, step uphill. The oddest part was that the number my code calls loss means nothing. Mine went up while the agent got better, so I judged training by the return alone. From there I tried three upgrades to the weight, one idea at a time. Reward-to-go stops an action from taking credit for rewards that came before it. A learned baseline compares each return with what was expected from that state. GAE blends the two with a knob λ that trades bias against variance. The gradient never changes. Only the weight on log π does.

![REINFORCE, reward-to-go, baseline and GAE on CartPole, three seeds each](/assets/images/ablation.png)

Then the experiment disagreed with me. I expected each trick to visibly help. Instead, reward-to-go closed most of the gap on its own. The baseline and GAE added nothing I could tell apart. Their bands overlap all the way to the 500 step ceiling. Probably CartPole is just too easy for variance to be the bottleneck.

![A sweep over λ on CartPole](/assets/images/V4Figure_1.png)

I also swept λ across its whole range, from pure bootstrapping at 0 to pure Monte Carlo at 1. The one clear failure was λ = 0, where each advantage is one reward plus whatever the critic guesses about the rest. Early on that guess is simply wrong, and the curve stalled near 150 while every other λ reached the ceiling. So on this small task, bias cost me far more than variance did.

One coincidence caught me off guard. For a softmax policy, ∇log π(a|s) works out to onehot(a) minus p. Back in June I derived the gradient of cross entropy by hand and got p minus y. The same expression with the sign flipped, because here I climb instead of descend. Line the two up and the sampled action sits exactly where the label y used to be. Nobody hands me labels here, so the policy treats the action it just took as one, and the weight in front decides how hard to push its probability up or down. Policy gradient is cross entropy where the action plays the label and the return sets the weight.

As usual, the code and the proofs behind it are on GitHub as vpg-from-scratch. If you are working through Spinning Up's policy optimization pages or the GAE paper, I hope they save you some time:)
