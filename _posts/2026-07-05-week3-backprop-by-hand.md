---
title: "Week 3: Backprop by hand"
date: 2026-07-05
---

This week I kept going with makemore, but I also reproduced the earlier steps without looking at my old code and found a few gaps in my understanding.

Part 3 finally gave me a way to think about the MNIST result from last week. In that experiment, scaling the initial weights of my MLP by 0.01 brought test accuracy from roughly 95% to 98.03%. At the time, I only knew that the smaller weights worked. Now I know why they helped: large initial weights produce large pre-activations, which push tanh toward ±1. Since the derivative of tanh is 1 − h², the gradient becomes tiny in this saturated region, so very little learning signal reaches the earlier layers.

Part 4, Becoming a Backprop Ninja, was the hardest video so far. I worked through the entire backward pass by hand, and the line that held me up longest was emb = C[Xb]. The forward pass is just a lookup, but the backward pass is a scatter-add: each gradient has to go back to the right row of C, and repeated indices have to collect more than one contribution. I was stuck on that for a while, so getting it right felt especially good.

Before this exercise, loss.backward() still felt like one large operation hidden behind PyTorch. Now I can see it as a chain of small local derivatives, each one doing something I can work through myself. That made this one of my favorite lessons in the series so far. When I eventually start training robots, I think understanding the tools I rely on, rather than simply knowing how to call them, will matter in the long run.

I kept up with the math too, reviewing eigenvalues and eigenvectors, diagonalization, singular value decomposition, and positive-definite matrices.

I also turned the exercise into a set of detailed notes on GitHub. I expanded each step of the backward pass, collected a few tricks that helped me work through the gradients, and organized Karpathy’s handwritten derivations into a form I found easier to follow. If you are working through the same video, I hope they help.
