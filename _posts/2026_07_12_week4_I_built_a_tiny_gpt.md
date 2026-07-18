---
title: "Week 4: I built a tiny GPT"
date: 2026-07-12
---

This week I worked through Andrej Karpathy's "Let's build GPT" lecture and built a small GPT from scratch. It models text one character at a time. The lecture starts from a plain bigram model and grows it, piece by piece, into a full Transformer. Watching that happen made the architecture feel far less mysterious than it did a month ago.

Along the way I implemented token and positional embeddings, masked self attention with multiple heads, feedforward layers, residual connections, and LayerNorm. The part that took me longest was understanding how Q, K, and V actually fit together. It was also the part I found most fascinating. In fact, it still surprises me that such a compact piece of code can imitate something as human as language.

I made a point of rebuilding the model myself instead of copying the video line by line. That meant justifying every tensor shape as I went. The habit paid off. I noticed a small discrepancy in the lecture code. The attention scores are scaled using C, the full embedding dimension. But each head's queries and keys only have dimension head_size. So I changed the scaling factor to k.shape[-1]**-0.5. Catching that mismatch purely by tracing shapes through the network was a small moment, but an unusually satisfying one.

For training data I chose Water Margin, one of the classic Chinese novels. The model has 1.84 million parameters. After some tuning, it brought training loss from 8.47 down to 3.57, and validation loss settled at 4.63. Around step 2,500, though, the validation loss stopped improving while training loss kept falling. That is a sign the model was starting to overfit the relatively small dataset. Even so, the generated passages already carry recognizable structure and style. I will take that as a win :)

![Training results](/assets/images/week4.png)

The math continued in parallel. This week I reviewed random variables, expectation and variance, the common probability distributions, covariance and correlation, conditional expectation, and the law of total expectation.

As usual, my notes are up on GitHub. I added explanations for a few details the lecture does not spell out. They are the kind that can quietly trip you up if you are coding along. So if you are working through the same lecture, I hope they save you some time. Next week I officially begin reinforcement learning. I am excited to finally get started!
