---
title: "Week 4: I built a tiny GPT"
date: 2026-07-12
---

This week, I worked through Karpathy's *Let's build GPT* lecture and built a small character-level GPT from scratch. Starting with a bigram model and ending with a complete Transformer made the architecture feel much less distant.

From token and positional embeddings to masked self-attention, multiple attention heads, feed-forward layers, residual connections, and LayerNorm. The part that took me longest was understanding how Q, K, and V fit together. It was also the part I found most fascinating. It was surprising that such a compact piece of code could mimic a simple human-like behavior.

Rebuilding the model without following the video line by line also made me pay closer attention to every tensor shape. I noticed a small discrepancy in the lecture code: the attention scores were scaled using C, the full embedding dimension, even though each head’s queries and keys have dimension head_size. I changed the scaling factor to k.shape[-1]**-0.5, and finding this mismatch by following the dimensions was a small but especially satisfying moment.

After the Shakespeare version worked, I applied the same model to a Chinese text and tuned the context length, embedding size, number of heads, and learning rate myself. The final model had 1.84 million parameters. Over 3,000 training steps, the training loss fell from 8.47 to 3.57, while the validation loss fell from 8.47 to 4.66. The validation loss reached its lowest reported value of 4.63 at step 2,500 before ticking up slightly. Watching the loss fall on a text I had chosen, using a model I had rebuilt and tuned myself, was the most rewarding part of the week.

![Training results for my Chinese character-level GPT](/assets/img/week4/chinese-gpt-training.png)

I kept up with the math too, reviewing random variables, expectation and variance, common probability distributions, covariance and correlation, conditional expectation, and the law of total expectation.

I also turned the lecture into a set of detailed notes on GitHub, filling in the intermediate steps and recording the implementation details I found easiest to miss, including the attention-scaling issue above. If you are working through the same lecture, I hope they help. Next week, I will officially begin reinforcement learning, and I am excited to finally start.
