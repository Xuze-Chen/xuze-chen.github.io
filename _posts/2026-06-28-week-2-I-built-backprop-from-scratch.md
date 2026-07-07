---
title: "Week 2: I built backprop from scratch"
date: 2026-06-28
---

This week I started Karpathy's Zero to Hero. The first video has you build micrograd, a tiny autograd engine in about a hundred lines. I typed it out along with the video on Monday, then rewrote it from scratch on Tuesday without looking. To check it, I ran the same expression through my engine and through PyTorch, and the gradients came out the same.

Backprop turned out to be much smaller than I expected. It's the chain rule plus some bookkeeping. The part that took me longest was gradient accumulation: when a value gets used in two places, the gradients from both paths have to add up, so the code says `+=` and not `=`.

The best moment of the week came from makemore. I built the same bigram name generator twice: once by counting letter pairs into a 27×27 table, once by training a one-layer network with gradient descent. The trained weights came out almost identical to the log of the counts. It makes sense, since both ways minimize the same loss, but seeing it happen still surprised me.

Saturday was MNIST. Instead of reaching for `nn.Module` and an optimizer, I kept everything the way makemore does it: plain weight matrices with `requires_grad=True` and a training loop I wrote myself. The network is the makemore one again, just with three characters of context swapped for 784 pixels.

The biggest lesson was initialization. With weights straight out of 'torch.randn', test accuracy got stuck around 95%. Scaling the initial weights by 0.01 brought it to 98.03%. Big random weights make big logits, so the network starts out confidently wrong and spends its first epochs recovering. I was surprised that such a small change made that much difference.

![MNIST training loss](/assets/images/week2_mnist.png)

Next week I'll keep going with the series. Karpathy is such a good teacher that I'm actually looking forward to it. This week was the first time I trained a model myself and watched it work, and I didn't expect that to feel as good as it did.

I also took notes along the way, covering what each video teaches plus the details and easy mistakes I had questions about, and used AI to turn them into a proper set of notes. They’re on my GitHub. If you're going through the same videos, I hope they help.
