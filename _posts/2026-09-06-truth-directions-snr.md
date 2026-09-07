---
layout: post
title: "Truth Directions: Signal-to-Noise and Geometry of Recoverability"
date: 2026-09-06
description: "In final revision. When is a linear truth direction in a language model's activations recoverable at all, and what does the estimator return when it is not?"
---

This post is in final revision and will appear at this address in the coming days.

<div class="tldr gray" markdown="1">

A mass-mean "truth direction" can genuinely separate true from false, but it can also be the estimator collapsing onto the most *salient* axis in the activations — the direction of largest within-class variance. Along that axis the class gap is small compared with the spread, so it carries almost none of the recoverable signal. A truth direction and a salient axis look alike on a benchmark and require a signal-to-noise reading to tell them apart. This post takes up the question of recoverability: *when* does a model contain a linear truth direction a probe can actually recover, which rises above a random-direction null and carries signal beyond the single most salient axis?

Systems neuroscience has spent decades asking what a downstream reader can recover from a population of noisy neural units. In that spirit I treat probing for a truth direction as a readout problem: I take the probe as a linear readout, quantify its separation with a detection-theoretic $$d'$$, and benchmark that $$d'$$ against an explicit random-direction null across the Pythia scale ladder. This reading yields three results:

**(1) Apparent separation is trivial.** Below Cover's capacity ($$N \ll 2d$$ throughout), random directions already separate the classes — so the null, not the raw score, is the bar.

**(2) The estimator returns the wrong object when the signal is weak,** it returns the dominant activation axis $$\hat v_1$$ rather than a truth direction. As a result steering along the retrieved direction moves behavior with the wrong sign. Recoverability comes down to whether the class gap grows with depth until it dominates the spread along that axis. On `cities` it does, on `counterfact` it never does, and this holds in both model families.

**(3) The failure is in the estimator, not the model:** the offending direction is identified in advance from the within-class spectrum, not from the steering outcome. Removing it, without leaving the linear class, corrects the sign of the steering behavior.

</div>