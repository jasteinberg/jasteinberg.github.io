---
layout: page
title: Research
permalink: /research/
---

<p class="research-hero-caption">
"Poets say science takes away from the beauty of the stars —
mere globs of gas atoms. Nothing is 'mere.' I too can see the stars on a desert
night, and feel them. But do I see less or more?" — Richard Feynman
</p>

I am interested in how complex systems with many interacting degrees of freedom can give rise to collective behavior that cannot be understood simply by knowing the rules of interaction between individual components. Over my career this has led me across many areas — from strongly correlated electron systems to learning and memory in neural networks, and now large language models.

<!-- Figure sizing: add  style="--fig-w: 45%"  to any <figure>/<img>/.fig-cell (accepts % or px; caps at column width). Hero width = --hero-w below. -->
<img class="research-hero"
     style="--hero-w: 320px"
     src="{{ site.baseurl }}/assets/spherical_cows_ads.png"
     alt="A Poincaré-disk tiling of spherical cows — anti-de Sitter space">

<p class="research-hero-caption">
A tessellation of anti-de Sitter space by spherical cows.
</p>

## Quantum condensed matter & holography
I first became interested in condensed matter physics in 2010, when the Nobel
Prize was awarded for the isolation of graphene. As an undergraduate at Penn I
spent a summer in experimental nanoscale physics fabricating graphene
transistors, then turned to the electronic band structure of topological
semimetals — predicting, from first principles, a set of symmetry-protected
bulk Dirac points in distorted spinels.

In graduate school I moved to *strongly correlated* electron systems, where
band theory and the quasiparticle picture fail. The projects I took on were
scattered across different systems, but they had a feature in common: near a
quantum critical point, the real-time dynamics — how a system relaxes,
transports charge, scrambles information — has no conventional perturbative regime. Quantum and thermal fluctuations act on a single
"Planckian" timescale $\tau_\varphi \sim \hbar / k_B T$, with no separation of
fast and slow modes to organize an expansion — and because the imaginary-time
mapping is faithful only out to $\sim \hbar / k_B T$, and analytic continuation
does not commute with perturbation theory, the real-time, low-frequency
($\omega \ll T$) regime is exactly the one ordinary methods cannot reach. What
distinguishes the projects is mainly the non-perturbative method each one uses.

<strong>NMR relaxation in transverse-field Ising chain</strong>
The transverse-field Ising chain is an interacting spin model which yields perhaps the most canonical example of a quantum phase transition. Remarkably, under Jordan–Wigner transformation followed by a Bogoliubov transformation, the TFIC maps onto a system of free relativistic fermions; in
that frame my collaborators and I computed the NMR relaxation rate $1/T_1$ across the
quantum-critical fan, traced a factor-of-two discrepancy with
experiment on $\mathrm{CoNb_2O_6}$ to multi-fermion processes the continuum theory had
dropped, and obtained a universal, parameter-free scaling law testable by THz,
neutron, and NMR probes.

<strong>Quantum quench of the SYK model</strong>
The SYK model consists of $N$ Majorana fermions with random all-to-all
couplings in zero spatial dimensions. While no perturbative expansion in the coupling exists, the model becomes tractable in the large-$N$ limit. In this regime, one discovers a building block for a quantum-critical metal with no quasiparticles, whose low-energy effective theory admits a holographic dual in $\text{AdS}_{2}$. 
To study the out-of-equilibrium behavior of the model, my collaborators and I used the Keldysh closed-time contour formalism to obtain the Kadanoff–Baym equations describing the evolution of the two-point function after a quantum quench. We found that in the large-$q$ limit of the model, the local Green's function thermalizes *instantaneously*.
That instantaneous local thermalization is controlled by the reparameterization
(Schwarzian) modes: any theory with a Schwarzian effective action should
locally thermalize at once, even as higher-point correlations relax slowly.

<strong>Many-body quantum chaos in QED₃</strong> $\text{QED}_{3}$ is a theory of $N_f$ massless fermions coupled to a $U(1)$ gauge field and appears throughout condensed matter as the proposed low-energy theory for several lattice systems such as the
kagome antiferromagnet for $N_f=2$. The theory flows to a strongly interacting infrared fixed point, but it can be studied perturbatively in the large $N_f$ limit. With my collaborator I calculated the Lyapunov exponent — which sets the rate of information scrambling and many-body quantum chaos — by resumming the
ladder (Bethe–Salpeter) series for an out-of-time-order correlator. The result describes a fast scrambler that never violates the chaos bound $\lambda_L \le 2\pi / \beta$ — even in the regime where the $1/N_f$ expansion
itself ceases to converge.

<strong>Charge diffusion and the butterfly effect in disordered holographic matter</strong>
The AdS/CFT correspondence in the large-$N$ limit relates
strongly coupled quantum field theories to classical gravity with the
radial direction playing the role of energy scale and a black-hole horizon
geometrizing dissipation ($T_\text{QFT} = T_\text{Hawking}$). It is conjectured that the effective low-energy theory on the black hole horizon can provide insight into the behavior of strongly correlated quantum matter, in particular the "strange" metal phase of the cuprates. Quantities such as transport coefficients that are much more easily computed in the classical gravity theory can then be mapped back to the condensed matter system. 

One hope for AdS/CFT in condensed matter was to discover bounds on transport coefficients in strange metals. In fact, one of the early results out of gauge-gravity duality suggested a universal viscosity bound relating the shear viscosity $\eta$ to the entropy density $s$. A related observation in condensed matter was that the diffusion constants of charge and energy might be bounded by a relaxation time set by some velocity scale. It was proposed that this velocity may be the butterfly velocity, which sets the speed of propagation of perturbations in many-body quantum systems. To test the robustness of this bound, my collaborator and I studied a holographic dual to a metal with quenched
disorder, solved Einstein's equations in a fluid–gravity expansion for the resulting inhomogeneous
geometry, and computed the charge diffusivity $D$ and butterfly velocity $v_{B}$. However, we found that the disorder-averaged result gave an upper bound on $D$ rather than the
conjectured *lower* bound.

These projects are described in more detail in my
[thesis defense talk]({{ site.baseurl }}/assets/papers/thesis_defense_final.pdf)
and [dissertation](https://nrs.harvard.edu/URN-3:HUL.INSTREPOS:37365950). In my earlier grad school work, my collaborators and I studied a confinement transition from a fractionalized Fermi liquid (FL\*) reported on [here](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.94.024502).

<figure class="research-figure research-figure-grid" style="--fig-w: 100%">
  <div class="fig-grid">
    <div class="fig-cell">
      <img src="{{ site.baseurl }}/assets/figures/RESEARCH_FIG_QCM_qcising.png"
           alt="Quantum-critical phase diagram of the transverse-field Ising chain">
      <span>The quantum-critical fan: a $T = 0$ transition at $h_c$ that
      controls physics at finite temperature, between renormalized-classical
      and quantum-disordered regimes.</span>
    </div>
    <div class="fig-cell">
      <img src="{{ site.baseurl }}/assets/figures/RESEARCH_FIG_QCM_kruskal.png"
           alt="Kruskal diagram of the eternal black hole with a shockwave">
      <span>The two-sided global Kruskal geometry of an AdS black hole — the holographic interpretation
of the butterfly effect: the geometric shock wave producing the shift in the horizon on the right.</span>
    </div>
    <div class="fig-cell">
      <img src="{{ site.baseurl }}/assets/figures/RESEARCH_FIG_QCM_keldysh.png"
           alt="Keldysh closed-time-path contour">
      <span>The Keldysh contour showing forward and backward time evolution can be used to study
      real-time, out-of-equilibrium quantum dynamics.</span>
    </div>
    <div class="fig-cell">
      <img src="{{ site.baseurl }}/assets/figures/RESEARCH_FIG_QCM_ladder.png"
           alt="Ladder-diagram resummation of the two-particle vertex">
      <span>The ladder resummation for the squared fermion anticommutator. This quantity measures the effect that one local measurement has on another, and in systems exhibiting many-body chaos it initially grows exponentially.</span>
    </div>
    <div class="fig-cell" style="--fig-w: 70%">
      <img src="{{ site.baseurl }}/assets/figures/RESEARCH_FIG_QCM_spinel.png"
           alt="Crystal structure of a distorted bismuth spinel with zigzag Bi chains">
      <span>Distorted bismuth spinels: a first-principles prediction of symmetry-protected bulk 3D Dirac points, from Bi zigzag chains threading the octahedral framework.</span>
    </div>
    <div class="fig-cell" style="--fig-w: 52%">
      <img src="{{ site.baseurl }}/assets/figures/RESEARCH_FIG_QCM_flstar.png"
           alt="Figure from the FL* superconductivity paper">
      <span>FL* superconductivity transition: a fermionic spinon excitation of a $\mathbb{Z}_{2}$ spin liquid represented as a bound state of a bosonic spinon and a vison. Upon doping, the $\mathbb{Z}_{2}$ spin liquid can lead to an FL* state. </span>
    </div>
  </div>
  <figcaption>
  </figcaption>
</figure>

## Computational neuroscience & NeuroAI

Toward the end of my PhD and during my postdoc my research interests shifted to computational neuroscience, studying learning and memory through minimal, coarse-grained network models — simple enough to analyze in closed form, yet pitched at the network level where cognition lives. The central question guiding my interests is the following: how can knowledge be learned, represented, and stored in the brain so that it is at once *robustly
decodable* — recoverable from partial or degraded cues — and *durable*, persisting even as the underlying neural circuits are remodeled over time? 

<strong>Circuit expansion for learning in neural networks</strong>
In one line of work, motivated by the roles of neurogenesis and synaptic pruning in the brain, my collaborators and I showed how expanding a neural network's input during learning can improve generalization. Geometrically, a sparse random expansion lifts the inputs into a higher-dimensional feature space where patterns that were not linearly separable become separable, with the sparsity of the expansion controlling the correlations between different training examples in the extended part of the network. Working in a teacher–student setting and computing the generalization error using the replica trick, we showed that expanding the input during learning — either by adding neurons with random activity or by passing the input through a random mapping that produces sparse outputs — lets the network find the robustly generalizable max-margin solution and acts as a form of slack regularization, provided the expansion keeps the overlap between training examples low. The improvement in generalization survives even when the added neurons are pruned away once training is complete, so the benefit lives in the learning process rather than in any structure that remains in the final network.

<strong>Associative memory of structured knowledge</strong>

In later work, my collaborator and I developed a model of associative memory for structured knowledge, where the stored items are not isolated patterns but sets of relations between entities and their roles, such as episodic memories, temporal sequences, cognitive maps, and semantic structures. The role–filler pairs in a structure are represented through quadratic binding operations from vector-symbolic architectures — such as circular convolution, as in holographic reduced representations — where the dimensionality of the role–filler pair is the same as that of each individual item. The pairs are then superposed into a single distributed vector, so an entire structure lives as one point in the network's state space. These structures are stored as fixed points of a recurrent network with associative (Hebbian) plasticity, and a signal-to-noise analysis shows the conditions under which the individual constituents can be decoded from a whole structure once it has been retrieved from partial cues — including, strikingly, from cues lying outside the network's basins of attraction, a mode of retrieval beyond ordinary pattern completion. The analysis also yields a scaling law, $L \sim N^{1/3}$, relating the length $L$ of a reliably retrievable structure to the network size $N$, the sub-linear growth reflecting the interference noise that accumulates as more relations are packed into one superposed vector.

This work is described in more detail in a [talk](https://www.youtube.com/watch?v=rh_-Vh9CGbg) I gave at the van Vreeswijk Theoretical Neuroscience Seminar (VVTNS).


<figure class="research-figure" style="--fig-w: 50%">
  <img src="{{ site.baseurl }}/assets/figures/RESEARCH_FIG_SPOL.png"
       alt="Four-panel schematic of a teacher–student perceptron: a teacher mapping inputs to a noisy output, the baseline student, the student with added expansion neurons during learning, and the expanded-input readout.">
  <figcaption>
  A student network learns from a noisy teacher and can achieve better generalization by expanding the network during learning.
  </figcaption>
</figure>
<figure class="research-figure" style="--fig-w: 70%">
  <img src="{{ site.baseurl }}/assets/figures/RESEARCH_FIG_NEURO.png"
       alt="Three-panel schematic of the Dictionary network: encoding role–filler pairs into a single working-memory vector by vector-symbolic binding, decoding a filler from a query via a learned cortical dictionary, and storing and retrieving structures from long-term associative memory.">
  <figcaption>
  A schematic of a Structured Knowledge Associative Memory (SKAM). The leftmost network is used to bind object/attribute pairs to form the knowledge structures, and the middle network is used to decode them. The rightmost network 
  illustrates the process of storing multiple binarized structures in a memory network and retrieving them by generating a retrieval cue from a subset of the relations in the structure.
  </figcaption>
</figure>

## AI alignment & interpretability

My current interests carry the same questions into transformers: what internal structure a network builds, when it appears during training, and how to read it back off the weights, with a particular interest in binding, honesty, and calibration. 

The binding problem from my associative-memory work reappears in transformers, where current interpretability methods recover a vocabulary of features but not yet the grammar that binds them.

In my current projects, I use mechanistic interpretability to localize the circuits and features that implement a behavior. Alongside this, I have a growing interest in alignment, with a focus on honesty — whether a model's internal representations carry a direction that tracks the truth of a statement independently of what the model asserts, and how that geometry relates to calibration and hallucination. A forthcoming [blog](/blog/) post develops these truth directions in detail.

---

*<small>Drafted with the assistance of Claude (Anthropic).</small>*
