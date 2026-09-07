---
published: true
layout: review
title: "The geometry of truth probes: a reader's map"
description: >-
  A synthesis of the linear-truth-probe literature, read through one question: when does a probe recover a legitimate truth direction, and what does the estimator return when it fails? With a guide to the datasets the field shares.
updated: 2026-09-07
---

*A working synthesis of the literature on linear probes for truth in language
models. Living document.*

This is a short literature review accompanying my
[blog](https://jasteinberg.github.io/blog/2026/truth-directions-snr/) post analyzing
truth probes. The goal is to make connections between various parts of the
literature that are not obvious when reading the papers in isolation. The organizing
question throughout is the following: Under what conditions does a probe recover a legitimate truth direction, and what does the estimator return when it does not? This question is explored in detail in the blog.

---

## The linear representation hypothesis

The linear representation hypothesis holds that a model encodes a concept as a
direction in activation space, so that it can be obtained via a linear readout. For
truth, **Marks & Tegmark, *The Geometry of Truth: Emergent Linear Structure in LLM
Representations of True/False Datasets*** —
[arXiv:2310.06824](https://arxiv.org/abs/2310.06824), COLM 2024 — propose a linear
readout obtained via a difference-in-means ("mass-mean") direction on true/false
statements. On their datasets they show that it separates held-out statements, transfers across
datasets, is causally implicated, and sharpens with scale. Two directions appear in their paper. The feature direction is
$\theta_{\mathrm{mm}} = \mu_1-\mu_0$, and for classification they read it through the
covariance, with weights $\theta_{\mathrm{mm}}^{\top}\Sigma^{-1}$ (their §5.1, equivalent
to LDA and recast as Mahalanobis whitening in their App. E). For intervention they steer
along $\theta_{\mathrm{mm}}$ itself. The blog finds that this pairing fails on
`counterfact`: steering along the raw direction moves behavior with the wrong sign, and
applying the same covariance correction to the steering direction restores it. The
correction they reserve for reading turns out to be needed for steering as well on that data. The
public datasets and an interactive explorer are at
[saprmarks.github.io/geometry-of-truth](https://saprmarks.github.io/geometry-of-truth/dataexplorer/),
with code at
[github.com/saprmarks/geometry-of-truth](https://github.com/saprmarks/geometry-of-truth).

**Zou et al., *Representation Engineering: A Top-Down Approach to AI
Transparency*** — [arXiv:2310.01405](https://arxiv.org/abs/2310.01405) (2023) — generalize the readout beyond truth: their linear artificial tomography (§3.1.1) designs stimuli for a concept, collects the neural activity they evoke, and fits a linear model to it, with honesty one target among many.

By contrast, **Bürger, Hamprecht & Nadler, *Truth is Universal: Robust Detection of
Lies in LLMs*** — [arXiv:2407.12831](https://arxiv.org/abs/2407.12831), NeurIPS
2024 — suggest a more complicated picture in which a single mass-mean vector is
just a first-order approximation. They show that truth lives in an *at least*
two-dimensional subspace, represented via a general truth direction $t_G$ that
holds for affirmative and negated statements alike, plus a polarity-sensitive
direction $t_P$. The subspace is fit by OLS on per-topic-centered activations
($\hat a_{ij} = \mu_i + \tau_{ij}\,t_G + \tau_{ij}\,p_i\,t_P$, their Eq. 3, with truth
label $\tau_{ij} \in \{-1,+1\}$ and dataset polarity $p_i \in \{-1,+1\}$). There is no
covariance correction and no null. Their layer selection maximizes the ratio of between- to
within-class variance. That is the criterion Bao et al. and Poulis et al. both adopt
from them. Poulis et al. (App. B.1) write it out explicitly as an isotropic Fisher ratio of squared
mean differences to variances *averaged across dimensions*. In the blog this becomes $d'(u)$, whose square is the same ratio evaluated along a single direction $u$ rather than averaged over all directions, that is, the class gap projected onto 
$u$ over the within-class standard deviation along $u$, then calibrated against a random-direction null.

In Bürger et al.'s decomposition, the affirmative-only direction is a linear
combination $t_A = \alpha\, t_G + \beta\, t_P$, where $t_G$ is the general
direction and $t_P$ is the polarity-sensitive direction. In the language of this
map, this reads as a *recoverability* result as much as a structural one, though of a
different kind than the rogue-dimension failure from the
[blog](https://jasteinberg.github.io/blog/2026/truth-directions-snr/).
There the failure is finite-sample: a weak class gap lets the estimated
$\hat\delta$ collapse onto the dominant variance axis, and a rank-one correction
repairs it. Here the failure is population-level identifiability: $t_P$ is stable,
reproducible structure, recovered as the second principal component of the
truth-related variance at cosine $0.97$ across six model families, and an
affirmative-only training distribution simply cannot separate it from $t_G$, so
the estimator returns a mixture no amount of data would fix. Both are
estimator-side accounts. In neither case does the model lack a truth direction, but the
remedies differ. One enriches the training distribution, the other corrects the
readout.

Two later observations bound the structural reading. **Poulis, Crovella & Terzi,
*Testing the Limits of Truth Directions in LLMs*** —
[arXiv:2604.03754](https://arxiv.org/abs/2604.03754) (2026) — find the 2D subspace
marks a transitional phase in depth: the polarity direction dominates the
truth-related variance early, the two tie at exactly the layer Bürger et al.
analyze, and the general direction takes over by the mid layers. This implies that
the second dimension is a stage of the computation rather than a depth-invariant
property of truth. Their probes trained on affirmatives at layers 4–10 are
*anti-predictive* on negations, AUROC $\approx 0$, which is the anti-transfer signature the
blog measures in Pythia, reproduced in Llama. They also identify a task-difficulty
boundary: truth directions break to chance once assessment requires counting over
more than two items. Bürger et al. themselves report residual, topic-specific
separability after projecting out both $t_G$ and $t_P$. Reading that alongside the results above, one concludes that the apparent
dimensionality of truth tracks whatever the training distribution happens to contain.
$t_G$ itself does generalize across polarity, which is an improvement over the
affirmative-only direction. Nonetheless, the truth *object* is better read as
one direction plus the confounds each training distribution fails to identify it
against.

The rotation of the recovered direction with depth is a general phenomenon rather than one specific to polarity. **Nordby, Pais & Parrack, *Linear Probe Accuracy Scales with Model Size and Benefits from
Multi-Layer Ensembling*** — [arXiv:2604.13386](https://arxiv.org/abs/2604.13386)
(2026) — find probe AUROC rises by roughly five points per decade of parameters
($R = 0.81$) across twelve models from three families, and that deception
directions *rotate* gradually across layers rather than sitting at one depth. This is the
standing objection to quoting a single best layer, and the reason the blog reports
its sweeps at every depth. Their ladder begins above the 70m and 410m models where
the blog's transition is visible, so they measure the slope in the regime where
probing already works while the blog measures its onset.

The below-chance transfer from affirmative to negated statements was reported before Bürger et al.'s
decomposition explained it. **Levinstein & Herrmann,
*Still No Lie Detector for Language Models: Probing Empirical and Conceptual
Roadblocks*** — [arXiv:2307.00175](https://arxiv.org/abs/2307.00175),
*Philosophical Studies* 182(7):1539–1565 (2025, online-first 2024) — show probes
trained on affirmative statements score *worse than chance* on negated ones in
five of six datasets, even after training on the positive analogues of the
negations they are tested on. From this they draw the general conclusion: a probe
most directly learns to predict the *label*, not the truth-value, and the two
coincide only to the extent that the dataset's labels track truth rather than
something correlated with it. This means that a probe can latch onto any property correlating with truth on the training set. The blog takes this seriously in two ways: every probe is benchmarked against a random-direction null, and when a probe fails the question asked is what the estimator has returned instead of a truth direction.

## The standard datasets

Almost every result in this map is measured on one of four families of data: the
curated statement sets, the topic sets, the CCS benchmark suite, and the QA transfer
sets. Reported accuracies are comparable only across papers drawing on the same
family. Several disagreements in this literature dissolve once the data is matched.
The construction details also matter
directly to the blog's argument: a concentrated within-class spectrum is the default
condition on this data, and what varies from set to set is whether the class gap grows
enough with depth to pull a difference-in-means estimator off the leading axis. Templated sets tend to be the ones where the gap grows and free-form sets the ones where it does not. The template is a marker of that difference, but not its cause.

**Family 1: the curated statement sets.** Marks & Tegmark's twelve datasets are the
common currency of the supervised branch, and the ones the blog uses.

| Dataset | Construction | $N$ |
|---|---|---|
| `cities` | "The city of [city] is in [country]." | 1496 |
| `neg_cities` | negations of `cities`, formed by adding "not" | 1496 |
| `sp_en_trans` | "The Spanish word '[word]' means '[English word]'." | 354 |
| `neg_sp_en_trans` | negations of `sp_en_trans` | 354 |
| `larger_than` | "x is larger than y." | 1980 |
| `smaller_than` | "x is smaller than y." | 1980 |
| `cities_cities_conj` | conjunctions of two `cities` statements | 1500 |
| `cities_cities_disj` | disjunctions of two `cities` statements | 1500 |
| `companies_true_false` | company claims, from [Azaria & Mitchell (2023)](https://arxiv.org/abs/2304.13734) | 1200 |
| `common_claim_true_false` | various claims, from Casper et al. (2023) | 4450 |
| `counterfact_true_false` | factual recall, from [Meng et al. (2022)](https://arxiv.org/abs/2202.05262) | 31960 |
| `likely` | nonfactual text with likely or unlikely final tokens | 10000 |

Marks & Tegmark divide these into two categories: curated and uncurated. The curated
sets are built to be
uncontroversial, unambiguous, and simple enough that the model plausibly understands
them. The uncurated three (`companies`, `common_claim`, `counterfact`) are harder
test sets adapted from other sources and carry no such guarantee. Papers that report a
single headline number across "the Marks & Tegmark datasets" are averaging over that
distinction.

Three construction choices defuse shortcuts a probe
could otherwise exploit. In `cities`, the false country is not sampled uniformly but
with probability equal to that country's frequency among the true statements, so that
statements ending in a given country name are not disproportionately true. This
prevents the shortcut of simply looking for the country token. In the compound sets,
each conjunct is drawn
true with probability $1/\sqrt{2}$ for the conjunction and $1 - 1/\sqrt{2}$ for the
disjunction, which balances the compound label while leaving the two conjuncts'
truth values uncorrelated, so a probe cannot score by reading only the first
statement. The `likely` set is not a truth dataset at all: it is LLaMA-13B's own
unconditioned generations, with the final token either the most likely or the
hundredth most likely continuation. It is a pure probability axis included as a
distractor, and a probe tracking salience rather than truth should fail there
specifically.

The negation sets carry most of the weight in the polarity literature. Following
Levinstein & Herrmann, a negated statement is formed by inserting "not", so it differs
from its affirmative partner by a single token while carrying the opposite label, which is exactly what makes the pair diagnostic, and what the general/polarity decomposition of
Bürger et al. is fit against. It also means the polarity direction $t_P$ is a
direction for one syntactic operation applied by one rule, not for negation in
general. `neg_cities` and `neg_sp_en_trans` supply a second control for free: on
those sets truth is strongly *anti*-correlated with the model's own probability for
the sentence, so a probe reading plausibility rather than truth scores below chance
rather than above it.

`counterfact_true_false` is the odd member of the family and the one the blog's failure case rests on. It is referred to as free-form because its statements share no common template or topic, and this property is implicit in how Marks & Tegmark describe their two groups. Their curated sets are built for *controllable*
structural and topical diversity: within any one dataset the statements follow a fixed
template and a single topic, and the variation is placed between datasets rather than
inside them. The uncurated sets are adapted from prior work to check that a direction
found on the curated data generalizes. Their statements are more diverse and by the
authors' own account sometimes ambiguous, malformed, controversial, or hard to
understand. `counterfact_true_false` is uncurated in exactly that sense. It comes from Meng et al.'s CounterFact, which was built for model *editing* rather
than probing. Marks &
Tegmark create `counterfact_true_false` by keeping the statements that form complete sentences, pairing each true version with
a false one drawn from CounterFact's suggested false modifications, and appending a
period. Their two example statements share no frame at all: one reporting the language
a person spoke and the other the official religion of a sultanate, whereas `cities` reuses
one frame with two slots filled in. Free-form is therefore the
condition under which the blog finds the mass-mean estimator collapsed onto the
leading within-class axis at every depth, while on templated `cities` that condition
lifts as the class gap grows. Its nominal $N$ of $31{,}960$ is also two orders of
magnitude above the smallest set here, so any comparison across these datasets has to
subsample.

**Family 2: the topic sets.** Bürger et al. work from statements curated by Azaria &
Mitchell and Marks & Tegmark, organized as six topics (`animal_class`, `cities`,
`element_symb`, `facts`, `inventors`, `sp_en_trans`), each in affirmative, negated,
conjunctive, and disjunctive form. Sizes run from $164$ (`animal_class`) to $1{,}496$
(`cities`), which is the smallest $N/d$ ratio in the map, against the $d = 4096$
models it is used on.

Bao et al. report two defects in this set. The `inventors`
topic is ambiguous through duplicated names, which they repair by marking the role
explicitly ("The inventor Thomas Edison lived in the U.S."). More consequentially,
Bürger et al.'s disjunctions write their subjects as pronouns where the conjunctions
write them in full, so a probe compared across conjunction and disjunction is partly
reading a syntactic difference rather than a logical one. Bao et al. restore the
subjects before comparing. Any conjunction-versus-disjunction gap quoted from the
unmodified set inherits that confound.

**Family 3: the CCS benchmark suite.** Burns et al. evaluate on ten datasets recast as
yes/no contrast pairs: IMDB and Amazon (sentiment), AG-News and DBpedia-14 (topic),
RTE and QNLI (entailment), COPA and Story-Cloze (story completion), BoolQ (question
answering), and PIQA (commonsense reasoning). These are not truth datasets in the
sense of Family 1. Their native labels are sentiment, topic, and entailment. "Truth"
is the correctness of a candidate label appended to the prompt. A method that finds
the most prominent feature and a method that finds truth are not distinguishable on
that data. For example, on IMDB the most prominent feature of the activation is
sentiment, and sentiment is what the label tracks, not truth. Farquhar et al. narrow
to three of the ten (IMDB, BoolQ, DBpedia), the others excluded for legal reasons or
for poor accuracy in the original.

**Family 4: the QA transfer sets.** Bao et al. test whether probes trained on atomic
statements carry to question answering, using MMLU and TriviaQA for parametric
knowledge and SciQ, BoolQ, and XSum for knowledge grounded in the prompt. The
distinction they draw is that contextual truthfulness and factual
correctness are different targets, since faithfully following a false context is
correct behavior for the first and not the second. A probe scored across both is
being scored against two different notions of what it is supposed to read.

Two properties cut across all four families. Every set is small relative to the
ambient dimension, $N$ of $10^2$–$10^3$ against $d$ of $10^3$–$10^4$, so all of this
work sits below Cover's separating capacity, which is what the blog's $N \gtrsim 2d$
table quantifies. Furthermore, the choice between a templated and a free-form set is not a
neutral one. In the blog's measurements it is the templated sets whose class gap grows
with depth. On `cities` the participation ratio climbs $1.9 \to 29.7$ by layer $28$
and the alignment of $\hat\theta$ with the leading within-class axis falls to $0.159$,
while free-form `counterfact` stays pinned at participation ratio $\approx 1$ and
alignment $\approx 1$ at every layer. Which family a result was measured on therefore
decides whether a difference-in-means estimator had anything to lock onto.

## Three critiques of linear probes

The following three references are often cited together to provide criticisms of
linear truth probes. They discuss three distinct failure modes described below:

- **Burns, Ye, Klein & Steinhardt, *Discovering Latent Knowledge in Language Models
  Without Supervision*** — [arXiv:2212.03827](https://arxiv.org/abs/2212.03827),
  ICLR 2023. Contrast-Consistent Search (CCS) finds a truth direction *without
  labels* by demanding logical consistency across a statement and its negation. That constraint does not single out truth: any property that flips under negation satisfies it, so what CCS returns is a direction consistent under negation, which on this data is typically the most prominent such feature rather than truth. 
- **Farquhar, Varma, Kenton, Gasteiger, Mikulik & Shah, *Challenges with
  Unsupervised LLM Knowledge Discovery*** —
  [arXiv:2312.10029](https://arxiv.org/abs/2312.10029) (2023) — prove that a class
  of arbitrary binary classifiers is optimal under the CCS consistency loss, and
  show empirically that unsupervised methods recover the *most prominent* feature,
  not knowledge: an **identifiability/salience** failure. The salience-beats-truth
  result is a signal-versus-noise story. Levinstein & Herrmann's conceptual
  argument (above) anticipates it: coherence alone cannot isolate truth, since
  $\Pr(x \wedge P(x)) + \Pr(\neg x \vee \neg P(x)) = 1$ holds for *any* sentence
  property $P$.
- **Roger, *What Discovering Latent Knowledge Did and Did Not Find*** —
  [Alignment Forum (2023)](https://www.alignmentforum.org/posts/bWxNPMy5MhPnQTzKz/what-discovering-latent-knowledge-did-and-did-not-find)
  — finds more than twenty mutually orthogonal probes whose accuracies match the one
  CCS returns, and notes that untrained, randomly initialized probes reach ~75% on the
  easy datasets (with UQA, once the CCS convention of flipping a below-chance probe is
  applied): a **dimensional-slack** failure. He separately finds that CCS test loss is
  often *higher* than a constant prediction, so the orthogonal probes match on accuracy
  rather than on loss. **Mallen & Belrose, *Eliciting Latent Knowledge from Quirky Language
  Models*** — [arXiv:2312.01037](https://arxiv.org/abs/2312.01037) (2023) — later
  formalize the observation into an explicit baseline: spherically uniform probe
  weights, sign resolved to $\geq 0.5$ AUROC on the source distribution, transfer
  quantified against the empirical distribution of $10^7$ random-probe AUROCs. This
  is the direct precedent for the blog's random-direction null. The per-layer nulls,
  $d'$, and the margin-above-null criterion are the contribution of the blog.

A counterpoint on the salience failure: **Laurito, Maiya, Dhimoïla, Yeung & Hänni,
*Cluster-Norm for Unsupervised Probing of Knowledge*** —
[arXiv:2407.18712](https://arxiv.org/abs/2407.18712), EMNLP 2024 — restore
separability against distracting salient features via cluster normalization.

To summarize, Burns et al. seek the truth direction via unsupervised methods,
Farquhar looks at *which* direction such an unsupervised objective selects, and Roger
analyzes how much apparent separation *any* direction achieves, which is set by the
ambient dimension relative to the sample size rather than by the probe's expressivity.

## The adjudicating tools

The following references provide the methodology for building controls to
evaluate probes:

- **Cover, *Geometrical and Statistical Properties of Systems of Linear
  Inequalities with Applications in Pattern Recognition*** — IEEE Trans.
  Electronic Computers **EC-14**(3):326–334 (1965)
  [[PDF](https://web.njit.edu/~usman/courses/cs675_fall19/10.1.1.366.5645.pdf)]. The
  function-counting theorem: for $N$ points in general position in $\mathbb{R}^d$
  there are $C(N,d) = 2\sum_{k<d}\binom{N-1}{k}$ homogeneously separable
  dichotomies, so nearly every labeling is linearly separable while $N < 2d$.
  That is the reason separability of a dichotomy is nearly information-free below the
  separating capacity $N = 2d$, and hence the reason a null is the baseline that
  matters.
- **Hewitt & Liang, *Designing and Interpreting Probes with Control Tasks*** —
  [arXiv:1909.03368](https://arxiv.org/abs/1909.03368), EMNLP 2019. Control
  tasks fit the probe to random labels, and the gap (selectivity) measures whether
  it reads structure or memorizes.
- **Kumar, *Pressure-Testing Deception Probes in LLMs: Scaling, Robustness, and
  the Geometry of Deceptive Representations*** —
  [arXiv:2605.27958](https://arxiv.org/abs/2605.27958) (2026). A permutation null
  applied to a spectrum rather than to a probe: PCA on the paired activation
  differences, each eigenvalue compared to a sign-flip null, and the number of
  significant components $k^{*}$ comes out zero both pooled and within every domain
  (Gemma 3, 1B–27B). Probes with $k \ge 5$ dimensions nonetheless beat a single
  direction, so the deception signal is spread across many directions none of which
  clears the null alone. This is the opposite regime to the blog's rogue dimension,
  where one direction dominates the spectrum and carries no label information. Kumar also
  traces probe fragility under stylistic shift to a narrow training distribution
  rather than to the representation, since style-augmented probes recover
  near-perfect detection on unseen styles. The domain is system-prompt-induced
  deception, not statement truth.
- **Bao et al., *Probing the Geometry of Truth: Consistency and Generalization***
  — [ACL Findings 2025](https://aclanthology.org/2025.findings-acl.38.pdf). A
  modern metric-rich reproduction on Llama-3.1-8B (AUROC / ECE / Brier).
- **Diedrichsen, Berlot, Mur, Schütt, Shahbazi & Kriegeskorte, *Comparing
  Representational Geometries Using Whitened Unbiased-Distance-Matrix
  Similarity*** — [arXiv:2007.02789](https://arxiv.org/abs/2007.02789), *Neurons,
  Behavior, Data Analysis, and Theory* (2021). The whitening move in its native
  setting: when noise is correlated across units, an unwhitened inner product
  reports on the noise rather than the geometry, so they whiten the estimation
  errors of the representational dissimilarity matrix. The blog's reason for
  reaching for $\Sigma^{-1}$ is theirs.

Hewitt & Liang's capacity story **does not apply to a mass-mean probe**, which
fits only two class means and has no capacity to shrink. When such a zero-capacity
probe still separates shuffled labels, the cause is ambient dimension (Cover), not
expressivity, so the fix is $N \gtrsim 2d$ or an explicit null, not a smaller
probe. Roger's 75% for a randomly initialized probe is the companion fact about a typical direction. Cover's theorem says that below capacity some direction fits almost any labeling. Roger's number says that a random direction, once its sign is resolved on the data, already scores well above chance. The blog's random-direction null measures the second quantity directly, per layer and with a distribution rather than a single number.

## Representation geometry and its connection to truth

The truth-direction papers above analyze geometry in their own terms, Marks & Tegmark
through PCA visualizations and Bürger et al. through a two-dimensional truth subspace.
However, none of them (nor Poulis et al., Levinstein & Herrmann, or Farquhar et al.)
cite the literature on activation anisotropy and outlier dimensions, or the whitening
and control-task methodology that goes with it. Instead, it enters through the steering
and generalization work of the next section, which draws on the papers described below.

- **Timkey & van Schijndel, *All Bark and No Bite: Rogue Dimensions in Transformer
  Language Models Obscure Representational Quality*** —
  [arXiv:2109.04404](https://arxiv.org/abs/2109.04404), EMNLP 2021. A handful of
  "rogue dimensions," often just 1–3, dominate cosine similarity and obscure
  representational quality, and standardizing them repairs the metric. This is the
  prior-art term for the confound the blog diagnoses.
- **Dettmers, Lewis, Belkada & Zettlemoyer, *LLM.int8(): 8-bit Matrix
  Multiplication for Transformers at Scale*** —
  [arXiv:2208.07339](https://arxiv.org/abs/2208.07339), NeurIPS 2022. *Outlier
  features*: whole channels with large magnitude across most tokens, emerging past
  ~6.7B parameters.
- **Sun, Chen, Kolter & Liu, *Massive Activations in Large Language Models*** —
  [arXiv:2402.17762](https://arxiv.org/abs/2402.17762), COLM 2024. A few scalar
  activations reach magnitudes in the thousands, emerge abruptly after one layer,
  sit on delimiter and start tokens, and act as fixed input-agnostic bias terms.
  (They note these are *distinct* from Dettmers' outlier features: a massive
  activation is a scalar at a few tokens, an outlier feature is a vector across
  most tokens.)
- **Roh, Cho & Kim, *Embracing Anisotropy: Turning Massive Activations into
  Interpretable Control Knobs for Large Language Models*** —
  [arXiv:2603.00029](https://arxiv.org/abs/2603.00029) (2026). The counterpoint
  reading of the lineage: massive-activation dimensions as intrinsic,
  domain-specialized functional units, identified by a training-free magnitude
  criterion and used to *restrict* steering to a sparse critical set.
  Their reading is coordinate-basis and domain-semantic where the blog's diagnosis is
  eigenbasis and estimator-level, and the two are compatible rather than conflicting: a direction can be
  functional for the model and still be nuisance for a truth estimator.

The bridge between these literatures is one I have not found stated elsewhere: the
mass-mean estimator has no signal to lock onto when the class-mean gap is small, so
it **silently returns the dominant axis of the within-class covariance**. This is why
the same method that works on
templated `cities` fails on free-form claims, measured directly as
$\cos(\hat\theta, \hat v_1) = 0.994$ with $d'_{\mathrm{mm}}$ at chance, and
$\lambda_1/\lambda_2 = 535$ on `counterfact` at layer $28$ of pythia-2.8b.

The relation to massive activations needs stating carefully, because the two notions
are not the same object. Sun et al.'s criterion ($>100$ in magnitude, $>1000\times$
the median) applies to individual activation coordinates, and is applied here to the
per-coordinate mean over statements. The rogue dimension is a property of the within-class *covariance*, and is basis-free. They
coincide at early layers and come apart at depth: on `counterfact` at layer $8$,
$\hat v_1$ carries $86\%$ of its mass on one coordinate whose mean activation is
$1446\times$ the median, so there it is a massive activation in their sense. By layer
$28$ the eigenvector has delocalized across five coordinates. And `cities` at layer
$28$ retains the massive activation ($563\times$ the median) while having no rogue
dimension at all. A massive activation is nearly constant across statements, so it
inflates the mean without inflating the covariance. It produces a rogue dimension only
when the remaining variance is small enough for it to dominate.

## Decodability versus causation

**Arditi et al., *Refusal in Language Models Is Mediated by a Single Direction***
— [arXiv:2406.11717](https://arxiv.org/abs/2406.11717), NeurIPS 2024 — show
refusal is causally mediated by a single difference-in-means direction, across 13
open-source chat models. They show ablating it bypasses refusal and adding it
induces refusal. Read against the truth results, this shows that decodability and
causal efficacy are different measurements: the same difference-in-means recipe
yields a direction that is causal for refusal and, on `counterfact`, one that decodes
nothing yet steers the wrong way. Therefore, each needs its own
control tests.

**Braun, Eickhoff, Krueger,
Bahrainian & Krasheninnikov, *Understanding (Un)Reliability of Steering Vectors in
Language Models*** — [arXiv:2505.22637](https://arxiv.org/abs/2505.22637), ICLR
2025 Workshop on Foundation Models in the Wild — characterize the regime in which
steering works at all. Steering is predicted to succeed when there is directional
coherence of activation differences and separability of positive and negative
activations along the difference-of-means line ($d'$), and to fail when the
behavior is not represented as a coherent linear direction. This is the closest
published neighbor to the blog's recoverability/SNR lens. The rogue-dimension
result supplies a mechanism and correction for one failure mode it characterizes,
without leaving the linear class. The per-sample version comes from **Tan, Chanin,
Lynch, Paige, Kanoulas, Garriga-Alonso & Kirk, *Analysing the Generalisation and
Reliability of Steering Vectors*** —
[arXiv:2407.12404](https://arxiv.org/abs/2407.12404), NeurIPS 2024. Their
*steerability*, the slope of a linear fit through a logit-difference propensity
curve, is the same summary the blog measures as $\chi$. The blog adds the
even/odd split separating degradation from signed effect, and the random-direction
null. The blog's harness displaces every position while their intervention displaces the last token position only, so magnitudes are not directly comparable. In addition, they
show that steerability is highly variable per input: for several concepts close to
half the inputs steer in the *opposite* direction to the one intended. The blog's
wrong-signed `counterfact` effect is the dataset-level version, a significantly
negative mean response with a mechanism and a rank-one correction.

**Cho, Wu, Da Costa &
Koshiyama, *The Confidence Manifold: Geometric Structure of Correctness
Representations in Language Models*** —
[arXiv:2602.08159](https://arxiv.org/abs/2602.08159) (2026) — supply a
random-*direction* steering null and a low-dimensional discriminative subspace (3–8 dims),
adjacent to the blog's participation-ratio result.

**Torop, Masoomi & Dy, *Inverted Detection and Control in Steering Vectors*** —
[arXiv:2608.02957](https://arxiv.org/abs/2608.02957) (2026) — find mean-difference
directions in attention-head outputs (Gemma 3 12B, Qwen 2.5 14B, Olmo 3 7B) that
discriminate the concept well yet consistently push behavior the other way, which they
distinguish from Tan's per-input anti-steerable cases and from Braun's
low-discriminability failures. The blog's `counterfact` inversion is the complementary
case, a wrong sign from a direction that reads nothing, and its mechanism, alignment
with a label-blind variance axis, is unavailable when the direction reads well.

**Ying, Ravfogel, Kriegeskorte & Hase, *The Truthfulness Spectrum Hypothesis*** —
[arXiv:2602.20273](https://arxiv.org/abs/2602.20273) (2026) — reconcile the
transfer literature with the truthfulness spectrum hypothesis: truth directions
occupy a spectrum from domain-general to domain-specific, so pairwise transfer can
fail while a jointly trained general direction still exists. To measure how far apart two probe directions are, they use a Mahalanobis cosine,
defined as the inner product taken in the metric of the activation covariance, which predicts
cross-domain generalization at $R^2 = 0.98$ while the raw cosine only obtains $0.56$. This is due to the representations being anisotropic, which is the
same property the blog measures as a participation ratio. Both papers adopt the
whitening procedure from Diedrichsen et al. (above), whom Kriegeskorte co-authors.
A follow-up by **Ying, Hase & Kriegeskorte, *Comparing Linear Probes with Mahalanobis
Cosine Similarity*** — [arXiv:2606.19603](https://arxiv.org/abs/2606.19603) (2026) —
explains why the metric works: under a Gaussian model both held-out AUROC and the
Mahalanobis cosine to the Fisher direction are monotone functions of one
signal-to-noise ratio, which is the blog's $d'(u)$ under the pooled within-class
covariance. They also list the conditions under which that law fails. One of them is a
difference-of-means probe that sits far from the Fisher direction, which is the blog's
regime. Ying et al.'s causal section reports a decoding/steering dissociation of its own:
steering along the domain-general direction slightly degrades truthfulness while
domain-specific directions improve it.

The decoding and steering results together sort into four categories:

|  | **Causally effective** | **Miscausal or inert** |
|---|---|---|
| **Decodable** | `cities` at depth; Arditi's refusal direction | Torop's inverted vectors (wrong sign); Ying et al. (slight degradation); Liu (null) |
| **Not decodable** | Nadaf's function vectors\* (\*logit-lens readout, not a trained probe) | deep `counterfact` (wrong sign); shallow `counterfact` (inert) |

*Decodable and causal*: Arditi's refusal direction, and `cities` at depth, where the
plain mass-mean direction steers correctly and whitening only adds estimation
noise, the regime mass-mean probing intends. *Decodable but miscausal*: three
signatures with three mechanisms. Torop et al.'s inverted vectors read the concept well
and push behavior the other way. Ying et al.'s domain-general direction slightly degrades
truthfulness because it conflates factual with sycophancy-related variance. The third
comes from **Liu, *Decodable but Not Corrected by Fixed
Residual-Stream Linear Steering: Evidence from Medical LLM Failure Regimes*** —
[arXiv:2605.05715](https://arxiv.org/abs/2605.05715) (2026) — a steering *null*
rather than a wrong sign: overthinking in medical QA is decodable while
twenty-nine fixed linear interventions produce $\Delta \approx 0$, diagnosed as
entanglement (88% of the contrastive signal shared with task computation, and
within-class variance $2$–$4\times$ the inter-centroid gap, a $d' < 1$ statement
without the formalism). A footnote finds a *within-class-whitened* erasure
direction reduces the damage, an independent nearby data point for the
$\hat\Sigma^{-1}$ correction.

*Steerable but not decodable*: **Nadaf, *Steerable but Not Decodable: Function
Vectors Operate Beyond the Logit Lens*** —
[arXiv:2604.02608](https://arxiv.org/abs/2604.02608) (2026) — function vectors
steer behavior while the unembedding-aligned readout decodes nothing at any
layer, across 12 tasks and 6 models, though the operationalization differs
(logit lens, not a trained probe).

*Not decodable but miscausal*: deep `counterfact`, where the plain mass-mean
direction decodes nothing, since the estimator has returned $\hat v_1$, yet steering
along it moves behavior significantly the wrong way. A rank-one correction restores
both the readout and the sign. *Not decodable and inert*: shallow `counterfact`, where no linear
correction recovers signal and the steering response is indistinguishable from that
of a random direction.

Overall, these results demonstrate that decodability and causal efficacy are
separate measurements with separate nulls.

---

*Further afield: **Kadavath et al., *Language Models (Mostly) Know What They
Know*** — [arXiv:2207.05221](https://arxiv.org/abs/2207.05221) (2022) — P(True)
self-evaluation and trained P(IK), with calibration on true/false improving with
scale. This is a possible later post rather than part of this
map.*

*<small>Drafted with the assistance of Claude (Anthropic).</small>*
