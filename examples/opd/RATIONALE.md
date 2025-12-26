Below is a careful read of the post you sent plus an analysis of your “EAGLE‑3 as on‑policy distillation (OPD) with strict masking” idea. I’ll keep this crisp and technical.

---

## What the post actually argues (key takeaways)

**1) OPD = on‑policy sampling + dense, per‑token supervision.**
Instead of (a) off‑policy SFT on teacher trajectories or (b) sparse, sequence‑level RL rewards, *on‑policy distillation* samples trajectories from the **student** and asks a **teacher** to grade every token (typically with a reverse‑KL loss). This combines the distributional relevance of RL (you learn on the states you visit) with dense credit assignment like SFT. The post frames the loss as per‑token reverse KL (D_{\mathrm{KL}}(\pi_{\text{student}}|\pi_{\text{teacher}})) on the states visited by the student. ([Thinking Machines Lab][1])

**2) Reverse‑KL is the simple, effective choice.**
They implement OPD with a per‑token reverse KL between student and teacher next‑token distributions on the student’s rollouts. Reverse‑KL is “mode seeking,” discourages placing mass outside the teacher’s support, and reduces exposure bias because the supervision is on student‑visited states. Importantly, it doesn’t need rollouts to finish (no sequence‑level return), so you can use partial rollouts. ([Thinking Machines Lab][1])

**3) Compute economics are favorable vs. RL and long SFT runs.**
The post reproduces Qwen3 results showing OPD reaches strong reasoning scores at a fraction of RL GPU‑hours, and provides a FLOPs comparison suggesting **~9–30×** cost reduction relative to very large off‑policy SFT runs (and ~10× less than RL in the cited setup), because the expensive teacher is only queried for log‑probs, while sampling comes from the cheap student. ([Thinking Machines Lab][1])

**4) OPD is handy for continual learning / de‑forgetting.**
They demonstrate using an earlier checkpoint as the “teacher” to *restore* lost behaviors after mid‑training on new corpora—another practical plus of OPD. ([Thinking Machines Lab][1])

---

## Where EAGLE‑3 stands today (what we know from the paper)

EAGLE‑3 drops feature‑regression, **trains the drafter with token‑level supervision**, and introduces **training‑time test (TTT)**: during training it unrolls the drafter for multiple steps and **feeds back its own predictions**, using **custom causal masks** so the drafter only “sees” information it would have at inference. This also lets EAGLE‑3 **fuse low/mid/high target features** at step 1, then **replace unavailable future target features with the drafter’s own outputs** at steps 2..k. These changes unlock a favorable scaling curve and higher acceptance length vs. EAGLE‑2. ([ar5iv][2])

Concretely (paper notation):

* **Step 1** (legal to use target features): concatenate low/mid/high features from the target forward pass at the current prefix, plus the embedding of the sampled last token; pass through a single‑layer decoder; project with the **target LM head** to get token logits.
* **Steps 2..k** (cannot use “future” target features): **substitute** the previous drafter output for the missing target features; keep rolling.
* **Masks**: attention masks are rewritten to reflect the *tree‑like / unrolled* dependencies (vectorized dot‑products avoid wasted matmuls).
* **Objective**: token‑level prediction (CE) over the native step + simulated steps; feature MSE is removed. ([ar5iv][2])

---

## Your theory: “treat EAGLE‑3 as on‑policy distillation, but mask so we never use post‑verification activations”

**Short answer:** This is a sound and likely beneficial reframing. It dovetails with EAGLE‑3’s TTT machinery; the key is to **keep the drafter’s inputs “legal”** (exactly what TTT masking already enforces), while using the **teacher only for *losses* (log‑probs)** on the states produced by the drafter. Below I formalize this and discuss why it should improve acceptance length.

### A precise formulation (drop‑in for EAGLE‑3 training)

Let (q_\phi(\cdot \mid x_{\le t})) be the **drafter** distribution obtained by passing the drafter feature through the **frozen target LM head** (as in EAGLE‑3), and let (p(\cdot \mid x_{\le t})) be the **target** distribution. During TTT:

1. **Rollout (masked):**

   * Step 1: inputs may include fused target features at prefix (x_{\le t}) (legal).
   * Steps (s \ge 2): **mask out** all target features that would require verifying speculative tokens; instead feed in the drafter’s own outputs from step (s-1). (This is unchanged from EAGLE‑3.) ([ar5iv][2])

2. **Per‑token OPD loss:** for each drafted token (\hat{y}*{t+s}\sim q*\phi(\cdot \mid \hat{x}*{\le t+s-1})), query **only** the teacher’s *log‑probs* on that *visited* prefix and add a **reverse‑KL** term:
   [
   \mathcal{L}*{\text{OPD}} ;=; \mathbb{E}*{\hat{y}\sim q*\phi}\Big[\log q_\phi(\hat{y}\mid \hat{x})-\log p(\hat{y}\mid \hat{x})\Big]
   ]
   Optionally keep a small CE term to ground truth (like EAGLE‑3) for stability on supervised data:
   [
   \mathcal{L} ;=; \lambda\cdot \text{CE}(\text{gold}) ;+; (1-\lambda)\cdot \mathcal{L}_{\text{OPD}}
   ]
   Crucially, no target *intermediate features* from unverified steps are provided to the drafter—only the teacher’s **scalar** log‑prob supervision is used to update (\phi). This avoids “cheating” and respects speculative legality. ([Thinking Machines Lab][1])

3. **Masking:** reuse EAGLE‑3’s TTT attention masks for the unrolled steps so each prediction only conditions on information the drafter would have at inference (student tokens + step‑1 fused features only). ([ar5iv][2])

This is OPD “inside” EAGLE‑3’s TTT loop: the *inputs* and *masking* are identical to EAGLE‑3; only the **loss** changes from CE‑only to CE+reverse‑KL (or reverse‑KL only).

### Why this should improve acceptance length

* **Acceptance in speculative decoding** depends on the ratio (p/q): a drafted token is accepted with probability (\min(1,, p(y)/q(y))), and the expected acceptance rate (\alpha) grows as the two distributions get closer (e.g., (\alpha = \sum_y \min(p(y),q(y)) = 1 - \text{TV}(p,q))). Training on **student states** with reverse‑KL directly **pushes (q) toward (p)** *where it matters*—on the drafter’s own rollouts—so (\alpha) should increase, lengthening accepted runs per verify step. ([scale-ml.org][3])
* **Reverse‑KL penalizes over‑confident mass** that the teacher deems low‑probability (the regime that causes rejections), while being forgiving of under‑confidence (which still yields acceptance if the same token is sampled). This asymmetry aligns with the acceptance rule. ([Thinking Machines Lab][1])
* **OPD reduces exposure bias in the exact regime EAGLE‑3 cares about.** EAGLE‑3 already addresses train/test drift with TTT unrolling; OPD adds *teacher* shaping *on those drifted states*, which CE‑only can’t provide unless the gold token happens to match the sampled student token. ([Thinking Machines Lab][1])

### Where this differs from current EAGLE‑3

* **Supervision source.** EAGLE‑3 trains with token‑level CE (to dataset gold) while simulating future steps. OPD *replaces or augments* CE with **reverse‑KL to the target** on student‑visited trajectories. The masking and feature legality constraints **do not change**. ([ar5iv][2])
* **Compute pattern.** EAGLE‑3 has an **offline** option that precomputes target features once. OPD is **on‑policy**, so you pay teacher forward passes each batch (log‑probs only), similar to the blog’s recipe—but still dramatically cheaper than RL and, in many stacks, competitive with large‑scale SFT in wallclock/GPU‑hours. ([Thinking Machines Lab][1])

---

## A slightly stronger variant: acceptance‑aware OPD

If you want the objective to **track acceptance** even more directly, add a hinge on the **over‑confidence margin** (\Delta = \big[\log q_\phi(\hat{y}) - \log p(\hat{y})\big]_+) on student tokens (zero if (q\le p), positive if (q>p)). Minimizing this term pushes the drafter to be *no more confident than the teacher* on its chosen tokens—precisely the condition that leads to (\Pr[\text{accept}]=1). In practice:

[
\mathcal{L}*{\text{acc}} = \mathbb{E}*{\hat{y}\sim q_\phi}\big[,[\log q_\phi(\hat{y}) - \log p(\hat{y})]_{+},\big]
]

Use (\mathcal{L} = \lambda\cdot \text{CE} + (1-\lambda)\cdot (\text{reverse‑KL} + \beta \mathcal{L}_{\text{acc}})).
This is consistent with the acceptance rule in speculative sampling and targets the specific failure mode that triggers rejections. ([scale-ml.org][3])

---

## Practical training recipe (drop‑in)

1. **Keep EAGLE‑3 TTT dataflow/masks unchanged.** Step‑1 can use fused low/mid/high target features; steps 2..k substitute drafter outputs; all future target features are masked from inputs. ([ar5iv][2])
2. **For each drafted token in the unroll**, compute:

   * **Student log‑prob** via target LM head on the drafter feature (this is EAGLE‑3’s normal head).
   * **Teacher log‑prob** by running the frozen target on the *same* prefix (composed of student tokens so far).
   * **Loss** = reverse‑KL (and optionally CE + acceptance‑hinge). ([Thinking Machines Lab][1])
3. **Batching & cost:** amortize teacher log‑prob queries across the unrolled tokens; you don’t need logits over the whole vocab if you implement sampled reverse‑KL (only the student‑sampled tokens). This mirrors the blog’s compute‑efficient trick. ([Thinking Machines Lab][1])

---

## Likely benefits and failure modes

**Pros**

* **Higher acceptance length** via better calibration of (q) to (p) on student states (acceptance increases with similarity between (p) and (q)). ([scale-ml.org][3])
* **Less reliance on gold data**: CE uses only the gold path; OPD supervises *every token* the drafter actually visits, a better match to speculative acceptance dynamics. ([Thinking Machines Lab][1])
* **No correctness risk**: at inference you still verify against the target; training can’t “break losslessness.” (EAGLE‑3 already preserves strict acceptance.) ([ar5iv][2])

**Cons / caveats**

* **Teacher compute during training:** OPD needs teacher log‑probs on student tokens; that removes EAGLE‑3’s fully‑offline training path. If your target is very large, this is the main cost consideration. The blog’s results suggest the economics still work out well vs. RL and large SFT. ([Thinking Machines Lab][1])
* **Mode‑seeking bias:** reverse‑KL can collapse to a dominant mode. Here that’s mostly fine—acceptance prefers agreement on high‑mass tokens—but you might add a small CE term or temperature‑tune to avoid over‑peaking. ([Thinking Machines Lab][1])
* **Teacher mismatch at temperature:** If you verify at (T=0) but train OPD at (T>0) (or vice‑versa), your supervision distribution shifts. Match the verification temperature. (EAGLE‑3 reports speedups at both (T=0) and (T=1).) ([ar5iv][2])

---

## How I would test this (minimal ablations)

1. **Loss sweep:** CE‑only (EAGLE‑3 baseline) vs. reverse‑KL‑only vs. CE+reverse‑KL vs. CE+reverse‑KL+accept‑hinge; measure **acceptance rate (\alpha)** and **avg accepted length** at fixed (\gamma). (Speedup correlates with (\alpha).) ([scale-ml.org][3])
2. **Unroll depth:** hold cost fixed, sweep TTT unroll (k\in{1,2,4}); check whether OPD stabilizes deeper unrolls (expect yes). ([ar5iv][2])
3. **Teacher scale:** target = verifier; optionally try a slightly *larger* teacher for OPD than the runtime target, then distill back to the runtime target distribution (sanity check that aligning to the *actual* verifier distribution is best).
4. **Acceptance‑aware hinge:** check whether adding ([\log q-\log p]_+) reduces rejections without harming sampling diversity.

---

## Bottom line

Your headcanon—*“treat EAGLE‑3 training as OPD, but enforce the same masking so the drafter never consumes post‑verification target activations”*—is **well‑motivated** and **compatible** with the paper’s TTT mechanics. It should **raise acceptance length** by aligning (q) to (p) *on the states the drafter actually visits*, with costs largely limited to teacher log‑prob queries during training (no change at inference). That’s exactly the strength the OPD post highlights: dense, on‑policy supervision at a fraction of RL cost. ([Thinking Machines Lab][1])

If you want, I can sketch code‑level changes against an open EAGLE‑3 training script (loss hook + teacher log‑prob RPC + mask reuse), but the conceptual changes are the bullets above.

---

### References cited

* **On‑Policy Distillation blog** (Thinking Machines): reverse‑KL formulation, compute economics, and examples. ([Thinking Machines Lab][1])
* **EAGLE‑3 paper**: training‑time test, masking, multi‑layer feature fusion, token‑level objective. ([ar5iv][2])
* **Speculative acceptance math** (relationship between acceptance and (p,q), and min(1,p/q) acceptance rule). ([scale-ml.org][3])

*Additional context on EAGLE‑3 performance claims and implementation details is available from the Microsoft Research page, Hugging Face paper card, and SGLang docs.* ([Microsoft][4])

[1]: https://thinkingmachines.ai/blog/on-policy-distillation/ "On-Policy Distillation - Thinking Machines Lab"
[2]: https://ar5iv.org/pdf/2503.01840 "[2503.01840] EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test"
[3]: https://scale-ml.org/posts/speculative-decoding.html?utm_source=chatgpt.com "Speculative decoding |"
[4]: https://www.microsoft.com/en-us/research/publication/eagle-3-scaling-up-inference-acceleration-of-large-language-models-via-training-time-test/ "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test - Microsoft Research"
