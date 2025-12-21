# Thinking in Loops: Scaling Visual ARC with Looped Transformers

Wen-Jie Shu¹²†, Xuerui Qiu³, Rui-Jie Zhu², Harold Haodong Chen¹, Yexin Liu¹, Harry Yang¹

¹HKUST, ²Bitdeer AI, ³CASIA

†Correspondence to: Wen-Jie Shu <[wenjieshu2003@gmail.com](mailto:wenjieshu2003@gmail.com)>

![param_efficiency_serif.png](param_efficiency_serif.png)

- **Cite this work**
    
    ```jsx
    @misc{shu2025loopvit,
      title         = {Thinking in Loops: Scaling Visual ARC with Looped Transformers},
      author        = {Wen-Jie Shu and Xuerui Qiu and Rui-Jie Zhu and Harold Haodong Chen and Yexin Liu and Harry Yang},
      howpublished  = {\url{https://www.notion.so/Thinking-in-Loops-Scaling-Visual-ARC-with-Looped-Transformers-2c7f56a03c7f8027b692dfe93a3abb18}},
      note          = {Notion Blog},
      year          = {2025},
      month         = {Nov},
      day           = {13}
    }
    ```
    
    ---
    

Historically, the ability to reason was widely considered the exclusive domain of Large Language Models or symbolic systems. Visual reasoning often felt like a contradiction in terms until the VARC framework redefined the boundaries. By successfully framing ARC-AGI as a vision-only generation task, [VARC](https://arxiv.org/pdf/2511.14761) proved that pixel-based architectures could handle abstract logic. However, while VARC solved the representation problem, the question of how to induce true Chain-of-Thought reasoning in vision remains under investigation.

Most current vision approaches rely on feed-forward architectures designed for fast pattern matching rather than deliberate thought. We look to the recent successes in Natural Language Processing for a better inductive bias. In NLP, **Looped Transformers have emerged as a powerful paradigm.** Because they recycle the same layers for multiple steps, they effectively introduce a "Latent Chain-of-Thought" at the architectural level. Given the success of recurrent models in language, ranging from pre-training efficiency ([Huginn](https://www.arxiv.org/pdf/2502.05171) and [Ouro](https://arxiv.org/pdf/2510.25741)) to specific synthetical reasoning tasks ([TRM](https://arxiv.org/pdf/2510.04871) and [HRM](https://arxiv.org/pdf/2506.21734)), it is only natural to ask if this architecture can unlock similar potential in Computer Vision.

In this blog, we introduce **Loop-ViT** to answer this question. This architecture brings iterative refinement to the visual domain. Our experiments reveal that this is more than just a successful adaptation. It represents a paradigm shift in efficiency.

We demonstrate that Loop-ViT works remarkably well. **As illustrated in the figure above, our approach sets a new Pareto frontier for model parameters versus accuracy on ARC-AGI.** The results speak for themselves:

- We prove that "thinking time" creates a more efficient scaling axis than model width. Our **5.9M** parameter looped model outperforms the **18M** VARC baseline (57.2% vs. 54.5% on ARC-1), achieving state-of-the-art results with **≈3× fewer parameters**.
- Scaling up computation through iteration yields diminishing returns less quickly than parameter scaling. A single **11.2M** looped model reaches **61.2%** accuracy on ARC-1, surpassing even the complex **VARC Ensemble** (60.4% with 73M Params). This demonstrates that a single model with ample "thinking time" can outperform a committee of feed-forward experts.

# **2 The Landscape: From "Seeing" to "Reasoning"**

## **2.1 Validating the Vision Perspective**

Historically, ARC was treated as a domain for symbolic programs or Large Language Models (LLMs), which convert visual grids into text tokens to leverage pre-training priors. However, the recent **VARC** framework challenged this dogma. By treating ARC as an image-to-image translation task, VARC demonstrated that standard vision backbones (like ViTs and U-Nets) could solve complex reasoning tasks purely from pixels.

> **The Takeaway:** VARC established a clean, vision-only testbed, proving that linguistic intermediates aren't strictly necessary. But it left one question unanswered: *is a single forward pass enough?*
> 

---

## **2.2 The Case for "Thinking Time" (Iterative Inference)**

While VARC showed *vision* is capable, its feed-forward architecture mimics "System 1" thinking—fast, intuitive, but prone to error on complex logic. Deep learning history, from [Recurrent Neural Networks](https://arxiv.org/pdf/2410.20672) to [Universal Transformers](https://arxiv.org/pdf/2511.14761), suggests that difficult problems require **depth-wise recurrence**—reusing parameters to refine representations over time.

This is where our work steps in. While scaling computation via iteration is gaining traction in [LLMs](https://arxiv.org/pdf/2410.20672) (e.g., Chain-of-Thought), **Looped Transformers** remain unexplored in vision-only rule induction. We argue that ARC is the perfect arena for this architecture: by replacing the "one-shot" prediction with a "looped" refinement process, we allow the model to hypothesize, check, and correct itself—all within a fixed parameter budget.

# **3 Methodology: Thinking in Loops**

![Figure 1: (A) VARC baseline pipeline. A standard feed-forward ViT produces a one-shot prediction from the task canvas. (B) Our looped pipeline. We keep the same VARC-style canvas representation and training/evaluation protocol, but replace the feed-forward backbone with a weight-tied (looped) transformer core executed for a fixed number of iterations K. Each iteration refines the hidden states and intermediate prediction, and the final output is taken from the last step.](image.png)

Figure 1: (A) VARC baseline pipeline. A standard feed-forward ViT produces a one-shot prediction from the task canvas. (B) Our looped pipeline. We keep the same VARC-style canvas representation and training/evaluation protocol, but replace the feed-forward backbone with a weight-tied (looped) transformer core executed for a fixed number of iterations K. Each iteration refines the hidden states and intermediate prediction, and the final output is taken from the last step.

## **3.1 The Setup: ARC on a Canvas**

We build directly upon the **VARC** paradigm. Instead of treating ARC as a sequence of discrete tokens, we treat it as a pure vision problem. Figure 1 contrasts the standard VARC pipeline (one-shot feed-forward backbone) with our looped variant. Crucially, we keep the canvas representation and the overall training/evaluation protocol unchanged, and modify only the backbone computation by reapplying the same transformer core for K latent iterations. 

- **Input:** We place all demonstration pairs (2–4 examples) and the query input onto a single, large fixed-size "canvas". This allows the model to "see" the entire task context at once, leveraging the spatial priors inherent in Convolutional Networks or Vision Transformers.
- **Goal:** The model must perform image-to-image translation, predicting the correct output pixels for the query region.

---

## **3.2 The Architecture: From Feed-Forward to Looped**

This is where we diverge. A standard VARC model is **feed-forward**: it encodes the canvas, processes it through a stack of transformer blocks, and outputs the prediction in one go. It has no opportunity to correct itself.

We introduce the **Looped Transformer**. The core idea is simple: instead of stacking $K$ *different* layers, we take a single block of layers and apply it $K$ times recursively.

> **💡 Key Concept: Weight Tying.** Imagine trying to solve a puzzle. You don't swap your brain for a new one every second; you use the *same* brain repeatedly to refine your thought. Similarly, our model reuses the **same parameters (**$F_\theta$**)** for every iteration. This allows us to increase the depth of reasoning (compute) without increasing the model size (memory).
> 

---

## **3.3 The Loop in Action**

Formally, we modify the inference pipeline as follows:

- **Initialization ($t=0$):** We embed the canvas into an initial hidden state:
    
    $$
    ⁍
    $$
    
- **The Loop ($t=1,..., K$):** We feed the hidden state back into the *same* Transformer core ($F_\theta$) repeatedly:

$$
⁍
$$

       *Note: $F_\theta$ is shared across all steps.*

- **Refinement & Prediction:** At each step, we can decode the hidden state to see what the model is "thinking":

$$
⁍
$$

       The final answer is simply the prediction at the last step $K$.

---

# **4 Experiments: Efficiency via Recurrence**

## **4.1 Setup: Standing on the Shoulders of VARC**

To ensure a fair comparison, we adopt the **VARC protocol** wholesale. We aim to isolate the benefit of "looping" by keeping the rest of the pipeline identical to the current state-of-the-art.

**Pipeline:** We use the exact same canvas representation, random scale/translation augmentations, and per-pixel classification objective as VARC.

**Two-Stage Training:**

- **Stage 1 (Offline):** We train a single model on ARC-1 training tasks (plus RE-ARC augmentations).
- **Stage 2 (Test-Time Training, TTT):** For every test task, we initialize a fresh task token and fine-tune using only the provided demonstrations. We use the robust augmentation strategy from VARC: combining flips, rotations ($90^\circ/180^\circ/270^\circ$), and 10 color permutations to generate 50 auxiliary tasks.

**Inference:** We employ **multi-view voting** with 510 random views to produce the final Pass@2 accuracy.

- **Optimization Details** (Click to expand)
    
    We mirror VARC’s hyperparameters exactly to ensure reproducibility:
    
    - **Optimizer:** Adam ($\beta=(0.9, 0.999)$)
    - **Learning Rate:** 3e-4 (cosine schedule, 10-epoch warmup)
    - **Epochs:** 100 for offline, 100 for TTT
    - **Batch Size:** 32 (offline), 8 (TTT)

---

## **4.2 Model Variants: Trading Width for Thinking Time**

![image.png](image%201.png)

![image.png](image%202.png)

![image.png](image%203.png)

![image.png](image%204.png)

![Figure 2: Offline training dynamics of Loop-ViT with fixed core depth (B=2, 4, 6, 8, 10) and different loop iterations (K ∈ {1,2,3}). We report grid-level exact match accuracy (entire output grid must be correct) on the training tasks (dashed) and a held-out evaluation split (solid) over epochs. Increasing K improves evaluation accuracy and reduces the train–eval generalization gap, suggesting iterative computation provides a beneficial inductive bias beyond feed-forward depth.](image%205.png)

Figure 2: Offline training dynamics of Loop-ViT with fixed core depth (B=2, 4, 6, 8, 10) and different loop iterations (K ∈ {1,2,3}). We report grid-level exact match accuracy (entire output grid must be correct) on the training tasks (dashed) and a held-out evaluation split (solid) over epochs. Increasing K improves evaluation accuracy and reduces the train–eval generalization gap, suggesting iterative computation provides a beneficial inductive bias beyond feed-forward depth.

How do we scale a Looped Transformer? We fix the embedding dimension ($d=512$, matching the VARC 18M Baseline) and explore two axes:

1. **Core Depth ($B$):** The number of physical Transformer blocks in the shared core (determines **Parameter Count**).
2. **Loop Steps ($K$):** How many times we recycle the core (determines **Compute/Inference Depth**).

We compare standard VARC models against our Loop-ViT variants. Note that while effective depth ($B \times K$) can grow large, the parameter count stays small because weights are tied.

| Core Depth (*B*) ↓  Loop Steps (*K*) → | **1 (No Loop)** | **2** | **3** |
| --- | --- | --- | --- |
| **B = 2** | 29.7 | 41.5 | 48.5 |
| **B = 4** | 41.0 | 52.2 | 57.4 |
| **B = 6** | 51.7 | 54.4 | 59.5 |
| **B = 8** | 53.1 | 55.2 | 58.8 |
| **B = 10** | 54.5 | 57.0 | 61.4 |

---

## **4.3 Main Results: Small Models, Big Gains**

![Figure 3: Training exact-match accuracy (grid-level) for Loop-ViT variants of different sizes (Small/Medium/Large) during offline training. Accuracy is measured as the fraction of tasks whose entire output grid is predicted correctly on the training split. Larger models fit the training set more strongly, motivating the need to evaluate improvements under held-out accuracy and to analyze whether gains arise from iterative refinement rather than memorization.](image%206.png)

Figure 3: Training exact-match accuracy (grid-level) for Loop-ViT variants of different sizes (Small/Medium/Large) during offline training. Accuracy is measured as the fraction of tasks whose entire output grid is predicted correctly on the training split. Larger models fit the training set more strongly, motivating the need to evaluate improvements under held-out accuracy and to analyze whether gains arise from iterative refinement rather than memorization.

We evaluated our final Loop-ViT variants (using a larger embedding $d=1024$ for Small and Medium,  $d=512$ for Large) on ARC-1 and ARC-2. Crucially, both ARC-1 and ARC-2 evaluations use the same checkpoint trained on the ARC-1 training set (plus [RE-ARC](https://github.com/michaelhodel/re-arc))**,** relying on Test-Time Training (TTT) to adapt to the specific tasks. The results challenge the "bigger is better" dogma.

<aside>
💡

***Key Findings:***

1. **Efficiency Victory:** Our small **5.9M Loop-ViT** achieves **57.2%** on ARC-1. This beats the standard **18M VARC model** (54.5%) while using **$\approx 3\times$ fewer parameters**.
2. **Scaling Up:** Our larger **11.2M looped model** reaches **61.2%** on ARC-1. Remarkably, this single model outperforms the elaborate **VARC Ensemble** (60.4% ), which combines multiple ViTs and U-Nets.
3. **Consistency:** These gains hold true on ARC-2 as well, where our looped models achieve **10.3%**, surpassing the single-model VARC baseline (8.3%).
</aside>

| Model | #params | K | ARC-AGI-1 | ARC-AGI-2 |
| --- | --- | --- | --- | --- |
| *large language models (LLMs)* |  |  |  |  |
| Deepseek R1 | 671B | - | 15.8 | 1.3 |
| Claude 3.7 8k | N/A | - | 21.2 | 0.9 |
| o3-mini-high | N/A | - | 34.5 | 3.0 |
| GPT-5 | N/A | - | 44.0 | 1.9 |
| Grok-4-thinking | 1.7T | - | 66.7 | 16.0 |
| Bespoke (Grok-4) | 1.7T | - | 79.6 | 29.4 |
| *recurrent models* |  |  |  |  |
| HRM | 27M | - | 40.3 | 5.0 |
| TRM | 7M | - | 44.6 | 7.8 |
| *vision models* |  |  |  |  |
| VARC | 18M | - | 54.5 | 8.3 |
| VARC (ensemble) | 73M | - | 60.4 | **11.1** |
| **Loop-ViT (Small)** | **3.1M** | 24 | 53.9 | 7.5 |
| **Loop-ViT (Medium)** | **5.9M** | 6 | 57.2 | 8.33 |
| **Loop-ViT (Large)** | **11.2M** | 6 | **61.2** | 10.3 |

---

# **5 Conclusion**

Our investigation reveals that **computational time** is a potent, yet underutilized, resource in visual reasoning. By introducing a Looped Vision Transformer, we demonstrated that **progressive refinement** yields better generalization than simply stacking more layers. Empirically, the results speak for themselves: a 5.9M looped model beats the 18M VARC baseline, and scaling to just 11.2M allows us to surpass the performance of large model ensembles. This confirms that for reasoning-heavy tasks, the ability to iterate and correct intermediate errors is just as critical as raw capacity.

---

# 6 Limitations and Future Work

## **6.1 Adaptive computation**

All experiments in this draft use a fixed (non-adaptive) number of loop iterations **K**, which may vary across model variants. While this setting cleanly isolates the effect of iterative computation, it is not compute-optimal: easy tasks may not require the full iteration budget, while harder tasks might benefit from more steps. A natural next step is to learn adaptive halting (e.g., early-exit policies or learned stopping criteria) to allocate iteration budget per instance, improving the accuracy–compute trade-off and enabling compute-aware deployment.

## **6.2 Beyond ARC: looped computation for other visual tasks**

ARC offers a controlled testbed for rule induction, but iterative refinement is broadly relevant to vision problems that require multi-step correction. An important direction is to integrate looped (weight-tied) transformers into other reasoning-heavy visual pipelines, such as image inpainting and editing (where the model progressively fixes structural inconsistencies), and video understanding/generation tasks (where temporal consistency often benefits from repeated refinement). We hope this work encourages the community to study computation scaling via iteration as a general-purpose tool for visual reasoning beyond ARC.

# Citation

If you find this blog useful, please consider citing:

<aside>
💡

@misc{shu2025loopvit,
title         = {Thinking in Loops: Scaling Visual ARC with Looped Transformers},
author        = {Wen-Jie Shu and Xuerui Qiu and Rui-Jie Zhu and Harold Haodong Chen and Yexin Liu and Harry Yang},
howpublished  = {\url{[https://www.notion.so/Thinking-in-Loops-Scaling-Visual-ARC-with-Looped-Transformers-2c7f56a03c7f8027b692dfe93a3abb18](https://www.notion.so/Thinking-in-Loops-Scaling-Visual-ARC-with-Looped-Transformers-2c7f56a03c7f8027b692dfe93a3abb18?pvs=21)}},
note          = {Notion Blog},
year          = {2025},
month         = {Nov},
day           = {13}
}

</aside>