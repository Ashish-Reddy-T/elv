# SpatialVLM: feasibility analysis and competitive landscape

**The SpatialVLM project is technically feasible but carries concentrated risk in two modules — GridCellRoPE3D (medium risk) and GATr integration (high risk) — while the remaining three modules rest on well-validated foundations.** The competitive landscape has intensified dramatically: SpaceMind reached **69.6% on VSI-Bench** in November 2025, and CVPR 2026 features two strong spatial VLM papers (G²VLM, VLM-3R). However, no competitor combines neuroscience-inspired positional encoding with equivariant geometric algebra — this intersection remains the project's strongest differentiator and most defensible contribution. The field is moving at roughly one major spatial VLM paper per month, creating significant timeline pressure despite the project's 1–2 year horizon.

---

## Module 1 — GridCellRoPE3D: strong foundations, manageable risk

**Technical Risk: MEDIUM**

The mathematical foundation for extending GridPE to 3D is sound. The original GridPE paper (Li et al., AAAI, arXiv 2406.07049) derives grid cell activation as a summation of Fourier basis functions from the VCO model, and this formulation explicitly generalizes to arbitrary dimensions. For 3D, the natural extension uses 4 tetrahedral basis vectors (FCC lattice directions at ~109.47° separation) rather than the 3 hexagonal directions used in 2D. The multi-scale encoding — with exponentially increasing periods matching biological grid cell modules — transfers directly.

Three independent lines of evidence validate 3D RoPE extensions in practice. **PointROPE** (LitePT, arXiv 2512.13689, ETH Zurich, December 2025) achieves 82.2 mIoU on NuScenes by simply splitting RoPE embedding dimensions across Cartesian axes — parameter-free and training-free. **PE-Field** (Bai et al., arXiv 2510.20385, October 2025) extends 2D RoPE to a volumetric 3D field using monocular depth estimation, achieving state-of-the-art novel view synthesis. Most critically, **MHRoPE** (arXiv 2510.23095, accepted ICLR 2026, adopted in Qwen3-VL) explicitly addresses the dimensionality scaling problem: rather than splitting a fixed 128-dimension head across more axes, MHRoPE dedicates entire attention heads to each spatial axis. The paper states that "partitioning fixed channel dimensions becomes untenable, whereas dedicating distinct heads to new dimensions offers a far more robust approach" — a direct architectural recommendation for adding depth as a fourth axis.

**The LoRA feasibility question has a positive answer.** A dedicated study on positional embedding interchangeability (arXiv 2408.11382) tested switching sinusoidal PE to RoPE in pretrained models using LoRA on self-attention modules only (2.35M trainable parameters, 1.11% of the model). The result was "negligible or no performance loss." DroPE (Sakana AI, 2025) demonstrated an even more radical change — completely dropping positional embeddings from a pretrained RoPE model — with brief recalibration recovering performance. These findings suggest that LoRA can adapt attention patterns to substantially different positional encoding geometries, though the GridCellRoPE3D change (from M-RoPE's Cartesian decomposition to tetrahedral Fourier basis) is more radical than anything tested so far.

**Depth noise from Depth Anything V2 is a real but manageable concern.** Wildlife monitoring benchmarks show MAE of **0.454m** with correlation of 0.962. Known failure modes include thin structures, reflective/transparent surfaces, and long-range inaccuracy. However, RoPE-style encodings care primarily about relative distances between tokens, not absolute positions. Systematic scale errors shift all z-positions uniformly, preserving relative structure. PE-Field already demonstrates that monocular depth noise is tolerable for 3D positional encoding in practice.

The interaction with Qwen2.5-VL's existing M-RoPE deserves careful handling. M-RoPE decomposes head dimensions into (temporal, height, width) chunks. Extending to (t, h, w, depth) requires either: (a) a four-way channel split, reducing frequency resolution per axis, or (b) the MHRoPE approach of head-level allocation. GC-VSA (arXiv 2503.08608, March 2025) provides additional design guidance — it implements 3D grid cell modules with cosine gratings along three axes, multi-scale encoding with a biologically motivated scale factor of 1.42, and explicitly notes the similarity to RoPE.

**Biggest threat:** The tetrahedral Fourier basis may not be the optimal 3D geometry. Neuroscience evidence from 3D grid cells in bats (Ginosar et al., Nature 2021; Grieves et al., Nature) shows irregular, complex distributions rather than clean tetrahedral symmetry. The idealized mathematical model may not capture the best encoding geometry for real-world indoor scenes.

**First ablation test:** Replace M-RoPE with a simple 4-axis Cartesian RoPE split (adding depth from Depth Anything V2 as the fourth axis) on Qwen2.5-VL-7B with LoRA fine-tuning. Measure spatial reasoning performance on CV-Bench 2D and VSI-Bench before and after. This tests whether depth-aware positional encoding improves spatial reasoning at all, before investing in the more complex tetrahedral Fourier basis.

---

## Module 2 — GATr integration presents the highest risk

**Technical Risk: HIGH**

The Geometric Algebra Transformer (Brehmer et al., NeurIPS 2023, Qualcomm AI Research) is an elegant architecture that provides **E(3)-equivariant** processing through Projective Geometric Algebra G(3,0,1). Each token is represented as a 16-dimensional multivector, and all operations (linear maps, attention, bilinear interactions) respect Euclidean symmetries. The architecture scales to thousands of tokens via dot-product attention with efficient implementations (Flash Attention, xformers). For the proposed configuration (8 blocks, 16 multivector channels, 32 scalar channels), each token carries 16 × 16 + 32 = **288 floats** — substantial but manageable.

However, four serious concerns emerge from follow-up research:

**PGA expressivity is limited without augmentation.** De Haan, Cohen, and Brehmer (AISTATS 2024, arXiv 2311.04744) found that basic projective geometric algebra is "not sufficiently expressive" — it cannot represent distances between points via inner product. Their improved version adds join bilinear layers. The planned implementation **must** use this improved PGA, not the basic version. This is non-obvious from the original GATr paper alone and represents a critical implementation detail.

**GATr convergence degrades dramatically when used end-to-end.** The hPGA-DP paper (Sun et al., July 2025, arXiv 2507.05695) — the first robotics application of GATr — reports that "P-GATr alone as denoising backbone has prohibitively slow convergence due to inherent geometric inductive biases and the complexity of multivector computations." Their solution was a hybrid architecture: GATr for state encoding/decoding only, with a standard U-Net/Transformer for the core computation. This finding directly challenges using GATr as a deep processing module for VLM features.

**Memory scaling is problematic for point clouds.** LaB-GATr (Suk et al., MICCAI 2024) found GATr "highly memory-intensive on large meshes" (>48GB VRAM for >200K vertices), motivating geometric tokenization with learned pooling. The hPGA-DP paper explicitly states that "embedding each point into a 16-dimensional multivector leads to impractical memory usage" for point clouds. Indoor scenes from monocular depth could easily produce 10,000+ 3D points, requiring aggressive subsampling.

**No one has connected GATr to VLMs.** The closest work is EquiLLM (arXiv 2502.11149, February 2025), which bridges equivariant geometric encoders with LLMs for molecular dynamics and protein design. Its critical architectural insight: the LLM receives **only invariant features** extracted from the equivariant encoder's output. Directional/spatial information is exclusively handled by equivariant modules. For GATr→LLM projection, this means extracting scalar/grade-0 components and inner products between multivectors, then projecting these invariants to the LLM hidden dimension. EquAct (Zhu et al., May 2025) demonstrates SE(3)-invariant language conditioning for robotics via Feature-wise Linear Modulation (iFiLM), providing another viable bridge pattern.

Compared to alternatives, GATr's main advantage is richness of representation. **EGNN** (Satorras et al., ICML 2021) is simpler but severely numerically unstable at depth ≥5 and uses only scalar distance features. **SE(3)-Transformers** (Fuchs et al., NeurIPS 2020) use expensive Clebsch-Gordan tensor products that scale poorly. **EquiformerV2** (ICLR 2024) excels at molecular data but is domain-specific. **Vector Neurons** (Deng et al., ICCV 2021) handle only SO(3), not full E(3). GATr remains the strongest option for general-purpose equivariant processing — the question is whether the domain gap from physics/biology to noisy indoor point clouds can be bridged.

**Biggest threat:** The convergence problem documented by hPGA-DP. If GATr cannot learn effectively from noisy monocular depth point clouds — which have fundamentally different statistical properties than the clean physics simulations and structured meshes it was designed for — the entire geometric reasoning module fails.

**First ablation test:** Process point clouds from Depth Anything V2 through an 8-block GATr, extract invariant features, and train a linear probe for simple spatial relationship classification (left/right/above/below/in-front/behind) on SpatialRGPT-Bench data. Compare against a standard PointNet++ baseline. If GATr doesn't outperform PointNet++ on this simple task with clean supervision, deeper integration is unlikely to succeed.

---

## Module 3 — dual encoding is the safest bet in the pipeline

**Technical Risk: LOW**

The SigLIP 2 + DINOv2 combination is the most empirically validated component of the architecture. SigLIP 2 (Google DeepMind, arXiv 2502.14786, February 2025) incorporates self-distillation and masked prediction losses that substantially improve dense feature quality over vanilla SigLIP, but **DINOv2 still leads on spatial tasks**. The EUPE paper (March 2025) confirms DINOv3 and PE-spatial outperform SigLIP 2 on NYUv2 depth and keypoint correspondence. As one March 2026 analysis summarizes: "SigLIP 2 captures semantics; DINOv2 captures visual structure, texture, shapes, and spatial composition." The complementarity remains strong.

Eagle (NVIDIA, ICLR 2025 Spotlight) validated that "simply concatenating visual tokens from complementary vision encoders is as effective as more complex mixing architectures." Its pre-alignment innovation — individually fine-tuning each non-text-aligned encoder with a frozen LLM before joint training — is the established approach for bridging DINOv2 to language model space. Cambrian-1 (NYU, NeurIPS 2024) evaluated **over 20 vision encoders** and found consistent performance improvements when adding more encoders, with its Spatial Vision Aggregator reducing token count while preserving multi-encoder benefits.

Multi-layer extraction from layers {4, 8, 16, 24, final} is strongly supported. Meta's Perception Encoder paper (April 2025, NeurIPS 2025 oral) is titled "The best visual embeddings are not at the output of the network" — its core thesis validates extracting intermediate features. LLaMA 3.2 Vision extracts from layers {4, 8, 16, 24, 31} for exactly this reason. DINOv2's own multi-layer probing (layers {10, 20, 30, 40} for ViT-g) shows significant gains over single-layer extraction for depth estimation.

The computational overhead is moderate. **SigLIP SO400M (~400M params, ~70–80 GFLOPs at 384px) plus DINOv2-L (~304M params, ~60–65 GFLOPs)** totals roughly 2× a single encoder's cost, but vision encoding is typically a small fraction of total VLM inference time. FastVLM (Apple, CVPR 2025) showed vision encoding can be under 15% of total latency. The dual-encoder total of ~704M params is far cheaper than scaling a single encoder to equivalent capability (InternViT-6B: 6B params; Perception Encoder-G: 1.9B params).

One alternative worth monitoring is Meta's **Perception Encoder** family (April 2025). PE-core, PE-lang, and PE-spatial demonstrate that a single contrastive model can match specialized encoders when features are extracted from the right intermediate layers. If PE weights become widely available and well-supported, it could simplify the architecture by replacing the dual encoder with a single PE model. However, the dual-encoder approach offers a stronger safety margin for spatial tasks.

**Biggest threat:** Token count explosion. Five layers × two encoders × ~729 patches each = ~7,290 tokens before compression. Even with aggressive pooling (SVA-style reduction to 576 tokens, as in Cambrian-1), this requires a well-designed learned aggregation layer that doesn't discard critical spatial information.

**First ablation test:** Compare Qwen2.5-VL-7B with its native SigLIP encoder against a version augmented with DINOv2-L features (channel-concatenated with Eagle-style pre-alignment) on VSI-Bench and SpatialRGPT-Bench. Test single-layer vs. multi-layer extraction (final only vs. {4, 8, 16, 24, final}) with learned weighted averaging. This establishes whether the dual encoder adds measurable spatial capability before building more complex fusion.

---

## Module 4 — gated cross-attention with norm balancing is novel and promising

**Technical Risk: MEDIUM**

The empirical motivation for this module is exceptionally strong. The "Beyond Semantics" paper (Qi et al., arXiv 2503.17349, CUNY, March 2025) provides three damning findings about current LLaVA-style VLMs. First, vision token embeddings have norms **10–100× larger** than text token embeddings, suppressing RoPE positional information and making the model insensitive to token ordering. Second, randomly permuting all vision tokens causes only **0.2–2.74% performance drops** on POPE, GQA, and CV-Bench — the model treats vision tokens as an unordered bag. Third, system prompts absorb ~72.1% of attention as attention sinks, with vision tokens dominating the remainder through brute-force norm dominance rather than meaningful cross-modal integration.

The "Why Is Spatial Reasoning Hard for VLMs?" paper (Chen et al., arXiv 2503.01773, ICML 2025) adds complementary evidence: image tokens constitute >90% of input but receive only ~10% of attention, spatial errors correlate with attention misdirection to irrelevant objects, and models show significant performance gaps between familiar relations ("left of") and unfamiliar ones ("behind"), suggesting reliance on language priors rather than visual evidence.

**Norm-balanced gated cross-attention addresses both problems simultaneously.** The Flamingo-style tanh(α) gating (initialized at 0, so the model starts as a pure text LLM and gradually learns to incorporate vision) prevents training instability. RMS norm matching of vision embeddings — calibrated to match text-token norms as proposed by "Beyond Semantics" — restores the influence of RoPE positional information. Cross-attention (rather than early fusion) separates the vision and text representation spaces, reducing the norm-dominance issue.

Recent spatial VLMs validate cross-attention fusion. **Spa3R** (arXiv 2602.21186, February 2026) found that residual cross-attention adapters outperform naive token appending by **+7.5%**, explicitly documenting "modality collapse" where VLMs ignore appended spatial tokens. **SpaceMind** uses camera-conditioned gating with biasing and importance weighting. **VLM-3R** (CVPR 2026) uses Spatial-Visual-View Fusion via cross-attention with LoRA fine-tuning. **LVLDrive** uses gradual fusion with a trainable gate that schedules introduction of 3D features over training to prevent catastrophic disturbance.

The LoRA + cross-attention training question is resolved by **Idefics2** (Laurençon et al., NeurIPS 2024): using LoRA to adapt backbone parameters while fully training new cross-attention layers yields more stable training than either fully frozen or fully trainable approaches. VLM-3R (CVPR 2026) confirms this pattern in practice — LoRA on the LLM backbone plus full training on 3D fusion cross-attention blocks. The zero-initialized MLP projector + residual connection pattern from Spa3R provides an additional safety mechanism.

**No existing paper combines norm balancing with gated cross-attention** — this appears to be a genuinely novel fusion design. SpaceMind's CGMF comes closest but uses camera conditioning rather than explicit norm calibration, and doesn't address the norm imbalance findings.

**Biggest threat:** Whether gated cross-attention truly solves the "bag of tokens" problem or merely adds parameters while the core issue (vision encoders not preserving spatial information in their feature ordering) persists. The "Beyond Semantics" finding applies specifically to LLaVA-style early fusion — cross-attention architectures may have different attention dynamics, but this hasn't been studied for norm-balanced designs specifically.

**First ablation test:** Apply RMS norm matching alone (without any architectural changes) to Qwen2.5-VL-7B's vision projector output and re-evaluate on spatial benchmarks. Measure the Position Sensitivity Index (PSI) and Cross-Modality Balance (CMB) metrics from "Beyond Semantics" before and after. If norm balancing alone doesn't improve PSI significantly, the premise for norm-balanced cross-attention weakens.

---

## Module 5 — RL post-training has validated recipes but navigation-specific pitfalls

**Technical Risk: MEDIUM**

GRPO has become the dominant RL algorithm for VLM post-training, with successful applications across spatial reasoning (SpatialThinker, MetaSpatial), navigation (VLN-R1, SeeNav-Agent), and general VLM improvement (VLM-R1, R1-V). The core mechanism — sampling G completions per prompt, normalizing rewards within the group as advantages, and applying a clipped PPO-style update — eliminates the need for a critic model and reduces compute by ~50% versus PPO.

The **vanishing advantages problem** documented by VL-Rethinker (Wang et al., arXiv 2504.08837, NeurIPS 2025) is a real concern. As training progresses, the proportion of examples with non-zero advantages drops from ~40% to below 20% within 256 gradient steps, causing the effective training batch to shrink and training to stall. Three mitigations exist: **Selective Sample Replay (SSR)** maintains a buffer of high-advantage experiences; **DAPO** and **MC-GRPO** modify the advantage normalization; **DrGRPO** normalizes with a global constant to eliminate length bias.

**fDPO from SpatialReasoner-R1** (Shen et al., arXiv 2506.21656, UIUC + Google) offers a compelling alternative. By applying different optimization parameters (β values) to descriptive grounding segments versus logical reasoning segments, fDPO achieves **+4.1% over standard DPO on spatial quality tasks and +9.0% on spatial quantity tasks**. The M3CTS data generation pipeline produces diverse reasoning trajectories with fine-grained spatial rewards. A key theoretical insight from "It Takes Two: Your GRPO Is Secretly DPO" shows the algorithms are fundamentally related — 2-GRPO (group size 2) matches 16-GRPO performance while reducing training time by >70%.

For navigation specifically, **vanilla GRPO handles only outcome rewards**, which is insufficient for long-horizon tasks with dozens of steps. Three extensions address this:

- **Time-Decayed Reward (TDR)** from VLN-R1 (arXiv 2506.17221): weights multi-step future actions with temporal decay
- **Step Reward Group Policy Optimization (SRGPO)** from SeeNav-Agent (arXiv 2512.02631): process rewards per navigation step with random step grouping
- **Value-model dense rewards** from OpenVLN: a separate value model assesses per-waypoint state quality, achieving +4.34% SR and +6.19% OSR on TravelUAV

**SpatialThinker** (Batra et al., arXiv 2511.07403, UC Santa Cruz) provides the strongest validation for the proposed reward design. Using **lexicographically gated multi-objective rewards** (format → accuracy → count → spatial) with only **7K synthetic training samples**, SpatialThinker-7B nearly doubles base-model gains versus sparse RL and surpasses GPT-4o on 3DSRBench (56.4%). The lexicographic gating is explicitly designed to prevent reward hacking — spatial rewards activate only when format and accuracy conditions are met.

RL post-training consistently shows better OOD transfer than SFT across R1-V, VLM-R1, MultihopSpatial, and SpatialThinker. Sim-to-sim transfer from Matterport3D to continuous environments (VLN-CE) has been validated (Krantz & Lee, ECCV 2022), showing +12% success rate improvement, though some performance degradation occurs.

**Staged curriculum training is well-validated** across multiple papers. MedCCO's close-ended → open-ended curriculum yields +11.4% on in-domain tasks. E2H Reasoner (arXiv 2506.06632) provides theoretical analysis showing curriculum RL requires fewer total samples. A critical caution from FASTCURL: vanilla staged RL suffers significant performance drops between stages and needs mixed-strategy transitions.

**Biggest threat:** Reward hacking in navigation, specifically geodesic progress gaming. Agents can oscillate near the goal accumulating small positive progress rewards without completing the task. SpatialThinker's lexicographic gating is the current best mitigation, but indoor navigation with continuous actions introduces more opportunities for exploitation than VQA-style tasks.

**First ablation test:** Apply GRPO with a binary success reward (no dense rewards) to Qwen2.5-VL-7B on a simple ObjectNav task in HM3D. Measure whether the model develops any spatial reasoning improvement from RL alone. Then add geodesic progress as a dense reward and check whether the lexicographic gating from SpatialThinker prevents reward hacking. This establishes whether RL adds value before investing in the full reward engineering pipeline.

---

## Qwen2.5-VL-7B is the clear backbone choice

The LLM backbone comparison yields a decisive winner. **Qwen2.5-VL-7B-Instruct** is the only 7B-class VLM with a natively multi-dimensional positional encoding in its LLM fusion stage. M-RoPE decomposes head dimensions into (temporal, height, width) chunks, with `position_ids` shaped as `[3, seq_len]`. Extending to a fourth depth axis requires adding one row and repartitioning the embedding dimension — a structurally straightforward modification. The key code locations are in HuggingFace Transformers' `modeling_qwen2_5_vl.py`, specifically `Qwen2_5_VLRotaryEmbedding` and `get_rope_index`.

Qwen2.5-VL-7B is empirically the **strongest 7B-class model on spatial benchmarks**. The "Spatial Reasoning is Not a Free Lunch" paper (arXiv 2603.12545, March 2026) found it "the strongest frontier model overall" for spatial reasoning, leading on CV-Bench 2D, MMVP, VSR, TopViewRS, and CountBenchQA. The paper attributes this partly to M-RoPE preserving 2D structure during multimodal fusion.

Its LoRA ecosystem is the strongest: supported by Unsloth, LLaMA-Factory, HuggingFace PEFT, and multiple community repositories. Apache 2.0 license is fully permissive. Known issues include a PEFT bug (#2880) where LoRA gradients on ViT QKV modules are zero unless `requires_grad=True` is manually set, and VRAM requirements of ~18GB for LoRA (12GB with QLoRA 4-bit).

**InternVL3-8B** is the runner-up with slightly better general multimodal scores, but its V2PE (Variable Visual Position Encoding) is fundamentally 1D — visual tokens get a sub-unit position increment δ < 1, compressing their "positional space" without encoding spatial structure. Modifying this for 3D would be far more invasive. **LLaVA-OneVision-7B** and **Molmo-7B-D** both use standard 1D RoPE with no spatial infrastructure.

One potential upgrade path: **Qwen3-VL-8B** (late 2025) inherits the M-RoPE framework with improved TM-RoPE (time-aligned), uses SigLIP 2 SO400M as its vision encoder, and achieves **85.8 on MathVista** versus Qwen2.5-VL's 68.2. If its architecture is sufficiently mature, it could be a drop-in upgrade.

---

## The competitive landscape is fast-moving with a clear uniqueness angle

The field has exploded in 2025–2026 with multiple strong spatial VLM systems now published at top venues. The current competitive state of play:

| System | Approach | VSI-Bench | Key Innovation |
|---|---|---|---|
| **SpaceMind** (Nov 2025) | InternViT + VGGT + Camera-Guided Fusion | **69.6%** | Camera as active guiding modality |
| **VLM-3R** (CVPR 2026) | VGGT geometry encoder + cross-attention fusion | 60.9% | 3D reconstructive instruction tuning |
| **Spa3R** (Feb 2026) | Self-supervised spatial field modeling | 58.6% | No explicit 3D supervision needed |
| **G²VLM** (CVPR 2026) | DINOv2 + Qwen2-VL Mixture-of-Transformer-Experts | — | Geometric + semantic expert sharing |
| **SpatialThinker** (Nov 2025) | GRPO + scene graph + dense rewards | Competitive | 7K samples, surpasses GPT-4o |
| **SpatialReasoner-R1** (Jun 2025) | fDPO + M3CTS LongCoT | — | +9.8% on SpatialRGPT-Bench |

**SpaceMind at 69.6% VSI-Bench sets a high bar.** It uses a data-centric RGB-only approach comparison showing ~45.4%, proving that architecture matters — data alone is insufficient. However, SpaceMind uses lightweight linear modules (camera-conditioned bias + gating) with no formal equivariance guarantees and no biologically inspired encodings.

**The SpatialVLM project differentiates on four axes that no competitor occupies.** First, neuroscience-inspired 3D positional encoding — no spatial VLM uses grid cell-inspired Fourier basis encodings. Second, equivariant geometric algebra — GATr provides provable E(3) equivariance through 16-dimensional Clifford algebra multivectors, fundamentally different from the ad-hoc fusion modules in SpaceMind or VLM-3R. Third, the combination of RL training with equivariant geometric processing is unprecedented. Fourth, norm-balanced gated cross-attention specifically targeting the "bag of tokens" problem with calibrated vision-text integration is novel.

The scooping risk is **asymmetric**: high for general "better spatial VLM" contributions (SpaceMind's team at Shanghai AI Lab could easily iterate), but low for the specific neuroscience + PGA + RL combination. The technical specificity is a natural moat — no competing group works at this three-way intersection.

Key groups to monitor: Shanghai AI Lab (SpaceMind, G²VLM, InternVL series), UT Austin VITA Group (VLM-3R), HUST Vision Lab (Spa3R), UIUC PLAN Lab (SpatialReasoner-R1), UC Santa Cruz (SpatialThinker), and Google Research (original SpatialVLM). Current SOTA on additional benchmarks: VLN-SRDF (ICLR 2025) **surpasses human performance** on R2R; Efficient-VLN reaches 64.2% SR on R2R-CE; ObjectNav methods achieve ~46% SR zero-shot on HM3D.

---

## Risk summary and recommended validation sequence

| Module | Risk | Biggest Threat | First Ablation |
|---|---|---|---|
| **1: GridCellRoPE3D** | MEDIUM | Tetrahedral geometry may not be optimal; LoRA adaptation to radical PE change untested at scale | Add 4th depth axis to M-RoPE (simple Cartesian split), LoRA fine-tune, measure PSI and spatial benchmark delta |
| **2: GATr** | **HIGH** | Convergence failure on noisy depth point clouds (documented in hPGA-DP); no VLM integration precedent | GATr → linear probe for spatial relationship classification vs. PointNet++ baseline on Depth Anything V2 point clouds |
| **3: Dual Encoding** | LOW | Token count explosion from multi-layer × dual encoder; learned aggregation may discard spatial information | DINOv2 channel-concat with Eagle pre-alignment vs. SigLIP-only on VSI-Bench/SpatialRGPT-Bench |
| **4: Gated Fusion** | MEDIUM | May not solve "bag of tokens" if the root cause is encoder-side, not fusion-side | RMS norm matching alone (no architecture change) on Qwen2.5-VL; measure PSI/CMB improvement |
| **5: RL Training** | MEDIUM | Reward hacking via geodesic progress gaming in continuous navigation | Binary GRPO on simple ObjectNav → add dense rewards with lexicographic gating → check for exploitation |

**Recommended validation order:** Run Module 2 (GATr) and Module 1 (GridCellRoPE3D) ablations first — these are the highest-risk, most novel components. If GATr fails to outperform simple baselines on spatial relationship classification from depth point clouds, consider a fallback to PointNet++ or even a simple MLP on pooled geometric features (sacrificing equivariance for reliability). If the 4-axis M-RoPE extension shows no spatial improvement, investigate MHRoPE-style head allocation before attempting the tetrahedral Fourier basis. Modules 3, 4, and 5 can proceed in parallel on well-established foundations while the critical modules are validated.

## Conclusion

SpatialVLM's architecture occupies a genuinely unique position in the competitive landscape — no existing system combines neuroscience-inspired positional encoding, equivariant geometric algebra, and RL post-training for spatial VLM reasoning. The scientific contribution is defensible precisely because of this unusual combination. The greatest risk lies in Module 2 (GATr), where the hPGA-DP convergence findings and the complete absence of VLM integration precedent warrant early kill-or-proceed testing. The greatest opportunity lies in the interaction between GridCellRoPE3D and GATr — if both work, the system provides both biologically motivated spatial encoding and mathematically principled geometric reasoning, a combination that would be extremely difficult for competitors to replicate quickly. The Qwen2.5-VL-7B backbone choice is optimal and likely to remain so given the M-RoPE → Qwen3-VL upgrade path. Timeline pressure from SpaceMind's 69.6% VSI-Bench result suggests prioritizing the two high-risk ablations within the first 2–3 months, then committing to a 6–8 month integration push if results are positive.