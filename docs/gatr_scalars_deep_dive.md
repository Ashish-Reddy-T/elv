# GATr in SpatialVLM: From `[B, 1369, 3]` to 32 Learned Scalars

This note explains **what the 32 scalar channels are**, **how they are created and updated**, and **how they interact with the 16×16 multivector stream** in the reference **GATr** implementation (`REPOS/geometric-algebra-transformer`) as wrapped by `GATrWrapper`.

It is written to complement `dimensions_updated.md` Step 8 and to answer: *“Is attention just dot products? What mixes with what?”*

---

## 1. Starting point: patch 3D positions

After depth backprojection and per-patch aggregation you have:

- **`points_3d`**: `[B, 1369, 3]` — one 3D point per DINOv2 patch (camera frame, metres before optional normalization).

SpatialVLM’s wrapper (`src/spatialvlm/geometry/gatr_wrapper.py`) then does:

1. **Optional normalization** (per batch): scale all coordinates by `max_i ‖p_i‖` so geometry sits in a stable numeric range.
2. **PGA point embed**: `embed_point(coords)` → **`mv_in`**: `[B, 1369, 1, 16]`  
   One multivector channel per token, 16 PGA basis coefficients.
3. **Scalar seed**: `scalar_in = ‖coords‖` per token → **`[B, 1369, 1]`**  
   This is **not** “one of the 32 outputs” yet; it is the **initial scalar signal** (distance from origin after normalization).

So before GATr’s stack you have **1 MV channel** and **1 scalar channel** per token.

---

## 2. Where do the “32 learned scalars” come from?

They are **not** hand-derived from depth. They are **32 channels of a learned latent**, produced by GATr’s **equivariant linear maps** and **geometric attention + MLP blocks**, exactly like hidden units in a transformer.

Concretely, `GATrWrapper` constructs:

```text
GATr(
  in_mv_channels=1,   out_mv_channels=16, hidden_mv_channels=16,
  in_s_channels=1,   out_s_channels=32,  hidden_s_channels=32,
  num_blocks=8,
)
```

The first operation inside `GATr.forward` is **`linear_in`**, an **`EquiLinear`** that maps:

- **`[B, 1369, 1, 16]` → `[B, 1369, 16, 16]`** (multivectors)
- **`[B, 1369, 1]` → `[B, 1369, 32]`** (scalars)

So the **32** is simply **`hidden_s_channels`**: the width of the **scalar hidden representation** for every patch token. After **8** `GATrBlock`s and a final **`linear_out`**, you still have **`scalar_out` with shape `[B, 1369, 32]`**.

**Intuition:** think of the scalar tensor as a **`[B, 1369, 32]` “side branch”** of plain real features living **next to** the multivector tensor **`[B, 1369, 16, 16]`**. Both are updated every block.

---

## 3. How scalars and multivectors talk to each other (the nitty-gritty)

GATr is **not** “only dot products on 3D points.” It is closer to: **two coupled streams** (MV + scalar) with **equivariant** ops on MVs and **standard** linear/inner-product ops on scalars, **wired together** at specific places.

### 3.1 `EquiLinear`: explicit cross-talk

From `gatr/layers/linear.py`, an equivariant linear layer does four conceptual things:

1. **MV → MV** via learned coefficients on **pin-equivariant basis maps** (structured 16×16 mixing **per MV channel**).
2. **Optional bias** only in the **scalar blade** of MV outputs (equivariance-safe).
3. **`s2mvs`**: `Linear(in_s_channels → out_mv_channels)` and the result is **added to `outputs_mv[..., 0]`** (the **scalar component** of each output multivector channel).  
   So **scalars directly push on the grade-0 part** of MV channels.
4. **`mvs2s`**: `Linear(in_mv_channels → out_s_channels)` applied to **`multivectors[..., 0]`** (scalar components of **all input MV channels**), producing part of the output scalars; plus **`s2s`**: `Linear(in_s → out_s)` if both exist.

So at **every `EquiLinear`**, information flows:

```text
MV (16-D per channel)  ←→  Scalars (width S)
     via scalar blade (index 0) and dedicated linear maps
```

That is already richer than “one dot product”: it is **bipartite mixing** between **(all MV scalar parts)** and **(all scalar channels)**.

### 3.2 One `GATrBlock` (attention + MLP)

From `gatr/layers/gatr_block.py`, each block is:

```text
mv, s  →  EquiLayerNorm  →  SelfAttention  →  residual
       →  EquiLayerNorm  →  GeoMLP         →  residual
```

Both **attention** and **MLP** take **(mv, s)** and return **(mv, s)**.

### 3.3 Geometric self-attention: the “dot product” analogy

Vanilla transformer attention uses **one** similarity:

```text
score(i,j) ∝ (W_q h_i) · (W_k h_j)
```

GATr’s **`GeometricAttention`** (see docstring in `gatr/layers/attention/attention.py`) uses a **weighted sum of three similarities**:

1. **`pga_inner_product(q_mv[i], k_mv[j])`** — a **geometric inner product** between **multivector** queries and keys (can encode spatial relationships; not the same as Euclidean dot of 3D points).
2. **`inner_product(φ(q_s[i]), ψ(k_s[j]))`** — learned/feature-mapped scalar interaction.
3. **`euclidean_inner_product(q_s[i], k_s[j])`** — ordinary dot product **on the scalar channels**.

Then softmax over `j`, and values are mixed:

```text
out_mv[i] = Σ_j α_ij v_mv[j] / norm
out_s[i]  = Σ_j α_ij v_s[j]  / norm
```

So **the same attention weights** blend **both** MV and scalar values for each token.

**Covariance analogy (careful):** attention scores are **similarities** between token representations, not covariances. The **only** loose analogy is: both involve **pairwise comparisons** across items (`i` vs `j`). Covariance needs **many samples** of two variables; attention needs **two vectors at positions i and j** at one forward pass.

### 3.4 `GeoMLP`: bilinear coupling (improved PGA path)

From `gatr/layers/mlp/mlp.py`, the MLP starts with **`GeometricBilinear`**, which uses **geometric product / equivariant join** style operations (with a **reference multivector** from the point cloud) to produce **new MV channels** and **new scalars**. That is another place where **geometry and scalars are jointly transformed**, not independently.

Later layers use **`ScalarGatedNonlinearity`** and **`EquiLinear`** again, so scalars and MVs keep **re-mixing**.

---

## 4. End-to-end shape walk (SpatialVLM defaults)

| Stage | Multivector tensor | Scalar tensor |
|------|----------------------|---------------|
| After embed + `scalar_in` | `[B, 1369, 1, 16]` | `[B, 1369, 1]` |
| After `linear_in` | `[B, 1369, 16, 16]` | `[B, 1369, 32]` |
| After each of 8 blocks | `[B, 1369, 16, 16]` | `[B, 1369, 32]` |
| After `linear_out` | `[B, 1369, 16, 16]` | `[B, 1369, 32]` |

The wrapper then builds **invariants for the LLM**:

- **`mv_norms`**: `‖mv_out[b,n,c,:]‖₂` for each of **16** MV channels → `[B, 1369, 16]`
- **`invariants`**: `concat(scalar_out, mv_norms)` → **`[B, 1369, 48]`**
- **`MLPProjector(48 → 4096)`** → GATr token features.

So the **32** are the **raw learned scalar channels**; the **16** norms are **rotation-invariant summaries** of the **16** MV channels. Together they feed the projector.

---

## 5. Direct answers to common confusions

**Q: Are the 32 scalars “PGA scalar components”?**  
**A:** No. They are **arbitrary learned real channels** in the **scalar stream**. Only **parts** of `EquiLinear` read/write the **grade-0 slot** of multivectors (`[..., 0]`), which *is* tied to PGA’s scalar blade, but the **32-D vector** is not “32 blades.”

**Q: Do the 16 MV channels interact?**  
**A:** Yes — via **`EquiLinear`** (channel mixing with equivariant structure), **attention** (which mixes **items** and **heads**), and **GeoMLP** (bilinear + gated layers).

**Q: Is this just depth repeated 32 times?**  
**A:** No. Depth/point geometry enters once; the **32** are **hidden activations** shaped by **all layers** and **all tokens’ context** via attention.

---

## 6. References in this repo

- Wrapper: `src/spatialvlm/geometry/gatr_wrapper.py`
- GATr core: `REPOS/geometric-algebra-transformer/gatr/nets/gatr.py`
- Block: `gatr/layers/gatr_block.py`
- Attention: `gatr/layers/attention/self_attention.py`, `gatr/layers/attention/attention.py`
- Coupling linear: `gatr/layers/linear.py`
- MLP: `gatr/layers/mlp/mlp.py`, `gatr/layers/mlp/geometric_bilinears.py`

For the original research framing, see the GATr paper (NeurIPS 2023) and the improved PGA / join discussion in `docs/critique.md`.
