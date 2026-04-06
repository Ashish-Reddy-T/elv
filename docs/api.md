# SpatialVLM API Reference

Auto-generated from `src/spatialvlm/` source files.

---

## Table of Contents

- [spatialvlm.config](#spatialvlmconfig)
- [spatialvlm.encoders](#spatialvlmencoders)
- [spatialvlm.geometry](#spatialvlmgeometry)
- [spatialvlm.fusion](#spatialvlmfusion)
- [spatialvlm.backbone](#spatialvlmbackbone)
- [spatialvlm.utils](#spatialvlmutils)
- [spatialvlm.data](#spatialvlmdata)
- [spatialvlm.training](#spatialvlmtraining)
- [spatialvlm.eval](#spatialvlmeval)

---

## `spatialvlm.config`

**Module:** `spatialvlm/config/model.py`

All pre-trained model constants are marked with VERIFY -- they must be confirmed by loading the model config at runtime.

### `class EncoderConfig`

Configuration for the dual vision encoder stage (Stage 1).

| Field | Type | Default | Description |
|---|---|---|---|
| `siglip_model_id` | `str` | `"google/siglip2-so400m-patch16-naflex"` | HuggingFace SigLIP2 model ID |
| `dinov2_model_id` | `str` | `"facebook/dinov2-large"` | HuggingFace DINOv2 model ID |
| `siglip_image_size` | `int` | `384` | SigLIP input resolution (384/16 = 576 patches) |
| `dinov2_image_size` | `int` | `518` | DINOv2 input resolution (518/14 = 1369 patches) |
| `siglip_patch_size` | `int` | `16` | SigLIP patch size |
| `dinov2_patch_size` | `int` | `14` | DINOv2 patch size |
| `siglip_extract_layers` | `list[int]` | `[9, 18, 27]` | Multi-layer extraction indices (1-indexed) |
| `dinov2_extract_layers` | `list[int]` | `[8, 16, 24]` | Multi-layer extraction indices (1-indexed) |
| `proj_output_dim` | `int` | `4096` | Projector output dim (must equal LLM hidden size) |

### `class GeometryConfig`

Configuration for the geometric branch (Stage 2).

| Field | Type | Default | Description |
|---|---|---|---|
| `depth_image_size` | `int` | `518` | Depth map resolution (matches DINOv2 input) |
| `depth_percentile` | `float` | `0.15` | 15th-percentile depth aggregation |
| `gatr_blocks` | `int` | `8` | Number of GATr equivariant blocks |
| `gatr_mv_channels` | `int` | `16` | Multivector channels per token |
| `gatr_s_channels` | `int` | `32` | Scalar channels per token |
| `pga_dim` | `int` | `16` | PGA basis dimension (mathematical constant) |
| `n_tetra_dirs` | `int` | `4` | Tetrahedral directions for GridCellRoPE3D |
| `n_freqs` | `int` | `8` | Golden-ratio spaced frequencies |
| `base_freq` | `float` | `10.0` | Base frequency f_k = base_freq * phi^k |
| `golden_ratio` | `float` | `1.618033988749895` | Golden ratio phi |

**Properties:**

- `gatr_invariant_dim -> int` -- Invariant features dim = gatr_s_channels + gatr_mv_channels (48).
- `rope3d_dims -> int` -- Output dims: n_tetra_dirs * n_freqs * 2 = 64.

### `class FusionConfig`

Configuration for the fusion stage (Stage 3).

| Field | Type | Default | Description |
|---|---|---|---|
| `sva_num_queries` | `int` | `576` | SVA query count (matches SigLIP patches) |
| `sva_kv_tokens` | `int` | `3314` | Total KV tokens (576 + 1369 + 1369) |
| `sva_num_layers` | `int` | `2` | Number of SVA cross-attention layers |
| `cross_attn_layers` | `list[int]` | `[4,8,12,16,20,24,28,32,36]` | Gated cross-attention injection layers |
| `norm_ema_momentum` | `float` | `0.99` | RMS norm matching EMA momentum |
| `use_typed_attention_bias` | `bool` | `True` | Enable 3x3 learned type bias matrix |

### `class BackboneConfig`

Configuration for the LLM backbone (Stage 4).

| Field | Type | Default | Description |
|---|---|---|---|
| `model_id` | `str` | `"Qwen/Qwen3-VL-8B-Instruct"` | HuggingFace backbone model ID |
| `hidden_size` | `int` | `4096` | LLM hidden dimension |
| `num_hidden_layers` | `int` | `36` | Number of transformer layers |
| `num_attention_heads` | `int` | `32` | Query attention heads |
| `num_key_value_heads` | `int` | `8` | KV heads (GQA 4:1) |
| `head_dim` | `int` | `128` | Per-head dimension |
| `mrope_section` | `list[int]` | `[24, 20, 20]` | M-RoPE sections [time, height, width] |
| `lora_rank` | `int` | `32` | LoRA rank |
| `lora_alpha` | `int` | `64` | LoRA alpha (effective scale = 2.0) |

### `class SpatialVLMConfig`

Top-level config composing all stage sub-configs.

| Field | Type | Default |
|---|---|---|
| `encoder` | `EncoderConfig` | `EncoderConfig()` |
| `geometry` | `GeometryConfig` | `GeometryConfig()` |
| `fusion` | `FusionConfig` | `FusionConfig()` |
| `backbone` | `BackboneConfig` | `BackboneConfig()` |

---

## `spatialvlm.encoders`

### `class SigLIP2Encoder(nn.Module)`

**Module:** `spatialvlm/encoders/siglip.py`

SigLIP2-SO400M encoder with multi-layer feature extraction. Extracts features from intermediate transformer layers and concatenates them along the channel dimension. All encoder parameters are frozen.

**Constructor:**

```python
SigLIP2Encoder(
    model_id: str = "google/siglip2-so400m-patch16-naflex",
    extract_layers: list[int] | None = None,  # default [9, 18, 27]
    device: torch.device | None = None,
    lazy_load: bool = False,
    local_files_only: bool = False,
    config_loader: Callable = AutoConfig.from_pretrained,
    model_loader: Callable = AutoModel.from_pretrained,
)
```

**Methods:**

- `load_model() -> None` -- Load SigLIP weights and register extraction hooks.
- `forward(pixel_values: Tensor[B, 3, H, W]) -> Tensor[B, n_patches, n_layers * hidden_size]` -- Extract multi-layer features. Default output: `[B, 576, 3456]` (576 = 24^2, 3456 = 3 * 1152).

**Properties:**

- `out_dim -> int` -- Output channel dimension = n_extract_layers * hidden_size.
- `n_patches -> int` -- Number of spatial patch tokens.

---

### `class DINOv2Encoder(nn.Module)`

**Module:** `spatialvlm/encoders/dinov2.py`

DINOv2-L encoder with multi-layer feature extraction. Strips CLS token, no spatial pooling (full 1369 tokens preserved). All encoder parameters are frozen.

**Constructor:**

```python
DINOv2Encoder(
    model_id: str = "facebook/dinov2-large",
    extract_layers: list[int] | None = None,  # default [8, 16, 24]
    device: torch.device | None = None,
    lazy_load: bool = False,
    local_files_only: bool = False,
    config_loader: Callable = AutoConfig.from_pretrained,
    model_loader: Callable = AutoModel.from_pretrained,
)
```

**Methods:**

- `load_model() -> None` -- Load DINOv2 weights and register extraction hooks.
- `forward(pixel_values: Tensor[B, 3, H, W]) -> Tensor[B, n_patches, n_layers * hidden_size]` -- Extract multi-layer features. Default output: `[B, 1369, 3072]` (1369 = 37^2, 3072 = 3 * 1024).

**Properties:**

- `out_dim -> int` -- Output channel dimension = n_extract_layers * hidden_size.
- `n_patches -> int` -- Number of spatial patch tokens (no CLS).

---

### `class MLPProjector(nn.Module)`

**Module:** `spatialvlm/encoders/projector.py`

Two-layer MLP projector: `Linear(in_dim, hidden_dim) -> GELU -> Linear(hidden_dim, out_dim)`.

Instantiated three times with different input dimensions:
- SigLIP2 projector: `MLPProjector(3456, 4096)` (~31M params)
- DINOv2 projector: `MLPProjector(3072, 4096)` (~29M params)
- GATr projector: `MLPProjector(48, 4096)` (~17M params)

**Constructor:**

```python
MLPProjector(in_dim: int, out_dim: int, hidden_dim: int | None = None)
```

**Methods:**

- `forward(x: Tensor[B, N, in_dim]) -> Tensor[B, N, out_dim]` -- Project token features to LLM hidden size.

---

## `spatialvlm.geometry`

### `backproject_depth_map()`

**Module:** `spatialvlm/geometry/backproject.py`

```python
def backproject_depth_map(
    depth: Tensor[B, H, W],
    intrinsics: CameraIntrinsics,
) -> Tensor[B, H, W, 3]
```

Backproject a batch of depth maps into 3D point maps. Zero depth values produce `(0, 0, 0)`.

---

### `aggregate_patches_percentile()`

**Module:** `spatialvlm/geometry/backproject.py`

```python
def aggregate_patches_percentile(
    point_map: Tensor[B, H, W, 3],
    depth: Tensor[B, H, W],
    patch_size: int = 14,
    percentile: float = 0.15,
) -> Tensor[B, n_patches, 3]
```

Aggregate 3D points to patch-level using k-th percentile depth. For H=W=518, patch_size=14: n_patches = 1369. Hypothesis H2e: 15th-percentile > mean aggregation.

---

### `pool_positions_to_sva_grid()`

**Module:** `spatialvlm/geometry/backproject.py`

```python
def pool_positions_to_sva_grid(
    positions: Tensor[B, N, 3],
    source_h: int = 37, source_w: int = 37,
    target_h: int = 24, target_w: int = 24,
) -> Tensor[B, target_h * target_w, 3]
```

Pool 3D positions from the DINOv2 patch grid (37x37) to the SVA query grid (24x24) via adaptive average pooling. Ensures geometric consistency between fused token content and positional encoding.

---

### `class GATrWrapper(nn.Module)`

**Module:** `spatialvlm/geometry/gatr_wrapper.py`

SpatialVLM wrapper around GATr for geometric token features. Pipeline: 3D points -> PGA multivectors -> GATr blocks -> invariant extraction -> MLP projection.

**Constructor:**

```python
GATrWrapper(
    num_blocks: int = 8,
    gatr_mv_channels: int = 16,
    gatr_s_channels: int = 32,
    projector_out_dim: int = 4096,
    join_reference: Literal["data", "canonical"] = "data",
    checkpoint_blocks: bool = True,
    normalize_inputs: bool = True,
    disable_cached_einsum: bool = True,
    eps: float = 1e-8,
    device: torch.device | None = None,
)
```

**Methods:**

- `forward(points_3d: Tensor[B, N, 3], return_invariants: bool = False) -> Tensor[B, N, projector_out_dim]` -- Run GATr and return projected features. Optionally returns `(projected, invariants)` tuple where invariants is `[B, N, 48]`.
- `uses_improved_pga() -> bool` -- Returns True if GATr blocks use improved PGA (GeometricBilinear).

**Properties:**

- `invariant_dim -> int` -- Invariant feature width: gatr_s_channels + gatr_mv_channels.

---

### `class GridCellRoPE3D(nn.Module)`

**Module:** `spatialvlm/geometry/gridcell_rope3d.py`

Rotary position encoding for 3D spatial coordinates. Tetrahedral Fourier basis inspired by grid cell theory. Zero learnable parameters.

- 4 tetrahedral directions (isotropic 3D coverage)
- 8 golden-ratio spaced frequencies: f_k = 10.0 * phi^k
- Output: 4 * 8 * 2 (sin/cos) = 64 dims (matches Qwen3's M-RoPE rotary pairs)

**Constructor:**

```python
GridCellRoPE3D()
```

**Methods:**

- `forward(positions: Tensor[B, N, 3]) -> Tensor[B, N, 64]` -- Compute 3D rotary position encoding. Layout: `[sin(d1*p*f0), cos(d1*p*f0), ...]`.

---

## `spatialvlm.fusion`

### `class SVACrossAttentionLayer(nn.Module)`

**Module:** `spatialvlm/fusion/sva.py`

Single cross-attention layer for SVA with optional typed attention bias (3x3 learned matrix over token-source types).

**Constructor:**

```python
SVACrossAttentionLayer(
    hidden_dim: int,
    num_heads: int,
    use_typed_attention_bias: bool = True,
    eps: float = 1e-6,
)
```

**Methods:**

- `forward(queries: Tensor[B, Nq, D], kv_tokens: Tensor[B, Nk, D], query_type_ids: Tensor[Nq], kv_type_ids: Tensor[Nk], kv_padding_mask: Tensor[B, Nk] | None = None) -> Tensor[B, Nq, D]`
- `build_typed_attention_mask(query_type_ids, kv_type_ids) -> Tensor | None` -- Build additive attention bias from the learned type matrix.

---

### `class SpatialVisionAggregator(nn.Module)`

**Module:** `spatialvlm/fusion/sva.py`

SVA module: 576 queries over 3314 visual KV tokens, stacked for multiple layers.

**Constructor:**

```python
SpatialVisionAggregator(
    hidden_dim: int,
    num_queries: int = 576,
    num_layers: int = 2,
    num_heads: int = 32,
    use_typed_attention_bias: bool = True,
)
```

**Methods:**

- `forward(siglip_tokens: Tensor[B, 576, D], dinov2_tokens: Tensor[B, 1369, D], gatr_tokens: Tensor[B, 1369, D], queries: Tensor[B, 576, D] | None = None, query_type_ids: Tensor[576] | None = None, kv_padding_mask: Tensor[B, 3314] | None = None) -> Tensor[B, 576, D]` -- Aggregate visual streams into 576 fused query tokens.

---

### `class RMSNormMatching(nn.Module)`

**Module:** `spatialvlm/fusion/norm_matching.py`

Scale vision tokens to match text token RMS norm magnitude. Tracks EMA of text token RMS norms. Zero learnable parameters.

**Constructor:**

```python
RMSNormMatching(ema_momentum: float = 0.99, eps: float = 1e-6)
```

**Methods:**

- `forward(vision_tokens: Tensor[B, N, D], text_tokens: Tensor[B, T, D] | None = None) -> Tensor[B, N, D]` -- Scale vision tokens to text-token magnitude. Updates EMA during training when text_tokens is provided.

---

### `class GatedCrossAttentionBlock(nn.Module)`

**Module:** `spatialvlm/fusion/gated_cross_attn.py`

Flamingo-style gated cross-attention block with zero-init gates and GQA. Pattern: `x <- x + tanh(alpha_attn) * CrossAttention(x, vision)` then `x <- x + tanh(alpha_ff) * FeedForward(x)`.

**Constructor:**

```python
GatedCrossAttentionBlock(
    hidden_dim: int,
    num_heads: int,
    num_kv_heads: int | None = None,
    ff_mult: int = 4,
    dropout: float = 0.0,
)
```

**Methods:**

- `forward(text_tokens: Tensor[B, T, D], vision_tokens: Tensor[B, N, D], vision_key_padding_mask: Tensor[B, N] | None = None) -> Tensor[B, T, D]` -- Inject visual context into language tokens.

---

## `spatialvlm.backbone`

### `class Qwen3VLBackbone(nn.Module)`

**Module:** `spatialvlm/backbone/qwen3_vl.py`

Wrapper around Qwen3-VL with optional LoRA and PEFT #2880 workaround.

**Constructor:**

```python
Qwen3VLBackbone(
    model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
    lora_rank: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.0,
    lora_target_modules: Sequence[str] = ("q_proj", "k_proj", "v_proj", "o_proj"),
    enable_lora: bool = True,
    freeze_base_model: bool = True,
    apply_peft_2880_workaround: bool = True,
    device: torch.device | None = None,
    torch_dtype: torch.dtype | None = None,
    lazy_load: bool = False,
    local_files_only: bool = False,
    # ... plus injectable config/model/factory args
)
```

**Methods:**

- `load_model() -> None` -- Load backbone weights when lazy initialization is enabled.
- `freeze_all_parameters() -> None` -- Freeze all current model parameters.
- `enable_peft_2880_workaround(...) -> tuple[int, int]` -- Set `requires_grad=True` on vision QKV modules to avoid PEFT bug #2880.
- `forward(*args, **kwargs) -> Any` -- Delegate forward pass to wrapped model.

**Properties:**

- `is_model_loaded -> bool`
- `stats -> Qwen3BackboneStats` -- Trainable/total params and PEFT #2880 info.

---

### `class Qwen3BackboneStats`

**Module:** `spatialvlm/backbone/qwen3_vl.py`

Small stats container for debug/verification.

| Field | Type |
|---|---|
| `trainable_params` | `int` |
| `total_params` | `int` |
| `peft_2880_modules_touched` | `int` |
| `peft_2880_params_touched` | `int` |

---

### `class RoutedPositionBatch`

**Module:** `spatialvlm/backbone/position_routing.py`

Position-routing outputs for mixed text+spatial sequences.

| Field | Type | Shape |
|---|---|---|
| `combined_tokens` | `Tensor` | `[B, T+N, D]` |
| `is_spatial_mask` | `Tensor` | `[B, T+N]` bool |
| `text_mrope_position_ids` | `Tensor` | `[B, 3, T]` |
| `spatial_rope3d` | `Tensor` | `[B, N, 64]` |

---

### `class PositionRouter`

**Module:** `spatialvlm/backbone/position_routing.py`

Routes text and spatial positional representations before LLM attention.

**Constructor:**

```python
PositionRouter(mrope_section: Sequence[int], expected_spatial_rotary_dim: int = 64)
```

**Methods:**

- `build_text_mrope_position_ids(batch_size: int, text_len: int, device: torch.device | None = None) -> Tensor[B, 3, T]` -- Build text position IDs for M-RoPE.
- `route(text_tokens: Tensor[B, T, D], spatial_tokens: Tensor[B, N, D], spatial_rope3d: Tensor[B, N, 64], text_mrope_position_ids: Tensor | None = None) -> RoutedPositionBatch` -- Create a routed batch for mixed text+spatial processing.

---

## `spatialvlm.utils`

### `class CameraIntrinsics`

**Module:** `spatialvlm/utils/camera.py`

Pinhole camera intrinsic parameters (OpenCV convention).

| Field | Type | Description |
|---|---|---|
| `fx` | `float` | Focal length X (pixels) |
| `fy` | `float` | Focal length Y (pixels) |
| `cx` | `float` | Principal point X |
| `cy` | `float` | Principal point Y |
| `width` | `int` | Image width |
| `height` | `int` | Image height |

---

### `make_pixel_grid()`

**Module:** `spatialvlm/utils/camera.py`

```python
def make_pixel_grid(
    width: int, height: int,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor[H, W, 2]
```

Create a dense grid of pixel `(u, v)` coordinates. Last dim is `(col, row)`, 0-indexed.

---

### `backproject_pixel()`

**Module:** `spatialvlm/utils/camera.py`

```python
def backproject_pixel(
    u: Tensor[...], v: Tensor[...],
    depth: Tensor[...],
    intrinsics: CameraIntrinsics,
) -> Tensor[..., 3]
```

Backproject pixels with depths into 3D camera-space coordinates using standard pinhole model.

---

## `spatialvlm.data`

### `class HabitatEnvConfig`

**Module:** `spatialvlm/data/habitat_env.py`

Configuration for constructing a Habitat environment wrapper.

| Field | Type | Default | Description |
|---|---|---|---|
| `config_path` | `str` | -- | Path to Habitat YAML config |
| `scene_id` | `str \| None` | `None` | Optional scene override |
| `width` | `int` | `518` | Sensor width (must be 518 for DINOv2 alignment) |
| `height` | `int` | `518` | Sensor height |
| `max_episode_steps` | `int \| None` | `None` | Optional episode step limit |
| `seed` | `int \| None` | `None` | Random seed |
| `extra_overrides` | `list[str]` | `[]` | Additional Habitat overrides |

**Methods:**

- `build_overrides() -> list[str]` -- Build Habitat override strings including required sensor resolution.

---

### `class HabitatEnvWrapper`

**Module:** `spatialvlm/data/habitat_env.py`

Light wrapper around Habitat env with tensorized observations. Validates that RGB/depth are exactly 518x518.

**Constructor:**

```python
HabitatEnvWrapper(env: Any, expected_width: int = 518, expected_height: int = 518, device: torch.device | None = None)
```

**Class Methods:**

- `from_config(cfg: HabitatEnvConfig, device: torch.device | None = None) -> HabitatEnvWrapper` -- Create wrapper from config + overrides.

**Methods:**

- `reset() -> dict[str, Tensor]` -- Reset environment and return tensorized observations.
- `step(action: int | str | Mapping) -> tuple[dict[str, Tensor], float, bool, dict]` -- Step environment (supports both gym 4-tuple and 5-tuple returns).
- `validate_observation_resolution(obs: Mapping[str, Tensor]) -> None` -- Ensure rgb/depth are exactly expected resolution.
- `close() -> None`

---

### `extract_rgb_depth()`

**Module:** `spatialvlm/data/habitat_env.py`

```python
def extract_rgb_depth(obs: Mapping[str, Tensor]) -> tuple[Tensor[H,W,3], Tensor[H,W]]
```

Extract and standardize RGB + depth tensors from observation mapping.

---

### `class NavSample`

**Module:** `spatialvlm/data/datasets.py`

Standardized navigation sample.

| Field | Type |
|---|---|
| `instruction` | `str` |
| `episode_id` | `str` |
| `source` | `str` |
| `payload` | `dict[str, Any]` |

---

### `class R2RCEDataset(Dataset[NavSample])`

**Module:** `spatialvlm/data/datasets.py`

R2R-CE loader.

- `from_file(path, split=None, limit=None) -> R2RCEDataset` (classmethod)

### `class RxRCEDataset(Dataset[NavSample])`

**Module:** `spatialvlm/data/datasets.py`

RxR-CE loader.

- `from_file(path, split=None, limit=None) -> RxRCEDataset` (classmethod)

### `class SQA3DDataset(Dataset[NavSample])`

**Module:** `spatialvlm/data/datasets.py`

SQA3D loader.

- `from_file(path, split=None, limit=None) -> SQA3DDataset` (classmethod)

---

### `build_dataset()`

**Module:** `spatialvlm/data/datasets.py`

```python
def build_dataset(name: str, path: str | Path, split: str | None = None, limit: int | None = None) -> Dataset[NavSample]
```

Factory for supported benchmark datasets. Supported names: `r2r`, `r2r-ce`, `vln-r2r`, `rxr`, `rxr-ce`, `vln-rxr`, `sqa3d`.

---

### `iter_instructions()`

**Module:** `spatialvlm/data/datasets.py`

```python
def iter_instructions(dataset: Iterable[NavSample]) -> Iterable[str]
```

Yield instruction strings from standardized samples.

---

### Preprocessing Functions

**Module:** `spatialvlm/data/preprocessing.py`

- `to_float_rgb01(rgb: Tensor) -> Tensor` -- Convert RGB to float [0,1]. Handles uint8 and float [0,255].
- `resize_rgb_bchw(rgb: Tensor[B,3,H,W], size=(518,518)) -> Tensor` -- Bilinear resize.
- `resize_depth_bhw(depth: Tensor[B,H,W], size=(518,518)) -> Tensor` -- Nearest-neighbor resize.
- `normalize_depth_bhw(depth: Tensor[B,H,W], max_depth=None, percentile=99.0, eps=1e-6) -> Tensor` -- Normalize depth to [0,1], robust to zeros/NaNs.
- `preprocess_rgb_depth(rgb: Tensor[B,3,H,W], depth: Tensor[B,H,W], size=(518,518), depth_percentile=99.0) -> tuple[Tensor, Tensor]` -- Full preprocessing pipeline for model-ready tensors.

---

## `spatialvlm.training`

### `class RewardConfig`

**Module:** `spatialvlm/training/rewards.py`

Configurable constants for reward computation.

| Field | Type | Default |
|---|---|---|
| `format_reward` | `float` | `1.0` |
| `collision_penalty` | `float` | `-2.0` |
| `goal_reward` | `float` | `10.0` |
| `consistency_penalty` | `float` | `-1.0` |
| `collision_clearance_threshold` | `float` | `0.1` |
| `goal_distance_threshold` | `float` | `1.0` |
| `progress_clip` | `tuple[float, float]` | `(-2.0, 2.0)` |
| `required_response_markers` | `tuple[str, ...]` | `("Reasoning:", "Action:")` |

---

### Reward Functions

**Module:** `spatialvlm/training/rewards.py`

- `format_reward_from_responses(responses: Sequence[str], reward_value=1.0, required_markers=("Reasoning:", "Action:"), device=None) -> Tensor[N]` -- Binary reward for response format compliance.
- `progress_reward(previous_geodesic: Tensor[N], current_geodesic: Tensor[N], clip_range=(-2.0, 2.0)) -> Tensor[N]` -- Dense reward from geodesic-distance improvement.
- `collision_penalty_from_clearance(clearance: Tensor[N], threshold=0.1, penalty=-2.0) -> Tensor[N]` -- Penalty when clearance is below threshold.
- `goal_reward(final_geodesic: Tensor[N], stopped: Tensor[N], threshold=1.0, reward_value=10.0) -> Tensor[N]` -- Terminal reward when agent stops within goal threshold.
- `consistency_reward(predicted_actions: Sequence[str|None], executed_actions: Sequence[str|None], mismatch_penalty=-1.0, device=None) -> Tensor[N]` -- Penalty for predicted/executed action mismatch.
- `consistency_reward_from_responses(responses: Sequence[str], executed_actions: Sequence[str|None], mismatch_penalty=-1.0, device=None) -> Tensor[N]` -- Extract action from response text, then compute consistency reward.
- `compute_reward_terms(responses, executed_actions, previous_geodesic, current_geodesic, clearance, final_geodesic, stopped, config=None) -> dict[str, Tensor]` -- Compute all dense reward terms (format, progress, collision, goal, consistency).
- `total_reward(reward_terms: Mapping[str, Tensor], weights: RewardWeights) -> Tensor` -- Weighted sum of reward terms using curriculum weights.

---

### `class RewardWeights`

**Module:** `spatialvlm/training/curriculum.py`

Scalar weights for dense spatial reward components.

| Field | Type |
|---|---|
| `format_weight` | `float` |
| `progress_weight` | `float` |
| `collision_weight` | `float` |
| `goal_weight` | `float` |
| `consistency_weight` | `float` |

**Methods:**

- `as_dict() -> dict[str, float]`

---

### `class CurriculumPoint`

**Module:** `spatialvlm/training/curriculum.py`

Anchor point for piecewise-linear curriculum interpolation.

| Field | Type |
|---|---|
| `epoch` | `int` |
| `weights` | `RewardWeights` |

---

### `class RewardCurriculum`

**Module:** `spatialvlm/training/curriculum.py`

Piecewise-linear reward schedule keyed by epoch.

**Constructor:**

```python
RewardCurriculum(points: list[CurriculumPoint])
```

**Class Methods:**

- `default() -> RewardCurriculum` -- Default 6-epoch progression (format -> spatial rewards).

**Methods:**

- `get_weights(epoch: int) -> RewardWeights` -- Interpolate weights at given epoch.

---

### `aggregate_weighted_rewards()`

**Module:** `spatialvlm/training/curriculum.py`

```python
def aggregate_weighted_rewards(reward_terms: dict[str, Tensor], weights: RewardWeights) -> Tensor
```

Aggregate reward terms (format, progress, collision, goal, consistency) into a single scalar reward tensor.

---

### `class PrealignConfig`

**Module:** `spatialvlm/training/prealign.py`

Hyperparameters for projector pre-alignment (Stage 1).

| Field | Type | Default |
|---|---|---|
| `learning_rate` | `float` | `1e-4` |
| `weight_decay` | `float` | `0.01` |
| `max_grad_norm` | `float` | `1.0` |
| `ignore_index` | `int` | `-100` |
| `projector_keywords` | `tuple[str, ...]` | `("projector",)` |

---

### `class PrealignmentTrainer`

**Module:** `spatialvlm/training/prealign.py`

Projector-focused trainer for Stage-1 pre-alignment. Freezes all parameters, then unfreezes only projector layers.

**Constructor:**

```python
PrealignmentTrainer(model: nn.Module, config: PrealignConfig, optimizer: Optimizer | None = None)
```

**Methods:**

- `step(batch: Mapping[str, Any]) -> PrealignStepOutput` -- One optimization step. Batch must include `labels: Tensor[B, T]`.
- `trainable_parameter_count() -> int`

---

### `masked_lm_loss()`

**Module:** `spatialvlm/training/prealign.py`

```python
def masked_lm_loss(logits: Tensor[B,T,V], labels: Tensor[B,T], ignore_index=-100) -> Tensor
```

Token-level cross-entropy with ignore index.

---

### `class SFTConfig`

**Module:** `spatialvlm/training/sft.py`

Hyperparameters for supervised fine-tuning (Stage 2).

| Field | Type | Default |
|---|---|---|
| `learning_rate` | `float` | `5e-5` |
| `weight_decay` | `float` | `0.01` |
| `max_grad_norm` | `float` | `1.0` |
| `ignore_index` | `int` | `-100` |
| `label_smoothing` | `float` | `0.0` |
| `trainable_keywords` | `tuple[str, ...]` | `("projector", "gatr", "sva", "cross_attn", "lora")` |

---

### `class SFTTrainer`

**Module:** `spatialvlm/training/sft.py`

Trainer for Stage-2 supervised fine-tuning.

**Constructor:**

```python
SFTTrainer(model: nn.Module, config: SFTConfig, optimizer: Optimizer | None = None)
```

**Methods:**

- `step(batch: Mapping[str, Any]) -> SFTStepOutput` -- One optimization step. Batch must include `labels`.
- `trainable_parameter_count() -> int`

---

### `supervised_loss()`

**Module:** `spatialvlm/training/sft.py`

```python
def supervised_loss(logits: Tensor[B,T,V], labels: Tensor[B,T], ignore_index=-100, label_smoothing=0.0) -> Tensor
```

Cross-entropy SFT objective with optional label smoothing.

---

### `class GRPOConfig`

**Module:** `spatialvlm/training/grpo.py`

Core GRPO hyperparameters.

| Field | Type | Default |
|---|---|---|
| `group_size` | `int` | `8` |
| `clip_epsilon` | `float` | `0.2` |
| `kl_beta` | `float` | `0.001` |
| `entropy_beta` | `float` | `0.0` |
| `normalize_advantages` | `bool` | `True` |
| `advantage_eps` | `float` | `1e-6` |
| `max_grad_norm` | `float` | `1.0` |
| `learning_rate` | `float` | `5e-7` |
| `weight_decay` | `float` | `0.0` |
| `replay_capacity` | `int` | `4096` |
| `replay_advantage_threshold` | `float` | `0.05` |

---

### `class GRPOTrainer`

**Module:** `spatialvlm/training/grpo.py`

Minimal GRPO trainer around pre-computed token logprobs. Includes `SelectiveSampleReplay` buffer.

**Constructor:**

```python
GRPOTrainer(model: nn.Module, config: GRPOConfig, optimizer: Optimizer | None = None)
```

**Methods:**

- `step(batch: Mapping[str, Any]) -> GRPOStepOutput` -- One GRPO step. Batch keys: `new_logprobs`, `old_logprobs`, `ref_logprobs`, plus `advantages` or `rewards`, optional `mask` and `entropy`.

---

### `class SelectiveSampleReplay`

**Module:** `spatialvlm/training/grpo.py`

Replay buffer storing high-advantage trajectories only (SSR mitigation for vanishing advantages).

**Constructor:**

```python
SelectiveSampleReplay(capacity: int, min_abs_advantage: float)
```

**Methods:**

- `add_batch(advantages: Tensor[N], payloads: list[dict]) -> int` -- Add samples exceeding advantage threshold. Returns count inserted.
- `sample(n: int) -> list[ReplaySample]` -- Random sample from buffer.

---

### `compute_group_advantages()`

**Module:** `spatialvlm/training/grpo.py`

```python
def compute_group_advantages(rewards: Tensor[G, K], eps=1e-6, normalize=True) -> Tensor[G, K]
```

Compute per-group normalized advantages. K = GRPO group size.

---

### `grpo_loss()`

**Module:** `spatialvlm/training/grpo.py`

```python
def grpo_loss(
    new_logprobs: Tensor[N, T], old_logprobs: Tensor[N, T],
    ref_logprobs: Tensor[N, T], advantages: Tensor[N],
    clip_epsilon: float, kl_beta: float,
    entropy_beta: float = 0.0, entropy: Tensor | None = None,
    mask: Tensor[N, T] | None = None,
) -> GRPOLossBreakdown
```

Compute GRPO objective. Returns `GRPOLossBreakdown(policy_loss, kl_loss, entropy_loss, total_loss, clip_fraction)`.

---

### `class FDPOConfig`

**Module:** `spatialvlm/training/fdpo.py`

Hyperparameters for fine-grained preference optimization. Separate betas for spatial grounding vs. logical reasoning segments.

| Field | Type | Default |
|---|---|---|
| `beta_grounding` | `float` | `0.1` |
| `beta_reasoning` | `float` | `0.05` |
| `learning_rate` | `float` | `5e-7` |
| `weight_decay` | `float` | `0.0` |
| `max_grad_norm` | `float` | `1.0` |
| `reference_free` | `bool` | `False` |
| `label_smoothing` | `float` | `0.0` |

---

### `class FDPOTrainer`

**Module:** `spatialvlm/training/fdpo.py`

Minimal optimizer wrapper for fDPO updates.

**Constructor:**

```python
FDPOTrainer(model: nn.Module, config: FDPOConfig, optimizer: Optimizer | None = None)
```

**Methods:**

- `step(batch: Mapping[str, Any]) -> FDPOStepOutput` -- One fDPO step. Batch keys: `chosen_logps`, `rejected_logps`, optional `chosen_ref_logps`, `rejected_ref_logps`, `segment_mask`.

---

### `fdpo_loss()`

**Module:** `spatialvlm/training/fdpo.py`

```python
def fdpo_loss(
    chosen_logps: Tensor[N], rejected_logps: Tensor[N],
    chosen_ref_logps: Tensor[N] | None, rejected_ref_logps: Tensor[N] | None,
    beta_grounding: float, beta_reasoning: float,
    segment_mask: Tensor[N] | None = None,
    reference_free: bool = False, label_smoothing: float = 0.0,
) -> FDPOLossBreakdown
```

Compute fDPO objective with segment-specific beta values. `segment_mask`: True = grounding, False = reasoning.

---

## `spatialvlm.eval`

### `class NavigationEpisodeResult`

**Module:** `spatialvlm/eval/metrics.py`

| Field | Type |
|---|---|
| `success` | `bool` |
| `path_length` | `float` |
| `shortest_path_length` | `float` |

---

### `class MetricBundle`

**Module:** `spatialvlm/eval/metrics.py`

Standardized metric tuple: `success_rate`, `spl`, `permutation_sensitivity`, `composite`.

---

### Metric Functions

**Module:** `spatialvlm/eval/metrics.py`

- `success_rate(successes: Sequence[bool]) -> float`
- `spl(episodes: Sequence[NavigationEpisodeResult], eps=1e-8) -> float` -- Success weighted by Path Length.
- `permutation_sensitivity_index(baseline_score, permuted_score, eps=1e-8) -> float` -- PSI = max(0, (baseline - permuted) / max(|baseline|, eps)).
- `weighted_composite(metrics: Mapping[str, float], weights: Mapping[str, float] | None = None) -> float`
- `compute_metric_bundle(episodes, baseline_score, permuted_score, composite_weights=None) -> MetricBundle` -- Compute SR, SPL, PSI, and composite.

---

### `class PermutationTestResult`

**Module:** `spatialvlm/eval/permutation_test.py`

| Field | Type |
|---|---|
| `baseline_score` | `float` |
| `permuted_mean` | `float` |
| `permuted_std` | `float` |
| `absolute_drop` | `float` |
| `relative_drop` | `float` |
| `empirical_pvalue` | `float` |
| `num_permutations` | `int` |

**Methods:**

- `is_spatially_grounded(min_relative_drop=0.15) -> bool`

---

### Permutation Test Functions

**Module:** `spatialvlm/eval/permutation_test.py`

- `permute_tokens(tokens: Tensor[B,N,D], spatial_mask: Tensor[B,N] | None = None, generator=None) -> Tensor[B,N,D]` -- Permute token order per sample (optionally only masked positions).
- `run_permutation_test(scoring_fn, batch, token_key="vision_tokens", spatial_mask_key=None, num_permutations=64, seed=0, eps=1e-8) -> PermutationTestResult` -- Run baseline-vs-permuted score comparison.

---

### `class BenchmarkSpec`

**Module:** `spatialvlm/eval/benchmarks.py`

| Field | Type |
|---|---|
| `benchmark_id` | `str` |
| `display_name` | `str` |
| `task_family` | `str` |
| `primary` | `bool` |
| `indoor` | `bool` |
| `requires_gt_depth` | `bool` |
| `notes` | `str` |

---

### `class BenchmarkResult`

**Module:** `spatialvlm/eval/benchmarks.py`

| Field | Type |
|---|---|
| `benchmark_id` | `str` |
| `score` | `float` |
| `metrics` | `dict[str, float]` |
| `metadata` | `dict[str, Any]` |

---

### `class BenchmarkRunner`

**Module:** `spatialvlm/eval/benchmarks.py`

Registry-driven benchmark runner.

**Constructor:**

```python
BenchmarkRunner(specs: Sequence[BenchmarkSpec], evaluators: Mapping[str, Callable])
```

**Methods:**

- `run(benchmark_ids: Sequence[str] | None = None) -> dict[str, BenchmarkResult]`

---

### Benchmark Suite Functions

**Module:** `spatialvlm/eval/benchmarks.py`

- `default_benchmark_suite(include_supplementary=False) -> list[BenchmarkSpec]` -- Default indoor-first primary evaluation suite (VLN-CE R2R, RxR, ObjectNav HM3D, SQA3D, VSI-Bench, NavTrust).
- `validate_primary_suite_is_indoor(specs: Sequence[BenchmarkSpec]) -> bool` -- Ensure all primary benchmarks are indoor.

---

### `class AblationSpec`

**Module:** `spatialvlm/eval/ablations.py`

| Field | Type |
|---|---|
| `ablation_id` | `str` |
| `name` | `str` |
| `hypotheses` | `tuple[str, ...]` |
| `overrides` | `dict[str, Any]` |

---

### `class AblationResult`

**Module:** `spatialvlm/eval/ablations.py`

| Field | Type |
|---|---|
| `ablation_id` | `str` |
| `score` | `float` |
| `delta_vs_baseline` | `float` |
| `metadata` | `dict[str, Any]` |

---

### `class AblationOrchestrator`

**Module:** `spatialvlm/eval/ablations.py`

Runs baseline + ablation variants via injected evaluator callback.

**Constructor:**

```python
AblationOrchestrator(base_config: Mapping, evaluator: Callable, specs: Sequence[AblationSpec] | None = None)
```

**Methods:**

- `run(ablation_ids: Sequence[str] | None = None) -> dict[str, AblationResult]` -- Run baseline and selected ablations.

---

### `default_ablation_specs()`

**Module:** `spatialvlm/eval/ablations.py`

```python
def default_ablation_specs() -> list[AblationSpec]
```

Default ablation matrix: no-gridcell-rope3d, no-gatr, siglip-only, dinov2-only, no-rms-norm-matching, no-typed-attn-bias.

---

### `class Phase9RunSpec`

**Module:** `spatialvlm/eval/phase9.py`

| Field | Type |
|---|---|
| `run_id` | `str` |
| `title` | `str` |
| `hypotheses` | `tuple[str, ...]` |
| `config_overrides` | `dict[str, Any]` |

---

### Phase 9 Functions

**Module:** `spatialvlm/eval/phase9.py`

- `phase9_run_specs() -> list[Phase9RunSpec]` -- Canonical 16-run ablation matrix.
- `missing_phase9_runs(results_by_id: Mapping) -> list[str]` -- Return missing run IDs.
- `phase9_coverage_complete(results_by_id: Mapping) -> bool`
- `permutation_smoking_gun_pass(ours_relative_drop, baseline_relative_drop, ours_min_drop=0.15, baseline_max_drop=0.03) -> bool` -- Check H3c criterion: ours >15% drop, baseline <3% drop.

---

### Paper Asset Functions

**Module:** `spatialvlm/eval/paper_assets.py`

- `load_phase9_results(path: Path) -> dict[str, Any]` -- Load Phase 9 results JSON.
- `render_ablation_table_tex(runs: dict) -> str` -- LaTeX tabular for ablation results.
- `render_main_results_table_tex(runs: dict) -> str` -- LaTeX tabular for main results.
- `write_permutation_csv(runs: dict, output_path: Path) -> None` -- Write permutation curve CSV.
- `write_paper_assets(runs: dict, paper_dir: Path) -> None` -- Generate all paper-ready assets (tables + figures).
