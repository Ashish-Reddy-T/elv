### 1. The Data Pipeline (`datasets.py` & `preprocessing.py`)
* **Coordinate Survival:** Ensure that when `preprocess_rgb_depth` finishes and hands the data back to the `NavSample` payload, the actual 3D point coordinates (or the depth map that generates them) are kept in the dictionary so they can be passed to the Qwen collator.

### 2. The Qwen Collator (`qwenvl/data/data_processor.py`)
* **Batch Injection:** Inside `_get_item` and `_get_packed_item`, extract your 3D coordinates from the `sources` dictionary and add them as a new key (e.g., `spatial_coords_3d`).
* **Collator Padding:** Inside `DataCollatorForSupervisedDataset` and `FlattenedDataCollatorForSupervisedDataset`, explicitly stack/pad your new `spatial_coords_3d` key and the `mm_token_type_ids` so they are successfully returned in the final batch dictionary (this guarantees they get passed as `**kwargs` into the model).

### 3. The Architecture Configurations (`spatialvlm/config/model.py`)
* **`GeometryConfig`:** Update the defaults. Change tetrahedral directions to 6 (Icosahedral). Change the golden ratio to the $e^{1/3}$ scalar. Update the math so `rope3d_dims` outputs `96`.
* **`FusionConfig`:** Change `sva_num_queries` from 576 to 1369.

### 4. The Geometry & Fusion Core (`spatialvlm/geometry/` & `fusion/`)
* **Deprecate Position Pooling:** Inside `backproject.py`, you can completely bypass or remove `pool_positions_to_sva_grid()`. The 1369 points extracted from the 15th-percentile aggregation now go straight into the RoPE math.
* **Update `GridCellRoPE3D`:** Inside `gridcell_rope3d.py`, swap the 4 hardcoded tetrahedral vectors for the 6 normalized icosahedral vectors. Implement the $e^{1/3}$ frequency scaling. Ensure the tensor reshapes to `[B, 1369, 96]`.
* **Update `SpatialVisionAggregator`:** Inside `sva.py`, change the `query_initialization` logic. Stop using `siglip_tokens` as the additive base for the queries; use `dinov2_tokens` instead to establish the structural anchor.

### 5. The Top-Level Hijack (`qwenvl/train/train_qwen.py`)
* **The Catcher (Top-Level Patch):** Before initializing the `Trainer`, write a function to monkey-patch `Qwen3VLForConditionalGeneration.forward`. This function must catch `spatial_coords_3d` and `mm_token_type_ids` from the kwargs, stash them as temporary attributes on `self.model.language_model.rotary_emb`, and then call the original forward pass.
* **The Math (RoPE Patch):** Write a second function to monkey-patch `Qwen3VLTextRotaryEmbedding.forward`. This function must call the original RoPE to get the standard 1D sequence rotations. Then, it must retrieve the stashed coordinates, run the 6-axis Icosahedral math, pad the resulting 48 pairs with 16 pairs of zeros/ones to reach the 64-pair (128-dim) requirement, use the stashed `mm_token_type_ids` to dynamically find the spatial token indices, and overwrite those specific positions in the standard RoPE tensor.