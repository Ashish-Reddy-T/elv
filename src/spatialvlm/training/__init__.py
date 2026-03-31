"""Stage 5: Training pipeline (pre-alignment, SFT, GRPO, fDPO)."""

from .curriculum import (
    CurriculumPoint,
    RewardCurriculum,
    RewardWeights,
    aggregate_weighted_rewards,
)
from .fdpo import FDPOConfig, FDPOTrainer, fdpo_loss
from .grpo import (
    GRPOConfig,
    GRPOTrainer,
    SelectiveSampleReplay,
    compute_group_advantages,
    grpo_loss,
)
from .prealign import PrealignConfig, PrealignmentTrainer, masked_lm_loss
from .sft import SFTConfig, SFTTrainer, supervised_loss

__all__ = [
    "RewardWeights",
    "CurriculumPoint",
    "RewardCurriculum",
    "aggregate_weighted_rewards",
    "PrealignConfig",
    "PrealignmentTrainer",
    "masked_lm_loss",
    "SFTConfig",
    "SFTTrainer",
    "supervised_loss",
    "GRPOConfig",
    "GRPOTrainer",
    "SelectiveSampleReplay",
    "compute_group_advantages",
    "grpo_loss",
    "FDPOConfig",
    "FDPOTrainer",
    "fdpo_loss",
]
