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
from .rewards import (
    RewardConfig,
    collision_penalty_from_clearance,
    compute_reward_terms,
    consistency_reward,
    consistency_reward_from_responses,
    format_reward_from_responses,
    goal_reward,
    progress_reward,
    total_reward,
)
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
    "RewardConfig",
    "format_reward_from_responses",
    "progress_reward",
    "collision_penalty_from_clearance",
    "goal_reward",
    "consistency_reward",
    "consistency_reward_from_responses",
    "compute_reward_terms",
    "total_reward",
    "FDPOConfig",
    "FDPOTrainer",
    "fdpo_loss",
]
