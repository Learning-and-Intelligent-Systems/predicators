"""Classification approaches for predicators."""
from predicators.classification_approaches.dino_similarity_approach import \
    DinoSimilarityApproach
from predicators.classification_approaches.vlm_classification_approach import \
    VLMClassificationApproach

__all__ = ["VLMClassificationApproach", "DinoSimilarityApproach"]
