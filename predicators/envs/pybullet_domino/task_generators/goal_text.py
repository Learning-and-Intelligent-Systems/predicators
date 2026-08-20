"""Natural-language goal descriptions for min-block domino tasks.

Centralizes the goal-NL strings that were duplicated verbatim across the
plain, turn, and heavy min-block task builders. All share the same
"arrange the blues, push the green, topple the purple, use as few blues
as possible, keep everything else staged and standing until the push"
instruction; the heavy variant also names the gray heavy blocks as fixed
scenery.
"""

# The evaluator's counterfactual verification, stated in the goal text
# so every enforced rule is inferable up front: without it, an agent
# whose rollouts pass only with the arm's help sees nothing but opaque
# solved=False verdicts (run_20260718_141716 burned two 45-minute
# attempts theorizing about an "unknown evaluator criterion").
CASCADE_VERIFICATION_NL = (
    " A solve only counts if the push itself causes the cascade: it is "
    "verified by replaying your push with every robot link except the "
    "fingertips made intangible, and the built layout must still cascade "
    "to the goal - topples that needed the arm's body earn nothing.")

MIN_BLOCK_GOAL_NL = (
    "Arrange the blue dominoes so that when the green domino is pushed, "
    "the purple domino is toppled -- using AS FEW blue dominoes as "
    "possible (possibly none). Only the blue dominoes may be rearranged: "
    "the green and purple dominoes must stay untouched at their staged "
    "poses, upright and never held, until the green is pushed, and nothing "
    "may topple before that push. Only the green domino may ever be "
    "pushed." + CASCADE_VERIFICATION_NL)

HEAVY_GOAL_NL = (
    "Arrange the blue dominoes so that when the green domino is pushed, "
    "the purple domino is toppled -- using AS FEW blue dominoes as "
    "possible (possibly none). Only the blue dominoes may be rearranged: "
    "the green and purple dominoes and the gray blocks must stay "
    "untouched at their staged poses, upright and never held, until the "
    "green is pushed, and nothing may topple before that push. Only the "
    "green domino may ever be pushed." + CASCADE_VERIFICATION_NL)
