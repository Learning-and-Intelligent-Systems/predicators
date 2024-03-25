def is_block_at_target(block_patch: ImagePatch, target_patch: ImagePatch) -> str:
    """
    Determines whether a given block is at the target position by checking if the block's image patch
    overlaps with the target's image patch.

    Parameters
    ----------
    block_patch : ImagePatch
        The ImagePatch object representing the block.
    target_patch : ImagePatch
        The ImagePatch object representing the target.

    Returns
    -------
    str
        A string "yes" if the block is at the target position (i.e., the patches overlap), otherwise "no".
    """
    # Check if the block overlaps with the target
    if block_patch.overlaps_with(target_patch.left, target_patch.lower, target_patch.right, target_patch.upper):
        return "yes"
    else:
        return "no"

# Example usage:
# Assuming block_patch and target_patch are predefined ImagePatch objects representing the block and the target respectively
# result = is_block_at_target(block_patch, target_patch)
# print(result)