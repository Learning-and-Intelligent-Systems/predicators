from typing import List

def is_block_on_table(environment_patch: ImagePatch, block_patch: ImagePatch) -> bool:
    """
    Determines whether the block currently lies on the table.

    Parameters
    ----------
    environment_patch : ImagePatch
        The ImagePatch object representing the entire environment observation.
    block_patch : ImagePatch
        The ImagePatch object representing the block in question.

    Returns
    -------
    bool
        True if the block lies on the table, False otherwise.
    """
    # Find all 'table' objects in the environment
    table_patches = environment_patch.find("table")
    
    # If there's no table in the environment, return False
    if not table_patches:
        return False

    # Iterate through all found tables to check if any block is on them
    for table_patch in table_patches:
        # Check if the block's lower edge is just above the table's upper edge,
        # indicating it's lying on the table. This is a simplified approach and
        # might need adjustments based on the actual coordinate system and perspective.
        if block_patch.lower >= table_patch.upper and block_patch.overlaps_with(
                table_patch.left, table_patch.lower, table_patch.right, table_patch.upper):
            return True
    
    # If none of the tables have the block on them, return False
    return False