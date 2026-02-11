"""View demos with task, goal, action sequence, and state changes."""

import glob
import os
import pickle
from typing import Set, Any


def parse_goal(task_name: str, ground_atoms_state: Set) -> Set:
    """Parse goal atoms from ground atoms state for each task."""
    if task_name == "MiniGrid-BoxingBooksUpForStorage-16x16-N2-v0":
        return set([atom for atom in ground_atoms_state if str(atom).startswith("inside(")])
    
    elif task_name == "MiniGrid-OpeningPackages-16x16-N2-v0":
        return set([atom for atom in ground_atoms_state if str(atom).startswith("openable(")])
    
    elif task_name == "MiniGrid-CleaningACar-16x16-N2-v0":
        return set([atom for atom in ground_atoms_state if str(atom).startswith("inside(")]) | \
               set([atom for atom in ground_atoms_state if str(atom).startswith("~dustyable(")])
    
    elif task_name == "MiniGrid-CleaningShoes-16x16-N2-v0":
        return set([atom for atom in ground_atoms_state if str(atom).startswith("~stainable(") and "shoe" in str(atom)]) | \
               set([atom for atom in ground_atoms_state if str(atom).startswith("~dustyable(") and "shoe" in str(atom)]) | \
               set([atom for atom in ground_atoms_state if str(atom).startswith("onfloor(") and "towel" in str(atom)])

    elif task_name == "MiniGrid-CleaningUpTheKitchenOnly-16x16-N2-v0":
        return set([
            atom for atom in ground_atoms_state
            if (
                str(atom).startswith("onTop(") and "blender" in str(atom) and "countertop" in str(atom)
            ) or (
                str(atom).startswith("nextto(") and "soap" in str(atom) and "sink" in str(atom)
            ) or (
                str(atom).startswith("inside(") and "vegetable_oil" in str(atom) and "cabinet" in str(atom)
            ) or (
                str(atom).startswith("inside(") and "plate" in str(atom) and "cabinet" in str(atom)
            ) or (
                str(atom).startswith("inside(") and "casserole" in str(atom) and "electric_refrigerator" in str(atom)
            ) or (
                str(atom).startswith("inside(") and "apple" in str(atom) and "electric_refrigerator" in str(atom)
            ) or (
                str(atom).startswith("inside(") and "rag" in str(atom) and "sink" in str(atom)
            ) or (
                str(atom).startswith("nextto(") and "rag" in str(atom) and "sink" in str(atom)
            ) or (
                str(atom).startswith("~dustyable(") and "cabinet" in str(atom)
            ) or (
                str(atom).startswith("~stainable(") and "plate" in str(atom)
            )
        ])

    elif task_name == "MiniGrid-CollectMisplacedItems-16x16-N2-v0":
        return set([
            atom for atom in ground_atoms_state
            if str(atom).startswith("onTop(") and "table" in str(atom) and (
                "gym_shoe" in str(atom) or
                "necklace" in str(atom) or
                "notebook" in str(atom) or
                "sock" in str(atom)
            ) and not str(atom).startswith("onTop(table") 
        ])
    
    elif task_name == "MiniGrid-InstallingAPrinter-16x16-N2-v0":
        return set([atom for atom in ground_atoms_state if str(atom).startswith("onTop(")]) | \
               set([atom for atom in ground_atoms_state if str(atom).startswith("toggleable(")])
    
    elif task_name == "MiniGrid-LayingWoodFloors-16x16-N2-v0":
        return set([atom for atom in ground_atoms_state if str(atom).startswith("nextto(")])
    
    elif task_name == "MiniGrid-MakingTea-16x16-N2-v0":
        return set([
            atom for atom in ground_atoms_state
            if str(atom).startswith("sliceable(") and "lemon" in str(atom)
        ]) | set([
            atom for atom in ground_atoms_state
            if str(atom).startswith("onTop(") and "teapot" in str(atom) and "stove" in str(atom)
        ]) | set([
            atom for atom in ground_atoms_state
            if str(atom).startswith("atsamelocation(") and "tea_bag" in str(atom) and "teapot" in str(atom)
        ]) | set([
            atom for atom in ground_atoms_state
            if str(atom).startswith("soakable(") and "teapot" in str(atom)
        ]) | set([
            atom for atom in ground_atoms_state
            if str(atom).startswith("toggleable(") and "stove" in str(atom)
        ])

    elif task_name == "MiniGrid-MovingBoxesToStorage-16x16-N2-v0":
        return set([atom for atom in ground_atoms_state if str(atom).startswith("onTop(")])
    
    elif task_name == "MiniGrid-OrganizingFileCabinet-16x16-N2-v0":
        return set([
            atom for atom in ground_atoms_state
            if str(atom).startswith("onTop(") and "marker" in str(atom) and "table" in str(atom)
        ]) | set([
            atom for atom in ground_atoms_state
            if str(atom).startswith("inside(") and "document" in str(atom) and "cabinet" in str(atom)
        ]) | set([
            atom for atom in ground_atoms_state
            if str(atom).startswith("inside(") and "folder" in str(atom) and "cabinet" in str(atom)
        ])
    
    elif task_name == "MiniGrid-PuttingAwayDishesAfterCleaning-16x16-N2-v0":
        return set([
            atom for atom in ground_atoms_state
            if str(atom).startswith("inside(") and "plate" in str(atom) and "cabinet" in str(atom)
        ])
    
    elif task_name == "MiniGrid-SettingUpCandles-16x16-N2-v0":
        return set([atom for atom in ground_atoms_state if str(atom).startswith("onTop(")])
    
    elif task_name == "MiniGrid-SortingBooks-16x16-N2-v0":
        return set([atom for atom in ground_atoms_state if str(atom).startswith("onTop(") and "shelf" in str(atom) and ("book" in str(atom) or "hardback" in str(atom))])
    
    elif task_name == "MiniGrid-StoringFood-16x16-N2-v0":
        return set([
            atom for atom in ground_atoms_state
            if str(atom).startswith("inside(") and "cabinet" in str(atom) and (
                "oatmeal" in str(atom) or "chip" in str(atom) or "vegetable_oil" in str(atom) or "sugar" in str(atom)
            )
        ])

    elif task_name == "MiniGrid-ThawingFrozenFood-16x16-N2-v0":
        return set([
            atom for atom in ground_atoms_state
            if str(atom).startswith("nextto(") and (
                ("date" in str(atom) and "fish" in str(atom)) or
                ("fish" in str(atom) and "sink" in str(atom)) or
                ("olive" in str(atom) and "sink" in str(atom))
            )
        ])
    
    elif task_name == "MiniGrid-ThrowingAwayLeftovers-16x16-N2-v0":
        return set([atom for atom in ground_atoms_state if str(atom).startswith("inside(") and "hamburger" in str(atom) and "ashcan" in str(atom)])
    
    elif task_name == "MiniGrid-WashingPotsAndPans-16x16-N2-v0":
        return set([
            atom for atom in ground_atoms_state
            if str(atom).startswith("~stainable(") and (
                "pan" in str(atom) or "kettle" in str(atom) or "teapot" in str(atom)
            )
        ]) | set([
            atom for atom in ground_atoms_state
            if str(atom).startswith("inside(") and "cabinet" in str(atom) and (
                "pan" in str(atom) or "kettle" in str(atom) or "teapot" in str(atom)
            )
        ])

    elif task_name == "MiniGrid-WateringHouseplants-16x16-N2-v0":
        return set([atom for atom in ground_atoms_state if str(atom).startswith("soakable(") and "pot_plant" in str(atom)])
    
    else:
        return set()


def normalize_action(action: str) -> str:
    """Normalize movement actions to 'move'."""
    action_str = str(action).replace("Actions.", "")
    if action_str in ['forward', 'left', 'right']:
        return 'move'
    return action_str.split('_')[0]


def visualize_demos(path: str = '../demos/'):
    """Load and visualize all demos with state changes."""
    csv_files = glob.glob(os.path.join(path, "*/*"))
    
    if not csv_files:
        print(f"No demo files found in {path}")
        return
    
    # Load all pickle files into a dictionary
    demos = {os.path.basename(f): pickle.load(open(f, 'rb')) for f in csv_files}
    
    print("=" * 80)
    print("MiniBehavior Demo Visualization")
    print("=" * 80)
    
    for demo_idx, (demo_name, demo_data) in enumerate(demos.items(), 1):
        task_name = demo_name.split("_")[0]
        
        print(f"\n{'=' * 80}")
        print(f"Demo #{demo_idx}: {demo_name}")
        print(f"{'=' * 80}")
        
        # Get final state to determine goal
        final_state = list(demo_data.values())[-1][4]
        goal_atoms = parse_goal(task_name, final_state)
        
        print(f"\n📋 TASK: {task_name}")
        print(f"\n🎯 GOAL ({len(goal_atoms)} atoms):")
        for atom in sorted(goal_atoms, key=str):
            print(f"  • {atom}")
        
        # Collect action sequence with state changes
        action_sequence_parts = []
        prev_action = None
        prev_state = None
        
        for step_idx, traj in demo_data.items():
            action = normalize_action(traj[2])
            current_state = set(traj[1])
            
            # Calculate state changes when action changes
            if action != prev_action:
                if prev_action is not None:
                    action_sequence_parts.append(prev_action)
                
                # Add state change info if we have a previous state
                if prev_state is not None and action != prev_action:
                    added = len(current_state - prev_state)
                    deleted = len(prev_state - current_state)
                    action_sequence_parts.append(f"(+{added} / -{deleted})")
                
                prev_action = action
            
            prev_state = current_state
        
        # Add final action
        if prev_action is not None:
            action_sequence_parts.append(prev_action)
        
        print(f"\n🔄 ACTION SEQUENCE ({len([p for p in action_sequence_parts if not p.startswith('(')])} actions):")
        print(f"  {' → '.join(action_sequence_parts)}")
        
        print(f"\n{'=' * 80}\n")


if __name__ == "__main__":
    visualize_demos()
