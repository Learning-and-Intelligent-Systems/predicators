import re
from collections import defaultdict
from pprint import pprint

input_text = """
* **Action:** Twist(robot1:robot, jug6:jug)
    * **preconditions (state_24):**
        * **GripperClosed(robot1:robot):** The robot's gripper is closed, allowing it to manipulate the jug.
        * **JugOnTopOf(jug6:jug, table2:table):** The jug is on the table, accessible for twisting.
    * **add effect (state_16):**
        * **JugTilted(jug6:jug):** The twisting action results in the jug being tilted.
    * **delete effect (state_16):**
        * **JugUpright(jug6:jug):** The jug is no longer upright after twisting.

* **Action:** PlaceJugInMachine(robot1:robot, jug6:jug, coffee_machine3:coffee_machine)
    * **preconditions (state_26):**
        * **GripperClosed(robot1:robot):** The robot needs to be gripping the jug to place it.
        * **JugOnTopOf(jug6:jug, table2:table):** The jug needs to be on the table to be placed in the machine.
    * **add effect (state_47):**
        * **JugInMachine(jug6:jug, coffee_machine3:coffee_machine):** The jug is now inside the coffee machine.
    * **delete effect (state_47):**
        * **JugOnTopOf(jug6:jug, table2:table):** The jug is no longer on top of the table.

* **Action:** TurnMachineOn(robot1:robot, coffee_machine3:coffee_machine)
    * **preconditions (state_31):**
        * **RobotFingersOpen(robot1:robot):** The robot's fingers are open, allowing it to interact with the machine's controls.
    * **add effect (state_37):**
        * **None observed:** This action doesn't seem to change the state predicates based on the provided information. We might need additional predicates to capture the machine's state.
    * **delete effect (state_37):**
        * **None observed:** This action doesn't seem to delete any state predicates based on the provided information.

* **Action:** PickJug(robot1:robot, jug6:jug)
    * **preconditions (state_25):**
        * **RobotFingersOpen(robot1:robot):** The robot needs open fingers to grasp the jug.
        * **JugOnTopOf(jug6:jug, table2:table):** The jug must be on the table to be picked up.
    * **add effect (state_6):**
        * **GripperClosed(robot1:robot):** The robot closes its gripper to pick up the jug.
    * **delete effect (state_6):**
        * **GripperOpen(robot1:robot):** The robot's gripper is no longer open after picking up the jug. 
"""
input_text = """# Action Preconditions and Effects
* Action: Twist(robot1:robot, jug6:jug)
    * preconditions (state_24):
        * RobotFingersOpen(robot1): robot1's fingers value is 0.4
    * add effect (state_16):
        * JugRotated(jug6): jug6's rot value is not 0.0
    * delete effect (state_16):
        * JugUpright(jug6): jug6's rot value is 0.0
* Action: PlaceJugInMachine(robot1:robot, jug6:jug, coffee_machine3:coffee_machine)
    * preconditions (state_26):
        * RobotFingersClosed(robot1): robot1's fingers value is 0.1
    * add effect (state_47):
        * JugInMachine(jug6, coffee_machine3): jug6 is in coffee_machine3
    * delete effect (state_47):
        * JugInHand(robot1, jug6): robot1 is holding jug6
* Action: TurnMachineOn(robot1:robot, coffee_machine3:coffee_machine)
    * preconditions (state_31):
        * RobotFingersOpen(robot1): robot1's fingers value is 0.4
    * add effect (state_37):
        * None
    * delete effect (state_37):
        * None
* Action: PickJug(robot1:robot, jug6:jug)
    * preconditions (state_25):
        * RobotFingersOpen(robot1): robot1's fingers value is 0.4
        * JugUpright(jug6): jug6's rot value is 0.0
    * add effect (state_6):
        * RobotFingersClosed(robot1): robot1's fingers value is 0.1
        * JugInHand(robot1, jug6): robot1 is holding jug6
    * delete effect (state_6):
        * RobotFingersOpen(robot1): robot1's fingers value is 0.4"""
input_text = """* Action: Twist(robot1:robot, jug6:jug)
    * preconditions (state_24):
        * RobotHolding(robot1, jug6): robot1 is holding jug6.
        * FingersClosed(robot1): robot1's fingers are closed.
    * add effect (state_16):
        * JugTipped(jug6): jug6 is tipped.
    * delete effect (state_16):
        * JugUpright(jug6): jug6 is upright.
* Action: PlaceJugInMachine(robot1:robot, jug6:jug, coffee_machine3:coffee_machine)
    * preconditions (state_26):
        * RobotHolding(robot1, jug6): robot1 is holding jug6.
        * FingersClosed(robot1): robot1's fingers are closed.
    * add effect (state_47):
        * JugInMachine(jug6, coffee_machine3): jug6 is inside coffee_machine3.
    * delete effect (state_47):
        * JugInMachineInverse(jug6, coffee_machine3): jug6 is not inside coffee_machine3.
* Action: TurnMachineOn(robot1:robot, coffee_machine3:coffee_machine)
    * preconditions (state_31):
        * FingersOpen(robot1): robot1's fingers are open.
    * add effect (state_37):
        * None observed from provided data.
    * delete effect (state_37):
        * None observed from provided data.
* Action: PickJug(robot1:robot, jug6:jug)
    * preconditions (state_25):
        * FingersOpen(robot1): robot1's fingers are open.
        * JugUpright(jug6): jug6 is upright.
    * add effect (state_6):
        * RobotHolding(robot1, jug6): robot1 is holding jug6.
        * FingersClosed(robot1): robot1's fingers are closed.
    * delete effect (state_6):
        * RobotHoldingInverse(robot1, jug6): robot1 is not holding jug6.
        * FingersOpen(robot1): robot1's fingers are open. """
# Define regex patterns
action_pattern = re.compile(r'\* Action:.*?\n(.*?)(?=\* Action:|\Z)', re.DOTALL)
preconditions_pattern = re.compile(r'\* preconditions \(state_(\d+)\):\n(.*?)(?=\* (add effect|delete effect)|$)', re.DOTALL)
add_effect_pattern = re.compile(r'\* add effect \(state_(\d+)\):\n(.*?)(?=\* (preconditions|delete effect)|$)', re.DOTALL)
delete_effect_pattern = re.compile(r'\* delete effect \(state_(\d+)\):\n(.*?)(?=\* (preconditions|add effect)|$)', re.DOTALL)

state_dict = defaultdict(lambda: (set(), set()))

for action_block in action_pattern.finditer(input_text):
    block_text = action_block.group(1)
    
    for match in preconditions_pattern.finditer(block_text):
        state = match.group(1)
        predicates = match.group(2)
        for predicate in predicates.strip().split('\n'):
            cleaned_predicate = predicate.strip()
            if cleaned_predicate:
                state_dict[f'state_{state}'][0].add(cleaned_predicate)

    for match in add_effect_pattern.finditer(block_text):
        state = match.group(1)
        predicates = match.group(2)
        for predicate in predicates.strip().split('\n'):
            cleaned_predicate = predicate.strip()
            if cleaned_predicate and cleaned_predicate != 'None':
                state_dict[f'state_{state}'][0].add(cleaned_predicate)

    for match in delete_effect_pattern.finditer(block_text):
        state = match.group(1)
        predicates = match.group(2)
        for predicate in predicates.strip().split('\n'):
            cleaned_predicate = predicate.strip()
            if cleaned_predicate and cleaned_predicate != 'None':
                state_dict[f'state_{state}'][1].add(cleaned_predicate)

pprint(state_dict)

predicate_format = re.compile(
    r"\* \*\*([A-Za-z0-9_]+)\(([^)]+)\):\*\* (.+)"
)
predicate_format = re.compile(
    r"\* ([A-Za-z0-9_]+)\(([^)]+)\): (.+)"
)
def filter_predicates(predicates_set):
    return {predicate for predicate in predicates_set if predicate_format.match(predicate)}

filtered_state_dict = defaultdict(lambda: (set(), set()), {
    state: (filter_predicates(preconditions), filter_predicates(effects))
    for state, (preconditions, effects) in state_dict.items()
})

pprint(filtered_state_dict)

# Iterate over the state_dict and modify predicates
state_str = []
for state in sorted(filtered_state_dict.keys()):
    add_set, del_set = filtered_state_dict[state]
    if not add_set and not del_set:
        continue
    state_str.append(state)
    if add_set:
        pos = "\n".join([f"{pred}: True" for pred in add_set])
        state_str.append(pos)
    if del_set:
        neg = "\n".join([f"{pred}: False" for pred in del_set])
        state_str.append(neg)
    # state_dict[state] = (
    #     {f"{pred}: True" for pred in add_set},
    #     {f"{pred}: False" for pred in del_set}
    # )
state_str = "\n".join(state_str)
# Print the modified state_dict
print(state_str)