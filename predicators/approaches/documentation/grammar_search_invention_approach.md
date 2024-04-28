# Grammar Search Invention Approach
This approach is primarily useful for inventing predicates via program synthesis from demonstrations, as described originally in:
[Predicate Invention for Bilevel Planning](https://arxiv.org/abs/2203.09634). Silver*, Chitnis*, Kumar, McClinton, Lozano-Perez, Kaelbling, Tenenbaum. AAAI 2023.

An example command for running the approach from that paper is:
```
python predicators/main.py --env cover --approach grammar_search_invention --excluded_predicates all --num_train_tasks 50
```

Last updated: 04/27/2024

## Inventing predicates via a txt file
Instead of generating demonstrations with an oracle (as is done for the typical grammar search invention approach), we can also generate them directly from a manually-specified txt file.

We assume the txt files have a particular structure that we leverage for parsing. To explain these components, consider this below example:

```
===
{*Holding(spoon): True.
*Submerged(teabag): False.
*Submerged(spoon): False.} ->
pick(teabag, hand)[] -> 
{*Holding(spoon): True.
*Submerged(teabag): False.
*Submerged(spoon): False.} ->
place_in(teabag, cup)[] -> 
{*Holding(spoon): True.
*Submerged(teabag): False.
*Submerged(spoon): False.} ->
pick(spoon, hand)[] -> 
{*Holding(spoon): True.
*Submerged(teabag): False.
*Submerged(spoon): False.} ->
place_in(spoon, cup)[] -> 
{*Holding(spoon): True.
*Submerged(teabag): False.
*Submerged(spoon): False.}
===
```

**Components**
- Separator: '===' is used to separate one trajectory from another (so a trajectory is sandwiched between two lines that have only '===' on them). In the above example, there is exactly one demonstration trajectory.
- State: Each state is a bulleted list of atoms enclosed between set brackets {}. In the above example, there are 5 states. Note importantly that the format of every atom should be `*<predicate_name>(<ob1_name>, <obj2_name>, ...).`. The period `.` at the end is very important.
- Skill: Each skill is sandwiched between two states and takes the format: `<skill_name>(<ob1_name>, <obj2_name>, ...)[<continuous_param_vector>]`. In the above example, there are 4 skills. Note that after every state, there is a `->` character, followed by a newline, then a skill followed by another `->` character and newline. This is also critical to parsing. Note also that the above example doesn't feature any continuous parameters.


### Future features to be added
* Enable pipeline to consider demonstrations that have low-level object-oriented state, as well as image observations.
* Enable invented VLM predicates to actually be used and run at test-time.
* Consider different VLM's