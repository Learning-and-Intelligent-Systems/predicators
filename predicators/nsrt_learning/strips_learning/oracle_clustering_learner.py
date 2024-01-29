"""Oracle for STRIPS learning."""

from typing import List, Set

from predicators.envs import get_or_create_env
from predicators.ground_truth_models import get_gt_nsrts, get_gt_options
from predicators.nsrt_learning.strips_learning import BaseSTRIPSLearner
from predicators.settings import CFG
from predicators.structs import NSRT, PNAD, Datastore, DummyOption, \
    LiftedAtom, Predicate, Segment, STRIPSOperator
from predicators import utils


class OracleSTRIPSLearner(BaseSTRIPSLearner):
    """Base class for a STRIPS learner that uses oracle operators but re-learns
    all the components via currently-implemented methods in the base class."""

    def _induce_add_effects_by_intersection(self,
                                            pnad: PNAD) -> Set[LiftedAtom]:
        """Given a PNAD with a nonempty datastore, compute the add effects for
        the PNAD's operator by intersecting all lifted add effects."""
        assert len(pnad.datastore) > 0
        for i, (segment, var_to_obj) in enumerate(pnad.datastore):
            objects = set(var_to_obj.values())
            obj_to_var = {o: v for v, o in var_to_obj.items()}
            atoms = {
                atom
                for atom in segment.add_effects
                if all(o in objects for o in atom.objects)
            }
            lifted_atoms = {atom.lift(obj_to_var) for atom in atoms}
            if i == 0:
                add_effects = lifted_atoms
            else:
                add_effects &= lifted_atoms
        return add_effects

    def _compute_datastores_given_nsrt(self, nsrt: NSRT) -> Datastore:
        datastore = []
        assert self._annotations is not None
        for seg_traj, ground_nsrt_list in zip(self._segmented_trajs,
                                              self._annotations):
            assert len(seg_traj) == len(ground_nsrt_list)
            for segment, ground_nsrt in zip(seg_traj, ground_nsrt_list):
                if ground_nsrt.parent == nsrt:
                    op_vars = nsrt.op.parameters
                    obj_sub = ground_nsrt.objects
                    var_to_obj_sub = dict(zip(op_vars, obj_sub))
                    datastore.append((segment, var_to_obj_sub))
        return datastore

    def _find_add_effect_intersection_preds(
            self, segments: List[Segment]) -> Set[Predicate]:
        unique_add_effect_preds: Set[Predicate] = set()
        for seg in segments:
            if len(unique_add_effect_preds) == 0:
                unique_add_effect_preds = set(atom.predicate
                                              for atom in seg.add_effects)
            else:
                unique_add_effect_preds &= set(atom.predicate
                                               for atom in seg.add_effects)
        return unique_add_effect_preds

    def _learn(self) -> List[PNAD]:
        # For this learner to work, we need annotations to be non-null
        # and the length of the annotations to match the length of
        # the trajectory set.
        # assert self._annotations is not None
        # assert len(self._annotations) == len(self._trajectories)
        # assert CFG.offline_data_method == "demo+gt_operators"
        assert self._clusters is not None

        # env = get_or_create_env(CFG.env)
        # env_options = get_gt_options(env.get_name())
        # gt_nsrts = get_gt_nsrts(env.get_name(), env.predicates, env_options)
        pnads: List[PNAD] = []
        import pdb; pdb.set_trace()

        # *** doing this is tricky
        # you know there's some intersection between two segments' effects
        # but each one of those segments may have some effects that the other doesn't have
        # so how do you do that intersection?
        # e.g. you may have ?block1 in segment one, and ?block0 and ?block1 in segment 2, all in a predicate NOT-OnTable(?block[x]).
        # how do you know if it's ?block1 -> ?block1 or ?block1 -> ?block0?
        # you could choose the mapping that maximizes the # predicates in effects in the final inferred operator
        # as you get more data, the extraneous ones will get removed?
        # but as you get more predicates in your grammar, you will have more extraneous ones?
        # number of bijections for a set onto itself, with n elements, is n!
        # but here, our n is small. and it's per type. so it's
        # a! * b! * c! ... for types numbers of objects in type a, b, c.
        # so here, with stack, with 2 blocks and 1 robot, we have 2! * 1 = 2 possibilities. Not bad.
        # but, we have M segments in total, and we have to do this mapping across all of these segments?
        # can you do this greedily?
        # start with one segment, and another. choose the mapping to maximize. and then the intersection of those
        # is the set of predicates you work with for the next one you consider.

        # as a subproblem, consider how to do the intersection itself
        #
        # def lifted_atom_set_intersection(a, b):
        #     # a is a set of lifted atoms
        #     # b is a set of lifted atoms
        #
        #
        #
        #     pass

        for name, v in self._clusters.items():


            preconds, add_effects, del_effects, segments = v

            seg_0 = segments[0]
            opt_objs = tuple(seg_0.get_option().objects)
            relevant_add_effects = [a for a in seg_0.add_effects if a.predicate in add_effects]
            relevant_del_effects = [a for a in seg_0.delete_effects if a.predicate in del_effects]
            objects = {o for atom in relevant_add_effects + relevant_del_effects for o in atom.objects} | set(opt_objs)
            objects_list = sorted(objects)

            params = utils.create_new_variables([o.type for o in objects_list])
            obj_to_var = dict(zip(objects_list, params))
            var_to_obj = dict(zip(params, objects_list))

            relevant_preconds = [a for a in seg_0.init_atoms if (a.predicate in preconds and set(a.objects).issubset(set(objects_list)))]
            op_add_effects = {atom.lift(obj_to_var) for atom in relevant_add_effects}
            op_del_effects = {atom.lift(obj_to_var) for atom in relevant_del_effects}
            op_preconds = {atom.lift(obj_to_var) for atom in relevant_preconds}

            # if name == "Op2-Stack":
            #     import pdb; pdb.set_trace()
            #     block_to_not_del = [p for p in op_preconds if p.predicate.name=="NOT-((0:block).pose_z<=[idx 0]0.461)"][0].entities[0]
            #     new_op_preconds = set()
            #     for p in op_preconds:
            #         if p.predicate.name == "NOT-OnTable" and block_to_not_del not in p.entities:
            #             continue
            #         new_op_preconds.add(p)
            #     op_preconds = new_op_preconds
            #     # this is dealing with the fact that segment 0 happens to be stacking onto a block
            #     # that's not on the table, but this isn't always true. you can be stacking onto a block that
            #     # is on the table.
            #     # need to fix this more generally without hardcoding it in.
            #     # also, if the map isn't 1:1 obvious, it's not clear how exactly how to correspond
            #     # obj_to_var in one segment versus another.
            #     import pdb; pdb.set_trace()

            option_vars = [obj_to_var[o] for o in opt_objs]
            option_spec = [seg_0.get_option().parent, option_vars]

            # if name == "Op2-Stack":
            #     import pdb; pdb.set_trace()

            op_ignore_effects = set()
            op = STRIPSOperator(name, params, op_preconds, op_add_effects, op_del_effects, op_ignore_effects)


            from itertools import permutations, product
            def get_mapping_between_params(params1):
                unique_types = sorted(set(elem.type for elem in params1))
                group_params_by_type = []
                for elem_type in unique_types:
                    elements_of_type = [elem for elem in params1 if elem.type == elem_type]
                    group_params_by_type.append(elements_of_type)

                all_mappings = list(product(*list(permutations(l) for l in group_params_by_type)))
                squash = []
                for m in all_mappings:
                    a = []
                    for i in m:
                        a.extend(i)
                    squash.append(a)

                return squash

            ops = []
            for seg in segments:
                opt_objs = tuple(seg.get_option().objects)
                relevant_add_effects = [a for a in seg.add_effects if a.predicate in add_effects]
                relevant_del_effects = [a for a in seg.delete_effects if a.predicate in del_effects]
                objects = {o for atom in relevant_add_effects + relevant_del_effects for o in atom.objects} | set(opt_objs)
                objects_list = sorted(objects)
                params = utils.create_new_variables([o.type for o in objects_list])
                obj_to_var = dict(zip(objects_list, params))
                var_to_obj = dict(zip(params, objects_list))
                relevant_preconds = [a for a in seg.init_atoms if (a.predicate in preconds and set(a.objects).issubset(set(objects_list)))]

                op_add_effects = {atom.lift(obj_to_var) for atom in relevant_add_effects}
                op_del_effects = {atom.lift(obj_to_var) for atom in relevant_del_effects}
                op_preconds = {atom.lift(obj_to_var) for atom in relevant_preconds}
                # t = (params, var_to_obj, obj_to_var, op_preconds, op_add_effects, op_del_effects)
                t = (params, objects_list, relevant_preconds, relevant_add_effects, relevant_del_effects)
                ops.append(t)


            # We would like to take the intersection of preconditions, add effects, and delete effects
            # between operators in a particular cluster to weed out ones that do not generalize, e.g. that
            # the block you are stacking on (in Op2-Stack), is also on another block (ratherr than on the table).
            # But the object -> variable mapping is not consistent, so this takes some extra effort. For example,
            # consider Op2-Stack, which operates on two blocks and one robot. Sometimes, blockn is put on blockn+1,
            # but other times, blockn+1 is put on blockn. Because the object -> variable mapping is done with sorted
            # objects (which sort by name and then by type), the stack operators created from some segments will have
            # the first parameter be the top block, while others will have the first operator be the second block. So,
            # if we just took an intersection of lifted atoms between the two operators, the predicates would would not
            # correspond to each other properly.
            if name == "Op2-Stack":
                op1 = ops[0]
                op1_params = op1[0]
                op1_objs_list = op1[1]
                op1_obj_to_var = dict(zip(op1_objs_list, op1_params))
                op1_preconds = {atom.lift(op1_obj_to_var) for atom in op1[2]}
                op1_add_effects = {atom.lift(op1_obj_to_var) for atom in op1[3]}
                op1_del_effects = {atom.lift(op1_obj_to_var) for atom in op1[4]}

                op1_preconds_str = set(str(a) for a in op1_preconds)
                op1_adds_str = set(str(a) for a in op1_add_effects)
                op1_dels_str = set(str(a) for a in op1_del_effects)

                # debug:
                # maybe explicitly go find operators where n+1 is put on n, and then where n is put on n+1
                # look at add effects and see the numbers
                # demo 7 seems to have n on n+1 at some point in it
                # demo 0 has n+1 on n
                import re
                def extract_numbers_from_string(input_string):
                    # Use regular expression to find all numeric sequences in the 'blockX' format
                    numbers = re.findall(r'\bblock(\d+)\b', input_string)
                    # Convert the found strings to integers and return a set
                    return set(map(int, numbers))
                # for x, seg in enumerate(segments):
                #     relevant = [str(a) for a in seg.add_effects if a.predicate.name == "On"]
                #     # now see if n+1 on n or n on n+1
                #     assert len(relevant) == 1
                #     z = relevant[0]
                #     nums = extract_numbers_from_string(z)
                #     assert len(nums) == 2
                #     nums_sorted = sorted(list(nums))
                #     higher_on_top = z.index(str(nums_sorted[1])) < z.index(str(nums_sorted[0]))
                    # if not higher_on_top:
                    #     print("HIGHER NOT ON TOP")
                    #     import pdb; pdb.set_trace()

                for i in range(1, len(ops)):
                    if i < 10:
                        continue

                    op2 = ops[i]
                    op2_params = op2[0]
                    op2_objs_list = op2[1]

                    mappings = get_mapping_between_params(op2_params)
                    mapping_scores = []
                    for m in mappings:

                        mapping = dict(zip(op2_params, m))

                        overlap = 0

                        # Get Operator 2's preconditions, add effects, and delete effects
                        # in terms of a particular object -> variable mapping.
                        new_op2_params = [mapping[p] for p in op2_params]
                        new_op2_obj_to_var = dict(zip(op2_objs_list, new_op2_params))
                        op2_preconds = {atom.lift(new_op2_obj_to_var) for atom in op2[2]}
                        op2_add_effects = {atom.lift(new_op2_obj_to_var) for atom in op2[3]}
                        op2_del_effects = {atom.lift(new_op2_obj_to_var) for atom in op2[4]}

                        # Take the intersection of lifted atoms across both operators, and
                        # count the overlap.
                        op2_preconds_str = set(str(a) for a in op2_preconds)
                        op2_adds_str = set(str(a) for a in op2_add_effects)
                        op2_dels_str = set(str(a) for a in op2_del_effects)

                        score1 = len(op1_preconds_str.intersection(op2_preconds_str))
                        score2 = len(op1_adds_str.intersection(op2_adds_str))
                        score3 = len(op1_dels_str.intersection(op2_dels_str))
                        score = score1 + score2 + score3

                        new_preconds = set(a for a in op1_preconds if str(a) in op1_preconds_str.intersection(op2_preconds_str))
                        new_adds = set(a for a in op1_add_effects if str(a) in op1_adds_str.intersection(op2_adds_str))
                        new_dels = set(a for a in op1_del_effects if str(a) in op1_dels_str.intersection(op2_dels_str))

                        mapping_scores.append((score, new_preconds, new_adds, new_dels))

                    import pdb; pdb.set_trace()

                    s, a, b, c = max(mapping_scores, key=lambda x: x[0])
                    op1_preconds = a
                    op1_add_effects = b
                    op1_del_effects = c
                    op1_preconds_str = set(str(a) for a in op1_preconds)
                    op1_adds_str = set(str(a) for a in op1_add_effects)
                    op1_dels_str = set(str(a) for a in op1_del_effects)

                    import pdb; pdb.set_trace()



                import pdb; pdb.set_trace()



            datastore = []
            counter = 0
            for seg in segments:
                seg_opt_objs = tuple(seg.get_option().objects)
                var_to_obj = {v: o for v, o in zip(option_vars, seg_opt_objs)}


                relevant_add_effects = [a for a in seg.add_effects if a.predicate in add_effects]
                relevant_del_effects = [a for a in seg.delete_effects if a.predicate in del_effects]

                seg_objs = {o for atom in relevant_add_effects + relevant_del_effects for o in atom.objects} | set(seg_opt_objs)
                seg_objs_list = sorted(seg_objs)
                remaining_objs = [o for o in seg_objs_list if o not in seg_opt_objs]
                # if you do this, then there's an issue in sampler learning, because it uses
                # pre.variables for pre in preconditions -- so it will look for ?x0 but not find it
                # and there is a key error
                # remaining_params = utils.create_new_variables(
                #     [o.type for o in remaining_objs], existing_vars = list(var_to_obj.keys()))

                from predicators.structs import Variable
                def diff_create_new_variables(types, existing_vars, var_prefix: str = "?x"):
                    pre_len = len(var_prefix)
                    existing_var_nums = set()
                    if existing_vars:
                        for v in existing_vars:
                            if v.name.startswith(var_prefix) and v.name[pre_len:].isdigit():
                                existing_var_nums.add(int(v.name[pre_len:]))
                    def get_next_num(used):
                        counter = 0
                        while True:
                            if counter in used:
                                counter += 1
                            else:
                                return counter
                    new_vars = []
                    for t in types:
                        num = get_next_num(existing_var_nums)
                        existing_var_nums.add(num)
                        new_var_name = f"{var_prefix}{num}"
                        new_var = Variable(new_var_name, t)
                        new_vars.append(new_var)
                    return new_vars
                remaining_params = diff_create_new_variables(
                    [o.type for o in remaining_objs], existing_vars = list(var_to_obj.keys())
                )

                var_to_obj2 = dict(zip(remaining_params, remaining_objs))
                # var_to_obj = dict(zip(seg_params, seg_objs_list))
                var_to_obj = {**var_to_obj, **var_to_obj2}
                datastore.append((seg, var_to_obj))
                if name == "Op2-Stack" and counter == 10:
                    # normally, block n+1 is stacked on block n, but here
                    # block2 is stacked on block3.
                    # so, when we sort the seg_objs_list, we have [block2, block3, robot]
                    # the operator params are such that [?x0:block, "?x1:block, "?x2: robot]
                    # ?x1 is stacked on ?x0.
                    # so, later, in "learn_option_specs()", we get the error: assert option_args == option.objects
                    # because option_args are [block2, robot], while the gt option objects are [block3, robot]
                    # so: how do we order var_to_obj here correctly?
                    # we want a consistent map - take one of the predicates that involves two blocks, and
                    # make sure the assignment of variables is the same (order-wise) as was used in the construction of
                    # params?
                    # that is, if we saw On(x1, x0) in params, then we must also have that here.
                    # how do you choose On to do this for?
                    # or, you can ensure the sub is correct for the option spec

                    # hardcode it for now

                    import pdb; pdb.set_trace()
                counter += 1

            option_vars = [obj_to_var[o] for o in opt_objs]
            option_spec = [seg_0.get_option().parent, option_vars]
            if name == "Op2-Stack":
                import pdb; pdb.set_trace()
            pnads.append(PNAD(op, datastore, option_spec))

            # might be able to handle ignore effects by something like what
            # _compute_pnad_ignore_effects in base_strips_learner.py does.
            # basically you can try to see if the ops would be used anywhere in
            # the demos, and and narrow it down via the demos. but you might
            # pick up some extraneous predicates in some of them, so you would
            # have to be careful and do some inference when trying them out on
            # the demos.

        import pdb; pdb.set_trace()
        return pnads

    # def _learn(self) -> List[PNAD]:
    #     # For this learner to work, we need annotations to be non-null
    #     # and the length of the annotations to match the length of
    #     # the trajectory set.
    #     assert self._annotations is not None
    #     assert len(self._annotations) == len(self._trajectories)
    #     assert CFG.offline_data_method == "demo+gt_operators"
    #
    #     env = get_or_create_env(CFG.env)
    #     env_options = get_gt_options(env.get_name())
    #     gt_nsrts = get_gt_nsrts(env.get_name(), env.predicates, env_options)
    #     pnads: List[PNAD] = []
    #     for nsrt in gt_nsrts:
    #         # If options are unknown, use a dummy option spec.
    #         if CFG.option_learner == "no_learning":
    #             option_spec = (nsrt.option, list(nsrt.option_vars))
    #         else:
    #             option_spec = (DummyOption.parent, [])
    #
    #         datastore = self._compute_datastores_given_nsrt(nsrt)
    #         pnad = PNAD(nsrt.op, datastore, option_spec)
    #         add_effects = self._induce_add_effects_by_intersection(pnad)
    #         preconditions = self._induce_preconditions_via_intersection(pnad)
    #         pnad = PNAD(
    #             pnad.op.copy_with(preconditions=preconditions,
    #                               add_effects=add_effects), datastore,
    #             option_spec)
    #         self._compute_pnad_delete_effects(pnad)
    #         self._compute_pnad_ignore_effects(pnad)
    #         pnads.append(pnad)
    #
    #     return pnads

    @classmethod
    def get_name(cls) -> str:
        return "oracle_clustering"
