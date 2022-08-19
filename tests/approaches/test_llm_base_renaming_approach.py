"""Test cases for the abstract base renaming open-loop LLM approach."""

from predicators import utils
from predicators.approaches.llm_base_renaming_approach import \
    LLMBaseRenamingApproach
from predicators.approaches.llm_open_loop_approach import \
    LLMOpenLoopApproach
from predicators.envs import create_new_env


class _MockLLMBaseRenamingApproach(LLMBaseRenamingApproach):

    @classmethod
    def get_name(cls):
        return "mock_llm_renaming"

    @property
    def _renaming_prefixes(self):
        return [" ", "\n"]

    @property
    def _renaming_suffixes(self):
        return ["("]

    def _create_replacements(self):
        return {"at": "foo", "satisfied": "foob", "move": "baz"}

    def create_original_prompt(self, init, goal, options):
        """Expose the original prompt."""
        return LLMOpenLoopApproach._create_prompt(self, init, goal, options)  # pylint: disable=protected-access

    def create_prompt(self, init, goal, options):
        """Expose the prompt."""
        return self._create_prompt(init, goal, options)

    def llm_prediction_to_option_plan(self, llm_prediction, objects):
        """Expose the option plan."""
        return self._llm_prediction_to_option_plan(llm_prediction, objects)


def test_llm_syntax_renaming_approach():
    """Tests for LLMSyntaxRenamingApproach()."""
    env_name = "pddl_easy_delivery_procedural_tasks"
    utils.reset_config({
        "env": env_name,
        "approach": "llm_syntax_renaming",
        "num_train_tasks": 1,
        "num_test_tasks": 1,
        "strips_learner": "oracle",
    })
    env = create_new_env(env_name)
    train_tasks = env.get_train_tasks()
    approach = _MockLLMBaseRenamingApproach(env.predicates, env.options,
                                            env.types, env.action_space,
                                            train_tasks)
    assert approach.get_name() == "mock_llm_renaming"
    task = train_tasks[0]
    init = utils.abstract(task.init, env.predicates)
    original_prompt = approach.create_original_prompt(init, task.goal, [])
    prompt = approach.create_prompt(init, task.goal, [])
    assert original_prompt.count("at(") == prompt.count("foo(")
    assert original_prompt.count("satisfied(") == prompt.count("foob(")
    llm_prediction = "\nbaz(loc-1:loc, loc-2:loc)\nbaz(loc-2:loc, loc-0:loc)\n"
    objs = set(task.init)
    option_plan = approach.llm_prediction_to_option_plan(llm_prediction, objs)
    option_plan_str = str(option_plan)
    assert option_plan_str == "[(SingletonParameterizedOption(name='move', types=[Type(name='loc'), Type(name='loc')]), [loc-1:loc, loc-2:loc]), (SingletonParameterizedOption(name='move', types=[Type(name='loc'), Type(name='loc')]), [loc-2:loc, loc-0:loc])]"  # pylint: disable=line-too-long
