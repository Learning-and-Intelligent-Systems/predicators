from predicators.third_party.fast_downward_translator.pddl_parser import \
    lisp_parser, parsing_functions

file_open = open


def open(domain_string, task_string):
    domain_pddl = lisp_parser.parse_nested_list(domain_string)
    task_pddl = lisp_parser.parse_nested_list(task_string)
    return parsing_functions.parse_task(domain_pddl, task_pddl)
