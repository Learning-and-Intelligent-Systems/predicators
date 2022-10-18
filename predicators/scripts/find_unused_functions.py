"""Report all unused top-level functions in a particular file.

Note: this can report false positives because we only check for the string:
"<function name>(", while sometimes functions are used in other ways.
"""

import re
import subprocess

FILENAME = "predicators/utils.py"
DIRS_TO_CHECK = [
    "predicators/",
    "scripts/",
    # "tests/",
]

with open(FILENAME, "r", encoding="utf-8") as f:
    lines = f.readlines()
for line in lines:
    if line.startswith("def "):  # top-level only!
        match = re.match(r"def (.+?)\(.*", line)
        assert match is not None, f"Malformed line: {line}"
        func_name = match.groups()[0]
        dirs = " ".join(DIRS_TO_CHECK)
        results = subprocess.getoutput(f"git grep '{func_name}(' {dirs}")
        assert results  # at least must match the definition line
        num_hits = len(results.split("\n"))
        assert num_hits > 0
        if num_hits == 1:
            print(func_name)
