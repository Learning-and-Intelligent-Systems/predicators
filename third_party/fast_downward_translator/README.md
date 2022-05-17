Fast Downward translator, with light modifications made by Rohan.

Code copied from the `src/translate/` directory in the [official repository at this commit from May 9, 2022](https://github.com/aibasel/downward/tree/3e3759d091196515fa68c44a729153100747c4bf). All credits go to the original authors.

To use, call the function `main()` in translate.py, which takes in domain and problem file strings and returns a `SASTask` object (sas_tasks.py) representing a ground planning problem.

Modifications:
* Changed input and output to not require file I/O.
* Removed [options.py](https://github.com/aibasel/downward/blob/3e3759d091196515fa68c44a729153100747c4bf/src/translate/options.py) and associated command-line arguments, replacing with the default values from the linked file.
* Removed tests.
* Changed imports to be absolute.
* Ran our code autoformatter.