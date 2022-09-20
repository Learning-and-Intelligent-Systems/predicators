# Running Experiments on Supercloud

## First Time Setup

### Getting Access to Supercloud

* If you do not yet have a supercloud account, you will need to [request one](https://supercloud.mit.edu/requesting-account). 
* After receiving an account, follow the instructions to ssh into supercloud.
* Once logged in to supercloud, [create an ssh key and add it to your github.com account](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent).
* The most important rule about using supercloud is that you should not run any intensive jobs on a login node. Instead, launch batch jobs, or use an interactive session. Both options are illustrated below.

### Installing this Repository

Run the following commands to install this repository.

```
# Set `PYTHONHASHSEED` to 0 by default (see our README.md for explanation).
echo "export PYTHONHASHSEED=0" >> ~/.bashrc
# Clone the repository.
git clone https://github.com/Learning-and-Intelligent-Systems/predicators_behavior.git
# Set up conda with Python 3.9.
module unload anaconda
module load anaconda/2021b
conda create --name predicators_behavior python=3.9
conda init bash  # will need to restart shell after this
conda activate predicators_behavior
# Install the predicators dependencies.
cd predicators_behavior
mkdir /state/partition1/user/$USER
export TMPDIR=/state/partition1/user/$USER
pip install -r requirements.txt
# Add a shortcut for activating the conda env and switching to this repository.
echo -e "predicate_behavior() {\n    cd ~/predicators_behavior\n    conda activate predicators_behavior\n}" >> ~/.bashrc
# Add a shortcut for displaying running jobs.
echo "alias sl='squeue --format=\"%.18i %.9P %.42j %.8u %.8T %.10M %.6D %R\"'" >> ~/.bashrc
source ~/.bashrc
```
To test if it worked:
```
# Start an interactive session.
LLsub -i
# Activate conda and switch to the repository.
predicate_behavior
# Run a short experiment.
python predicators/main.py --env cover --approach oracle --seed 0
# Exit the interactive session.
exit
```
Note that supercloud sometimes hangs, so the experiment may take a few minutes to get started. But once it does, you should see 50/50 tasks solved, and the script should terminate in roughly 2 seconds (as reported at the bottom).

## Running Experiments

To get started, activate the conda environment and switch to the repository. If you followed the instructions above, you can do both with `predicate_behavior`.

Before running any experiments, it is good practice to make sure that you have a clean workspace:
* Make sure that you have already backed up any old results that you want to keep.
* Remove all previous results: `rm -f results/* logs/* saved_approaches/* saved_datasets/*`.
* Make sure you are on the right branch (`git branch`) with a clean diff (`git diff`).

To run our default suite of experiments (will take many hours to complete, we recommend letting it run overnight):
```
./scripts/supercloud/run_nightly_experiments.sh
```

Upon running that script, you should see many printouts, such as:
```
Running command: sbatch --time=99:00:00 --partition=xeon-p8 --nodes=1 --exclusive --job-name=cover_oracle --array=456-465 -o logs/cover__oracle__%a______cover_oracle__%j.log temp_run_file.sh
```

After experiments are running:
* To monitor experiments that are running, use `sl`.
* To see individual logs, look in the `logs/` directory.
* To cancel all jobs, use `scancel -u $USER`.
* To see a summary of results so far, run `python scripts/analyze_results_directory.py`.
* To download results onto your local machine, use `scp -r`. The most important directory to back up is `results/`, but we also recommend backing up `logs/`, `saved_datasets/`, and `saved_approaches/`.

## Contributing

If any of the above steps do not work perfectly or lack clarity, please update this document with a pull request!
