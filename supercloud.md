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
git clone git@github.com:Learning-and-Intelligent-Systems/predicators.git
# Set up conda with Python 3.9.
module unload anaconda
module load anaconda/2021b
conda create --name predicators python=3.9
conda init bash  # will need to restart shell after this
conda activate predicators
# Install the predicators dependencies.
cd predicators
mkdir /state/partition1/user/$USER
export TMPDIR=/state/partition1/user/$USER
pip install -r requirements.txt
# Add a shortcut for activating the conda env and switching to this repository.
echo -e "predicate() {\n    cd ~/predicators\n    conda activate predicators\n}" >> ~/.bashrc
```
To test if it worked:
```
# Start an interactive session.
LLsub -i
# Activate conda and switch to the repository.
predicate
# Run a short experiment.
python src/main.py --env cover --approach oracle --seed 0
# Exit the interactive session.
exit
```
Note that supercloud sometimes hangs, so the experiment may take a while to get started. But once it does, you should see 50/50 tasks solved, and the script should terminate in roughly 2 seconds (as reported at the bottom).

## Running Experiments

Before running any experiments, it is good practice to make sure that you have a clean workspace:
* Make sure that you have already backed up any old results that you want to keep.
* Remove all previous results: `rm -rf results/* supercloud_* saved_*`.
* Make sure you are on the right branch (`git branch`) with a clean diff (`git diff`).

To run our default suite of experiments (will take many hours to complete):
```
# Activate conda and switch to the repository.
predicate
# Launch experiments.
./analysis/run_supercloud_experiments.sh
```

Upon running that script, you should see many printouts, such as:
```
Running command: sbatch -p normal --time=99:00:00 --partition=xeon-p8 --nodes=1 --exclusive --job-name=run.sh -o supercloud_logs/%j_log.out run_534559589715824.sh
Started job, see log with:
tail -n 10000 -F supercloud_logs/14438294_log.out
```

After experiments are running:
* To monitor experiments that are running, use `squeue`.
* As indicated by the printouts, to see individual logs, you can use `tail -n 10000 -F supercloud_logs/14438294_log.out`.
* To cancel all jobs, use `scancel -u $USER`.
* To see a summary of results, do `python analysis/analyze_results_directory.py`.
* To download results onto your local machine, use `scp -r`. The most important directory to back up is `results/`, but we also recommend backing up `supercloud_logs/`, `saved_datasets/`, and `saved_approaches/`.

### Contributing

If any of the above steps do not work perfectly or lack clarity, please update this document with a pull request!
