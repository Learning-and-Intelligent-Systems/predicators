import pickle

filename = 'results.pkl'

with open(filename, 'rb') as file:
    results = pickle.load(file)

filename = 'HITL_results.pkl'

with open(filename, 'rb') as file:
    results_hitl = pickle.load(file)

filename = 'HITL_more_results.pkl'

with open(filename, 'rb') as file:
    results_more_hitl = pickle.load(file)


import pandas as pd

# Assuming `results` is your dictionary of lists of tuples
df_matches = pd.DataFrame(results["tot_matches"], columns=["num_trajs", "run_i", "num_match"])
df_soft_matches = pd.DataFrame(results["tot_soft_matches"], columns=["num_trajs", "run_i", "num_soft_match"])
df_exsoft_matches = pd.DataFrame(results["tot_exsoft_matches"], columns=["num_trajs", "run_i", "num_exsoft_match"])
df_num_ops = pd.DataFrame(results["tot_num_ops"], columns=["num_trajs", "run_i", "num_op_sets", "num_actions"])

agg_matches = df_matches.groupby("num_trajs")["num_match"].agg(["mean", "std"]).reset_index()
agg_soft = df_soft_matches.groupby("num_trajs")["num_soft_match"].agg(["mean", "std"]).reset_index()
agg_exsoft = df_exsoft_matches.groupby("num_trajs")["num_exsoft_match"].agg(["mean", "std"]).reset_index()
agg_ops = df_num_ops.groupby("num_trajs")[["num_op_sets", "num_actions"]].agg(["mean", "std"]).reset_index()

df_matches_hitl = pd.DataFrame(results_hitl["tot_matches"], columns=["num_trajs", "run_i", "num_match"])
df_soft_matches_hitl = pd.DataFrame(results_hitl["tot_soft_matches"], columns=["num_trajs", "run_i", "num_soft_match"])
df_exsoft_matches_hitl = pd.DataFrame(results_hitl["tot_exsoft_matches"], columns=["num_trajs", "run_i", "num_exsoft_match"])
df_num_ops_hitl = pd.DataFrame(results_hitl["tot_num_ops"], columns=["num_trajs", "run_i", "num_op_sets", "num_actions"])

agg_matches_hitl = df_matches_hitl.groupby("num_trajs")["num_match"].agg(["mean", "std"]).reset_index()
agg_soft_hitl = df_soft_matches_hitl.groupby("num_trajs")["num_soft_match"].agg(["mean", "std"]).reset_index()
agg_exsoft_hitl = df_exsoft_matches_hitl.groupby("num_trajs")["num_exsoft_match"].agg(["mean", "std"]).reset_index()
agg_ops_hitl = df_num_ops_hitl.groupby("num_trajs")[["num_op_sets", "num_actions"]].agg(["mean", "std"]).reset_index()

df_matches_more_hitl = pd.DataFrame(results_more_hitl["tot_matches"], columns=["num_trajs", "run_i", "num_match"])
df_soft_matches_more_hitl = pd.DataFrame(results_more_hitl["tot_soft_matches"], columns=["num_trajs", "run_i", "num_soft_match"])
df_exsoft_matches_more_hitl = pd.DataFrame(results_more_hitl["tot_exsoft_matches"], columns=["num_trajs", "run_i", "num_exsoft_match"])
df_num_ops_more_hitl = pd.DataFrame(results_more_hitl["tot_num_ops"], columns=["num_trajs", "run_i", "num_op_sets", "num_actions"])

agg_matches_more_hitl = df_matches_more_hitl.groupby("num_trajs")["num_match"].agg(["mean", "std"]).reset_index()
agg_soft_more_hitl = df_soft_matches_more_hitl.groupby("num_trajs")["num_soft_match"].agg(["mean", "std"]).reset_index()
agg_exsoft_more_hitl = df_exsoft_matches_more_hitl.groupby("num_trajs")["num_exsoft_match"].agg(["mean", "std"]).reset_index()
agg_ops_more_hitl = df_num_ops_more_hitl.groupby("num_trajs")[["num_op_sets", "num_actions"]].agg(["mean", "std"]).reset_index()

import matplotlib.pyplot as plt

plt.figure()
# No-HITL
plt.errorbar(agg_matches["num_trajs"], agg_matches["mean"], yerr=agg_matches["std"], label="Match (no-HITL)")
plt.errorbar(agg_soft["num_trajs"], agg_soft["mean"], yerr=agg_soft["std"], label="Soft Match (no-HITL)")
plt.errorbar(agg_exsoft["num_trajs"], agg_exsoft["mean"], yerr=agg_exsoft["std"], label="ExSoft Match (no-HITL)")

# HITL
plt.errorbar(agg_matches_hitl["num_trajs"], agg_matches_hitl["mean"], yerr=agg_matches_hitl["std"], linestyle='--', label="Match (HITL)")
plt.errorbar(agg_soft_hitl["num_trajs"], agg_soft_hitl["mean"], yerr=agg_soft_hitl["std"], linestyle='--', label="Soft Match (HITL)")
plt.errorbar(agg_exsoft_hitl["num_trajs"], agg_exsoft_hitl["mean"], yerr=agg_exsoft_hitl["std"], linestyle='--', label="ExSoft Match (HITL)")

# More HITL
plt.errorbar(agg_matches_more_hitl["num_trajs"], agg_matches_more_hitl["mean"], yerr=agg_matches_more_hitl["std"], linestyle=':', label="Match (HITL+)")
plt.errorbar(agg_soft_more_hitl["num_trajs"], agg_soft_more_hitl["mean"], yerr=agg_soft_more_hitl["std"], linestyle=':', label="Soft Match (HITL+)")
plt.errorbar(agg_exsoft_more_hitl["num_trajs"], agg_exsoft_more_hitl["mean"], yerr=agg_exsoft_more_hitl["std"], linestyle=':', label="ExSoft Match (HITL+)")

plt.xlabel("Number of Trajectories")
plt.ylabel("Matches")
plt.title("Match Types vs Number of Trajectories (HITL vs No-HITL vs. More-HITL)")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()  # Adjust layout to make space for legend
plt.grid(True)
plt.show()

plt.figure()
# No-HITL
plt.errorbar(agg_ops["num_trajs"], agg_ops[("num_op_sets", "mean")], yerr=agg_ops[("num_op_sets", "std")], label="# Operators (no-HITL)")
plt.errorbar(agg_ops["num_trajs"], agg_ops[("num_actions", "mean")], yerr=agg_ops[("num_actions", "std")], label="# Actions (no-HITL)")

plt.xlabel("Number of Trajectories")
plt.ylabel("Count")
plt.title("Operator and Action Counts vs Number of Trajectories (No-HITL)")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()  # Adjust layout to make space for legend
plt.grid(True)
plt.show()
