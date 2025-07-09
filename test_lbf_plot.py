import pickle
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

# Load results
with open("HITL_experiment_results_random.pkl", "rb") as f:
    results = pickle.load(f)

NUM_OPERATORS = 10  # ensure consistency if changed in experiment

# Helper: aggregate to mean/std
def aggregate(metric):
    agg = defaultdict(list)
    for num_trajs, _, score in results[metric]:
        agg[num_trajs].append(score)
    means = {k: np.mean(v) for k, v in agg.items()}
    stds = {k: np.std(v) for k, v in agg.items()}
    return means, stds

# Compute cumulative scores
def compute_total(eq, cov, ofit):
    cov_total = {k: eq[k] + cov[k] for k in eq}
    ofit_total = {k: cov_total[k] + ofit[k] for k in eq}
    return cov_total, ofit_total

# Get all metrics
eq, eq_std = aggregate("equivalent")
cov, cov_std = aggregate("covered")
ofit, ofit_std = aggregate("overfit")
miss, miss_std = aggregate("missed")

eq1, eq1_std = aggregate("hitl_1_equivalent")
cov1, cov1_std = aggregate("hitl_1_covered")
ofit1, ofit1_std = aggregate("hitl_1_overfit")
miss1, miss1_std = aggregate("hitl_1_missed")

eq5, eq5_std = aggregate("hitl_5_equivalent")
cov5, cov5_std = aggregate("hitl_5_covered")
ofit5, ofit5_std = aggregate("hitl_5_overfit")
miss5, miss5_std = aggregate("hitl_5_missed")

# Compute cumulative
cov_total, ofit_total = compute_total(eq, cov, ofit)
cov1_total, ofit1_total = compute_total(eq1, cov1, ofit1)
cov5_total, ofit5_total = compute_total(eq5, cov5, ofit5)

# Plot
plt.figure(figsize=(14, 8))
x_vals = sorted(eq)

def plot_with_error(x, y_mean, y_std, label, color, linestyle):
    y = [y_mean[k] for k in x]
    err = [y_std.get(k, 0) for k in x]
    plt.errorbar(x, y, yerr=err, label=label, fmt=linestyle, color=color, capsize=4)

# Baseline
# plot_with_error(x_vals, eq, eq_std, "Exact", "black", "o-")
# plot_with_error(x_vals, cov_total, cov_std, "Covered (incl. exact)", "black", "--")
# plot_with_error(x_vals, ofit_total, ofit_std, "Overfit (incl. cov)", "black", ":")
# plot_with_error(x_vals, miss, miss_std, "Missed", "black", "-.")

# # HITL-1
plot_with_error(x_vals, eq1, eq1_std, "HITL-1 Exact", "blue", "o-")
plot_with_error(x_vals, cov1_total, cov1_std, "HITL-1 Covered", "blue", "--")
plot_with_error(x_vals, ofit1_total, ofit1_std, "HITL-1 Overfit", "blue", ":")
# plot_with_error(x_vals, miss1, miss1_std, "HITL-1 Missed", "blue", "-.")

# HITL-5
# plot_with_error(x_vals, eq5, eq5_std, "HITL-5 Exact", "green", "o-")
# plot_with_error(x_vals, cov5_total, cov5_std, "HITL-5 Covered", "green", "--")
# plot_with_error(x_vals, ofit5_total, ofit5_std, "HITL-5 Overfit", "green", ":")
# plot_with_error(x_vals, miss5, miss5_std, "HITL-5 Missed", "green", "-.")

plt.xlabel("Number of Demonstrations")
plt.ylabel("Operators")
plt.title("Operator Learning Comparison: Exact, Covered, Overfit, Missed")
plt.legend(loc="upper left", fontsize="small", ncol=2)
plt.grid(True)
plt.tight_layout()
plt.savefig("operator_learning_summary.png")
plt.ylim(0, 10)
plt.show()
