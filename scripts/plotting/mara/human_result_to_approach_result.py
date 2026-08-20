"""Convert human evaluation results to approach result format."""
import csv
import os
import pickle

from predicators.settings import CFG

# Input and output file paths
csv_file = os.path.join("scripts", "plotting", "mara", "results",
                        "slido_results.csv")
pickle_file = os.path.join(CFG.results_dir,
                           "all_tasks__human__0______all-human__None.pkl")

# Read CSV and extract {env_name}_accuracy : accuracy_float
data_dict = {}
with open(csv_file, mode="r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    accuracys = []
    for row in reader:
        env_name = row.get("env_name")  # Adjust if column name differs
        if env_name is not None:
            env_name = env_name.lower()
        accuracy_str = row.get(
            "correct_percentage")  # Adjust if column name differs
        if env_name is not None and accuracy_str is not None:
            try:
                accuracy = float(accuracy_str)
                data_dict[f"{env_name}_accuracy"] = accuracy
                accuracys.append(accuracy)
            except ValueError:
                pass  # skip rows where accuracy is not a valid float
    # add average accuracy
    if accuracys:
        average_accuracy = sum(accuracys) / len(accuracys)
        data_dict["avg_accuracy"] = average_accuracy

# Save dictionary to pickle file
with open(pickle_file, "wb") as pf:
    pickle.dump(data_dict, pf)

print(f"Extracted {len(data_dict)} items and saved to '{pickle_file}'.")
