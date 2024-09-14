import re

log_filepath = "bridge_policy_experiment_logs/coffee__rl_bridge_policy__RLBRIDGE_coffee-oracle__1__27003151.log"
with open(log_filepath, "r") as f:
    log_txt = f.readlines()

training_time_rewards_list = []
for line in log_txt:
    match = re.search(r'WE GOT REWARDS:\s+(\d+)', line)
    if match:
        training_time_rewards_list.append(float(match.group(1)))

print(f"Got {len(training_time_rewards_list)} rewards!\n{training_time_rewards_list}")
