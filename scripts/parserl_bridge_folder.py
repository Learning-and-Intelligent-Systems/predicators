import re
import os


# Folder path where your files are located
folder_path = "lockeexperiments/logs"

# Iterate through all the files in the folder
counter = 0
smooth_test_rwd =[]
train_rwd =[]
for filename in os.listdir(folder_path):
    # Check if "mapleq" is in the filename
    #CHANGE THIS TO FIND THE WANTED FILES
    # print(filename)
    if "grid_row_door__rl_bridge_policy__RLBRIDGE_gridrowdoor-rl_rwd_shape" in filename:
        file_path = os.path.join(folder_path, filename)
        print(file_path)
        with open(file_path, "r") as f:
            log_txt = f.readlines()
 
        training_time_rewards_list = []
        for line in log_txt:
            file_path = os.path.join(folder_path, filename)
                
            match = re.search(r'WE GOT REWARDS:\s+(\d+)', line)
            if match:
                training_time_rewards_list.append(float(match.group(1)))

        # print(f"Got {len(training_time_rewards_list)} training rewards!\n{training_time_rewards_list}")
        counter+=1
        train_rwd.append(training_time_rewards_list)


        testing_time_rewards_list = []
        for line in log_txt:
            file_path = os.path.join(folder_path, filename)
                
            match = re.search(r'^SMOOTH REWARDS\s+\[([^\]]+)\]', line)
            if match:
                testing_time_rewards_list.append(float(match.group(1)))

        # print(f"Got {len(testing_time_rewards_list)} SMOOTH TEST rewards!\n{testing_time_rewards_list}")
        smooth_test_rwd.append(testing_time_rewards_list)
print("number of files: ", counter)
print("TRAIN RWDS", train_rwd)
print("TEST RWDS", smooth_test_rwd)





# Folder path where your files are located
folder_path = "1232amexperiments/logs"

# Iterate through all the files in the folder
counter = 0
smooth_test_rwd =[]
train_rwd =[]
for filename in os.listdir(folder_path):
    # Check if "mapleq" is in the filename
    #CHANGE THIS TO FIND THE WANTED FILES
    if "grid_row_door__rl_bridge_policy__RLBRIDGE_gridrowdoor-rl_rwd_shape__" in filename:
        file_path = os.path.join(folder_path, filename)
        print(file_path)
        with open(file_path, "r") as f:
            log_txt = f.readlines()
 
        training_time_rewards_list = []
        for line in log_txt:
            file_path = os.path.join(folder_path, filename)
                
            match = re.search(r'WE GOT REWARDS:\s+(\d+)', line)
            if match:
                training_time_rewards_list.append(float(match.group(1)))

        # print(f"Got {len(training_time_rewards_list)} training rewards!\n{training_time_rewards_list}")
        counter+=1
        train_rwd.append(training_time_rewards_list)


        testing_time_rewards_list = []
        for line in log_txt:
            file_path = os.path.join(folder_path, filename)
                
            match = re.search(r'^SMOOTH REWARDS\s+([\d\s]+)', line)
            if match:
                for smooth_reward in match.group(1).split(" "):
                    testing_time_rewards_list.append(float(smooth_reward))

        # print(f"Got {len(testing_time_rewards_list)} SMOOTH TEST rewards!\n{testing_time_rewards_list}")
        smooth_test_rwd.append(testing_time_rewards_list)
print("number of files: ", counter)
print("TRAIN RWDS", train_rwd)
print("TEST RWDS", smooth_test_rwd)




# Folder path where your files are located
folder_path = "1230amexperiments/logs"

# Iterate through all the files in the folder
counter = 0
smooth_test_rwd =[]
train_rwd =[]
for filename in os.listdir(folder_path):
    # Check if "mapleq" is in the filename
    #CHANGE THIS TO FIND THE WANTED FILES
    if "grid_row_door__rl_bridge_policy__RLBRIDGE_gridrowdoor-rl_rwd_shape__" in filename:
        file_path = os.path.join(folder_path, filename)
        print(file_path)
        with open(file_path, "r") as f:
            log_txt = f.readlines()
 
        training_time_rewards_list = []
        for line in log_txt:
            file_path = os.path.join(folder_path, filename)
                
            match = re.search(r'WE GOT REWARDS:\s+(\d+)', line)
            if match:
                training_time_rewards_list.append(float(match.group(1)))

        # print(f"Got {len(training_time_rewards_list)} training rewards!\n{training_time_rewards_list}")
        counter+=1
        train_rwd.append(training_time_rewards_list)


        testing_time_rewards_list = []
        for line in log_txt:
            file_path = os.path.join(folder_path, filename)
                
            match = re.search(r'^SMOOTH REWARDS\s+([\d\s]+)', line)
            if match:
                testing_time_rewards_list.append(float(match.group(1)))

        # print(f"Got {len(testing_time_rewards_list)} SMOOTH TEST rewards!\n{testing_time_rewards_list}")
        smooth_test_rwd.append(testing_time_rewards_list)
print("number of files: ", counter)
print("TRAIN RWDS", train_rwd)
print("TEST RWDS", smooth_test_rwd)

