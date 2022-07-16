import os
import shutil
import subprocess
import time

theproc = subprocess.run(["python", "test_easy.py"], shell=True)
time.sleep(1)
dir_list = os.listdir("results")
for i in dir_list:
    shutil.move("results/" + i, "varun_results/easy")
theproc = subprocess.run(["python", "test_medium.py"], shell=True)
time.sleep(1)
dir_list = os.listdir("results")
for i in dir_list:
    shutil.move("results/" + i, "varun_results/medium")
theproc = subprocess.run(["python", "test_hard.py"], shell=True)
time.sleep(1)
dir_list = os.listdir("results")
for i in dir_list:
    shutil.move("results/" + i, "varun_results/hard")
