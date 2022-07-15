import sys
import shutil
import os
import subprocess
import signal
import time

#theproc = subprocess.run(["python","test_easy.py"], shell = True)
#time.sleep(1)
#os.kill(theproc.pid, signal.CTRL_BREAK_EVENT)
#dir_list = os.listdir("results")
#for i in dir_list:
#    shutil.move("results/" + i, "varun_results/easy")
theproc = subprocess.run(["python","test_medium.py"], shell = True)
time.sleep(1)
dir_list = os.listdir("results")
for i in dir_list:
    shutil.move("results/" + i, "varun_results/medium")
theproc = subprocess.run(["python","test_hard.py"], shell = True)
time.sleep(1)
dir_list = os.listdir("results")
for i in dir_list:
    shutil.move("results/" + i, "varun_results/hard")
#investigate how to allow normal execution of rest of code after main code ends
