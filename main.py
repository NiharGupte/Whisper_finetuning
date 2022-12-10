import os

model_list = ["openai/whisper-tiny"]
dir_list = ["tiny"]

for m, d in zip(model_list, dir_list):
    cmd = "python whisper_train.py --model_name " + m + " --model_dir " + d
    os.system(cmd)