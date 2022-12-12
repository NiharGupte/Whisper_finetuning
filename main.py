# import os
#
# model_list = ["openai/whisper-tiny"]
# dir_list = ["tiny"]
#
# for m, d in zip(model_list, dir_list):
#     cmd = "python whisper_train.py --model_name " + m + " --model_dir " + d
#     os.system(cmd)
import os

model_list = ["openai/whisper-tiny", "openai/whisper-base", "openai/whisper-medium"]
dir_list = ["tiny", "base", "medium"]

for m, d in zip(model_list, dir_list):
    cmd = "python data_preprocess.py --model_name " + m
    os.system(cmd)
    cmd = "python whisper_train.py --model_name " + m + " --model_dir " + d
    os.system(cmd)
    cmd = "python whisper_eval.py --model_name " + m + " --finetuned_model_name " + d
    os.system(cmd)

    os.system("rm -r" + m + "_test.hf")
