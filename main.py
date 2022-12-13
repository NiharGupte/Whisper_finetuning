import os

model_list = ["openai/whisper-medium"]
dir_list = ["medium"]

for m, d in zip(model_list, dir_list):
    cmd = "python data_preprocess.py --model_name " + m + " --denoise 1"
    os.system(cmd)
    cmd = "python whisper_train.py --model_name " + m + " --model_dir " + d
    os.system(cmd)
    cmd = "python whisper_eval.py --model_name " + m + " --finetuned_model_name " + d + " --denoising 0"
    os.system(cmd)
    os.system("rm -r " + m + "_test.hf") #
