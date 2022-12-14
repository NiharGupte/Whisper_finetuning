import os

model_list = ["openai/whisper-medium"]
dir_list = ["medium"]

nv = 0

for m, d in zip(model_list, dir_list):
    cmd = "python data_preprocess.py --model_name " + m + " --need_validation " + str(nv)
    os.system(cmd)
    cmd = "python whisper_train.py --model_name " + m + " --model_dir " + d
    os.system(cmd)
    if nv==1:
      cmd = "python whisper_eval.py --model_name " + m + " --finetuned_model_name " + d + " --denoising 1"
      os.system(cmd)
    os.system("rm -r " + m + "_test.hf")
