import argparse
from datasets import load_dataset, DatasetDict
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor
from datasets import Audio
import torchaudio
import evaluate
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from transformers import WhisperForConditionalGeneration
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer
import os
import scipy.io.wavfile as swi
from speechbrain.pretrained import SepformerSeparation as separator

args = argparse.ArgumentParser()

args.add_argument("--model_name", type=str, required=True)
args.add_argument("--denoise", type=int, required=True)

a = args.parse_args()

model_name = a.model_name
denoise = a.denoise
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
tokenizer = WhisperTokenizer.from_pretrained(model_name, language="english", task="transcribe")
processor = WhisperProcessor.from_pretrained(model_name, language="english", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(model_name)
denoise_model = separator.from_hparams(source="speechbrain/sepformer-wham-enhancement",
                                       savedir='pretrained_models/sepformer-wham-enhancement')

if denoise==0:
    def prepare_dataset(batch):
        # load and resample audio data from 48 to 16kHz
        audio = batch["audio"]

        # compute log-Mel input features from input audio array
        batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

        # encode target text to label ids
        batch["labels"] = tokenizer(batch["text"]).input_ids
        return batch
else:

    def prepare_dataset(batch):
        # load and resample audio data from 48 to 16kHz
        audio = batch["audio"]
        # print("first : ",audio["array"].shape)
        swi.write("temp.wav", data=audio["array"], rate=16000)

        est_sources = denoise_model.separate_file(path="temp.wav")
        audio["array"] = torchaudio.functional.resample(est_sources[:, :, 0].detach().cpu(), orig_freq=8000,
                                                        new_freq=16000).reshape((-1,))
        # print("Second : ",audio["array"].shape)
        # compute log-Mel input features from input audio array
        batch["input_features"] = \
        feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

        # encode target text to label ids
        batch["labels"] = tokenizer(batch["text"]).input_ids
        # print(batch, batch.keys())
        return batch
#
common_voice = load_dataset("pokameswaran/ami-6h")
common_voice = common_voice.remove_columns(["file", "length", "segment_id", "segment_start_time", "segment_end_time"])
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))
common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=2)

data_name = model_name + "_test.hf"
common_voice.save_to_disk(data_name)

