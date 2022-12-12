import argparse
from datasets import load_dataset, DatasetDict
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor
from datasets import Audio
import torch
import evaluate
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from transformers import WhisperForConditionalGeneration
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer
import os

args = argparse.ArgumentParser()

args.add_argument("--model_name", type=str, required=True)

a = args.parse_args()

model_name = a.model_name

feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
tokenizer = WhisperTokenizer.from_pretrained(model_name, language="english", task="transcribe")
processor = WhisperProcessor.from_pretrained(model_name, language="english", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(model_name)


def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["text"]).input_ids
    return batch

#
common_voice = load_dataset("pokameswaran/ami-6h")
common_voice = common_voice.remove_columns(["file", "length", "segment_id", "segment_start_time", "segment_end_time"])
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))
common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=2)

data_name = model_name + "_test.hf"
common_voice.save_to_disk(data_name)

