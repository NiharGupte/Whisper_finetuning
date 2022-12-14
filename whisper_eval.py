import argparse
import datasets
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
import torchaudio
import scipy.io.wavfile as swi
from speechbrain.pretrained import SepformerSeparation as separator

model = separator.from_hparams(source="speechbrain/sepformer-wham-enhancement",
                               savedir='pretrained_models/sepformer-wham-enhancement')

args = argparse.ArgumentParser()

args.add_argument("--model_name", type=str, required=True)
args.add_argument("--finetuned_model_name", type=str, required=True)
args.add_argument("--denoising", type=bool, required=True)

a = args.parse_args()
#
model_name = a.model_name
finetuned_model_name = a.finetuned_model_name
denoise = a.denoising

feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
tokenizer = WhisperTokenizer.from_pretrained(model_name, language="english", task="transcribe")
processor = WhisperProcessor.from_pretrained(model_name, language="english", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(finetuned_model_name)
denoise_model = separator.from_hparams(source="speechbrain/sepformer-wham-enhancement",
                                       savedir='pretrained_models/sepformer-wham-enhancement')

if denoise:
    print("Applying denoising ...")


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


    common_voice = load_dataset("pokameswaran/ami-6h")
    common_voice = common_voice.remove_columns(
        ["file", "length", "segment_id", "segment_start_time", "segment_end_time"])
    common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))
    common_voice = common_voice["test"]
    common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names, num_proc=2)

    common_voice.save_to_disk(model_name + "_denoised_test.hf")
else:
    print("Loading data from disk ... ")
    data_path = model_name + "_test.hf"
    common_voice = datasets.load_from_disk(data_path)
    common_voice = common_voice["test"]


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors

        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

metric = evaluate.load("wer")


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


training_args = Seq2SeqTrainingArguments(
    output_dir="./cache",  # change to a repo name of your choice
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=500,
    gradient_checkpointing=True,
    fp16=True,
    num_train_epochs=5,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_eval_batch_size=2,
    predict_with_generate=True,
    generation_max_length=225,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False
)

trainer_eval = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=None,
    eval_dataset=common_voice,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

y = trainer_eval.evaluate()

with open("model_comp.txt", "a") as f:
    string = "Model : " + model_name + "\n"
    f.write(string)
    f.write(str(y))
    f.write("\n")