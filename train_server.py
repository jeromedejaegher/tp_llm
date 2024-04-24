import os

# Choix du GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np

import torch

from datasets import load_dataset
from transformers import CamembertForSequenceClassification, CamembertTokenizer
from transformers import DataCollatorWithPadding
import evaluate

from transformers import TrainingArguments, Trainer
import pynvml


ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
print(f"ROOT_DIR : {ROOT_DIR}")
resume=True
nb_epochs = 3
batch_size = 10 # 16 -> 12 Gb; 12 -> 10 Gb

nb_devices = torch.cuda.device_count()
print(f"nb_devices : {nb_devices}")

def print_gpu_utilization(nb_devices):
    pynvml.nvmlInit()
    for i in range(nb_devices):
      handle = pynvml.nvmlDeviceGetHandleByIndex(i)
      info = pynvml.nvmlDeviceGetMemoryInfo(handle)
      print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result, nb_devices):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization(nb_devices=nb_devices)

# load dataset
train_ds, val_ds, test_ds = load_dataset(
    path='allocine', cache_dir = "datasets/sentiment_analysis",
    split=['train', 'validation', 'test']
)

# Load an encoder and a classifier instance
# You can replace "camembert-base" with any other model from the table, e.g. "camembert/camembert-large".
tokenizer = CamembertTokenizer.from_pretrained("camembert/camembert-base-wikipedia-4gb", cache_dir = "model/camembert_token")
model = CamembertForSequenceClassification.from_pretrained("Jerome-Dej/camembert_classif", cache_dir = "model/camembert_classif_rte")


def tokenize_function(examples, max_length=512):
  """tokenize in pytorch function.
  """
  return tokenizer(examples["review"], max_length=512, truncation=True, padding=True, return_tensors="pt")

tokenized_train_x = train_ds.map(tokenize_function, batched=True)
tokenized_val_x   =   val_ds.map(tokenize_function, batched=True)
tokenized_test_x  =  test_ds.map(tokenize_function, batched=True)


accuracy = evaluate.load("accuracy")

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


training_args = TrainingArguments(
    output_dir="model/camembert_classif_rte",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size, 
    per_device_eval_batch_size=batch_size,
    num_train_epochs=nb_epochs,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=True,
    disable_tqdm=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_x,
    eval_dataset=tokenized_val_x,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

result = trainer.train()
print_summary(result=result, nb_devices=nb_devices)

trainer.push_to_hub("tp_llm")

model_dir = os.path.join(ROOT_DIR, "model/camembert_classif_manu")
if not os.path.isdir(model_dir):
  os.makedirs(model_dir, exist_ok=True)
trainer.save_model (model_dir)



