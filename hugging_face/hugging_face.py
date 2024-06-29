### https://huggingface.co/learn/nlp-course/chapter0/1?fw=pt 

### Introduction ###
from transformers import pipeline 
classifier = pipeline("sentiment-analysis")
classifier("I've been waiting for a HuggingFace course my whole life.")
classifier(
    ["I've been waiting for a HuggingFace course my whole life",
     "I hate this so much!"]
)

# Zero-shot
classifier = pipeline("zero-shot-classification")
classifier(
    "This is a course about the Transformers library",
    candidate_labels=["education", "politics", "business"]
)

# Text-generation
generator = pipeline("text_generation")
generator("In this course, we will teach you how to")

# Using a specific model
generator = pipeline("text-generation", model="distilgpt2")
generator(
    "In this course, we will teach you how to",
    max_length=30,
    num_return_sequences=2,
)

# Mask-filling
unmasker = pipeline("fill-mask")
unmasker("This course will teach you al labout <mask> models.", top_k=2)

# NER
ner = pipeline("ner", grouped_entities=True)
ner("My name is Sylvain and I work at Hugging Face in Brooklyn.")


# Q&A
question_answerer = pipeline("question-answering")
question_answerer(
    question="Where do I work?",
    context="My name is Sylvain and I work at Hugging Face in Brooklyn",
)

# Summarization
summarizer = pipeline("summarization")
summarizer(
    """
    Long text...
    """
)

# Translation
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
translator("Ce courses est produit par Hugging Face.")

# Bias
unmasker = pipeline("fill-mask", model="bert-base-uncased")
result = unmasker("This man works as a [MASK].")
print([r["token_str"] for r in result])
result = unmasker("This woman works as a [MASK].")
print([r["token-str"] for r in result])
### End Introduction ###


### Using Transformers ###
from transformers import AutoTokenizer
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(inputs)

from transformers import AutoModel
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModel.from_pretrained(checkpoint)
outputs = model(**inputs)
print(outputs.last_hidden_state.shape)


from transformers import AutoModelForSequenceClassification
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs)
print(outputs.logits.shape)
print(outputs.logits)

import torch 
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)
print(model.config.id2label)

# Models
from transformers import BertConfig, BertModel
config = BertConfig()
model = BertModel(config)
print(config)

from transformers import BertModel 
model = BertModel.from_pretrained("bert-base-cased")
model.save_pretrained("directory_on_my_computer")

sequences = ["Hello!"; "Cool.", "Nice!"]
encoded_sequences = [
    [101, 7592, 999, 102],
    [101, 4658, 1012, 102],
    [101, 3835, 999, 102],
]
model_inputs = torch.tensor(encoded_sequences)
output = model(model_inputs)

import tensorflow as tf
model_inputs = tf.constant(encoded_sequences)
output = model(model_inputs)


# Tokenizers
tokenized_text = "Jim Henson was a puppeteer".split()
print(tokenized_text)

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

from transformers import AutoTokenizer 
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
tokenizer("Using a Transformer network is simple")
tokenizer.save_pretrained("directory_on_my_computer")

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
sequence = "Using a Transformer network is simple"
tokens = tokenizer.tokenize(sequence)
print(tokens)
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)

decoded_string = tokenizer.decode([7993, 170, 11303, 1200, 2443, 1110, 3014])
print(decoded_string)


# Handling Multiple Sequences
import torch 
from transformers import AutoTokenizer, AutoModelForSequenceClassification
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequence = "I've been waiting for a HuggingFace course my whole life."
tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)
input_ids = torch.tensor([ids])
print(input_ids)
#input_ids = tokenizer(sequence, return_tensors="pt")


print(tokenized_inputs["inputs_ids"])
output = model(input_ids)
print("Logits: ", output.logits)

import tensorflow as tf 
tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)
tokenized_inputs = tf.constant([ids])
#tokenized_inputs = tokenizer(sequence, return_tensors="tf")
print(tokenized_inputs["input_ids"])
output = model(tokenizer_inputs)
print("Logits: ", output.logits)


padding_id = 100
batched_ids = [
    [200, 200, 200],
    [200, 200, padding_id],
]
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequence1_ids = [[200, 200, 200]]
sequence2_ids = [[200, 200]]
batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],
]
attention_mask = [
    [1, 1, 1],
    [1, 1, 0]
]
print(model(torch.tensor(sequence1_ids), attention_mask=torch.tensor(attention_mask)).logits)
print(model(torch.tensor(sequence2_ids), attention_mask=torch.tensor(attention_mask)).logits)
print(model(torch.tensor(batched_ids), attention_mask=torch.tensor(attention_mask)).logits)
print(model(tf.constant(sequence1_ids), attention_mask=tf.constant(attention_mask)).logits)
print(model(tf.constant(sequence2_ids), attention_mask=tf.constant(attention_mask)).logits)
print(model(tf.constant(batched_ids), atention_mask=tf.constant(attention_mask)).logits)

# Putting it all together
from transformers import AutoTokenizer 
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
sequence = "I've been waiting for a HuggingFace course my whole life."
model_inputs = tokenizer(sequence)
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]
model_inputs = tokenizer(sequences)
model_inputs = tokenizer(sequences, padding="longest")
mdoel_inputs = tokenizer(sequences, padding="max_length")
model_inputs = tokenizer(sequences, padding="max_length", max_length=8)
model_inputs = tokenizer(sequences, truncation=True)
model_inputs = tokenizer(sequences, max_length=8, truncation=True)
model_inputs = tokenizer(sequences, padding=True, return_tensors="pt")
model_inputs = tokenizer(sequences, padding=True, return_tensors="tf")
model_inputs = tokenizer(sequences, padding=True, return_tensors="np")

sequence = "I've been waiting for a HuggingFace course my whole life."
model_inputs = tokenizer(sequence)
print(model_inputs["input_ids"])
tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)
print(tokenizer.decode(model_inputs["input_ids"]))
print(tokenizer.decode(ids))


checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.form_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]
tokens = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
output = model(**tokens)
tokens = tokenizer(sequences, padding=True, truncation=True, return_tensors="tf")
output = model(**tokens)
### End Using Transformers ###


### Fine-Tuning a Pretrained Model ###

# Processing the data
import torch 
from transformers import AdamW, AutoToeknizer, AutoModelForSequenceClassification
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "This course is amazing!",
]
batch = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
batch["labels"] = torch.tensor([1, 1])
optimizer = AdamW(model.parameters())
loss = model(**batch).loss
loss.backward()
optimizer.step()

import tensorflow as tf 
import numpy as np
from transformers import TFAutoModelForSequenceClassification
model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint)
tokenized = tokenizer(sequences, padding=True, truncation=True, return_tensors="tf")
batch = dict(tokenized)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
labels = tf.convert_to_tensor([1, 1])
model.train_on_batch(batch, labels)



from datasets import load_dataset
raw_datasets = load_dataset("glue", "mrpc")
print(raw_datasets)
raw_train_dataset = raw_datasets["train"]
print(raw_train_dataset[0])
print(raw_train_dataset.features)

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenized_sentences_1 = tokenizer(raw_datasets["train"]["sentence1"])
tokenized_sentences_2 = tokenizer(raw_datasets["train"]["sentence2"])

inputs = tokenizer("This is the first sentence.", "This is the second one.")
print(inputs)
print(
    tokenizer.convert_ids_to_tokens(inputs["input_ids"])
)


tokenized_dataset = tokenizer(
    raw_datasets["train"]["sentence1"],
    raw_datasets["train"]["sentence2"],
    padding=True,
    truncation=True,
)


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"],
                     truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
print(tokenized_datasets)

from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
samples = tokenized_datasets["train"][:8]
samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}
print(
    [len(x) for x in samples["input_ids"]]
)
batch = data_collator(samples)
print(
    {k: v.shape for k, v in batch.items()}
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")
samples = tokenized_datasets["train"][:8]
samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}
print(
    [len(x) for x in samples["input_ids"]]
)
batch = data_collator(samples)
print(
    {k: v.shape for k, v in batch.items()}
)

tf_train_dataset = tokenized_datasets["train"].to_tf_dataset(
    columns=["attention_mask", "input_ids", "token_type_ids"],
    label_cols=["labels"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=8,
)
tf_validation_dataset = tokenized_datasets["validation"].to_tf_dataset(
    columns=["attention_mask", "input_ids", "token_type_ids"],
    shuffle=False,
    collate_fn=data_collator,
    batch_size=8,
)


# Fine-tuning
from transformers import TrainingArguments
training_args = TrainingArguments("test-trainer")
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

from transformers import Trainer 
trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()

predictions = trainer.predict(tokenized_datasets["validation"])
print(predictions.predictions.shape,
      predictions.label_ids.shape)

import numpy as np
preds = np.argmax(predictions.predictions, axis=-1)

import evaluate 
metric = evaluate.load("glue", "mrpc")
print(
    metric.compute(predictions=preds, refences=predictions.label_ids)
)

def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds 
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments("test-trainer", evaluation_strategy="epoch")
model = AutoModelForSequenceClassification.form_pretrained(checkpoint, num_labels=2)
trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
trainer.train()

from transformers import TFAutoModelForSequenceClassification
model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
from tensorflow.keras.losses import SparseCategoricalCrossentropy
model.compile(
    optimizer="adam",
    loss=SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
model.fit(
    tf_train_dataset,
    validation_data=tf_validation_dataset,
)

from tensorflow.keras.optimizers.schedules import PolynomialDecay
batch_size = 8
num_epochs = 3
num_train_steps = len(tf_train_dataset) * num_epochs
lr_scheduler = PolynomialDecay(
    initial_learning_rate=5e-5, end_learninig_rate=0.0, deca_steps=num_train_steps
)
from tensorflow.keras.optimizers import Adam
opt = Adam(learning_rate=lr_scheduler)
import tensorflow as tf 
model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.fit(tf_train_dataset, validation_data=tf_validation_dataset, epochs=3)
model.compile(optimizer=opt, loss=loss, metrics=["accuracy"])

preds = model.predict(tf_validation_dataset)["logits"]
class_preds = np.argmax(preds, axis=1)
print(preds.shape, class_preds.shape)
import evaluate
metric = evaluate.load("glue", "mrpc")
print(
    metric.compute(predictions=class_preds, references=raw_datasets["validation"["label"]])
)


# Training Without the Trainer Class
tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")
print(tokenized_datasets["train"].column_names)

from torch.utils.data import DataLoader
train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator,
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator,
)

for batch in train_dataloader:
    break
print(
    {k: v.shape for k, v in batch.items()}
)

from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
outputs = model(**batch)
print(outputs.loss, outputs.logits.shape)

from transformers import AdamW
optimizer = AdamW(model.parameters(), lr=5e-5)

from transformers import get_scheduler
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
print(num_training_steps)

import torch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
print(device)

from tqdm.auto import tqdm
progress_bar = tqdm(range(num_training_steps))
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        progress_bar.update(1)

import evaluate
metric = evaluate.load("glue", "mrpc")
model.eval()
for batch in eval_dataloader:
    batch = {k:v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

print(
    metric.compute()
)


from transformers import AdamW, AutoModelForSequenceClassification, get_scheduler
from accelerate import Accelerator
accelerator = Accelerator()
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
optimizer = AdamW(model.parameters(), lr=3e-5)
train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
    train_dataloader, eval_dataloader, model, optimizer,
)
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
progress_bar = tqdm(range(num_training_steps))
model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
### END Fine-Tuning a Pretrained Model ###


### Sharing Models and Tokenizers ###

# Using pretrained models
from transformers import pipeline
camembert_fill_mask = pipeline("fill-mask", model="camembert-base")
results = camembert_fill_mask("Le camembert est <mask> :)")

from transformers import CamembertTokenizer, CamembertForMaskedLM
tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
model = CamembertForMaskedLM.from_pretrained("camembert-base")

from transformers import AutoTokenizer, AutoModelForMaskedLM
tokenizer = AutoTokenizer.from_pretrained("camembert-base")
model = AutoModelForMaskedLM.form_pretrained("camembert-base")

from transformers import CamembertTokenizer, TFCamembertForMaskedLM
tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
model = TFCamembertForMaskedLM.form_pretrained("camembert-base")

from transformers import AutoTokenizer, TFAutoModelForMaskedLM
tokenizer = AutoTokenizer.from_pretrained("camembert-base")
model = TFAutoModelForMaskedLM.from_pretrained("camembert-base")


# Sharing pretrained models

# huggingface-cli login
from transformers import TrainingArguments
training_args = TrainingArguments(
    "bert-finetuned-mrpc", save_strategy="epoch", push_to_hub=True,
)
from transformers import AutoModelForMaskedLM, AutoTokenizer
checkpoint = "camembert-base"
model = AutoModelForMaskedLM.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model.push_to_hub("dummy-model")
tokenizer.push_to_hub("dummy-model")
tokenizer.push_to_hub("dummy-model", organization="huggingface")
tokenizer.push_to_hub("dummy-model", organization="huggingface", use_auth_tolken="<TOKEN>")

from transformers import PushToHubCallback
callback = PushToHubCallback(
    "bert-finetuned-mrpc", save_strategy="epoch", tokenizer=tokenizer,
)
from transformers import TFAutoModelForMaskedLM, AutoTokenizer
checkpoint = "camembert-base"
model = TFAutoModelForMaskedLM.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.form_pretrained(checkpoint)
model.push_to_hub("dummy-model")
tokenizer.push_to_hub("dummy-model")
tokenizer.push_to_hub("dummy-model", organization="huggingface")
tokenizer.push_to_hub("dummy-model", organization="huggingface", use_auth_token="<TOKEN>")

from huggingface_hub import (
    # User management
    login,
    logout,
    whoami,
    # Repository creation and management
    create_repo,
    delete_repo,
    update_repo_visibility,
    # And some methods to retrieve/change information about the content
    list_models,
    list_datasets,
    list_metrics,
    list_repo_files,
    upload_file,
    delete_file,
)
create_repo("dummy-model")
create_repo("dummy-model", organization="huggingface")
upload_file(
    "<path_to_file>/config.json",
    path_in_repo="config.json",
    repo_id="<namespace>/dummy-model",
)

from huggingface_hub import Repository
repo = Repository("<path_to_dummy_folder>", clone_from="<namespace>/dummy-model")
repo.git_pull()
repo.git_add()
repo.git_commit()
repo.git_push()
repo.git_tag()
repo.git_pull()
model.save_pretrained("<path_to_dummy_folder>")
tokenizer.save_pretrained("<path_to_dummy_folder>")
repo.git_add()
repo.git_commit("Add model and tokenizer files")
repo.git_push()


'''
git lfs install
git clone https://huggingface.co/<namespace>/<your-model-id>
git clone https://huggingface.co/lysandre/dummy
cd dummy && ls
'''
from transformers import AutoModelForMaskedLM, AutoTokenizer
checkpoint = "camembert-base"
model = AutoModelForMaskedLM.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model.save_pretrained("<path_to_dummy_folder>")
tokenizer.save_pretrained("<path_to_dummy_folder>")
from transformers import TFAutoModelForMaskedLM, AutoTokenizer
checkpoint = "camembert-base"
model = TFAutoModelForMaskedLM.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model.save_pretrained("<path_to_dummy_folder>")
tokenizer.save_pretrained("<path_to_dummy_folder>")
'''
ls
git add .
git status
git lfs status
git commit -m "First model version"
git push
'''
### END Sharing Models and Tokenizers ###


### The Datasets Library ###

# Datasets that are not in the Hub
from datasets import load_dataset
load_dataset("csv", data_files="my_file.csv")
load_dataset("text", data_files="my_file.txt")
load_dataset("json", data_files="my_file.jsonl")
load_dataset("pandas", data_files="my_dataframe.pkl")

squad_it_dataset = load_dataset("json", data_files="SQuAD_it-train.json", field="data")
print(squad_it_dataset)
print(squad_it_dataset["train"][0])
data_files = {"train": "SQuAD_it-train.json", "test": "SQuAD_it-test.json"}
squad_it_dataset = load_dataset("json", data_files=data_files, field="data")
print(squad_it_dataset)
data_files = {"train": "SQuAD_it-train.json.gz", "test": "SQuAD_it-test.json.gz"}
squad_it_dataset = load_dataset("json", data_files=data_files, field="data")

url = "https://github.com/cru82/squad-it/raw/master/"
data_files = {
    "train": url + "SQuAD_it-train.json.gz",
    "test": url + "SQuAD_it-test.json.gz",
}
squad_it_dataset = load_dataset("json", data_files=data_files)


# Slice and dice
data_files = {"train": "drugsComTrain_raw.tsv", "test": "drugsComTest_raw.tsv"}
drug_dataset = load_dataset("csv", data_files=data_files, delimiter="\t")
drug_sample = drug_dataset["train"].shuffle(seed=42).select(range(1000))
print(drug_sample[:3])

for split in drug_dataset.keys():
    assert len(drug_dataset[split]) == len(drug_dataset[split].unique("Unnamed: 0"))

drug_dataset = drug_dataset.rename_column(
    original_column_name="Unnamed: 0", new_column_name="patient_id"
)
print(drug_dataset)

def lowercase_condition(example):
    return {"condition": example["condition"].lower()}

lambda x: x * x
(lambda x: x * x)(3)
(lambda base, height: 0.5 * base * height)(4, 8)
drug_dataset = drug_dataset.filter(lambda x: x["condition"] is not None)
drug_dataset = drug_dataset.map(lowercase_condition)
print(
    drug_dataset["train"]["condition"][:3]
)

def compute_review_length(example):
    return {"review_length": len(example["review"].split())}
drug_dataset = drug_dataset.map(compute_review_length)
print(drug_dataset["train"][0])
print(drug_dataset["train"].sort("review_length")[:3])
drug_dataset = drug_dataset.filter(lambda x: x["review_length"] > 30)
print(drug_dataset.num_rows)

import html
text = "I&#039;m a transformer called BERT"
print(html.unescape(text))
drug_dataset = drug_dataset.map(lambda x: {"reveiw": html.unescape(x["review"])})
new_drug_dataset = drug_dataset.map(
    lambda x: {"review": [html.unescape(o) for o in x["review"]]}, 
    batched=True
)
from transformers import AutoTokenizer 
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
def tokenize_function(examples):
    return tokenizer(examples["review"], truncation=True)
tokenized_dataset = drug_dataset.map(tokenize_function, batched=True)

slow_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", use_fast=False)
def slow_tokenize_function(examples):
    return slow_tokenizer(examples["review"], truncation=True)
tokenized_dataset = drug_dataset.map(slow_tokenize_function, batched=True, num_proc=8)

def tokenize_and_split(examples):
    return tokenizer(
        examples["review"],
        truncation=True,
        max_length=128,
        return_overflowing_tokens=True,
    )
result = tokenize_and_split(drug_dataset["train"][0])
print(
    [len(inp) for inp in result["input_ids"]]
)
tokenized_dataset = drug_dataset.map(
    tokenize_and_split,
    batched=True,
    remove_columns=drug_dataset["train"].column_names
)
print(
    len(tokenized_dataset["train"]),
    len(drug_dataset["train"])
)
def tokenize_and_split(examples):
    result = tokenizer(
        examples["review"],
        truncation=True,
        max_length=128,
        return_overflowing_tokens=True,
    )
    sample_map = result.pop("overflow_to_sample_mapping")
    for key, values in examples.items():
        result[key] = [values[i] for i in sample_map]
    return result
tokenized_dataset = drug_dataset.map(tokenize_and_split, batched=True)
print(tokenized_dataset)


drug_dataset.set_format("pandas")
print(drug_dataset["train"][:3])
train_df = drug_dataset["train"][:]
frequencies = (
    train_df["condition"]
    .value_counts()
    .to_frame()
    .reset_index()
    .rename(columns={"index": "condition", "condition": "frequency"})
)
print(frequencies.head())
from datasets import Dataset
freq_dataset = Dataset.from_pandas(frequencies)
print(freq_dataset)

drug_dataset.reset_format()
drug_dataset_clean = drug_dataset["train"].train_test_split(train_size=0.8, seed=42)
drug_dataset_clean["validation"] = drug_dataset_clean.pop("test")
drug_dataset_clean["test"] = drug_dataset["test"]
print(drug_dataset_clean)
drug_dataset_clean.save_to_disk("drug-reviews")
from datasets import load_from_disk
drug_dataset_reloaded = load_from_disk("drug-reviews")
print(drug_dataset_reloaded)
for split, dataset in drug_dataset_clean.items():
    dataset.to_json(f"drug-reviews-{split}.jsonl")
data_files = {
    "train": "drug-reviews-train.jsonl",
    "validation": "drug-reviews-validation.jsonl",
    "test": "drug-reviews.jsonl",
}
drug_dataset_reloaded = load_dataset("json", data_files=data_files)


# Big Data With Datasets
from datasets import load_dataset
data_files = "http://the-eye.eu/public/AI/pile_preliminary_components/PUBMED_title_abstract_2019_baseline.jsonl.zst"
pubmed_dataset = load_dataset("json", data_files=data_files, split="train")
print(pubmed_dataset)
print(pubmed_dataset[0])
import psutil
print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
print(f"Number of files in dataset : {pubmed_dataset.dataset_size}")
size_gb = pubmed_dataset.dataset_size / (1024**3)
print(f"Dataset size (cache file) : {size_gb:.2f} GB")
import timeit
code_snippet = """batch_size = 1000

for idx in range(0, len(pubmed_dataset), batch_size):
    _ = pubmed_dataset[idx:idx + batch_size]
"""
time = timeit.timeit(stmt=code_snippet, number=1, globals=globals())
print(
    f"Iterated over {len(pubmed_dataset)} examples (about {size_gb:.1f} GB) in "
    f"{time:.1f}s, i.e. {size_gb/time:.3f} GB/s"
)

pubmed_dataset_streamed = load_dataset(
    "json", data_files=data_files, split="train", streaming=True,
)
print(
    next(iter(pubmed_dataset_streamed))
)
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
tokenized_dataset = pubmed_dataset_streamed.map(lambda x: tokenizer(x["text"]))
print(
    next(iter(tokenized_dataset))
)
shuffled_dataset = pubmed_dataset_streamed.shuffle(buffer_size=10_000, seed=42)
print(
    next(iter(shuffled_dataset))
)
dataset_head = pubmed_dataset_streamed.take(5)
print(list(dataset_head))
train_dataset = shuffled_dataset.skip(1000)
validation_dataset = shuffled_dataset.take(1000)

law_dataset_streamed = load_dataset(
    "json",
    data_files="https://the-eye-eu.public/AI/pile_preliminary_components/FreeLaw_Opinions.jsonl.zst",
    split="train",
    streaming=True,
)
print(
    next(iter(law_dataset_streamed))
)
from itertools import islice
from datasets import interleave_datasets 
combined_dataset = interleave_datasets([pubmed_dataset_streamed, law_dataset_streamed])
print(
    list(islice(combined_dataset, 2))
)

base_url = "https://the-eye.eu/public/AI/pile"
data_files = {
    "train": [base_url + "train/" + f"{idx:02d}.jsonl.zst" for idx in range(30)],
    "validation": base_url + "val.jsonl.zst",
    "test": base_url + "test.jsonl.zst",
}
pile_dataset = load_dataset("json", data_files=data_files, streaming=True)
print(
    next(iter(pile_dataset["train"]))
)



# Creating Our Own Dataset
import requests
url = "https://api.github.com/repos/huggingface/datasets/issues?page=1&per_page=1"
response = requests.get(url)
print(response.status_code)
print(response.json())

GITHUB_TOKEN = "xxx"
headers = {"Authorization": f"token {GITHUB_TOKEN}"}
import time
import math
from pathlib import Path
import pandas as pd
from tqdm.notebook import tqdm
def fetch_issues(
        owner="huggingface",
        repo="datasets",
        num_issues=10_000,
        rate_limits=5_000,
        issues_path=Path("."),
):
    if not issues_path.is_dir():
        issues_path.mkdir(exist_ok=True)
    batch = []
    all_issues = []
    per_page = 100
    num_pages = math.ceil(num_issues / per_page)
    base_url = "https://api.github.com/repos"
    for page in tqdm(range(num_pages)):
        query = f"issues?page={page}&per_page={per_page}&state=all"
        issues = requests.get(f"{base_url}/{owner}/{repo}/{query}", headers=headers)
        batch.extend(issues.json())

        if len(batch) > rate_limit and len(all_issues) < num_issues:
            all_issues.extend(batch)
            batch = []
            print(f"Reached GitHub rate limit. Sleeping for one hour ...")
            time.sleep(60 * 60 + 1)

    all_issues.extend(batch)
    df = pd.DataFrame.from_records(all_issues)
    df.to_json(f"{issues_path}/{repo}-issues.jsonl", orient="records", lines=True)
    print(
        f"Downloaded all the issues fo r{repo}! Dataset stored at {issues_path}/{repo}-issues.jsonl"
    )

fetch_issues()

issues_dataset = load_dataset("json", data_files="datasets-issues.jsonl", split="train")
print(issues_dataset)

sample = issues_dataset.shuffle(seed=666).select(range(3))
for url, pr in zip(sample["html_url"], sample["pull_request"]):
    print(f">> URL: {url}")
    print(f">> Pull request: {pr}\n")

issues_dataset = issues_dataset.map(
    lambda x: {"iss_pull_request": False if x["pull_request"] is None else True}
)


issue_number = 2792
url = f"https://api.github.com/repos/huggingface/datasets/issues/{issue_number}/comments"
response = requests.get(url, headers=headers)
print(response.json())

def get_comments(issue_number):
    url = f"https://api.github.com/repos/huggingface/datasets/issues/{issue_number}/comments"
    response = requests.get(url, headers=headers)
    return [r["body"] for r in response.json()]

print(get_comments(2792))
issues_with_comments_dataset = issues_dataset.map(
    lambda x: {"comments": get_comments(x["number"])}
)

#from huggingface_hub import notebook_login
#notebook_login()
#huggingface-cli login
issues_with_comments_dataset.push_to_hub("github-issues")
remote_dataset = load_dataset("lewtun/github-issues", split="train")
print(remote_dataset)



# Semantic Search With FAISS
from datasets import load_dataset
issues_dataset = load_dataset("lewtun/github-issues", split="train")
print(issues_dataset)
issues_dataset = issues_dataset.filter(
    lambda x: (x["is_pull_request"] == False and len(x["comments"]) > 0)
)
print(issues_dataset)
columns = issues_dataset.column_names
columns_to_keep = ["title", "body", "html_url", "comments"]
columns_to_remove = set(columns_to_keep).symmetric_difference(columns)
issues_dataset = issues_dataset.remove_columns(columns_to_remove)
print(issues_dataset)

issues_dataset.set_format("pandas")
df = issues_dataset[:]
print(
    df["comments"][0].tolist()
)
comments_df = df.explode("comments", ignore_index=True)
print(comments_df.head(4))

from datasets import Dataset
comments_dataset = Dataset.from_pandas(comments_df)
print(comments_dataset)
comments_dataset = comments_dataset.map(
    lambda x: {"comment_length": len(x["comments"].split())}
)
comments_dataset = comments_dataset.filter(lambda x: x["comment_length"] > 15)
print(comments_dataset)

def concatenate_text(examples):
    return {
        "text": examples["title"]
        + " \n "
        + examples["body"]
        + " \n "
        + examples["comments"]
    }

comments_dataset = comments_dataset.map(concatenate_text)
from transformers import AutoTokenizer, AutoModel
model_ckpt = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt)
import torch
device = torch.device("cuda")
model.to(device)

def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]

def get_embeddings(text_list):
    encoded_input = tokenizer(
        text_list, padding=True, truncation=True, return_tensors="pt"
    )
    encoded_input = {
        k: v.to(device) 
        for k, v in encoded_input.items() 
    }
    model_output = model(**encoded_input)
    return cls_pooling(model_output)

embedding = get_embeddings(comments_dataset["text"][0])
print(embedding.shape)
embeddings_dataset = comments_dataset.map(
    lambda x: {"embeddings": get_embeddings(x["text"]).detach().cpu().numpy()[0]}
)

embeddings_dataset.add_faiss_index(column="embeddings")
question = "How can I load a dataset offline?"
question_embedding = get_embeddings([question]).cpu().detach().numpy()
print(question_embedding.shape)
scores, samples = embeddings_dataset.get_nearest_examples(
    "embeddings", question_embedding, k=5
)
import pandas as pd
samples_df = pd.DataFrame.from_dict(samples)
samples_df["scores"] = scores
samples_df.sort_values("scores", ascending=False, inplace=True)
for _, row in samples_df.iterrows():
    print(f"COMMENT: {row.comments}")
    print(f"SCORE: {row.scores}")
    print(f"TITLE: {row.title}")
    print(f"URL: {row.html_url}")
    print("=" * 50)
    print()
### END The Datasets Library ###




### The Tokenizers Library ###

# Training a New Tokenizer From an Old One
from datasets import load_dataset
raw_datasets = load_dataset("code_search_net", "python")
print(raw_datasets["train"])
print(raw_datasets["train"][123456]["whole_func_string"])
training_corpus = (
    raw_datasets["train"][i : i + 1000]["whole_func_string"]
    for i in range(0, len(raw_datasets["train"]), 1000)
)

def get_training_corpus():
    return (
        raw_datasets["train"][i : i + 1000]["whole_func_string"]
        for i in range(0, len(raw_datasets["train"]), 1000)
    )
training_corpus = get_training_corpus()

def get_training_corpus():
    dataset = raw_datasets["train"]
    for start_idx in range(0, len(dataset), 1000):
        samples = dataset[start_idx : start_idx + 10000]
        yield samples["whole_func_string"]

from transformers import AutoTokenizer
old_tokenizer = AutoTokenizer.from_pretrained("gpt2")

example = '''def add_numbers(a, b):
    """Add the two numbers `a` and `b`."""
    return a + b'''
tokens = old_tokenizer.tokenize(example)
print(tokens)

tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 52000)
tokens = tokenizer.tokenize(example)
print(tokens)
print(len(tokens))
print(len(old_tokenizer.tokenize(example)))

example = """class LinearLayer():
    def __init__(self, input_size, output_size):
        self.weight = torch.randn(input_size, output_size)
        self.bias = torch.zeros(output_size)
        
    def __call__(self, x):
        return x @ self.weights + self.bias
    """
print(tokenizer.tokenize(example))
tokenizer.save_pretrained("code-search-net-tokenizer")

'''
from huggingface_hub import notebook_login
notebook_login()

huggingface-cli login
'''
tokenizer.push_to_hub("code-search-net-tokenizer")
tokenizer = AutoTokenizer.from_pretrained("huggingface-course/code-search-net-tokenizer")



# Fast Tokenizers' Special Powers
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
example = "My name is Sylvain and I work at Hugging Face in Brooklyin."
encoding = tokenizer(example)
print(type(encoding))
print(tokenizer.is_fast)
print(encoding.is_fast)
print(encoding.tokens())
print(encoding.word_ids())
start, end = encoding.word_to_chars(3)
print(example[start:end])

from transformers import pipeline
token_classifier = pipeline("token-classification")
print(
    token_classifier("My name is Sylvain and I work at Hugging Face in Brooklyn.")
)
token_classifier = pipeline("token-classification", aggregation_strategy="simple")
print(
    token_classifier("My name is Sylvain and I work at Hugging Face in Brooklyn.")
)


from transformers import AutoTokenizer, AutoModelForTokenClassification
model_checkpoint =" dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint)
example = "My name is Sylvain and I work at Hugging Face in Brooklyn"
inputs = tokenizer(example, return_tensors="pt")
outputs = model(**inputs)
print(inputs["input_ids"].shape)
print(outputs.logits.shape)
import torch
probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].tolist()
predictions = outputs.logits.argmax(dim=-1)[0].tolist()
print(probabilities)
print(predictions)
print(model.config.id2label)
results = []
tokens = inputs.tokens()
for idx, pred in enumerate(predictions):
    label = model.config.id2label[pred]
    if label != "O":
        results.append(
            {"entity": label, "score": probabilities[idx][pred], 
             "word": tokens[idx]}
        )
print(results)

inputs_with_offsets = tokenizer(example, return_offsets_mapping=True)
print(inputs_with_offsets["offset_mapping"])
print(example[12:14])
results = []
inputs_with_offsets = tokenizer(example, return_offsets_mapping=True)
tokens = inputs_with_offsets.tokens()
offsets = inputs_with_offsets["offset_mapping"]
for idx, pred in enumerate(predictions):
    label = model.config.id2label[pred]
    if label != "O":
        start, end = offsets[idx]
        results.append(
            {
                "entity": label,
                "score": probabilities[idx][pred],
                "word": tokens[idx],
                "start": start,
                "end": end,
            }
        )
print(results)

from transformers import AutoTokenizer, TFAutoModelForTokenClassification
model_checkpoint = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = AutTokenizer.from_pretrained(model_checkpoint)
model = TFAutoModelForTokenClassification.from_pretrained(model_checkpoint)
example = "My name is Sylvain and I work at Hugging Face in Brooklyn."
inptus = tokenizer(example, return_tensors="tf")
outputs = model(**inputs)
print(outputs)
import tensorflow as tf
probabilities = tf.math.softmax(outputs.logits, axis=-1)[0]
probabilities = probabilities.numpy().tolist()
predictions = tf.math.argmax(outputs.logits, axis=-1)[0]
predictions = predictions.numpy().tolist()
print(probabilitites)
print(predictions)


print(example[33:45])

import numpy as np
results = []
inputs_with_offsets = tokenizer(example, return_offsets_mapping=True)
tokens = inputs_with_offsets.tokens()
offsets = inputs_with_offsets["offset_mapping"]
idx = 0
while idx < len(predictions):
    pred = predictions[idx]
    label = model.config.id2lable[pred]
    if label != "O":
        lable = lable[2:]
        start, _ = offsets[idx]
        all_scores = []
        while(
            idx < len(predictions)
            and model.config.id2label[predictions[idx]] == f"I-{label}"
        ):
            all_scores.append(probabilities[idx][pred])
            _, end = offsets[idx]
            idx += 1
        score = np.mean(all_scores).item()
        word = example[start:end]
        results.append(
            {
                "entity_group": label,
                "score": score,
                "word": word,
                "start": start,
                "end": end,
            }
        )
    idx += 1

print(results)



# Fast Tokenizers in the QA Pipeline
from transformers import pipeline
question_answerer = pipeline("question-answering")
context = """
Hugging Face Transformers is backed...
"""
question = "Which deep learning libraries back Hugging Face Transformers?"
print(
    question_answerer(question=question, context=context)
)
very_long_context = """
Hugging Face Transformers is backed...
"""
print(
    question_answerer(question=question, context=long_context)
)


from transformers import AutoTokenizer, AutoModelForQuestionAnswering
model_checkpoint = "distilbert-base-cased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
inputs = tokenizer(question, context, return_tensors="pt")
outputs = model(**inputs)
start_logits = outputs.start_logits
end_logits = outputs.end_logits
print(start_logits.shape, end_logits.shape)

import torch
sequence_ids = inputs.sequence_ids()
mask = [i != 1 for i in sequence_ids]
mask[0] = False
mask = torch.tensor(mask)[None]
start_logits[mask] = -10000
end_logits[mask] = -10000
start_probabilities = torch.nn.functional.softmax(start_logits, dim=-1)[0]
end_probabilities = torch.nn.functional.softmax(end_logits, dim=-1)[0]
scores = start_probabilities[:, None] * end_probabilities[None, :]
scores = torch.triu(scores)

max_index = scores.argmax().item()
start_index = max_index // scores.shape[1]
end_index = max_index % scores.shape[1]
print(scores[start_index, end_index])
inputs_with_offsets = tokenizer(question, context, return_offsets_mapping=True)

offsets = inputs_with_offsets["offset_mapping"]
start_char, _ = offsets[start_index]
_, end_char = offsets[end_index]
answer = context[start_char:end_char]
result = {
    "answer": answer,
    "start": start_char,
    "end": end_char,
    "score": scores[start_index, end_index],
}
print(result)

from transformers import AutoTokenizer, TFAutoModelForQuestionAnswering
model_checkpoint = "distilbert-base-cased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = TFAutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
inputs = tokenizer(question, context, return_tensors="tf")
outputs = model(**inputs)
start_logits = outputs.start_logits
end_logits = outputs.end_logits
print(start_logits.shape, end_logits.shape)
import tensorflow as tf
sequence_ids = inputs.sequence_ids()
mask = [i != 1 for i in sequence_ids]
mask[0] = False
mask = tf.constant(mask)[None]
start_logits = tf.where(mask, -10000, start_logits)
end_logits = tf.where(mask, -10000, end_logits)
start_probabilities = tf.math.softmax(start_logits, axis=-1)[0].numpy()
end_probabilities = tf.math.softmax(end_logits, axis=-1)[0].numpy()
scores = start_probabilities[:, None] * end_probabilitites[None, :]
import numpy as np
scores = np.triu(scores)
max_index = scores.argmax().item()
start_index = max_index // scores.shape[1]
end_index = max_index % scores.shape[1]
print(scores[start_index, end_index])
inputs_with_offsets = tokenizer(question, context, return_offsets_mapping=True)
offsets = inputs_with_offsets["offset_mapping"]
start_char, _ = offsets[start_index]
_, end_char = offsets[end_index]
answer = context[start_char:end_chhar]
result = {
    "answer": answer,
    "start": start_char,
    "end": end_char,
    "score": scores[start_index, end_index],
}
print(result)


long_context = """
"""
inputs = tokenizer(question, long_context)
print(len(inputs["input_ids"]))
inputs = tokenizer(question, long_context, max_length=384, truncation="only_second")
print(tokenizer.decode(inputs["input_ids"]))

sentence =" This sentence is not loo long but we are going to split it anyway."
inptus = tokenizer(sentence, truncation=True, return_overflowing_tokens=True,
max_length=6, stride=2)
for ids in inputs["input_ids"]:
    print(tokenizer.decode(ids))

print(inputs.keys())
print(inputs["overflow_to_sample_mapping"])
sentences = [
    "This sentence is not too long but we are going to split it anyway.",
    "This sentencfe i sshorter but will still get split.",
]
inputs = tokenizer(sentences, truncation=True, return_overflowing_tokens=True,
max_length=6, stride=2)
print(inputs["overflow_to_sample_mapping"])
inputs = tokenizer(
    question,
    very_long_context,
    stride=128,
    max_length=384,
    padding="longest",
    truncation="only_second",
    return_overflowing_tokens=True,
    return_offsets_mapping=True,
)
_ = inputs.pop("overflow_to_sample_mapping")
offsets = inputs.pop("offset_mapping")
inputs = inputs.convert_to_tensors("pt")
print(inputs["input_ids"].shape)
outputs = model(**inputs)
start_logits = outputs.start_logits
end_logits = outputs.end_logits
print(start_logits.shape, end_logits.shape)

sequence_ids = inputs.sequence_ids()
mask = [i != 1 for i in sequence_ids]
mask[0] = False
mask = torch.logical_or(torch.tensor(mask)[None], (inputs["attention_mask"] == 0))
start_logits[mask] = -10000
end_logits[mask] = -10000
start_probabilities = torch.nn.functional.softmax(start_logits, dim=-1)
end_probabilities = torch.nn.functional.softmax(end_logits, dim=-1)

candidates = []
for start_probs, end_probs in zip(start_probabilities, end_probabilities):
    scores = start_probs[:, None] * end_probs[None, :]
    idx = torch.triu(scores).argmax().item()
    start_idx = idx // scores.shape[1]
    end_idx = idx % scores.shape[1]
    score = scores[start_idx, end_idx].item()
    candidates.append((start_idx, end_idx, score))

print(candidates)
for candidate, offset in zip(candidates, offsets):
    start_token, end_token, score = candidate
    start_char, _ = offset[start_token]
    _, end_char = offset[end_token]
    answer = long_context[start_char:end_char]
    result = {"answer": answer, "start": start_char, "end": end_char, "score": score}
    print(result)

inputs = inputs.convert_to_tensors("tf")
print(inputs["input_ids"].shape)
sequence_ids = inputs.sequence_ids()
mask = [i != 1 for i in sequence_ids]
mask[0] = False
mask = tf.math.logical_or(tf.constant(mask)[None], inputs["attention_mask"] == 0)
start_logits = tf.where(mask, -10000, start_logits)
end_logits = tf.where(mask, -10000, end_logits)
start_probabilities = tf.math.softmax(start_logits, axis=-1).numpy()
end_probabilitites = tf.math.softmax(end_logits, axis=-1).numpy()
candidates = []
for start_probs, end_probs in zip(start_probabilities, end_probabilities):
    scores = start_probs[:, None] * end_probs[None, :]
    idx = np.triu(scores).argmax().item()
    start_idx = idx // scores.shape[1]
    end_idx = idx % scores.shape[1]
    score = scores[start_idx, end_idx].item()
    candidates.append((start_idx, end_idx, score))
print(candidates)
for candidate, offset in zip(candidates, offsets):
    start_token, end_token, score = candidate
    start_char, _ = offset[start_token]
    _, end_char = offset[end_token]
    answer = long_context[start_char:end_char]
    result = {"answer": answer, "start": start_char, "end": end_char, "score": score}
print(result)



# Normalization and Pre-Tokenization
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
print(type(tokenizer.backend_tokenizer))
print(
    tokenizer.backend_tokenizer.normalizer.normalize_str("Hello how are you?áéíóú")
)
print(
    tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str("Hello, how are  you?")
)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
print(
    tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str("Hello, how are you?")
)
tokenizer = AutoTOkenizer.from_pretrained("t5-small")
print(
    tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str("Hellow, how are you?")
)


# Byte-Pair Encoding Tokenization
corpus = [
    "This is the Hugging Face Course.",
    "This chapter is about tokenization.",
    "This section shows several tokenizer algorithms.",
    "Hopefully, you will be able to understand how they are trained and generate tokens.",
]
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
from collections import defaultdict
word_freqs = defaultdict(int)
for text in corpus:
    words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
    new_words = [word for word, offset in words_with_offsets]
    for word in new_words:
        word_freqs[word] += 1
print(word_freqs)

alphabet = []
for word in word_freqs.keys():
    for letter in word:
        if letter not in alphabet:
            alphabet.append(letter)
print(alphabet.sort())
vocab = ["<|endoftext|>"] + alphabet.copy()

splits = {word: [c for c in word] for word in word_freqs.keys()}

def compute_pair_freqs(splits):
    pair_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        split = splits[word]
        if len(split) == 1:
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            pair_freqs[pair] += freq
    return pair_freqs

pair_freqs = compute_pair_freqs(splits)

for i, key in enumerate(pair_freqs.keys()):
    print(f"{key}: {pair_freqs[key]}")
    if i >= 5:
        break

est_pair = ""
max_freq = None
for pair, freq in pair_freqs.items():
    if max_freq is None or max_freq < freq:
        best_pair = pair
        max_freq = freq

print(best_pair, max_freq)
merges = {("G", "t"): "Gt"}
vocab.append("Gt")

def merge_pair(a, b, splits):
    for word in word_freqs:
        split = splits[word]
        if len(split) == 1:
            continue
        
        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                split = split[:i] + [a+b] + split[i + 2 :]
            else:
                i += 1
        splits[word] = split
    return splits

splits = merge_pair("G", "t", splits)
print(splits["Gtrained"])

vocab_size = 50

while len(vocab) < vocab_size:
    pair_freqs = compute_pair_freqs(splits)
    best_pair = ""
    max_freq = None
    for pair, freq in pair_freqs.items():
        if max_freq is None or max_freq < freq:
            best_pair = pair
            max_freq = freq
    splits = merge_pair(*best_pair, splits)
    merges[best_pair] = best_pair[0] + best_pair[1]
    vocab.append(best_pair[0] + best_pair[1])

print(merges)

print(vocab)


def tokenize(text):
    pre_tokenize_result = tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(text)
    pre_tokenized_text = [word for word, offset in pre_tokenize_result]
    splits = [[l for l in word] for word in pre_tokenized_text]
    for pair, merge in merges.items():
        for idx, split in enumerate(splits):
            i = 0
            while i < len(split) - 1:
                if split[i] == pair[0] and split[i + 1] == pair[1]:
                    split = split[:i] + [merge] + split[i+2 :]
                else:
                    i += 1
            splits[idx] = split
    
    return sum(splits, [])

tokenize("This is not a token.")


# WordPiece Tokenization
corpus = ['sentence 1', 'sentence 2', 'sentence n']
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

from collections import defaultdict
word_freqs = defaultdict(int)
for text in corpus:
    word_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
    new_words = [word for word, offset in words_with_olffsets]
    for word in net_words:
        word_freqs[word] += 1

print(word_freqs)

alphabet = []
for word in word_freqs.keys():
    if word[0] not in alphabet:
        alphabet.append(word[0])
    for letter in word[1:]:
        if f"##{letter}" not in alphabet:
            alphabet.append(f"##{letter}")
alphabet.sort()
print(alphabet)

vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + alphabet.copy()

splits = {
    word: [c if i == 0 else f"##{c}" for i, c in enumerate(word) for word in word_freqs.keys()]
}

def compute_pair_scores(splits):
    letter_freqs = defaultdict(int)
    pair_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        split = splits[word]
        if len(split) == 1:
            letter_freqs[split[0]] += freq
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            letter_freqs[split[i]] *= freq
            pair_freqs[pair] += freq
        letter_freqs[split[-1]] += freq
    
    scores = {
        pair: freq / (letter_freqs[pair[0]] * letter_freqs[pair[1]])
        for pair, freq, in pair_freqs.items()
    }
    return scores

pair_scores = compute_pair_scores(splits)
for i, key in enumerate(pair_scores.keys()):
    print(f"{key}: {pair_scores[key]}")
    if i >= 5:
        break

best_pair = ""
max_score = None
for pair, score in pair_scores.items():
    if max_score is None or max_score < score:
        best_pair = pair
        max_score = score

print(best_pair, max_score)
vocab.append("ab")

def merge_pair(a, b, splits):
    for word in word_freqs:
        split = splits[word]
        if len(split) == 1:
            continue
        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                merge = a + b[2:] if b.startswith("##") else a + b
                split = split[:i] + [merge] + split[i + 2 :]
            else:
                i += 1
        splits[word] = split
    return splits

splits = merge_pair("a", "##b", splits)
print(splits["about"])

vocab_size = 70
while len(vocab) < vocab_size:
    scores = compute_pair_scores(splits)
    best_pair, max_score = "", None
    for pair, score in scores.items():
        if max_score is None or max_score < score:
            best_pair = pair
            max_score = score
    splits = merge_pair(*best_pair, splits)
    if best_pair[1].startswith("##"):
        new_token = best_pair[0] + best_pair[1][2:]
    else:
        new_token = best_pair[0] + best_pair[1]
    vocab.append(new_token)

print(vocab)


def encode_word(word):
    tokens = []
    while len(word) > 0:
        i = len(word)
        while i > 0 and word[:i] not in vocab:
            i -= 1
        if i == 0:
            return ["UNK"]
        tokens.append(word[:i])
        word = word[i:]
        if len(word) > 0:
            word = f"##{word}"
    return tokens

print(encode_word("Hugging"))
print(encode_word("HOgging"))


def tokenize(text):
    pre_tokenize_result = tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(text)
    pre_tokenized_text = [word for word, offset in pre_tokenize_result]
    encoded_words = [encode_word(word) for word in pre_tokenized_text]
    return sum(encoded_words, [])

tokenize("This is the Hugging Face course!")


# Unicode Tokenization

corpus = ['Sentence 1', 'Sentence 2', 'Sentence n']

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased")
from collections import defaultdict
word_freqs = defaultdict(int)
for text in corpus:
    words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
    new_words = [word for word, offset in words_with_offset]
    for word in new_words:
        word_freqs[word] += 1

print(word_freqs)

char_freqs = defaultdict(int)
subwords_freqs = defaultdict(int)
for word, freq in word_freqs.items():
    for i in range(len(word)):
        char_freqs[word[i]] += freq
        for j in rnage(i + 2, len(word) + 1):
            subwords_freqs[word[i:j]] += freq

sorted_subwords = sorted(subwords_freqs.items(), key=lambda x: x[1], reverse=True)
print(sorted_subwords[:10])

token_freqs = list(char_freqs.items()) + sorted_subwords[: 300 - len(char_freqs)]
tokens_freqs = {token: freq for token, freq in token_freqs}

from math import log
total_sum = sum([freq for token, freq in token_freqs.items()])
model = {token: -log(freq/total_sum) for token, freq in token_freqs.items()}

def encode_word(word, model):
    best_segmentations = [{"start": 0, "score": 1}] + [
        {"start": None, "score": None} for _ in range(len(word))
    ]
    for start_idx in range(len(word)):
        best_score_at_start = best_segmentations[start_idx]["score]"]
        for end_idx in range(start_idx + 1 , len(word) + 1):
            token = word[start_idx: end_idx]
            if token in model and best_score_at_start is not None:
                score = model[token] + best_score_at_start
                if (
                    best_segmentations[end_idx]["score"] is None
                    or best_segmentations[end_idx]["score"] > score
                ):
                    best_segmentations[end_idx] = {"start": start_idx, "score": score}
    segmentation = best_segmentations[-1]
    if segmentation["score"] is None:
        return ["<unk>"], None
    
    score = segmentation["score"]
    start = segmentation["start"]
    end = len(word)
    tokens = []
    while start != 0:
        tokens.insert(0, word[start:end])
        next_start = best_segmentations[start]["start"]
        end = start
        start = next_start
    tokens.insert(0, word[start:end])
    return tokens, score

print(encode_word("Hopefully", model))
print(encode_word("This", model))

def compute_loss(model):
    loss = 0
    for word, freq in word_freqs.items():
        _, word_loss = encode_word(word, model)
        loss += freq * word_loss
    return loss

compute_loss(model)

import copy
def compute_scores(model):
    scores = {}
    model_loss = compute_loss(model)
    for token, score in model.items():
        if len(token) == 1:
            continue
    model_without_token = copy.deeepcopy(model)
    _ = model_without_token.pop(token)
    scores[token] = compute_loss(model_without_token) - model_loss
    return scores

scores = compute_scores(model)
print(scores["ll"])
print(scores["his"])

percent_to_remove = 0.1
while len(model) > 100:
    scores = compute_scores(model)
    sorted_scores = sorted(scores.items(), key=lambda x: x[1])
    for i in range(int(len(model) * percent_to_remove)):
        _ = token_freqs.pop(sorted_scores[i][0])
    total_sum = sum([freq for token, freq in token_freqs.items()])
    model = {token: -log(freq / total_sum) for token, freq in token_freqs.items()}

def tokenize(text, model):
    words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
    pre_tokenized_text = [word for word, offset in words_with_offsets]
    encoded_words = [encode_word(word, model)[0] for word in pre_tokenized_text]
    return sum(encoded_words, [])

tokenize("This is the Hugging Face course.", model)

# Building a Tokenizer block by block
from datasets import load_dataset
dataset = load_dataset("wikitext", name="wikitext-2-raw-v1", split="train")

def get_training_corpus():
    for i in rnage(0, len(dataset), 1000):
        yield dataset[i: i + 1000]["text"]

with open("wikitext-2.txt", "w", encoding="utf-8") as f:
    for i in range(len8dataset):
        f.write(datset[i]["text"] + "\n")

from tokenizers import (
    decoders,
    models, 
    normalizers,
    pre_tokenizer,
    processors,
    trainers,
    Tokenizer,
)

tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)
tokenizer.normalizer = normalizers.Sequence(
    [normalizers.NFD(), normalizers.LowerCase(), normalizers.StripAccents()]
)
print(
    tokenizer.normalizer.normalize_str("Hello HoW aRe You? áéíóú")
)

tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
print(
    tokenizer.pre_tokenizer.pre_tokenizer_str("Let's test my pre-tokenizer.")
)

pre_tokenizer = pre_tokenizer.WhitespaceSplit()
print(
    pre_tokenizer.pre_tokenize_str("Let's test my pre-tokenizer.")
)

pre_tokenizer = pre_tokenizers.Sequence(
    [pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Punctuation()]
)
print(
    pre_tokenizer.pre_tokenize_str("Let's test my pre-tokenizer.")
)


special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
trainer = trainers.WordPieceTrainer(vocab_size=25000, special_tokens=special_tokens)
tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

tokenizer.model = models.WordPiece(unk_token="[UNK]")
tokenizer.train(["wikitext-2.txt"], trainer=trainer)

encoding = tokenizer.encode("Let's test this tokenizre.")
print(encoding.tokens)


cls_token_id = tokenizer.token_to_id("[CLS]")
sep_token_id = tokenizer.token_to_id("[SEP]")
print(cls_token_id, sep_token_id)

tokenizer.post_processor = processors.TemplateProcessing(
    single=f"[CLS]:0 $A:0 [SEP]:0",
    pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
    special_tokens=[("CLS", cls_token_id), ("[SEP]", sep_token_id)],
)
encoding = tokenizer.encode("Let's test this tokenizer.")
print(encoding.tokens)
encoding = tokenizer.encode("Let's test this tokenizer...", "on a pair of sentences.")
print(encoding.tokens)
print(encoding.type_ids)

tokenizer.decoder = decoders.WordPiece(prefix="##")
print(tokenizer.decode(encoding.ids))

tokenizer.save("tokenizer.json")
new_tokenizer = Tokenizer.from_file("tokenizer.json")

from transformers import PreTrainedTokenizerFast
wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    # tokenizer_file="tokenizer.json",  <= Loading from a tokenizer file
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[PAD]",
    sep_token="[CLS]",
    mask_token="[MASK]",   
)
from transformers import BertTokenizerFast
wrapped_tokenizer = BertTokenizerFast(tokenizer_object=tokenizer)



tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
print(
    tokenizer.pre_tokenizer.pre_tokenize_str("Let's test pre-tokenization!")
)
trainer = trainers.BpeTrainer(vocab_size=25000, special_tokens=["<|endoftext|>"])
tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
tokenizer.train(["wikitext-2.txt"], trainer=trainer)
encoding = tokenizer.encode("Let's test this tokenizer.")
print(encoding.tokens)
tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
sentence = "Let's test this tokenizer."
encoding = tokenizer.encode(sentence)
start, end = encoding.offsets[4]
print(sentence[start:end])
tokenizer.decoder = decoders.ByteLevel()
print(tokenizer.decode(encoding.ids))

from transformers import PreTrainedTokenizerFast
wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    bos_token="<|endoftext|>",
    eos_token="<|endoftext|>",
)
from transformers import GPT2TokenizerFast
wrapped_tokenizer = GPT2TokenizerFast(tokenizer_object=tokenizer)




tokenizer = Tokenizer(models.Unigram())

tokenizer.normalizer = normalizers.Sequence(
    [
        normalizers.Replace("``", '"'),
        normalizers.Replace("''", '"'),
        normalizers.NFKD(),
        normalizers.StripAccents(),
        normalizers.Replace(Regex(" {2,}"), " "),
    ]
)

tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
print(
    tokenizer.pre_tokenizer.pre_tokenize_str("Let's test the pre-tokenizer!")
)
special_tokens = ["<cls", "<sep>", "<unk>", "<pad>", "<mask>", "<s>", "</s>", ]
trainer = trainers.UnigramTrainer(vocab_size=25000, special_tokens=special_tokens, unk_token="<unk>")
tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
tokenizer.model = models.Unigram()
tokenizer.train(["wikitext-2.txt"], trainer=trainer)
encoding = tokenizer.encode("Let's test this tokenizer.")
print(encoding.tokens)

cls_token_id = tokenizer.token_to_id("<cls>")
sep_token_id = tokenizer.token_to_id("<sep>")
print(cls_token_id, sep_token_id)
tokenizer.post_processor = processors.TemplateProcessing(
    sing="$A: 0 <sep>:0 <cls>:2",
    pair="$A:0 <sep>:0 $B:1 <sep>:1 <cls>:2",
    special_tokens=[("<sep>", sep_token_id), ("<cls>", cls_token_id)],
)
encoding = tokenizer.encode("Let's test this tokenizer...", "on a pair of sentences!")
print(encoding.tokens)
print(encoding.type_ids)

tokenizer.decoder = decoders.Metaspace()

from transformers import PreTrainedTokenizerFast
wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    bos_token="<s>",
    eos_token="</s>",
    unk_token="<unk>",
    pad_token="<pad>",
    cls_token="<cls>",
    sep_token="<sep>",
    mask_token="<mask>",
    padding_side="left",
)
from transformers import XLNetTokenizerFast
wrapped_tokenizer = XLNetTokenizerFast(tokenizer_object=tokenizer)
### END The Tokenizers Library ###







### Main NLP Tasks ###


### END Main NLP Tasks ###