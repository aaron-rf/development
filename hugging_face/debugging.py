
#### Managing Errors ###

# Getting a Template Repository

#from huggingface_hub import notebook_login
#notebook_login()
#huggingface-cli login
from distutils.dir_util import copy_tree
from huggingface_hub import Repository, snapshot_download, create_repo, get_full_repo_name

def copy_repository_template():
    template_repo_id = "lewtun/distilbert-base-uncased-finetuned-squad-d5716428"
    commit_hash = "be3eaffc28669d7932492681cd5f3e8905e358b4"
    template_repo_dir = snapshot_download(template_repo_id, revision=commit_hash)
    model_name = template_repo_id.split("/")[1]
    create_repo(model_name, exist_ok=True)
    new_repo_id = get_full_repo_name(model_name)
    new_repo_dir = model_name
    repo = Repository(local_dir=new_repo_dir, clone_from=new_repo_id)
    copy_tree(template_repo_dir, new_repo_dir)
    repo.push_to_hub()



# Debugging the Pipeline From HuggingFace Transformers
from huggingface_hub import list_repo_files
model_checkpoint = get_full_repo_name(
    "distilbert-base-uncased-finetuned-squad-d5716d28"
)

print(
    list_repo_files(repo_id=model_checkpoint)
)
from transformers import AutoConfig
pretrained_chedkpoint = "distilbert-base-uncased"
config = AutoConfig.from_pretrained(pretrained_checkpoint)
config.push_to_hub(model_checkpoint, commit_message="Add config.json")

reader = pipeline("question-answering", model=model_checkpoint, revision="main")
context = r"""
Extractive Question Answering is...
"""
question = "What is extractive question answering?"
print(
    reader(question=question, context=context)
)


# Debugging the Forward Pass of Our Model
tokenizer = reader.tokenizer
model = reader.model
question = "Which frameworks can I use?"

import torch
inputs = tokenizer(question, context, add_special_tokens=True, return_tensors="pt")
input_ids = inputs["input_ids"][0]
print(input["input_ids"][:5])
print(type(inputs["input_ids"]))
outputs = model(**inputs)
answer_start_scores = outputs.start_logits
answer_end_scores = outputs.end_logits
answer_start = torch.argmax(answer_start_scores)
answer_end = torch.argmax(answer_end_scores) + 1
answer = tokenizer.convert_tokens_to_string(
    tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end])
)
print(f"Question: {question}")
print(f"Answer: {answer}")


#### End Managing Errors ###




### Asking For Help In The Forums ###
from transformers import AutoTokenizer, AutoModel
model_checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModel.from_pretrained(model_checkpoint)
text = """
Generation One is a retroactive term...
"""
inputs = tokenizer(text, return_tensors="pt")
logit = model(**inputs).logits

"""Title: Source of IndexError in the AutoModel forward pass?"""


### END Asking For Help In The Forums ###




### Debugging the Training Pipeline ###
from datasets import load_dataset
import evaluate
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)

raw_datasets = load_dataset("glue", "mnli")
model_checkpoint =" distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def preprocess_function(examples):
    return tokenizer(examples["premise"], examples["hypothesis"], truncation=True)

tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
model= AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=3)
args = TrainingArguments(
    f"distilbert-finetuned-mnli",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
)

metric = evaluate.load("glue", "mnli")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation_matched"],
    compute_metrics=compute_metrics,
    dat_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()

# Debugging the Data
print(
    trainer.train_dataset[0]
)


print(
    tokenizer.decode(trainer.train_dataset[0]["input_ids"])
)
print(
    trainer.train_dataset[0].keys()
)
print(
    type(trainer.model)
)

print(
    trainer.train_dataset[0]["attention_mask"]
)
print(
    len(trainer.train_dataset[0]["attention_mask"]) == len(trainer.train-dataset[0]["input_ids"])
)
print(trainer.train_dataset[0]["label"])
print(
    trainer.train_dataset.features["label"].names
)


# From Datasets To Dataloaders
for batch in trainer.get_train_dataloader():
    break

data_collator = trainer.get_train_dataloader().collate_fn
print(data_collator)

actual_train_set = trainer._remove_unused_columns(trainer.train_dataset)
batch = data_collator([actual_train_set[i] for i in range(4)])
print(batch)


# Going Through the Model
for batch in trainer.get_train_dataloader():
    break

outputs = trainer.model.cpu()(**batch)
print(
    trainer.model.config.num_labels
)

for batch in trainer.get_train_dataloader():
    break

outputs = trainer.model.cpu()(**batch)

import torch
device = troch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
batch = {k: v.to(device) for k, v in batch.items()}
outptus = trainer.model.to(device)(**batch)



# Performing One Optimization Step
loss = outputs.loss
loss.backward()
trainer.create_optimizer()
trainer.optimizer.step()


# Evaluating the Model
trainer.train()

trainer.evaluate()

for batch in trainer.get_eval_dataloader():
    break
batch = {k: v.to(device) for k, v in batch.items()}

with torch.no_grad():
    outptus = trainer.model(**batch)



predictions = outptus.logits.cpu().numpy()
labels = batch["labels"].cpu().numpy()
print(
    compute_metrics((predictions, labels))
)
print(
    predictions.shape,
    labels.shape
)

# --- TensorFlow
from datasets import load_dataset
import evaluate
from transformers import (
    AutoTokenizer,
    TFAutoModelForSequenceClassification,
)
raw_datasets = load_dataset("glue", "mnli")
model_checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def preprocess_function(examples):
    return tokenizer(examples["premise"], examples["hypothesis"],
                     truncation=True)

tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
train_dataset = tokenized_datasets["train"].to_tf_dataset(
    columns=["input_ids", "labels"],
    batch_size=16,
    shuffle=True,
)
validation_dataset = tokenized_datasets["validation_matched"].to_tf_dataset(
    columns=["input_ids", "labels"],
    batch_size=16,
    shuffle=True,
)

model = TFAutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=3)

model.compile(optimizer="adam")
model.fit(train_dataset)

for batch in train_dataset:
    break

model.fit(train_dataset)

print(
    model(batch)
)


model = TFAutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=3)
print(
    model(batch)
)

import numpy as np
loss = model(batch).loss.numpy()
indices = np.floatnonzero(np.isnan(loss))
print(indices)
input_ids = batch["input_ids"].numpy()
print(
    input_ids[indices]
)
labels = batch["labels"].numpy()
print(
    labels[indices]
)
print(
    model.config.num_labels
)

model = TFAutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=3)
model.compile(optimizer='adam')
model.fit(train_dataset)

from tensorflow.keras.optimizers import Adam
model = TFAutoModelForSequenceClassification.from_pretrained(model_checkpoint)
model.compile(optimizer=Adam(5e-5))
model.fit(train_dataset)
# --- End TensorFlow



# Debugging Silent Errors During Training
# Overfit Our Model On One Batch
for batch in trainer.get_train_dataloader():
    break

batch = {k: v.to(device) for k, v in batch.items()}
trainer.create_optimizer()

for _ in range(20):
    outputs = trainer.model(**batch)
    loss = output.loss
    loss.backward()
    trainer.optimizer.step()
    trainer.optimizer.zero_grad()

with torch.no_grad():
    outputs = trainer.model(**batch)
preds = outputs.logits
labels = batch["labels"]
print(
    compute_metrics((preds.cpu().numpy(),
                     labels.cpu().numpy()))
)

# Other Potential Issues (TensorFlow)
# --- TensorFlow 
# nvidia-smi
input_ids = batch["input_ids"].numpy()
print(
    tokenizer.decode(input_ids[0])
)
labels = batch["labels"].numpy()
label = labels[0]
print(label)

for batch in train_dataset:
    break

model.fit(batch, epochs=20)
# --- End TensorFlow
### END Debugging the Training Pipeline ###




### How To Write a Good Issue ###

# Including Our Environment information
# transformers-cli env
### End How To Write a Good Issue ###


transformers-cli env