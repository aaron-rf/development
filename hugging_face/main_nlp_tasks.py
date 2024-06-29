

### Token Classification ###
# Preparing the data
from datasets import load_dataset
raw_datasets = load_dataset("conll2003")
print(raw_datasets)
print(raw_datasets["train"][0]["tokens"])
print(raw_datasets["train"][0]["ner_tags"])
ner_feature = raw_datasets["train"].features["ner_tags"]
print(ner_feature)
label_names = ner_features.feature.names
print(label_names)

words = raw_datasets["train"][0]["tokens"]
labels = raw_datasets["train"][0]["ner_tags"]
line1 = ""
line2 = ""
for word, label in zip(words, labels):
    full_label = label_names[label]
    max_length = max(len(word), len(full_label))
    line1 += word + " " * (max_length - len(word) + 1)
    line2 += full_label + " " * (max_length - len(full_label) + 1)

print(line1)
print(line2)


from transformers import AutoTokenizer
model_checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
print(tokenizer.is_fast)
inputs = tokenizer(raw_datasets["train"][0]["tokens"], is_split_into_words=True)
print(inputs.tokens)
print(inputs.word_ids())

def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            new_labels.append(-100)
        else:
            label = labels[word_id]
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels

labels = raw_datasets["train"][0]["ner_tags"]
word_ids = inputs.word_ids()
print(labels)
print(align_labels_with_tokens(labels, word_ids))

def tokenize_and_algin_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    all_labels = examples["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))
    
    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs

tokenized_datasets = raw_datasets.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=raw_datasets["train"].column_names,
)


from transformers import DataCollatorForTokenClassification
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
batch = data_collator([tokenized_datasets["train"][i] for i in range(2)])
print(batch["labels"])
for i in range(2):
    print(tokenized_datasets["train"][i]["labels"])

#pip install seqeval
import evaluate
metric = evaluate.load("seqeval")

labels = raw_datasets["train"][0]["ner_tags"]
labels = [label_names[i] for i in labels]
print(labels)
predictions = labels.copy()
predictions[2] = "O"
print(
    metric.compute(predictions=[predictions], references=[labels])
)

import numpy as np
def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    true_labels = [[label_names[l] for l in label if l != -100]
                   for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }

id2label = {i: label for i, label in enumerate(label_names)}
label2id = {v: k for k, v in id2label.items()}
from transformers import AutoModelForTokenClassification
model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    id2lable=id2label,
    label2id=label2id,
)
print(model.config.num_labels)

# from huggingface_hum import notebook_login
# notebook_login
# huggingface-cli login
from transformers import TrainingArguments
args = TrainingARguments(
    "bert-finetuned-ner",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=True,
)

from transformers import Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)
trainer.train()
trainer.push_to_hub(commit_message="Training complete")


# TensorFlow
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, return_tensors="tf") 
tf_train_dataset = tokenized_datasets["train"].to_tf_dataset(
    columns=["attention_mask", "input_ids", "labels", "token_type_ids"],
    shuffle=True,
    batch_size=16,
)
tf_eval_dataset = tokenized_datasets["validation"].to_tf_dataset(
    columns=["attention_mask", "input_ids", "labels", "token_type_ids"],
    collate_fn=data_collator,
    shuffle=False,
    batch_size=16,
)

from transformers import TFAutoModelForTokenClassification
model = TFAutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    id2label=id2label,
    label2id=label2id,
)
print(model.config.num_labels)
from transformers import create_optimizer
import tensorflow as tf
tf.keras.mixed_precision.set_global_policy("mixed_float16")
num_epochs = 3
num_train_steps = len(tf_train_dataset) * num_epochs
optimzier, schedule = create_optimizer(
    init_lr=2e-5,
    num_warmup_steps=0,
    num_train_steps=num_train_steps,
    weight_decay_rate=0.01,
)
model.compile(optimizer=optimizer)
from transformers.keras_callbacks import PushToHubCallback
callback = PushToHubCallback(output_dir="bert-finetuned-ner", tokenizer=tokenizer)
model.fit(tf_train_dataset, validation_data=tf_eval_dataset, 
          callbacks=[callback], epochs=num_epochs)

import evaluate
metric = evaluate.load("seqeval")
predictions = labels.copy()
import numpy as np
all_predictions = []
all_labels = []
for batch in tf_eval_dataset:
    logits = model.predict_on_batch(batch)["logits"]
    labels = batch["labels"]
    predictions = np.argmax(logits, axis=-1)
    for prediction, lable in zip(predictions, labels):
        for predicted_idx, label_idx in zip(prediction, label):
            if label_idx == -100:
                continue
            all_predictions.append(label_names[predicted_idx])
            all_labels.append(label_names[label_idx])

metric.compute(predictions=[all_predictions], references=[all_labels])
# END TensorFlow


# A Custom Training Loop
from torch.utils.data import DataLoader
train_dataloader = DataLoader(
    tokenized_datasets["train"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=8,
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"],
    collate_fn=data_collator,
    batch_size=8,
)

model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    id2lable=id2label,
    label2id=label2id,
)

from torch.optim import AdamW
optimizer = AdamW(model.parameters(), lr=2e-5)

from accelerate import Accelelrator
accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)

from transformers import get_scheduler
num_train_epochs = 3
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

from huggingface_hub import Repository, get_full_repo_name
mmodel_name = "bert-finetuned-ner-accelerate"
repo_name = get_full_repo_name(model_name)
print(repo_name)
output_dir = "bert-finetuned-ner-accelerate"
repo = Repository(output_dir, clone_from=repo_name)

def postprocess(predictions, labels):
    predictions = predictions.detach().cpu().clone().numpy()
    labels = labels.detach().cpu().clone().numpy()
    true_labels = [[label_names[l] for l in label if l != -100]
                   for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    return true_labels, true_predictions


from tqdm.auto import tqdm
import torch
progress_bar = tqdm(range(num_training_steps))
for epoch in range(num_train_epochs):
    model.train()
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss()
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    model.eval()
    for batch in eval_dataloader:
        with torch.no_grad():
            outputs = model(**batch)

        predictions = outputs.logits.argmax(dim=-1)
        labels = batch["labels"]
        predictions = accelerator.pad_across_processes(predictions, dim=1, pax_index=-100)
        lables = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
        predictions_gathered = accelerator.gather(predictions)
        lables_gathered = accelerator.gather(labels)
        true_predictions, true_labels = postprocess(predictions_gathered, labels_gathered)
        metric.add_batch(predictions=true_predictions, references=true_labels)
    
    results = metrics.compute()
    print(
        f"epoch {epoch}:",
        {
            keys: results[f"overall_{key}"]
            for key in ["precision", "recall", "f1", "accuracy"]
        },
    )

    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)
        repo.push_to_hub(
            commit_message=f"Training in progress epoch {epoch}", blocking=False
        )


from transformers import pipeline
mdoel_checkpoint = "huggingface-course/bert-finetuned-ner"
token_classifier = pipeline(
    "token-classification", modeL=model_checkpoint,
    aggregation_strategy="simple",
)
print(
    token_classifier("My name is Sylvain and I work at Hugging Face in Brooklyn.")
)
### End Token Classification ###



### Fine-Tuning a Masked Language Model ###
from transformers import AutoModelForMaskedLM
model_checkpoint = "distilibert-base-uncased"
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
distilbert_num_parameters = model.num_parameters() / 1_000_000
print(f"'>>> DistilBET number of paramters: {round(distilbert_num_parameters)}M'")
print(f"'>>> BERT number of parameters: 110M'")

text = "This is a great [MASK]"
from transformers import AutoTokenizer
tokenizer = AutoTOkenizer.from_pretrained(model_checkpoint)
import torch
inputs = tokenizer(text, return_tensors="pt")
token_logits = model(**inputs).logits
mask_token_index = torch.where(inptus["input_ids"] == tokenizer.mask_token_id)[1]
mask_token_logits = token_logits[0, mask_token_index, :]
top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
for token in top_5_tokens:
    print(f"'>>> {text.replace(tokenizer.mask_token, tokenizer.decode([token]))}'")

# TensorFlow
from transformers import TFAutoModelForMaskedLM
model_checkpoint = "distilibert-base-uncased"
model = TFAutoModelForMaskedLM.from_pretrained(model_checkpoint)
print(model.summary)
text = "This is a great [MASK]"
from transformers import AutoTokenizer
tokenizer = AutoTOkenizer.from_pretrained(model_checkpoint)
import numpy
import tensorflow as tf
inputs = tokenizer(text, return_tensors="np")
token_lgotis = model(**inputs).logits
mask_token_index = np.argwhere(inputs["input_ids"] == tokenizer.mask_token_id)[0, 1]
mask_token_logits = token_logits[0, mask_token_index, :]
top_5_tokens = np.argsort(-mask_token_logits)[:5].tolist()
for token in top_5_tokens:
    print(f">>> {text.replace(tokenizer.mask_token, tokenizer.decode([token]))}")
# --- End TensorFlow ---


# Loading the Data
from datasets import load_dataset
imdb_dataset = load_dataset("imdb")
print(imdb_dataset)
sample = imdb_dataset["train"].shuffle(seed=42).select(range(3))
for row in sample:
    print(f"\n'>>> Review: {row['text']}'")
    print(f"'>>> Label: {row['label']}'")


# Preprocessing the Data
def tokenize_function(examples):
    result = tokenizer(examples["text"])
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result

tokenized_dataset = imdb_dataset.map(
    tokenize_function, batched=True, remove_columns=["text", "label"]
)
print(tokenized_datasets)
print(tokenizer.model_max_length)
chunk_size = 128

tokenized_samples = tokenized_datasets["train"][:3]
for idx, sample in enumerate(tokenized_samples["input_ids"]):
    print(f"'>>> Review {idx} length: {len(sample)}'")
concatenated_examples = {
    k: sum(tokenized_samples[k], []) for k in tokenized_samples.keys()
}
total_length = len(concatenated_examples["input_ids"])
print(f"'>>> Concatenated reviews length: {total_length}'")
chunks = {
    k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
    for k, t in concatenated_examples.items()
}
for chunk in chunks["input_ids"]:
    print(f"'>>> Chunk length: {len(chunk)}'")

def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // chunk_size) * chunk_size
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

lm_datasets = tokenized_datasets.map(group_texts, batched=True)
print(lm_datasets)
print(tokenizer.decode(lm_datasets["train"][1]["input_ids"]))
print(tokenizer.decode(lm_datasets["train"][1]["labels"]))

# Fine-Tuning DistilBERT with the Trainer API
from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
samples = [lm_datasets["train"][i] for i in range(2)]
for sample in samples:
    _ = sample.pop("word_ids")
for chunk in data_collator(samples)["input_ids"]:
    print(f"\n'>>> {tokenizer.decode(chunk)}'")

import collections
import numpy as np
from transformers import default_data_collator
wwm_probability = 0.2

def whole_word_masking_data_collator(features):
    for feature in features:
        word_ids = feature.pop("word_ids")
        mapping = collections.defaultdict(list)
        current_word_index = -1
        current_word = None
        for idx, word_id in enumerate(word_ids):
            if word_id is not None:
                if word_id != current_word:
                    current_word = word_id
                    current_word_index += 1
                mapping[current_word_index].append(idx)
        mask = np.random.binomial(1, wwm_probability, (len(mapping), ))
        input_ids = feature["input_ids"]
        labels = feature["labels"]
        new_labels = [-100] * len(labels)
        for word_id in np.where(mask)[0]:
            word_id = word_id.item()
            for idx in mapping[word_id]:
                new_labels[idx] = labels[idx]
                input_ids[idx] = tokenizer.mask_token_id
        feature["labels"] = new_labels

    # TensorFlow
    # from transformers.data.data_collator import tf_default_data_collator
    # return tf_default_data_collator(features)
    # --- End TensorFlow
    return default_data_collator(features)

samples = [lm_datasets["train"][i] for i in range(2)]
batch = whole_word_masking_data_collator(samples)
for chunk in batch["input_ids"]:
    print(f"\n'>>> {tokenizer.decode(chunk)}")

train_size = 10_000
test_size = int(0.1 * train_size)
downsampled_dataset = lm_datasets["train"].train_test_split(
    train_size=train_size, test_size=test_size, seed=42
)
print(downsampled_dataset)

# from huggingface_hub import notebook_login
# notebook_login()
# huggingface-cli login
from transformers import TrainingArguments
batch_size = 64
loggint_steps = len(downsampled_dataset["train"]) // batch_size
model_name = model_checkpoint.split("/")[-1]
training_args = TrainingArguments(
    output_dir=f"{model_name}-finetuned-imdb",
    overwrite_outputdir=True,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    push_to_hub=True,
    fp16=True,
    logging_steps=logging_steps,
)
from transformers import Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=downsampled_dataset["train"],
    eval_dataset=downsampled_dataset["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# TensorFlow
tf_train_dataset = model.prepare_tf_dataset(
    downsampled_dataset["train"],
    collate_fn=data_collator,
    shuffle=True,
    batch_size=32,
)
tf_eval_dataset = model.prepare_tf_dataset(
    downsampled_dataset["test"],
    collate_fn=data_collator,
    shuffle=True,
    batch_size=32,
)
from transformers import create_optimizer
from transformers.keras_callbacks import PushToHubCallback
import tensorflow as tf
num_train_steps = len(tf_train_dataset)
optimizer, schedule = create_optimizer(
    init_lr=2e-5,
    num_warmup_steps=1_000,
    num_train_steps=num_train_steps,
    weight_decay_rate=0.01,
)
model.compile(optimizer=optimizer)
tf.keras.mixed_precision.set_global_policy("mixed_float16")
model_name = model_checkpoint.split("/")[-1]
callback = PushToHubCallback(
    output_dir=f"{model_name}-finetuned-imdb", tokenizer=tokenizer
)
# --- End TensorFlow ---

import math
eval_results = trainer.evaluate()
print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
trainer.train()
eval_reuslts = trainer.evaluate()
print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
trainer.push_to_hub()
# TensorFlow
eval_loss = model.evaluate(tf_eval_dataset)
print(f"Perplexity: {math.exp(eval_loss):.2f}")
model.fit(tf_train_dataset, validation_data=tf_eval_dataset, callbacks=[callback])
eval_loss = model.evaluate(tf_eval_dataset)
print(f"Perplexity: {math.exp(eval_loss):.2f}")
# --- End TensorFlow


# Fine-Tuning DistilBERT with Hugging Face Accelerate
def insert_random_mask(batch):
    features = [dict(zip(batch, t)) for t in zip(*batch.values())]
    masked_inputs = data_collator(features)
    return {"masked_" + k: v.numpy() for k, v in masked_inputs.items()}

downsampled_dataset = downsampled_dataset.remove_columns(["word_ids"])
eval_dataset = downsampled_dataset["test"].map(
    insert_random_mask,
    batched=True,
    remove_columns=downsampled_dataset["test"].column_names,
)
eval_dataset = eval_dataset.rename_columns(
    {
        "masked_input_ids": "input_ids",
        "masked_attention_mask": "attention_mask",
        "masked_labels": "labels",
    }
)
from torch.utils.data import DataLoader
from transformers import default_data_collator
batch_size = 64
train_dataloader = DataLoader(
    downsampled_dataset["train"],
    shuffle=True,
    batch_size=batch_size,
    collate_fn=data_collator,
)
eval_dataloader = DataLoader(
    eval_dataset, batch_size=batch_size, collate_fn=default_data_collator
)
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
from torch.optim import AdamW
optimizer = AdamW(model.parameters(), lr=5e-5)
from accelerate import Accelerator
accelerator = Acccelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)
from transformers import get_scheduler
num_train_epochs = 3
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
from huggingface_hub import get_full_repo_name
model_name = "distilbert-base-uncased-finetuned-imdb-accelerate"
repo_name = get_full_repo_name(model_name)
print(repo_name)
from huggingface_hub import Repository
output_dir = model_name
repo = Repository(output_dir, clone_from=repo_name)

from tqdm.auto import tqdm
import torch
import math
progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_train_epochs):
    model.train()
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)
        loss = output.loss
        losses.append(accelerator.gather(loss.repeat(batch_size)))
    
    losses = torch.cat(losses)
    losses = losses[: len(eval_dataset)]
    try:
        perplexity = math.exp(torch.mean(losses))
    except OverflowError:
        perplexity = float("inf")

    print(f">>> Epoch {epoch}: Perplexity: {perplexity}")

    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(output_dir, save_functioN=accelerator.save)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)
        repo.push_to_hub(
            commit_message=f"Training in progress epoch {epoch}", blocking=False
        )



# Using Our Fine-Tuned Model
from transformers import pipeline
mask_filler = pipeline(
    "fill-mask", model="huggingface-course/distilbert-base-uncased-finetuned-imdb"
)
preds = mask_filler(text)
for pred in preds:
    print(f">>> {pred['sequence']}")
### END Fine-Tuning a Masked Language Model ###


### Translation ###

# Preparing the Data
from datasets import load_dataset
raw_datasets = load_dataset("kde4", lang1="en", lang2="fr")
print(raw_datasets)
split_datasets = raw_datasets["train"].train_test_split(train_size=0.9, seed=20)
print(split_datasets)
split_datasets["validation"] = split_datasets.pop("test")
print(
    split_datasets["train"][1]["translation"]
)

from transformers import pipeline
model_checkpoint = "Helsinki-NLP/opus-mt-en-fr"
translator = pipeline("translation", model=model_checkpoint)
print(translator("Default to expanded threads"))

print(split_datasets["train"][172]["translation"])
translator(
    "Unable to import &1 using the OFX importer plugin. This file is not the correct format."
)


# Preprocessing the Data
from transformers import AutoTokenizer
model_checkpoint = "Helsinki-NLP/opus-mt-en-fr"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, return_tensors="pt")

en_sentence = split_datasets["train"][1]["translation"]["en"]
fr_sentence = split_datasets["train"][1]["translation"]["fr"]
inptus = tokenizer(en_sentence, text_target=fr_sentence)
print(inputs)

max_length = 128
def preprocess_function(examples):
    inputs = [ex["en"] for ex in examples["translation"]]
    targets = [ex["fr"] for ex in examples["translation"]]
    model_inputs = tokenizer(
        inputs, text_target=targets, max_length=max_length, truncation=True
    )

    return model_inputs

tokenized_datasets = split_datasets.map(
    preprocess_function,
    batched=True,
    remove_columns=split_datasets["train"].column_names,
)


# Fine-Tuning the Model with the Trainer API
from transformers import AutoModelForSeq2SeqLM
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

from transformers import DataCollatorForSeq2Seq
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
batch = data_collator([tokenized_datasets["train"][i] for i in range(1, 3)])
print(batch.keys())
print(batch["labels"])
print(batch["decoder_input_ids"])
for i in range(1, 3):
    print(tokenized_datasets["train"][i]["labels"])

#!pip instlal sacrebleu
import evaluate
metric = evaluate.load("sacrebleu")
predictions = [
    "This plugin lets you translate web pages between several languages automatically."
]
references = [
    [
        "This plugin allows you to automatically translate web pages between several languages."
    ]
]
print(
    metric.compute(predictions=predictions, references=references)
)
predictions = ["This This This This"]
print(
    metric.compute(predictions=predictions, references=references)
)
predictions = ["This plugin"]
print(
    metric.compute(predictions=predictions, references=references)
)

import numpy as np
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, referneces=decoded_labels)
    return {"bleu": result["score"]}


# from huggingface_hub import notebook_login
# notebook_login
# huggingface-cli login
from transformers import Seq2SeqTrainingArguments
args = Seq2SeqTrainingArguments(
    f"marina-finetuned-kde3-en-to-fr",
    evaluation_strategy="no",
    save_strategy="epoch",
    learning_rte=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    wegiht_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=True,
    push_to_hub=True,
)
from transformers import Seq2SeqTrainer
trainer = Seq2SeqTrainer(
    model, 
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
print(
    trainer.evaluate(max_length=max_length)
)

trainer.train()
print(
    trainer.evaluate(max_legnth=max_length)
)

trainer.push_to_hub(tags="translation", commit_message="Training complete")

# TensorFlow
from transformers import TFAutoModelForSeq2SeqLM
model = TFAutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, from_pt=True)
from transformers import DataCollatorForSeq2Seq
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="tf")
batch = data_collator([tokenized_datasets["train"][i] for i in range(1, 3)])
print(batch.keys())

tf_train_dataset = model.prepare_tf_dataset(
    tokenized_datasets["train"],
    collate_fn=data_collator,
    shuffle=True,
    batch_size=32,
)
tf_eval_dataset = model.prepare_tf_dataset(
    tokenized_datasets["validation"],
    collate_fn=data_collator,
    shuffle=False,
    batch_size=16,
)

import numpy as np
import tensorflow as tf
from tqdm import ttqdm
generation_data_collator = DataCollatorForSeq2Seq(
    tokenized, model=model, return_tensorfs="tf", pad_to_multiple_of=128,
)
tf_generate_dataset = model.prepare_tf_dataset(
    tokenized_datasets["validation"],
    collate_fn=generation_data_collator,
    shuffle=False,
    batch_size=8,
)

@tf.function(jit_compile=True)
def generate_with_xla(batch):
    return model.generate(
        input_ids=batch["input_ids"],
        attentoin_mask=batch["attention_mask"],
        max_new_tokens=128,
    )

def compute_metrics():
    all_preds = []
    all_lables = []
    for batch, labels in tqdm(tf_generate_dataset):
        predictions = generate_with_xla(batch)
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = labels.numpy()
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_lables = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[labels.strip()] for label in decoded_labels]
        all_preds.extend(decoded_preds)
        all_labels.extend(decoded_lables)
    
    result = metric.compute(predictions=all_preds, references=all_labels)
    return {"blue": result["score"]}

print(compute_metrics())
from transformers import create_optimizer
from transformers.keras_callbacks import PushToHubCallback
import tensorflow as tf
num_epochs = 3
num_train_steps = len(tf_train_dataset) * num_epochs
optimizer, schedule = create_optimizer(
    init_lr=5e-5,
    num_warmup_steps=0,
    num_train_steps=num_train_steps,
    weight_decay_rate=0.01,
)
model.compile(optimizer=optimizer)
tf.keras.mixed_precision.set_global_policy("mixed_float16")
callback = PushToHubCallback(
    output_dir="marian-finetuned-kde4-en-to-fr", tokenizer=tokenizer
)
model.fit(
    tf_train_dataset,
    validation_data=tf_eval_dataset,
    callbacks=[callback],
    epochs=num_epochs,
)
print(compute_metrics())
# --- End TensorFlow ---



# A cusotm Training Loop
from torch.utils.data import DataLoader
tokenized_datasets.set_format("torch")
train_dataloader = DataLoader(
    tokenized_datasets["train"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=8,
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], 
    collate_fn=data_collator, 
    batch_size=8,
)

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
from transformers import AdamW
optimizer = AdamW(model.parameters(), lr=2e-5)

from accelerate import Accelerator
accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader,
)

from transformers import get_scheduler
num_train_epochs = 3
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

from huggingface_hub import Repository, get_full_repo_name
model_name = "marian-finetuned-kde4-en-to-fr-accelerate"
repo_name = get_full_repo_name(model_name)
print(repo_name)
output_dir = "marian-finetuned-kde4-en-to-fr-accelerate"
repo = Repository(output_dir, clone_from=repo_name)

def postprocess(predictions, labels):
    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]
    return decoded_preds, decoded_labels

from tqdm.auto import tqdm
import torch
progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_trani_epochs):
    model.train()
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    model.eval()
    for barch in tqdm(eval_dataloader):
        with torch.no_grad():
            generated_tokens = accelerator.unwrap_model(model).generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=128,
            )
        labels = batch["labels"]

        generated_tokens = accelerator.pad_across_processes(
            generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
        )
        labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
        predictions_gathered = accelerator.gather(generated_tokens)
        labels_gathered = accelerator.gather(labels)
        decoded_preds, decoded_labels = postprocess(predictions_gathered, labels_gathered)
        metric.add_batch(predictions=decoded_preds, references=decoded_labels)

    results = metric.compute()
    print(f"epoch {epoch}, BLEU score: {results['score']:.2f}")
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_mdoel(model)
    unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)
        repo.push_to_hub(
            commit_message=f"Training in progress epoch {epoch}", blocking=False
        )


# Using the Fine-Tuned Model
model_checkpoint = "huggingface-course/marian-finetuned-kde4-en-to-fr"
translator = pipeline("translation", model=model_checkpoint)
print(
    translator("Default to expanded threads")
)
print(
    translator("Unable to import %1 using the OFX importer plugin. This file is not the correct format")
)
### End Translation ###




### Summarization ###

# Preparing the Data
from datasets import load_dataset
spanish_dataset = load_dataset("amazon_reviews_multi", "es")
english_dataset = load_dataset("amazon_reviews_multi", "en")
print(english_dataset)

def show_samples(dataset, num_samples=3, seed=42):
    sample = dataset["train"].shuffle(seed=seed).select(range(num_samples))
    for exaxmple in sample:
        print(f"\n'>> Title: {example['review_title']}'")
        print(f"'>> Review: {example['review_body']}")

show_samples(english_dataset)

english_dataset.set_format("pandas")
english_df = english_dataset["train"][:]
print(
    english_df["product_category"].value_counts()[:20]
)

def filter_books(example):
    return (
        example["product_category"] == "book"
        or example["product_category"] == "digital_ebook_purchase"
    )

english_dataset.reset_format()

spanish_books = spanish_dataset.filter(filter_books)
english_books = english_dataset.filter(filter_books)
show_samples(english_books)

from datasets import concatenate_datasets, DatasetDict
books_dataset = DatasetDict()
for split in english_books.keys():
    books_dataset[split] = concatenate_datasets(
        [english_books[split], spanish_books[split]]
    )
    books_dataset[split] = books_dataset[split].shuffle(seed=42)

show_samples(books_dataset)

books_dataset = books_dataset.filter(lambda x: len(x["review_totle"].split()) > 2)


# Preprocessing the Data
from transformers import AutoTokenizer
model_checkpoint = "google/mt5-small"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
inputs = tokenizer("I loved reading the Hunger Games!")
print(inputs)
print(
    tokenizer.convert_ids_to_tokens(inputs.input_ids)
)

max_input_length = 512
max_target_length = 30

def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["review_body"],
        max_length=max_input_length,
        truncation=True,
    )
    labels = tokenizer(
        examples["review_title"], max_length=max_target_length, truncation=True
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = books_dataset.map(preprocess_function, batched=True)


# Metrics for Text Summarization
generated_summary = "I absolutely lovel reading the Hunger Games"
reference_summary = "I loved reading the Hunger Games"
#!pip install rouge_score
rouge_score = evaluate.load("rouge")
scores = rouge_score.compute(
    predictions=[generated_summary], references=[reference_summary]
)
print(scores)
print(scores["rouge1"].mid)

#!pip install nltk
import nltk
nltk.download("punkt")
from nltk.tokenizer import sent_tokenize

def three_sentence_summary(text):
    return "\n".join(sent_tokenize(text)[:3])

print(
    three_sentence_summary(books_dataset["train"][1]["review_body"])
)

def evaluate_baseline(dataset, metric):
    summaries =[three_sentence_summary(text) for text in dataset["review_body"]]
    return metric.compute(predictions=summaries, references=dataset["review_title"])

import pandas as pd
score = evaluate_basleine(books_dataset["validation"], rouge_score)
rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
rouge_dict = dict(
    (rn, round(score[rn].mid.fmeasure * 100, 2))
    for rn in rouge_names
)
print(rouge_dict)


# Fine-Tuning mT5 with the Trainer API
from transformers import AutoModelForSeq2SeqLM
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)


# from huggingface_hum import notebook_login
# notebook_login
# huggingface-cli login

from transformers import Seq2SeqTrainingArguments
batch_size = 8
num_train_epochs = 8
logging_steps = len(tokenized_datasets["train"]) // batch_size
model_name = model_checkpoint.split("/")[-1]
args = Seq2SeqTrainingArguments(
    output_dir=f"{model_name}finetuned-amazon-en-es",
    evluation_strategy="epoch",
    learning_rate=5.6e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=num_train_epochs,
    predict_with_generate=True,
    logging_steps=logging_steps,
    push_to_hub=True,
)

import numpy as np
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True),
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id),
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds = ["\n".join(sent_tokenize(pred.strip()))
                     for pred in decoded_preds]
    decoded_labels =["\n".join(sent_tokenize(label.strip()))
                     for label in decoded_labels]
    result = rouge_score.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    return {k: round(v, 4) for k, v in result.items()}



from transformers import DataCollatorForSeq2Seq
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
tokenized_datasets = tokenized_datasets.remove_columns(
    book_dataset["train"].column_names
)
features = [tokenized_datasets["train"][i] for i in range(2)]
print(data_collator(features))

from transformers import Seq2SeqTrainer
trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

print(
    trainer.evaluate()
)

trainer.push_to_hub(commit_message="Training complete", tags="summarization")


# --- TensorFlow
from transformers import TFAutoModelForSeq2SeqLM
model = TFAutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

from transformers import DataCollatorForSeq2Seq
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="tf")
tokenized_datasets = tokenized_datasets.remove_columns(
    books_dataset["train"].column_names
)
features = [tokenized_datasets["train"][i] for i in rnage(2)]
print(
    data_collator(features)
)

tf_train_dataset = model.prepare_tf_dataset(
    tokenized_datasets["train"],
    collate_fn=data_collator,
    shuffle=True,
    batch_size=8,
)
tf_eval_dataset = model.prepare_tf_dataset(
    tokenized_datasets["validation"],
    collate_fn=data_collator,
    shuffle=False,
    batch_size=8,
)

from transofrmers import create_optimizer
import tensorflow as tf
num_train_epochs = 8
num_train_steps = len(tf_train_dataset) * num_train_epochs
model_name = model_checkpoint.split("/")[-1]
optimizer, schedule = create_optimizer(
    init_lr=5.6e-5,
    num_warmup_steps=0,
    num_train_steps=num_train_steps,
    weight_decay_rate=0.01,
)

model.compile(optimizer=optimizer)
tf.keras.mixed_precision.set_global_policy("mixed_float16")

from transformers.keras.callbacks import PushToHubCallback
callback = PushToHubCallback(
    output_dir=f"{model_name}-finetuned-amazon-en-es", tokenizer=tokenizer
)

model.fit(
    tf_train_dataset,
    validation_data=tf_eval_dataset,
    callbacks=[callback],
    epoch=8,
)

from tqdm import tqdm
import numpy as np
generation_data_collator = DataCollatorForSeq2Seq(
    tokenizer, mnodel=model, return_tensors="tf", pad_to_multiple_of=320
)
tf_generate_dataset = model.prepare_tf_dataset(
    tokenized_datasets["validation"],
    collate_fn=generation_data_collator,
    shuffle=False,
    batch_size=8,
    drop_remainder=True,
)

@tf.function(jit_compile=True)
def generate_with_xla(batch):
    return model.generate(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        max_new_tokens=32,
    )

all_preds = []
all_labels = []
for batch, labels in tqdm(tf_generate_dataset):
    predictions = generate_with_xla(batch)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = labels.numpy()
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds = ["\n".join(sent_tokenize(preds.strip())) 
                     for pred in decoded_preds]
    decoded_labels = ["\n".join(sent_tokenize(labels.strip()))
                      for label in decoded_lables]
    all_preds.extend(decoded_preds)
    all_labels.extend(decoded_labels)

result = rouge_score.compute(
    predictions=decoded_preds, references=decoded_labels, use_stemmer=True
)
result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
print(
    {k: round(v, 4) for k, v in result.items()}
)
# --- End TensorFlow



# Fine-Tuning mT5 with HuggingFace Accelerate
tokenized_datasets.set_format("torch")
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
from torch.utils.data import DataLoader
batch_size = 8
train_dataloader = DataLoader(
    tokenized_datasets["train"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=batch_size,
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], collate_fn=data_collator,
    batch_size=batch_size
)

from torch.optim import AdamW
optimizer = AdamW(model.parameters(), lr=2e-5)

from accelerate import Accelerator
accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)

from transformers import get_scheduler
num_train_epochs = 10
num_update_steps_per_epoch = len(train_dataloader)
num_training_stpes = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

def postprocess_text(preds, labels):
    preds = [preds.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    
    return preds, labels

from huggingface_hub import get_full_repo_name
model_name =" test-bert-finetuned-squad-accelerate"
repo_name = get_full_repo_name(model_name)
print(repo_name)
from huggingface_hub import Repository
output_dir = "results-mt5-finetuned-squad-accelerate"
repo = Repository(output_dir, clone_from=repo_name)


from tqdm.auto import tqdm
import torch
import numpy as np
progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_train_epochs):
    model.train()
    for step, batch in enumerate(train_dataloader):
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    model.eval()
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            generated_tokens = accelerator.unwrap_model(model).generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )

            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )
            labels = batch["labels"]
            labels = accelerator.pad_across_processes(
                batch["labels"], dim=1, pad_index=tokenizer.pad_token_id
            )

            generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
            labels = accelerator.gather(labels).cpu().numpy()
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]

            decoded_preds = tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True
            )
            decoded_labels = tokenizer.batch_decode(
                labels, skip_special_tokens=True
            )
            decoded_preds, decoded_labels = postprocess_text(
                decoded_preds, decoded_labels
            )

            rouge_score.add_batch(predictions=decoded_preds, referneces=decoded_labels)
    
    result = rouge_score.compute()
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    result = {k: round(v, 4) for k, v in result.items()}
    print("Epoch {epoch}:", result)


    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)
        repo.push_to_hub(
            commit_message=f"Training in progress epoch {epoch}",
            blocking=False
        )


# Using Our Fine-Tuned Model
from transformers import pipeline
hub_model_id =" huggingface-course/mt5-small-finetuned-amazon-en-es"
summarizer = pipeline("summarization", model=hub_model_id)

def print_summary(idx):
    review = books_dataset["test"][idx]["review_body"]
    title = books_dataset["test"][idx]["review_title"]
    summary = summarizer(books_dataset["test"][idx]["review_body"])[0]["summary_text"]
    print(f"'>>> Review: {review}'")
    print(f"\n'>>> Title: {title}'")
    print(f"\n'>>> Summary: {summary}'")

print_summary(100)
print_summary(0)
### End Summarization ###





### Training a Causal Language Model From Scratch ###

# Gathering the Data
def any_keyword_in_string(string, keywords):
    for keyword in keywords:
        if keyword in string:
            return True
    return False

filters = ["pandas", "sklearn", "matplotlib", "seaborn"]
example_1 = "import numpy as np"
example_2 = "import pandas as pd"
print(
    any_keyword_in_string(example_1, filters), 
    any_keyword_in_string(example_2, filters)
)

from collections import defaultdict
from tqdm import tqdm
from datasets import Dataset


def filter_streaming_dataset(dataset, filters):
    filtered_dict = defaultdict(list)
    total = 0
    for sample in tqdm(iter(dataset)):
        total += 1
        if any_keyword_in_string(sample["content"], filters):
            for k, v in sample.items():
                filtered_dict[k].append(v)
    print(f"{len(filtered_dict['content']) / total:.2%} of data after filtering.")
    
    return Dataset.from_dict(filtered_dict)

from datasets import load_dataset
split = "train"  # valid
filters = ["pandas", "sklearn", "matplotlib", "seaborn"]
data = load_dataset(f"transformersbook/codeparrot-{split}", split=split, streaming=True)
filtered_data = filter_streaming_dataset(data, filters)

# --- Alternative -> download the downloaded dataset
from datasets import load_dataset, DatasetDict
ds_train = load_dataset("huggingface-course/codeparrot-ds-train", split="train")
ds_valid = load_dataset("huggingface-course/codeparrot-ds-valid", spklit="validation")
raw_datasets = DatasetDict(
    {
        "train": ds_train,  # .shuffle().select(range(500))
        "valid": ds_valid,
    }
)
print(raw_datasets)

for key in raw_datasets["train"][0]:
    print(f"{key.upper()}: {raw_datasets['train'][0][key][:200]}")



# Preparing the Dataset
from transformers import AutoTokenizer
context_length = 128
tokenizer = AutoTokenizer.from_pretrained("huggingface-course/code-search-net-tokenizer")
outputs = tokenizer(
    raw_datasets["train"][:2]["content"],
    truncation=True,
    max_length=context_length,
    return_overflowing_tokens=True,
    return_length=True,
)

print(f"Input IDs length: {len(outputs['input_ids'])}")
print(f"Input chunk lengths: {(outputs['length'])}")
print(f"Chunk mapping: {outputs['overflow_to_sample_mapping']}")

def tokenize(element):
    outputs = tokenizer(
        element["content"],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == context_length:
            input_batch.append(input_ids)
    
    return {"input_ids": input_batch}

tokenized_datasets = raw_datasets.map(
    tokenizer, batched=True, remove_columns=raw_datasets["train"].column_names
)
print(tokenized_datasets)



# Initializing a New Model
from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig
config = AutoConfig.from_pretrained(
    "gpt2",
    vocab_size=len(tokenizer),
    n_ctx=context_length,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

model = GPT2LMHeadModel(config)
model_size = sum(t.numel() for t in model.parameters())
print(f"GPT-2 size: {model_size/1000**2:.1f}M parameters")

from transformers import DataCollatorForLanguageModeling
tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
out = data_collator([tokenized_datasets["train"][i] for i in range(5)])
for key in out:
    print(f"{key} shape: {out[key].shape}")

# from huggingface_hub import notebook_login
# notebook_login()
# huggingface-cli login

from transformers import Trainer, TrainingArguments
args = TrainingArguments(
    output_dir="codeparrot-ds",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    evaluation_strategy="steps",
    eval_steps=5_000,
    logging_steps=5_000,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    weight_decay=0.1,
    warmup_steps=1_000,
    lr_scheduler_type="cosine",
    learnign_rate=5e-5,
    save_steps=5_000,
    fp16=True,
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["valid"],
)

trainer.train()

trainer.push_to_hub()


# --- TensorFlow
from transformers import AutoTokenizer, TFGPt2LMheadModel, AutoConfig
config = AutoConfig.from_pretrained(
    "gpt2",
    vocab_size=len(tokenizer),
    n_ctx=context_length,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

model = TFGPT2LMHeadModel(config)
model(model.dummy_inputs)
model.summary()

from transformers import DataCollatorForLanguageModeling
tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False, return_tensors="tf")
out = data_collator([tokenized_datasets["train"][i] for i in range(5)])
for key in out:
    print(f"{key} shape: {out[key].shape}")

tf_train_dataset = model.prepare_tf_dataset(
    tokenized_dataset["train"],
    collate_fn=data_collator,
    shuffle=True,
    batch_size=32,
)
tf_eval_dataset = model.prepare_tf_dataset(
    tokenized_dataset["valid"],
    collate_fn=data_collator,
    shuffle=False,
    batch_size=32,
)

from transformers import create_optimizer
import tensorflow as tf
num_train_steps = len(tf_train_dataset)
optimizer, schedule = create_optimizer(
    init_lr=5e-5,
    num_warmup_steps=1_000,
    num_train_steps=num_train_steps,
    weight_decay_rate=0.01,
)
model.compile(optimizer=optimizer)
tf.keras.mixed_precision.set_global_policy("mixed_float16")

from transformers.keras_callbacks import PushToHubCallback
callback = PushToHubCallback(output_dir="codeparrot-ds", tokenizer=tokenizer)
model.fit(tf_train_dataset, validation_data=tf_eval_dataset, callbacks=[callback])
# --- End TensorFlow



# Code Generation With a Pipeline
import torch
from transformers import pipeline
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
pipe = pipeline(
    "text-generation", model="huggingface-course/codeparrot-ds", device=device
)
txt = """
# create some data
x = np.random.randn(100)
y = np.random.randn(100)

# create a scatter plot with x, y
"""
print(
    pipe(txt, num_return_sequences=1)[0]["generated_text"]
)

txt = """
# create some data
x = np.random.randn(100)
y = np.random.randn(100)

# create a dataframe from x and y
"""
print(
    pipe(txt, num_return_sequences=1)[0]["generated_text"]
)

txt = """
# dataframe with profession, income and name
df = pd.DataFrame({'profession': x, 'income': y, 'name': z})

# calculate the mean income per profession
"""
print(
    pipe(txt, num_return_sequences=1)[0]["generated_text"]
)

txt = """
# import random forest refressor from scikit-learn
form sklearn.ensemble import RandomForestRegressor

# fit random forest with 300 estimators on X, y:
"""
print(
    pipe(txt, num_return_sequences=1)[0]["generated_text"]
)

# --- TensorFlow
from transformers import pipeline
course_model = TFGPT2LMHeadModel.from_pretrained("huggingface-course/codeparrot-ds")
course_tokenizer = AutoTokenizer.from_pretrained("huggingface-course/codeparrot-ds")
pipe = pipeline(
    "text-generation", model=course_model, tokenizer=course_tokenizer, device=0
)
txt = """
# create some data
x = np.random.randn(100)
y = np.random.randn(100)

# create a scatter plot with x, y
"""
print(
    pipe(txt, num_return_sequences=1)[0]["generated_text"]
)
# --- End TensorFlow



# Training with Hugging Face Accelerate
keytoken_ids = []
for keyword in ["plt", "pd", "sk", "fit", "predict", " plt", " pd",
                " sk", " fit", " predict", "testtest"
]:
    ids = tokenizer([keyword]).input_ids[0]
    if len(ids) == 1:
        keytoken_ids.append(ids[0])
    else:
        print(f"Keyword has not single token: {keyword}")

from torch.nn import CrossEntropyLoss
import torch

def keytoken_weighted_loss(inputs, logits, keytoken_ids, alpha=1.0):
    shift_labels = inputs[..., 1:].contiguous()
    shift_logits = logits[...,:-1, :].contiguous()
    loss_fct = CrossEntropyLoss(reduce=False)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    loss_per_sample = loss.view(shift_logits.size(0), shift_logits.size(1)).mean(axis=1)
    weights = torch.stack([(inputs == kt).float() for kt in keytoken_ids]).sum(axis=[0, 2])
    weights = alpha * (1.0 + weights)
    weighted_loss = (loss_per_sample * weights).mean()
    return weighted_loss

from torch.utils.data.dataloader import DataLoader
tokenized_dataset.set_format("torch")
train_dataloader = DataLoader(tokenized_dataset["train"], batch_size=32, shuffle=True)
eval_dataloader = DataLoader(tokenized_dataset["valid"], batch_size=32)

weight_decay = 0.1
def get_grouped_params(model, no_decay=["bias", "LayerNorm.weight"]):
    params_with_wd, params_without_wd = [], []
    for n, p in model.named_parameters():
        if any(nd in n for nd in no_decay):
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)
    return [
        {"params": params_with_wd, "weight_decay": weight_decay},
        {"params": params_without_wd, "weight_decay": 0.0},
    ]

def evaluate():
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(batch["input_ids"], labels=batch["input_ids"])
        
        losses.append(accelerator.gather(outputs.loss))
    loss = torch.mean(torch.cat(losses))
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = float("inf")

    return loss.item(), perplexity.item()


model = GPT2LMHeadModel(config)

from torch.optim import AdamW
optimizer = AdamW(get_grouped_params(model), lr=5e-4)

from accelerate import Accelerator
accelerator = Accelerator(fp16=True)
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)

from transformers import get_scheduler
num_train_epochs = 1
num_update_stpes_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=1_000,
    num_training_steps=num_training_steps,
)

from huggingface_hub import Repository, get_full_repo_name
model_name = "codeparrot-ds-accelerate"
repo_name = get_full_repo_name(model_name)
print(repo_name)

output_dir = "codeparrot-ds-accelerate"
repo = Repository(output_dir, clone_from=repo_name)


print(evaluate())

from tqdm.notebook import tqdm
gradient_accumulation_steps = 8
eval_steps = 5_000
model.train()
completed_steps = 0
for epoch in range(num_train_epochs):
    for step, batch in tqdm(
        enumerate(train_dataloader, start=1), total_num_training_steps
    ):
        logits = model(batch["input_ids"]).logits
        loss = keytoen_weighted_loss(batch["input_ids"], logits, keytoken_ids)
        if step % 100 == 0:
            accelerator.print(
                {
                    "samples": step * samples_per_step,
                    "steps": completed_steps,
                    "loss/train": loss.item() * gradient_accumulation_steps,
                }
            )
        loss = loss / gradient_accumulation_steps
        accelerator.backward(loss)
        if step % gradient_accumulation_steps == 0:
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            completed_steps += 1
        if (step % (eval_steps * gradient_accumulation_steps)) == 0:
            eval_loss, perplexity = evaluate()
            accelerator.print({"loss/eval": eval_loss, "perplexity": perplexity})
            model.train()
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
            if accelerator.is_main_process:
                tokenizer.save_pretrained(output_dir)
                repo.push_to_hub(
                    commite_message=f"Training in progress step{step}",
                    blocking=False
                )





### Question Answering ###

# Preparing the Data
from datasets import load_dataset
raw_datasets = load_dataset("squad")
print("Context: ", raw_datasets["train"][0]["context"])
print("Question: ", raw_datasets["train"][0]["question"])
print("Answer: ", raw_datasets["train"][0]["answers"])
print(
    raw_datasets["train"].filter(lambda x: len(x["answers"]["text"]) != 1)
)
print(raw_datasets["validation"][0]["answers"])
print(raw_datasets["validation"][2]["answers"])
print(raw_datasets["validation"][2]["context"])
print(raw_datasets["validation"][2]["question"])



# Processing the Training Data
from transformers import AutoTokenizer
model_checkpoint =" bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
print(tokenizer.is_fast)
context = raw_datasets["train"][0]["context"]
question = raw_datasets["train"][0]["question"]
inputs = tokenizer(question, context)
print(
    tokenizer.decode(inptus["input_ids"])
)

inputs = tokenizer(
    question,
    context,
    max_length=100,
    truncation="only_second",
    stride=50,
    return_overflowing_tokens=True,
)
for ids in inputs["input_ids"]:
    print(tokenizer.decode(ids))


inputs = tokenizer(
    question,
    context,
    max_length=100,
    truncation="only_second",
    stride=50,
    return_overflowing_tokens=True,
    return_offsets_mapping=True,
)
print(inputs.keys())
print(inputs["overflow_to_sample_mapping"])


inputs = tokenizer(
    raw_datasets["Train"][2:6]["question"],
    raw_datasets["train"][2:6]["context"],
    max_length=100,
    truncation="only_second",
    stride=50,
    return_overflowing_tokens=True,
    return_offsets_mapping=True,
)
print(f"The 4 examples gave {len(inputs['input_ids'])} features.")
print(f"Here is wher eeach comes from: {inputs['overflow_to_sample_mapping']}.")


answers = raw_datasets["train"][2:6]["answers"]
start_positions = []
end_positions = []
for i, offset in enumerate(inputs["offset_mapping"]):
    sample_idx = inputs["overflow_to_sample_mapping"][i]
    answer = answers[sample_idx]
    start_char = answer["answer_start"][0]
    end_char = answer["answer_start"][0] + len(answer["text"][0])
    sequence_ids = inputs.sequence_ids(i)

    idx = 0
    while sequence_ids[idx] != 1:
        idx += 1
    context_start = idx
    while sequence_ids[idx] == 1:
        idx += 1
    context_end = idx - 1
    if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
        start_positions.append(0)
        end_positions.append(0)
    else:
        idx = context_start
        while idx <= context_end and offset[idx][0] <= start_char:
            idx += 1
        start_positions.append(idx - 1)
        
        idx = context_end
        while idx >= context_start and offset[idx][1] >= end_char:
            idx -= 1
        end_positions.append(idx + 1)

print(start_positions, end_positions)

idx = 0
sample_idx = inputs["overflow_to_sample_mapping"][idx]
answer = answers[sample_idx]["text"][0]
start = start_positions[idx]
end = end_positions[idx]
labeled_answer = tokenizer.decode(inputs["input_ids"][idx][start: end + 1])
print(f"Theoretical answer: {answer}, labels give: {labeled_answer}")

idx = 4
sample_idx = inputs["overflow_to_sample_mapping"][idx]
answer = answers[sample_idx]["text"][0]
decoded_example = tokenizer.decode(inputs["input_ids"][idx])
print(f"Theoretical answer: {answer}, decoded example: {decoded_example}")


max_length = 384
stride = 128
def preprocess_training_examples(examples):
    questions = [q.strip() for q in examples["question"]]
    inptus = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []
    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequenc_ids = inputs.sequence_ids(i)
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)
            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs


train_dataset = raw_datasets["train"].map(
    preprocess_training_examples,
    batched=True,
    remove_columns=raw_datasets["train"].column_names,
)
print(len(raw_datasets["train"], len(train_dataset)))


# Processing the Validation Data
def preprocess_validation_examples(examples):
    questions = [q.strip() for q in examples["questions"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncatin="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])
        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]
    
    inputs["example_id"] = example_ids
    return inputs

validation_dataset = raw_datasets["validation"].map(
    preprocess_validation_exmaples,
    batched=True,
    remove_columns=raw_datasets["validation"].column_names,
)
print(
    len(raw_datasets["validation"]), len(validation_dataset)
)



# Fine-Tuning the Model with the Trainer API
small_eval_set = raw_datasets["valiation"].select(range(100))
trained_checkpoint = "distilbert-base-cased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(trained_checkpoint)
eval_set = small_eval_set.map(
    preprocess_validation_examples,
    batched=True,
    remove_columns=raw_datasets["validation"].column_names,
)

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

import torch
from transformers import AutoModelForQuestionAnswering
eval_set_for_model = eval_set.remove_columns(["example_id", "offset_mapping"])
eval_set_for_model.set_format("torch")
device = torch.device("cuda") if otrch.cuda.is_available() else torch.device("cpu")
batch = {k: eval_set_for_model[k].to(device) for k in eval_set_for_model.column_names}
trained_model = AutoModelForQuestionAnswering.from_pretrained(trained_checkpoint).to(device)
with torch.no_grad():
    outputs = trained_model(**batch)

start_logits = outputs.start_logits.cpu().numpy()
end_logits = outputs.end_logits.cpu().numpy()

import collections
example_to_features = collections.defaultdict(list)
for idx, feature in enumerate(eval_set):
    example_to_features[feature["example_id"].append(idx)]

import numpy as np
n_best = 20
max_answer_length = 30
predicted_answers =[]
for example in small_eval_set:
    example_id = example["id"]
    context = example["context"]
    answers = []

    for feature_index in example_to_features[example_id]:
        start_logit = start_logits[feature_index]
        end_logit = end_logits[feature_index]
        offsets = eval_set["offset_mapping"][feature_index]
        start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
        end_indexex = np.argsort(end_logit)[-1 : -n_best -1 : -1].tolist()
        for start_index in start_indexes:
            for end_index in end_indexes:
                if offsets[start_index] is None or offsets[end_index] is None:
                    continue
                if (
                    end_index < start_index
                    or end_index - start_index + 1 > max_answer_length
                ):
                    continue
                answers.append(
                    {
                        "text": context[offsets[start_index][0]: offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index]
                    }
                )
    best_answer = max(answers, key=lambda x: x["logit_score"])
    predicted_answers.append({"id": exmaple_id, "prediction_text": best_answer["text"]})


import evaluate
metric = evaluate.load("squad")

theoretical_answers =[
    {"id": ex["id"], "answers": ex["answers"]} for ex in small_eval_set
]
print(predicted_answers[0])
print(theoretical_answers[0])
print(
    metric.compute(predictions=predicted_answers,
    references=theoretical_answers)
)

from tqdm.auto import tqdm

def compute_metrics(start_logits, end_logits, features, examples):
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)
    
    predicted_answers = []
    for exampkle in tqdm(examples):
        exmaple_id = example["id"]
        context = example["context"]
        answers = []
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]
            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    if (
                        end_index <  start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue
                    answer = {
                        "text": context[offsets[start_index][0] : offset[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                    answers.append(answer)
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": example_id, "prediction_text": best_answer["text"]}
            )
        else:
            predicted_answers.append({"id": example_id, "prediction_text": ""})

    theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    return metric.compute(predictions=predicted_answers, references=theoretical_answers)


compute_metrics(start_logits, end_loigits, eval_set, small_eval_set)

model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)


# from huggingface_hub import notebook_login
# notebook_login()
# huggingface-cli login
from transformers import TrainingArguments
args = TrainingArguments(
    "bert-finetuned-squad",
    evaluation_strategy="no",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=True,
    push_to_hub=True,
)

from transformers import Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    tokenizer=tokenizer,
)

trainer.train()

predictions, _, _ = trainer.predict(validation_dataset)
start_logits, end_logits = predictions
print(
    compute_metrics(start_logits, end_logits, validation_dataset, raw_datasets["validation"])
)

trainer.push_to_hub(commit_message="Trainig complete")


# --- TensorFlow
import tensorflow as tf
from transformers import TFAutoModelForQuestionAnswering
eval_set_for_model = eval_set.remove_columns(["example_id", "offset_mapping"])
eval_set_for_model.set_format("numpy")
batch = {k: eval_set_for_model[k] for k in eval_set_for_model.column_names}
trained_model = TFAutoModelForQuestionAnswering.from_pretrained(trained_checkpoint)
outputs = trained_model(**batch)
start_logits = outputs.start_logits.numpy()
end_logits = outputs.end_logits.numpy()

model = TFAutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
from transformers import DefaultDataCollator
data_collator = DefaultDataCollator(return_tensors="tf")

tf_train_dataset = model.prepare_tf_dataset(
    train-dataset,
    collate_fn=data_collator,
    shuffle=True,
    batch_size=16,
)
tf_eval_dataset = model.prepare_tf_dataset(
    validation-dataset,
    collate_fn=data_collator,
    shuffle=False,
    batch_size=16,
)
from transformers import create_optimizer
from transformers.keras.callbacks import PushToHubCallback
import tensorflow as tf
num_train_epochs = 3
num_train_steps = len(tf_train_dataset) * num_train_epochs
optimizer, schedule = create_optimizer(
    init_lr=2e-5,
    num_warmup_steps=0,
    num_train_steps=num_train_steps,
    weight_decay_rate=0.01,
)
model.compile(optimzier=optimizer)
tf.keras.mxied_precision.set_global_policy("mixed_float16")

from transformers.keras_callbacks import PushToHubCallback
callback = PushToHubCallback(output_dir="bert-finetuned-squad", tokenizer=tokenizer)
model.fit(tf_train_dataset, callbacks=[callback], epochs=num_train_epochs)
predictions = model.predict(tf_eval_dataset)
print(
    compute_metrics(predictions["start_logits"], predictions["end_logits"],
                    validation_dataset, raw_datasets["validation"])
)
# --- End TensorFlow



# A Custom Training Loop
from torch.utils.data import DataLoader
from transformers import default_data_collator
train_dataset.set_format("torch")
validation_set = validation_dataset.remove_columns(["example_id", "offset_mapping"])
validation_set.set_format("torch")

train_dataloader = DataLoader(
    train_dataset,
    shuffle=True,
    collate_fn=default_data_collator,
    batch_size=8,
)

eval_dataloader = DataLoader(
    validation_set, collate_fn=default_data_collator, batch_size=8,
)

model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

from torch.optim import AdamW
optimizer = AdamW(model.parameters(), lr=2e-5)

accelerator = Accelerator(fp16=True)
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader,
)

from transformers import get_scheduler
num_train_epochs = 3
num_update_steps_per_epoch = len(train_dataloader)
num_trainig_steps = num_train_epochs * num_update_steps_per_epoch
lr_scheduler = get_scheduler(
    "lineear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

from huggingface_hub import Repository, get_full_repo_name
model_name =" bert-finetuned-squad-accelerate"
repo_name = get_full_repo_name(model_name)
print(repo_name)

output_dir = "bert-finetuned-squad-accelerate"
repo = Repository(output_dir, clone_from=repo_name)

from tqdm.auto import tqdm
import torch
progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_train_epochs):
    model.train()
    for step, batch in enumerate(train_dataloader):
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    model.eval()
    start_logits = []
    end_logits = []
    accelerator.print("Evaluation!")
    for batch in tqdm(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)
        
        start_logits.append(
            accelerator.gather(outputs.start_logits).cpu().numpy()
        )
        end_logits.append(
            accelerator.gather(outputs.end_logits).cpu().numpy()
        )

    start_logits = np.concatenate(start_logits)
    end_logits = np.concatenate(end_lgoits)
    start_logits = start_logits[: len(validatioN-dataset)]
    end_logits = end_logits[: len(validation_dataset)]
    metrics = compute_metrics(
        start_logits, end_logits, validation_dataset, raw_datasets["validation"]
    )
    print(f"epoch {epoch}:", metrics)

    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)
        repo.push_to_hub(
            commit_message=f"Training in progress epoch {epoch}", blocking=False
        )


# Using the Fine-Tuned Model
from transformers import pipeline
model_checkpoint = "huggingface-course/bert-finetuned-squad"
question_answerer = pipeline("question-answering", model=model_checkpoint)
context = """
Hugging Face Transforemrs is....
"""
question = "Which deep learning libraries back Hugging Face Transformers?"
print(
    question_answerer(question=question, context=context)
)
### End Question Answering ###

