import codecs
import configparser
import nltk
import os

from datasets import Dataset
from language_model import create_tokenizer, create_model, load_tokenizer, load_model
from nltk.tokenize import sent_tokenize 
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

NSLOTS = int(os.environ.get("NSLOTS", 1))
punkt_cache = "/projectnb/scottml/punkt_cache/"
nltk.data.path.append(punkt_cache)

config = configparser.ConfigParser()
config.read('marthabot.cfg')

# input_data = config["Data"].get("input_data")
# raw_output = config["Data"].get("raw_output")
clean_data = config["Data"].get("clean")


# Optional clean
# files = list_files_in_directory(input_data)
# for file in files:
#     text = extract_text_from_pdf(file, 12, 284)


# mg_blood_memory = raw_output + "/blood_memory.txt"
# with open(mg_blood_memory, "w") as file:
#     file.write(clean_blood_memory(text))

# Otherwise use preprocessed

blood_memory_txt = clean_data + "/blood_memory_clean.txt"
with codecs.open(blood_memory_txt, 'r', encoding='utf-8-sig') as file:
    clean_text = file.read()

# Tokenize the full text of blood memory into sentences
sentences = sent_tokenize(clean_text)

def sentence_generator(sentences):
    for sentence in sentences:
        yield {"text": sentence}

dataset = Dataset.from_generator(sentence_generator, 
                                 gen_kwargs={"sentences": sentences})
dataset = dataset.train_test_split(test_size=0.2)

## Create tokenizer and tokenize dataset
hf_token = config["Access"].get("hf_token")

def get_latest_checkpoint():
    checkpoint_dir = "marthabot/"
    checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")]
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[-1]))
    latest_checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    return latest_checkpoint_path

use_checkpoint = False

if not use_checkpoint:
    ## create tokenizer
    tokenizer = create_tokenizer(hf_token)
    ## create model
    model = create_model(hf_token)

else:
    latest_checkpoint_path = get_latest_checkpoint()
    tokenizer=load_tokenizer(latest_checkpoint_path)  
    model = load_model(latest_checkpoint_path)

def preprocess_function(examples):
    # Tokenize the input text
    tokenized_inputs = tokenizer([" ".join(x) for x in examples["text"]],
                                 padding=True,
                                 truncation=True,
                                 max_length=100)
    
    # Copy input_ids to labels
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
    
    return tokenized_inputs

# tokenized_dataset = dataset.map(preprocess_function, 
#                                 batched=True, 
#                                 num_proc=NSLOTS)

# tokenizer.pad_token = tokenizer.eos_token
# data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, 
#                                                 mlm=False)


# ## Train
# training_args = TrainingArguments(
#     output_dir="marthabot",
#     eval_strategy="epoch",
#     num_train_epochs=50,
#     learning_rate=2e-5,
#     weight_decay=0.01,
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=16,
#     fp16=True #Start off with this true 
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_dataset["train"],
#     eval_dataset=tokenized_dataset["test"],
#     data_collator=data_collator,
#     tokenizer=tokenizer,
# )

# if use_checkpoint:
#     trainer.train(resume_from_checkpoint=latest_checkpoint_path)
# else:
#     trainer.train()

# ## Evaluate
# eval_results = trainer.evaluate()
# print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
