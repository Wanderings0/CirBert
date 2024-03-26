import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertConfig, Trainer, TrainingArguments, BertForSequenceClassification
from tqdm import tqdm
from CirBert import GetCirBertForSequenceClassification
import wandb, argparse, random, os
from datasets import load_dataset, load_from_disk
from utils import get_encoded_dataset

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True

argparser = argparse.ArgumentParser()
# different block size for different layers
argparser.add_argument('--block_size_selfattention', type=int, default=2)
argparser.add_argument('--block_size_attention_output', type=int, default=2)
argparser.add_argument('--block_size_intermediate', type=int, default=2)
argparser.add_argument('--block_size_output', type=int, default=2)

# whether to use circulate matrix for different layers
argparser.add_argument('--cir_selfattention', type=bool, default=True)
argparser.add_argument('--cir_attention_output', type=bool, default=True)
argparser.add_argument('--cir_intermediate', type=bool, default=True)
argparser.add_argument('--cir_output', type=bool, default=True)

# hyperparameters
argparser.add_argument('--lr', type=float, default=5e-5)
argparser.add_argument('--batch_size', type=int, default=8)
argparser.add_argument('--num_epochs', type=int, default=1)
argparser.add_argument('--seed', type=int, default=42)
argparser.add_argument('--max_length', type=int, default=32)
argparser.add_argument('--dataset', type=str, default='cola')

args = argparser.parse_args()

config = BertConfig.from_pretrained('model/bert-base-uncased')

config.update(args.__dict__)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
set_seed(config.seed)

# load raw dataset 
dataset = load_dataset('data/glue/cola')

# 只取dataset的10%作为训练集
dataset = dataset['validation'].train_test_split(test_size=0.1)

# tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# data preprocessing and tokenization
def preprocess1(examples):
    return tokenizer(examples['sentence'], truncation=True, max_length=config.max_length, padding='max_length')


encoded_dataset = dataset.map(preprocess1, batched=True)
# print(encoded_dataset)
config.num_labels = 2
# encoded_dataset,config.num_labels = get_encoded_dataset(config.dataset, tokenizer, config.max_length)

# dataloader
train_dataset = encoded_dataset['train']
test_dataset = encoded_dataset['test']

train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)


model = GetCirBertForSequenceClassification(config, weights_path='./model/bert-base-uncased/pytorch_model.bin')

# Initialize Weights & Biases
wandb.init(project="cirbert", name=f"{config.dataset}_cir{config.cir_attention_output}_block{config.block_size_attention_output}")

# Define the training arguments
training_args = TrainingArguments(
    output_dir=f'./results/{config.dataset}',
    learning_rate=config.lr,
    num_train_epochs=config.num_epochs,
    per_device_train_batch_size=config.batch_size,
    per_device_eval_batch_size=config.batch_size,
    gradient_accumulation_steps=4,
    seed=config.seed,
    weight_decay=1e-4,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    greater_is_better=True,
    report_to='wandb',
    auto_find_batch_size=True,
)



def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = (preds == labels).mean()
    return {'accuracy': acc}


# Create the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Evaluate the model
eval_result = trainer.evaluate()
print(f"Evaluation result: {eval_result}")

wandb.finish()