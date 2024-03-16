import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertConfig
from tqdm import tqdm
from CirBert import GetCirBertForSequenceClassification
import wandb, argparse,random,os
from datasets import load_dataset, load_from_disk

# TODO: 


# 3. train is slow, the bottleneck may be the trans_to_cir function.
# 4. better logging

# 6. bert-large-uncased

# 8. try Trainer



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
argparser.add_argument('--lr', type=float, default=2e-5)
argparser.add_argument('--batch_size', type=int, default=640)
argparser.add_argument('--num_epochs', type=int, default=1)
argparser.add_argument('--seed', type=int, default=42)
argparser.add_argument('--max_length', type=int, default=128)
argparser.add_argument('--num_workers', type=int, default=2)

args = argparser.parse_args()

config = BertConfig.from_pretrained('model/bert-base-uncased')

config.update(args.__dict__)
# print(config)
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
set_seed(config.seed)


# load raw dataset
dataset = load_dataset('data/ag_news/data')
# print(dataset)

# tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def preprocess_data(data):
    input_ids = []
    attention_masks = []
    labels = []
    for sample in data:
        # print(sample)
        text, label = sample['text'], sample['label']
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=config.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
        labels.append(label)
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.LongTensor(labels)
    # print(input_ids.shape, attention_masks.shape, labels.shape)
    return input_ids, attention_masks, labels

# prepare the data
train_inputs, train_masks, train_labels = preprocess_data(dataset['train'])
test_inputs, test_masks, test_labels = preprocess_data(dataset['test'])




# dataloader

train_data = DataLoader(
    list(zip(train_inputs, train_masks, train_labels)),
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=config.num_workers
)
test_data = DataLoader(
    list(zip(test_inputs, test_masks, test_labels)),
    batch_size=config.batch_size,
    shuffle=False,
    num_workers=config.num_workers
)
print(f'len of train_data is {len(train_data)}')
print(f'len of test_data is {len(test_data)}')
# print(train_data.dataset[0])
# save the train_data and test_data
# torch.save(train_data, 'train_data.pth')
# torch.save(test_data, 'test_data.pth')

# exit(0)

# read the train_data and test_data
# train_data = torch.load('train_data.pth')
# test_data = torch.load('test_data.pth')

config.num_labels = 4

model = GetCirBertForSequenceClassification(config,weights_path='./model/bert-base-uncased/pytorch_model.bin')
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
criterion = nn.CrossEntropyLoss()

def evaluate(model, test_loader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, masks, labels in test_loader:
            inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
            outputs = model(inputs, masks)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return total_loss/total, 100*correct / total
print("config is:")
print(config)
print("_______________________________")
wandb.init(project="CirBert", config=config)
wandb.run.name = f"agnews"

best_acc = 0.0
best_model_state = None
for epoch in range(config.num_epochs):
    model.train()
    total_loss = 0
    total = 0
    for inputs, masks, labels in tqdm(train_data):
        total+=1
        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs, masks)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        # if total % 20 == 0:
        #     print(f'Epoch {epoch+1}/{config.num_epochs}, Step {total}/{len(train_data)}, Train Loss: {loss.item():.4f}')
        
    avg_loss = total_loss / len(train_data)
    val_loss, val_acc = evaluate(model, test_data, device)
    print(f'Epoch {epoch+1}/{config.num_epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    wandb.log({"train_loss": avg_loss, "val_loss": val_loss, "val_acc": val_acc})
    if val_acc > best_acc:
        best_acc = val_acc
        best_model_state = model.state_dict()
if best_model_state is not None:
    if not os.path.exists('./model_best'):
        os.makedirs('./model_best')
    torch.save(best_model_state, f'./model_best/bert-base-best.pth')

wandb.finish()
