import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertConfig
from tqdm import tqdm
from CirBert import GetCirBertForSequenceClassification
import wandb, argparse,random,os
from datasets import load_dataset, load_from_disk
from utils import get_encoded_dataset


device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


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
argparser.add_argument('--batch_size', type=int, default=128)
argparser.add_argument('--num_epochs', type=int, default=3)
argparser.add_argument('--seed', type=int, default=42)
argparser.add_argument('--max_length', type=int, default=128)
argparser.add_argument('--dataset', type=str, default='cola')
argparser.add_argument('--device', type=int, default=2)



args = argparser.parse_args()

config = BertConfig.from_pretrained('model/bert-base-uncased')

config.update(args.__dict__)
# print(config)
device = torch.device("cuda:"+str(config.device) if torch.cuda.is_available() else "cpu")

set_seed(config.seed)

# tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# load raw dataset
encoded_dataset, config.num_labels = get_encoded_dataset(config.dataset, tokenizer, config.max_length)

train_dataset = encoded_dataset['train']
if config.dataset == 'mnli':
    validation_dataset = encoded_dataset['validation_mathched']
    test_dataset = encoded_dataset['test_matched']
else:
    validation_dataset = encoded_dataset['validation']
    test_dataset = encoded_dataset['test']


train_data = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
validation_data = DataLoader(validation_dataset, batch_size=config.batch_size, shuffle=False)
test_data = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)


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
        for data in tqdm(test_loader):
            inputs = torch.stack(data['input_ids'],dim=1).to(device)
            masks = torch.stack(data['attention_mask'],dim=1).to(device)
            token_type_ids = torch.stack(data['token_type_ids'],dim=1).to(device)
            labels = data['label'].to(device)
            outputs = model(inputs, masks,token_type_ids)[-1]
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return total_loss/total, 100*correct / total
print("The Config is:")
print(config)
print("-"*20)
wandb.init(project="CirBert", config=config)
wandb.run.name = f"{config.dataset}"

best_acc = 0.0
best_model_state = None
for epoch in range(config.num_epochs):
    model.train()
    total_loss = 0
    total = 0
    for data in tqdm(train_data):
        total+=1

        inputs = torch.stack(data['input_ids'],dim=1).to(device)
        masks = torch.stack(data['attention_mask'],dim=1).to(device)
        token_type_ids = torch.stack(data['token_type_ids'],dim=1).to(device)

        labels = data['label'].to(device)
        optimizer.zero_grad()
        outputs = model(inputs, masks,token_type_ids)[-1]
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        # if total % 20 == 0:
        #     print(f'Epoch {epoch+1}/{config.num_epochs}, Step {total}/{len(train_data)}, Train Loss: {loss.item():.4f}')
        
    avg_loss = total_loss / len(train_data)
    val_loss, val_acc = evaluate(model, validation_data, device)
    print(f'Epoch {epoch+1}/{config.num_epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    wandb.log({"train_loss": avg_loss, "val_loss": val_loss, "val_acc": val_acc})
    if val_acc > best_acc:
        best_acc = val_acc
        best_model_state = model.state_dict()
if best_model_state is not None:
    if not os.path.exists('./model_best'):
        os.makedirs('./model_best')
    torch.save(best_model_state, f'./model_best/bert-base-{config.dataset}.pth')

test_loss, test_acc = evaluate(model, test_data, device)
print("-"*20)
print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, Best Val Acc: {best_acc:.2f}%')

wandb.finish()
