import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from tqdm import tqdm

# 加载数据集
from torchtext.datasets import AG_NEWS
train_dataset, test_dataset = AG_NEWS(root='.data', split=('train', 'test'))

# 创建分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 数据预处理
def preprocess_data(data):
    input_ids = []
    attention_masks = []
    labels = []
    for sample in data:
        text, label = sample.text, sample.label
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
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
    return input_ids, attention_masks, labels

# 准备数据集
train_inputs, train_masks, train_labels = preprocess_data(train_dataset)
test_inputs, test_masks, test_labels = preprocess_data(test_dataset)

# 创建数据加载器
batch_size = 32
train_data = DataLoader(
    list(zip(train_inputs, train_masks, train_labels)),
    batch_size=batch_size,
    shuffle=True
)
test_data = DataLoader(
    list(zip(test_inputs, test_masks, test_labels)),
    batch_size=batch_size,
    shuffle=False
)

# 加载预训练权重和配置
pretrained_weights = torch.load('bert-base-uncased.pth')
config = BertConfig(
    vocab_size=30522,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    max_position_embeddings=512,
    type_vocab_size=2,
    layer_norm_eps=1e-12,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    block_size=4
)

# 创建模型和分类头
model = CirBertModel(config, pretrained_weights)
classifier = nn.Linear(config.hidden_size, 4)  # AGNews有4个类别

# 优化器和损失函数
optimizer = torch.optim.AdamW(list(model.parameters()) + list(classifier.parameters()), lr=2e-5)
criterion = nn.CrossEntropyLoss()

# 训练函数
def train_epoch(model, classifier, data_loader, optimizer, criterion):
    model.train()
    classifier.train()
    total_loss = 0
    for batch in tqdm(data_loader, total=len(data_loader)):
        input_ids, attention_mask, labels = batch
        outputs = model(input_ids, attention_mask)
        logits = classifier(outputs[:, 0])  # 只使用[CLS]的输出
        loss = criterion(logits, labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return total_loss / len(data_loader)

# 测试函数
def test_epoch(model, classifier, data_loader):
    model.eval()
    classifier.eval()
    total_loss = 0
    total_correct = 0
    with torch.no_grad():
        for batch in tqdm(data_loader, total=len(data_loader)):
            input_ids, attention_mask, labels = batch
            outputs = model(input_ids, attention_mask)
            logits = classifier(outputs[:, 0])
            loss = criterion(logits, labels)
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            total_correct += (predictions == labels).sum().item()
    accuracy = total_correct / len(data_loader.dataset)
    return total_loss / len(data_loader), accuracy

# 训练循环
num_epochs = 3
best_accuracy = 0
for epoch in range(num_epochs):
    train_loss = train_epoch(model, classifier, train_data, optimizer, criterion)
    test_loss, test_accuracy = test_epoch(model, classifier, test_data)
    print(f'Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}, Test Accuracy = {test_accuracy:.4f}')
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        torch.save(model.state_dict(), 'best_model.pth')
        torch.save(classifier.state_dict(), 'best_classifier.pth')