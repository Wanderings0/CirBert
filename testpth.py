
import torch

path1 = './model/bert-base-uncased/pytorch_model.bin'
path1 = './model_best/cir-bert-base-qqp.pth'
path2 = './model_best/bert-base-qqp.pth'

#加载参数
params1 = torch.load(path1,map_location='cpu')
updated_weights = {}
for name,parameters in params1.items():
    if 'LayerNorm.gamma' in name:
        name = name.replace('LayerNorm.gamma','LayerNorm.weight')
    if 'LayerNorm.beta' in name:
        name = name.replace('LayerNorm.beta','LayerNorm.bias')
    if 'cls' in name:
        continue
    updated_weights[name] = parameters
    # 给classifier层的权重赋初值
params1 = updated_weights
params2 = torch.load(path2,map_location='cpu')

#比较参数是否相同
for i in params1.keys():
    param1 = params1[i].to('cpu')
    param2 = params2[i].to('cpu')
    if not torch.equal(param1,param2):
        print(i)
        print('-------------------')

print(params1['bert.encoder.layer.11.output.dense.weight'])
print(params2['bert.encoder.layer.11.output.dense.weight'])
print(params1['bert.encoder.layer.11.output.dense.weight']-params2['bert.encoder.layer.11.output.dense.weight'])
