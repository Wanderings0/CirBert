import math,time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertEmbeddings, BertPooler

# TODO: 


#       4. if the out_features%block_size!=0, we need to do extra work
#       5. BertForSequencetoSequence need decoder part

class CirMatrix(nn.Module):
    def __init__(self,in_features,out_features,block_size=2,cir=False,weight=None):
        super(CirMatrix,self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.cir = cir
        self.block_size = block_size
        if weight is None:
            self.weight = nn.Parameter(torch.zeros(out_features,in_features))
            self.bias = nn.Parameter(torch.zeros(out_features))
            init.kaiming_uniform_(self.weight)
            init.zeros_(self.bias)
        else:
            self.weight, self.bias = weight
            

    def trans_to_cir(self):
        q = self.out_features // self.block_size
        p = self.in_features // self.block_size
        assert self.out_features % self.block_size == 0
        assert self.in_features % self.block_size == 0
        tmp = self.weight.reshape(q, self.block_size, p, self.block_size)
        tmp = tmp.permute(0, 2, 1, 3)
        w = torch.zeros(q, p, self.block_size, self.block_size)
        tmp_compress = torch.zeros(q,p,self.block_size)
        for i in range(self.block_size):
            diagonal = torch.diagonal(tmp,offset=i,dim1=2,dim2=3)
            if i>0:
                part_two = torch.diagonal(tmp,offset=-self.block_size+i,dim1=2,dim2=3)
                diagonal = torch.cat([diagonal,part_two],dim=2)
            # diagonal.shape (q,p,block_size)
            mean_of_diagonal = diagonal.mean(dim=2)
            # mean_of_diagonal.shape (q,p)
            tmp_compress[:,:,i] = mean_of_diagonal
        for i in range(self.block_size):
            w[:,:,:,i] = tmp_compress.roll(shifts=i,dims=2)

        # TODO: 
        # if the out_features%block_size!=0, we need to do extra work
            
        w = w.permute(0,2,1,3).reshape(self.out_features,self.in_features)

        return w
    
    def forward(self,x):
        if self.cir:
            weight = self.trans_to_cir().to(x.device)
        else:
            weight = self.weight.to(x.device)
        return torch.matmul(x,weight.t())
    
class CirSelfAttention(nn.Module):
    def __init__(self, config):
        super(CirSelfAttention, self).__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = CirMatrix(config.hidden_size, self.all_head_size,block_size=config.block_size_selfattention,cir=config.cir_selfattention)
        self.key = CirMatrix(config.hidden_size, self.all_head_size,block_size=config.block_size_selfattention,cir=config.cir_selfattention)
        self.value = CirMatrix(config.hidden_size, self.all_head_size,block_size=config.block_size_selfattention,cir=config.cir_selfattention)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states, attention_mask):

        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        # print(f'mixed_query_layer.shape: {mixed_query_layer.shape}')
        # print(f'mixed_key_layer.shape: {mixed_key_layer.shape}')
        # print(f'mixed_value_layer.shape: {mixed_value_layer.shape}')

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        # print(f'query_layer.shape: {query_layer.shape}')
        # print(f'key_layer.shape: {key_layer.shape}')
        # print(f'value_layer.shape: {value_layer.shape}')
        
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # print(f'attention_scores.shape: {attention_scores.shape}')
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        # print(f'attention_mask.shape: {attention_mask.shape}')
        attention_scores = attention_scores + attention_mask
        
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer
    
class CirBertAttention(nn.Module):
    def __init__(self, config):
        super(CirBertAttention, self).__init__()
        self.self = CirSelfAttention(config)
        self.output = CirMatrix(config.hidden_size, config.hidden_size,block_size=config.block_size_attention_output,cir=config.cir_attention_output)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output)
        attention_output = self.LayerNorm(attention_output + input_tensor)
        attention_output = self.dropout(attention_output)
        return attention_output
    
class CirIntermediate(nn.Module):
    def __init__(self, config):
        super(CirIntermediate, self).__init__()
        self.dense = CirMatrix(config.hidden_size, config.intermediate_size,block_size=config.block_size_intermediate,cir=config.cir_intermediate)
        # if isinstance(config.hidden_act, str):
        #     self.intermediate_act_fn = ACT2FN[config.hidden_act]
        # else:
        #     self.intermediate_act_fn = config.hidden_act
        self.intermediate_act_fn = nn.functional.gelu

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states
    
class CirOutput(nn.Module):
    def __init__(self, config):
        super(CirOutput, self).__init__()
        self.dense = CirMatrix(config.intermediate_size, config.hidden_size,block_size=config.block_size_output,cir=config.cir_output)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
    
class CirBertLayer(nn.Module):
    def __init__(self, config):
        super(CirBertLayer, self).__init__()
        self.attention = CirBertAttention(config)
        self.intermediate = CirIntermediate(config)
        self.output = CirOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
    
class CirBertEncoder(nn.Module):
    def __init__(self, config):
        super(CirBertEncoder, self).__init__()
        self.layer = nn.ModuleList([CirBertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers
    
class CirBertModel(nn.Module):
    def __init__(self, config):
        super(CirBertModel, self).__init__()
        self.embeddings = BertEmbeddings(config)
        self.encoder = CirBertEncoder(config)
        self.pooler = BertPooler(config)
        self.config = config

    # def load_weight(self,pretrained_weights):
    #     #load embedding layer
    #     self.embeddings.word_embeddings.weight = pretrained_weights['word_embeddings.weight']

    def forward(self, input_ids, attention_mask,token_type_ids=None):
        embedding_output = self.embeddings(input_ids=input_ids,token_type_ids=token_type_ids)
        all_encoder_layers = self.encoder(embedding_output, attention_mask)
        sequence_output = all_encoder_layers[-1]
        pooled_output = self.pooler(sequence_output)
        return sequence_output, pooled_output

class CirBertForSequenceClassification(nn.Module):
    def __init__(self, config):
        super(CirBertForSequenceClassification, self).__init__()
        self.bert = CirBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        # print(self.classifier.weight.shape)
        self.config = config

    def forward(self, input_ids, attention_mask, token_type_ids=None,labels=None):
        _,pooled_output = self.bert(input_ids, attention_mask,token_type_ids)
        sequence_output = self.dropout(pooled_output)
        logits = self.classifier(sequence_output)
        outputs = (logits,)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            outputs = (loss,)+ outputs
        return outputs
    
def weight_keys_pair(pretrained_weights,model):
    # print unpaired keys after necessary changes 
    # so that we can use load_state_dict(strict=False)
    updated_weights = {}
    for name,parameters in pretrained_weights.items():
        if 'LayerNorm.gamma' in name:
            name = name.replace('LayerNorm.gamma','LayerNorm.weight')
        if 'LayerNorm.beta' in name:
            name = name.replace('LayerNorm.beta','LayerNorm.bias')
        if 'attention.output.dense' in name:
            name = name.replace('attention.output.dense','attention.output')
        if 'attention.output.LayerNorm' in name:
            name = name.replace('attention.output.LayerNorm','attention.LayerNorm')
        updated_weights[name] = parameters

    for i,j in zip(model.state_dict().keys(),updated_weights.keys()):
        if i!=j:
            print(f'{i} in model and {j} in pretrained_weights unpaired!')


def GetCirBertForSequenceClassification(config,weights_path=None):
    model = CirBertForSequenceClassification(config)
    if weights_path is not None:
        pretrained_weights = torch.load(weights_path)
        weight_keys_pair(pretrained_weights,model)
        model.load_state_dict(pretrained_weights,strict=False)
        # handle the classifier layer weights with initial value
        init.kaiming_uniform_(model.classifier.weight)
        init.zeros_(model.classifier.bias)
        print('weight loaded!')
    return model


if __name__ == "__main__":
    # test CirMatrix
    cir = CirMatrix(4,4,block_size=2,cir=True)
    print(cir.weight.shape)
    print(cir.weight)
    print(cir.trans_to_cir())
    exit(0)

    # test CirBertModel
    config = BertConfig.from_pretrained('./model/bert-large-uncased')
    config.block_size_selfattention = 2
    config.block_size_attention_output = 2
    config.block_size_intermediate = 2
    config.block_size_output = 2


    config.cir_attention_output = True
    config.cir_selfattention = True
    config.cir_intermediate = True
    config.cir_output = True

    config.num_labels = 5

    model = CirBertForSequenceClassification(config)
    # print(config)
    
    # print(model.bert.embeddings)

    # 
    # for name,parameters in model.named_parameters():
    #     print(name,':',parameters.size())


    # input_ids = torch.tensor([[1,2,3,4,5],[1,2,3,4,5]])
    # attention_mask = torch.tensor([[1,1,1,1,1],[1,1,1,1,1]])

    # sequence_output= model(input_ids,attention_mask)
    # print(sequence_output.shape)


    # load pretrained weights from huggingface
    pretrained_weights = torch.load('./model/bert-large-uncased/pytorch_model.bin')

    # to see where unpaired
    weight_keys_pair(pretrained_weights,model)

    model.load_state_dict(pretrained_weights,strict=False)




    print('weight loaded!')

    #test trans_to_cir()
    weight = model.bert.encoder.layer[0].intermediate.dense
    print(weight.weight)
    print(weight.trans_to_cir())


    

    
    

    