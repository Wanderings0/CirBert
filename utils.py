from datasets import load_dataset

def get_encoded_dataset(dataset_name, tokenizer, max_length):
    match dataset_name:
        case 'agnews':
            dataset = load_dataset('data/ag_news/data')
            class_nums = 4
            def preprocess(examples):
                return tokenizer(examples['text'], truncation=True, max_length=max_length, padding='max_length')
            encoded_dataset = dataset.map(preprocess, batched=True)
            return encoded_dataset, class_nums
        case 'cola':
            dataset = load_dataset('data/glue/cola')
            class_nums = 2
            def preprocess(examples):
                return tokenizer(examples['sentence'], truncation=True, max_length=max_length, padding='max_length')
            encoded_dataset = dataset.map(preprocess, batched=True)
            return encoded_dataset, class_nums
        case 'mnli':
            dataset = load_dataset('data/glue/mnli')
            class_nums = 3
            def preprocess(examples):
                return tokenizer(examples['premise'], examples['hypothesis'], truncation=True, max_length=max_length, padding='max_length')
            encoded_dataset = dataset.map(preprocess, batched=True)
            return encoded_dataset, class_nums
        case 'mrpc':
            dataset = load_dataset('data/glue/mrpc')
            class_nums = 2
            def preprocess(examples):
                return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, max_length=max_length, padding='max_length')
            encoded_dataset = dataset.map(preprocess, batched=True)
            return encoded_dataset, class_nums
        case 'qnli':
            dataset = load_dataset('data/glue/qnli')
            class_nums = 2
            def preprocess(examples):
                return tokenizer(examples['question'], examples['sentence'], truncation=True, max_length=max_length, padding='max_length')
            encoded_dataset = dataset.map(preprocess, batched=True)
            return encoded_dataset, class_nums
        case 'qqp':
            dataset = load_dataset('data/glue/qqp')
            class_nums = 2
            def preprocess(examples):
                return tokenizer(examples['question1'], examples['question2'], truncation=True, max_length=max_length, padding='max_length')
            encoded_dataset = dataset.map(preprocess, batched=True)
            return encoded_dataset, class_nums
        case 'rte':
            dataset = load_dataset('data/glue/rte')
            class_nums = 2
            def preprocess(examples):
                return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, max_length=max_length, padding='max_length')
            encoded_dataset = dataset.map(preprocess, batched=True)
            return encoded_dataset, class_nums
        case 'sst2':
            dataset = load_dataset('data/glue/sst2')
            class_nums = 2
            def preprocess(examples):
                return tokenizer(examples['sentence'], truncation=True, max_length=max_length, padding='max_length')
            encoded_dataset = dataset.map(preprocess, batched=True)
            return encoded_dataset, class_nums
        # stsb is regression task
        # case 'stsb':
        #     dataset = load_dataset('data/glue/stsb')
        #     class_nums = 1
        #     def preprocess(examples):
        #         return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, max_length=max_length, padding='max_length')
        #     encoded_dataset = dataset.map(preprocess, batched=True)
        #     return encoded_dataset, class_nums
        case 'wnli':
            dataset = load_dataset('data/glue/wnli')
            class_nums = 2
            def preprocess(examples):
                return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, max_length=max_length, padding='max_length')
            encoded_dataset = dataset.map(preprocess, batched=True)
            return encoded_dataset, class_nums
        case _:
            raise ValueError(f"Dataset {dataset_name} is not supported.")
        
if __name__ == '__main__':
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encoded_dataset, class_nums = get_encoded_dataset('agnews', tokenizer, 256)
    print(encoded_dataset)
    print(class_nums)
    encoded_dataset, class_nums = get_encoded_dataset('cola', tokenizer, 256)
    print(encoded_dataset)
    print(class_nums)
    
    # encoded_dataset, class_nums = get_encoded_dataset('not_supported', tokenizer, 256)
    # print(encoded_dataset)
    # print(class_nums)

