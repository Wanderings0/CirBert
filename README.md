# CirBert Model
## Load model and data
We load Bert-base-uncased model and data from the Hugging Face Transformers library. On the server, we use huggingface-cli. Please run the following code in order that you have the corresponding model and data path with the code. More details can be found in the [HF-Mirror](https://hf-mirror.com/).

```bash
export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download --resume-download bert-base-uncased --local-dir ./model

huggingface-cli download --repo-type dataset --resume-download nyu-mll/glue --local-dir ./data

huggingface-cli download --repo-type dataset --resume-download  ag_news --local-dir ./data
```

## Code and Run

The CirBertForSequenceClassification model is implemented in the `CirBert.py` file. And please run the `run.sh` file to train the model and evaluate it.
Available dataset options are below(`stsb` and `ax` not implemented). You can change the dataset by changing the `dataset` variable in the `run.sh` file.

| Dataset | Train Size | Dev Size | Test Size | Task Type |
|---------|------------|----------|-----------|-----------|
| cola    | 8,551      | 1,043    | 1,063     | Single Sentence Classification |
| sst2   | 67,349     | 872      | 1,821     | Single Sentence Classification |
| mrpc    | 3,668      | 408      | 1,725     | Sentence Pair Classification |
| stsb   | 5,749      | 1,500    | 1,379     | Regression |
| qqp     | 363,849    | 40,430   | 390,965   | Sentence Pair Classification |
| mnli    | 392,702    | 9,815    | 9,832     | Sentence Pair Classification |
| qnli    | 104,743    | 5,463    | 5,463     | Sentence Pair Classification |
| rte     | 2,490      | 276      | 3,000     | Sentence Pair Classification |
| wnli    | 635        | 71       | 146       | Sentence Pair Classification |
| ax      | - (Question Answering) | 1,104   | 19,368    | Question Answering |

```bash
bash run.sh
```

## Multi-GPU Training
Transformers library supports multi-GPU training by using Trainer Class. I have implemented part of the multi-GPU training. But Some strange bug occurs when I run the `Trainer.py`. If you have any idea, please let me know. 
 






