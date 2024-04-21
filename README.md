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


## Hyperparameter Tuning

| Dataset | Batch Size | Max Length | Learning Rate | Epoch | Circular | Accuracy | Ref  Accuracy | Data Size |
|---------|------------|------------|---------------|-------|----------|----------|---------------|-----------|
|mrpc     | 16         | 512        |4e-5           | 2     | No       | 0.8441   |0.8407 |3.6k|
|mrpc     | 32         | 512        |4e-5           | 2     | Yes      | 0.6765   |-      |3.6k|
|agnews   | 64         | 32         |2e-5           | 2     | No       | 0.9357   |0.9475 |120k|
|agnews   | 64         | 32         |2e-5           | 2     | Yes      | 0.9111   |-      |120k|
|qnli     | 32         | 512        |2e-5           | 1     | No       | 0.9083   |0.9066 |105k|
|qnli     | 32         | 512        |1e-5           | 3     | Yes      | 0.8080   |-      |105k|
|qnli     | 16         | 512        |2e-5           | 3     | Yes      | 0.8149   |-      |105k|
|qnli     | 32         | 512        |3e-5           | 3     | Yes      | 0.8224   |-      |105k|
|qnli     | 32         | 512        |4e-5           | 3     | Yes      | 0.8230   |-      |105k|
|qqp      | 32         | 512        |2e-5           | 1     | No       | 0.8970   |0.9071 |363k|
|qqp      | 32         | 512        |2e-5           | 1     | Yes      | 0.8392   |-      |363k|
|wnli     | 32         | 512        |4e-5           | 1     | No (No scheduler)   | 0.5634   |0.5634 |0.63k|
|wnli     | 32         | 512        |4e-5           | 1     | Yes      | 0.5634   |-      |0.63k|
|rte      | 32         | 512        |4e-5           | 3     | No       | 0.6968   |0.6570 |2.5k|
|rte      | 32         | 512        |4e-5           | 3     | Yes      | 0.5271   |-      |2.5k|



python train.py --lr 2e-4 --batch_size 32 --max_length 512 --num_epochs 10 --dataset rte --device 4 --cir_selfattention 1 --cir_attention_output 1 --cir_intermediate 1 --cir_output 1 > out/rte2.txt

 






