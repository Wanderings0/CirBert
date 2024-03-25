#! /bin/bash

# Run this command you can get the result in output.txt
python train.py --lr 2e-5 --batch_size 640 --num_epochs 1 --max_length 32 --dataset agnews --seed 42

# Run this command you can get the result in output2.txt

# python train.py --lr 2e-5 --batch_size 640 --num_epochs 1 --max_length 32 --dataset agnews --seed 42 --cir_selfattention False --cir_attention_output False --cir_intermediate False --cir_output False

# the other datasets may need to adjust the hyperparameters, especially the batch_size and max_length,(64,256) may be a good choice

# mrpc
# python train.py --lr 2e-5 --batch_size 32 --num_epochs 5 --max_length 512 --dataset mrpc --seed 42 --device 6 > results/mrpc_cir.txt

# mnli
# python train.py --lr 2e-5 --batch_size 32 --num_epochs 1 --max_length 512 --dataset mnli --seed 42 --device 4 > results/mnli_cir.txt

# cola
# python train.py --lr 5e-5 --batch_size 128 --num_epochs 5 --max_length 128 --dataset cola --seed 42 --device 2 > results/cola_cir.txt 