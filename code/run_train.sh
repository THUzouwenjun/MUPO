#!/bin/bash

# seeds=(12345 22345 32345 42345 52345)
seeds=(12345)

for seed in "${seeds[@]}"
do
  echo "Running with seed: $seed"
  python3 example_train/mupo_bypass.py --seed $seed
done
