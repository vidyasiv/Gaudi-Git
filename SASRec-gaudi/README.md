#### Datasets:

We implement `data_preprocess.py` and `utils.py` for Amazon 2023 datasets.

By running `main.py`, the datasets are automatically downloaded and preprocessed.


#### Model Training:

E.g., train SASRec on `All_Beauty`
```
python main.py --dataset All_Beauty --maxlen 10 --device hpu
```

- **Issue**: When using the HPU, the loss does not decrease during training.

#### Run (CPU):
```
python main_rawdata.py --dataset All_Beauty --maxlen 10 --device cpu
```

- **Results**: Training works correctly on the CPU, with the loss decreasing as expected.

#### Run (HPU with `nn.Parameter`):
```
python main_rawdata.py --dataset All_Beauty --maxlen 10 --device hpu --nn_parameter
```

- **Results**: When using `nn.Parameter` instead of `nn.Embedding`, training works correctly on the HPU, and the loss decreases as expected.

