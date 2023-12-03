# AA-VAE-PyTorch
PyTorch Code for the paper "Adversarial and Attentive Variational Sequential Recommendation: An Information Bottleneck Perspective". 

Submitted to ICDE 2024.
Paper authors: Yue Ding, Yifan Zhou, Zhe Xie, Chang Liu, Chengxuan Liu, Hongtao Lu


## Usage
- python 3.6+
- PyTorch
- tqdm
- tensorboardX
- numpy

Run `train.py`:

```
python3 train.py
```

The defaut dataset is the Yelp sample dataset . If you want to train this model on your own datasets, you can save your preprocessed dataset files under `datasets/`. You also need to add one item in `dataset_info.json`, which contains the information of the count of users and items as well as the `seq_len` to use in the model.
