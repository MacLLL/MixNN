# Mitigating Noisy Supervision Using Synthetic Samples with Soft Labels

This repository is the official implementation of "Mitigating Noisy Supervision Using Synthetic Samples with Soft Labels".


## Requirements
- Python 3.8.3
- Pytorch 1.8.1
- nmslib 2.1.1

## Usage
For example, to train the model using SELC under class-conditional noise in the paper, run the following commands:
```train
python3 train_cifar_with_MixNN.py
```
It can config with noise_mode, noise_rate, batch size and epochs. Similar commands can also be applied to other label noise scenarios.
### Hyperparameter options:
```
--data_path             path to the data directory
--noise_mode            label noise model(e.g. sym, asym)
--r                     noise level (0.0, 0.2, 0.4, 0.6, 0.8)
--loss                  loss functions (e.g. ANNLoss)
--alpha                 alpha in target estimation
--batch_size            batch size
--lr                    learning rate
--lr_s                  learning rate schedule
--op                    optimizer (e.g. SGD)          
--num_epochs            number of epochs
```


## Citing this work
If you use this code in your work, please cite the accompanying paper:
```
@inproceedings{lu2021mixnn,
  title={MixNN: Combating Noisy Labels in Deep Learning by Mixing with Nearest Neighbors},
  author={Lu, Yangdi and He, Wenbo},
  booktitle={2021 IEEE International Conference on Big Data (Big Data)},
  pages={847--856},
  year={2021},
  organization={IEEE}
}
```
