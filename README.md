<div align="center">

# A Neural Network Warm-Start Approach for the Inverse Acoustic Obstacle Scattering Problem

Mo Zhou, Jiequn Han, Manas Rachh, Carlos Borges

[![Journal](https://img.shields.io/badge/JCP-2023-19b547.svg)](https://doi.org/10.1016/j.jcp.2023.112341)
[![arXiv](http://img.shields.io/badge/arXiv-2212.08736-b31b1b.svg)](https://arxiv.org/abs/2212.08736)

</div>


## Dependencies
* MATLAB (for running Gauss-Newton inverse solver). Make sure the folder ``starn`` is in your Matlab search path, set by running ``startup.m``
* Python (for training your own model and evaluation). Quick installation of conda environment: ``conda env create -f environment.yml``

## Running
All the commands below can also be found in ``run.m``. They are all run in MATLAB, except the training of model.
### Run the Gauss-Newton inverse solver with a pretrained model
Solve an inverse problem whose initial guess provided by a pretrained model is stored already
```
starn_inverse_singlefreq(1, 'nn_stored', './pretrained/star10_kh10_n48/valid_predby_pretrained.mat');
```

Solve an inverse problem whose initial guess is obtained by calling a pretrained model (:warning: requiring a file named "env_path.txt" in the current directory providing the location to python binary file, like the absolute path to '~/miniconda3/envs/invscatnn/bin/python')
```
starn_inverse_singlefreq(1, 'nn', './pretrained/star10_kh10_n48/pretrained');
```

Solve an inverse problem with zero initial guess
```
starn_inverse_singlefreq(1, 'random', './configs/nc3.json');
```
### Run the linear sampling method to solve an inverse problem
```
starn_lsm_fft(1, './configs/nc10.json');
```
### Run the whole pipeline of data generation, training, inverse solving on a small dataset
Step 1: data generation
```
starn_data_generation(-1, './configs/nc3.json');
```
Step 2: training, run the python file ``train.py``, making sure all the dependencies are installed

Step 3: inverse solving
```
starn_inverse_singlefreq(1, 'nn', './data/star3_kh10_n48_100/test');
```

## Citation
If you find this work helpful, please consider starring this repo and citing our paper using the following Bibtex.
```bibtex
@article{zhou2023neural,
  title={A Neural Network Warm-Start Approach for the Inverse Acoustic Obstacle Scattering Problem},
  author={Zhou, Mo and Han, Jiequn and Rachh, Manas and Borges, Carlos},
  journal={Journal of Computational Physics},
  pages={112341},
  year={2023},
  publisher={Elsevier}
}
```
