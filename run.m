clear
close all

data_idx = 1;
cfg3_path = './configs/nc3.json';
cfg10_path = './configs/nc10.json';
pred_path = './pretrained/star10_kh10_n48/valid_predby_pretrained.mat';
model_path = './pretrained/star10_kh10_n48/pretrained';

%% Run the Gauss-Newton inverse solver with a pretrained model

% data_type = 'nn_stored'
starn_inverse_singlefreq(data_idx, 'nn_stored', pred_path);

% data_type = 'nn'
% requiring a file named "env_path.txt" in the current directory providing
% the location to python binary file, like the absolute path to
% '~/miniconda3/envs/invscatnn/bin/python'
% evaluate the code below

% starn_inverse_singlefreq(data_idx, 'nn', model_path);

% data_type = 'random', does not require a pretrained model, 
% evaluate the code below
% starn_inverse_singlefreq(data_idx, 'random', cfg3_path);

%% Run the inverse solver with linear sampling method
% evaluate the code below
% starn_lsm_fft(data_idx, cfg10_path);

%% Run the whole pipeline of data generation, training, inverse solving on a small dataset
% Step 1: data generation, evaluate the code below
% starn_data_generation(-1, cfg3_path);

% Step 2: training, run the python file train.py, making sure all the
% dependencies are installed

% Step 3: inverse solving, evalute the code below
% starn_inverse_singlefreq(1, 'nn', './data/star3_kh10_n48_100/test');
