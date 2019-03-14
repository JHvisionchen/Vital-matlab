clear all;
clc;

addpath('./utils');
addpath('./models');
addpath('./vital');
addpath('./tracking');

run ./matconvnet/matlab/vl_setupnn ;

global gpu;
gpu=true;
    
% path to the folder with OTB sequences
base_path = '/media/cjh/datasets/tracking/OTB100/';
% choose name of the OTB sequence
sequence_name = choose_video(base_path);

test_seq=sequence_name;
conf = genConfig('otb',test_seq,base_path);

net=fullfile('./models/otbModel.mat');

result = vital_run(conf.imgList, conf.gt(1,:), net, true);


    


