% main
clear all
img_path = '/data/zwzhou/Data/MOT16/';               % dir of dataset
dataset_types={'train','test'};                 % dataset type
dataset_names={{'MOT16-02','MOT16-04','MOT16-05','MOT16-09','MOT16-10','MOT16-11','MOT16-13'},...
                {'MOT16-01','MOT16-03','MOT16-06','MOT16-07','MOT16-08','MOT16-12','MOT16-14'}};
detector_name = {'POI'};                        % different detection results 
close all
args.visualize = false;                         % flag of whether display the tracking results and generate corresponding video
args.res_path = './results/';                   % path for store results

args.gpu = true;
if args.gpu
    gpuDevice(1);
end
args.miss_th = 0.6;
args.conf_th = 0.2;
args.response_th = 0.2;
args.overlap_th = 0.2;
args.w1 = 0.5;
args.w2 = 0.5;
args.w3 = 0.6;
args.lambda = 1e-4;
args.padding = [1.0;2.0];
args.interp_factor = 0.3;
args.num_scale = 3;
args.scale_step = 1.0150; 
args.scale_penalty = 0.9925;
args.net_input_size = [150;65];

args.net_file = './model/DCFNet-net-7-125-2.mat';
args.det_conf_th = 0.1;
args.nms_th = 0.6;
args.min_length = 15;
args.consec_miss_length = 10;
args.max_miss_ratio = 0.6;
args.trk_confidence_th = 0.2;
cd amilan_mot_devkit;
compile;
cd ../
benchmarkGtDir =[img_path, 'train/'];
disp('======================================================')
i=1;

% path of detection results 
params = [];
for j=1:7             
    det_path = ['./detections/',detector_name{1}, '/', dataset_types{i},'/', dataset_names{i}{j},'.txt'];
    master([img_path, dataset_types{i}], det_path, dataset_names{i}{j}, args);
    if args.visualize
        visualize(img_path, dataset_types{i}, dataset_names{i}{j}, args.res_path);
    end
    disp('======================================')
    cd amilan_mot_devkit;
    allMets = evaluateTracking(['a1-' num2str(j) '.txt'],'../results/', benchmarkGtDir,'MOT16');
    params = [params; det_conf_th, nms_th, min_length, consec_miss_length, conf_th, overlap_th,...
            response_th, allMets.m(end-2:end)];
    disp('+++++++++++++++++++++++++++++++++++++++')
    cd ../
end    

save(dataset_names{i}{j},'params');
[~,idx]=max(params(:,end));
best_param = params(idx,:);
args.det_conf_th = best_param(1);
args.nms_th = best_param(2);
args.min_length = best_param(3);
args.consec_miss_length = best_param(4);
args.conf_th = best_param(5);
args.overlap_th = best_param(6);
args.response_th = best_param(7);
det_path = ['./detections/',detector_name{1}, '/', dataset_types{2},'/', dataset_names{2}{j},'.txt'];
master([img_path, dataset_types{i}], det_path, dataset_names{2}{j}, args);
end

               % better params
% dataset         det_conf_th       nms_th     min_length  consec_miss_length    
% MOT16-02
% MOT16-04
% MOT16-05
% MOT16-09
% MOT16-10
% MOT16-11
% MOT16-13


% test the tracking result on train dataset



