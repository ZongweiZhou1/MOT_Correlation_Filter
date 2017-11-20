% MOTracker 
function master(img_path, det_path, dataset_name, args)
    % params: det_path, it should be a text file and each line should like 'frameId, -1, x, y, w, h, confidence, ...' 
    % each line corresponding to a detection
 
    % load the det data
    det_records = dlmread(det_path, ',');
    det_records = det_records(det_records(:,7)>=args.det_conf_th,:);
    
    % load image file names
    img_names = dir([img_path,'/',dataset_name,'/img1/*.jpg']);
    img_paths = fullfile(img_path, dataset_name, 'img1', {img_names.name});
    end_frameId = max(det_records(:,1));                                   % all videos should  start from 1 and end at end_frameId
    
    MOTracker = init_MOTracker(args);                                      % initialize some common perportities of the trackers
    if MOTracker.gpu
        MOTracker.net = vl_simplenn_move(MOTracker.net, 'gpu');            % if gpu is used, net should be move on gpu env
    end
                                                                           % for reduce the storage occupied, read the imgs in batch
    res=[];
    for i= 1:end_frameId
        % read batch imgs in multi threads
        ims = vl_imreadjpeg(img_paths(i),'numThreads',4);
        if MOTracker.gpu
            frames = gpuArray(ims{1});
        end
        featmap = vl_simplenn(MOTracker.net, frames);
        featmap = featmap(end).x;        
        frameId=i;                 
        fprintf([num2str(i),' : ']);  
        if mod(frameId,100)==0
            fprintf('\n');
        end
        dets = det_records(det_records(:,1)==frameId,:);                   % find detections in current frame
        dets = preprocess(dets, MOTracker.nms_th);                         % here, we just use NMS on the detections, 
        % extract feature
        if frameId==1                                               
            % initialize trackers
            MOTracker = AddTracker(MOTracker, dets, featmap, frameId);     % create SOTracker for every valid det in first frame.
        else
            % update MOTracker
            [res, MOTracker] = UpdateTracker(MOTracker, dets, featmap,frameId,res); % update exist SOTrackers or add new SOTrackers
        end
        
    end

    for i=1:size(MOTracker.trackers,2)
       res = [res; saveTrajectory(MOTracker, MOTracker.trackers{i})];
    end
    dlmwrite(['results/', dataset_name, '.txt'], res);
end

function MOTracker = init_MOTracker(args)
    MOTracker={};
    MOTracker.trackerNum=0;                         % number of SOTrackers
    
    tracking_env();                                 % set env variables
    cur_path = fileparts(mfilename('fullpath'));
    net_file = args.net_file;
    net = load(fullfile(cur_path, net_file));
    MOTracker.net = vl_simplenn_tidy(net.net);      % load the base net used to extract deep features
    MOTracker.gpu=args.gpu;                         % whether gpu is used
    
    param={};                                       % params for identity SOTracker, which can be further assigned indivitually
    param.gpu = args.gpu;
    param.interp_factor = args.interp_factor;       % learning rate to update the regression weight
    param.miss_th = args.miss_th;
    param.conf_th = args.conf_th;
    param.consec_miss_length = args.consec_miss_length;
    param.response_th = args.response_th;
    param.overlap_th = args.overlap_th;
    param.w1 = args.w1;
    param.w2 = args.w2;
    param.w3 = args.w3;
    param.lambda = args.lambda;
    param.padding = args.padding;
    param.num_scale = args.num_scale;
    param.scale_step = args.scale_step;
    param.net_input_size = args.net_input_size;
    param.scale_penalty = args.scale_penalty;
    
    MOTracker.param = param;
    MOTracker.det_conf_th = args.det_conf_th;
    MOTracker.nms_th = args.nms_th;
    MOTracker.min_length = args.min_length;
    MOTracker.max_miss_ratio = args.max_miss_ratio;
    MOTracker.trk_confidence_th = args.trk_confidence_th;
    MOTracker.trackers=[];    
end

function dets = preprocess(dets, nms_thresh)
    % NMS for the detections, ofcourse many more preprocess can added
    % NMS
    type = '';
    if isempty(dets)
        dets=[];
        return
    end
    
    x1 = dets(:,3);
    y1 = dets(:,4);
    x2 = x1+dets(:,5);
    y2 = y1+dets(:,6);
    s = dets(:,7);
    
    area = (x2-x1+1).*(y2-y1+1);
    [~, I] = sort(s);
    pick = zeros(size(dets,1));
    counter = 1;
    while ~isempty(I)
        last = length(I);
        i = I(last);
        pick(counter)=i;
        counter = counter+1;
        
        xx1 = max(x1(i), x1(I(1:last-1)));
        yy1 = max(y1(i), y1(I(1:last-1)));
        xx2 = min(x2(i), x2(I(1:last-1)));
        yy2 = min(y2(i), y2(I(1:last-1)));
        w = max(0.0, xx2-xx1+1);
        h = max(0.0, yy2-yy1+1);
        inter = w.*h;
        
        if strcmp(type, 'MIN')
            o = inter ./ min(area(i), area(I(1:last-1)));
        else
            o = inter ./ (area(i) + area(I(1:last-1)) - inter);
        end
        
        I = I(o <= nms_thresh);
    end
    pick = pick(1:(counter-1));
    temp=[];
    for i=1:numel(pick)
        temp = [temp; dets(pick(i),:)];
    end
    dets = temp;
end

function MOTracker = AddTracker(MOTracker, dets, featmap, frameId)
    param = MOTracker.param;
    for i=1:size(dets,1)
        param.region = dets(i, 3:6);                         % box of ROI 
        param.confidence = dets(i,7);                        % regard the detection confidence as trajectory confidence in first time.
        param.trackId = MOTracker.trackerNum +1;             % track id assigned to this Tracker
        param.frameId = frameId;                             
        MOTracker.trackerNum = MOTracker.trackerNum + 1;
        % create new SOTracker
        MOTracker.trackers{size(MOTracker.trackers,2)+1} = SOTracker(featmap, param);        
    end
end

function [res,MOTracker] = UpdateTracker(MOTracker, dets, featmap, frameId, res)
    det_flags = ones(size(dets,1),1);               % flag vector, 1: not assigned, 0: assigned
    % sort the tracker according to confidence and assign dets to tracker by greed search
    trackerInfo=[];                                 % each row store two value: index, confidence
    dead_trackers=[];
    for i=1:size(MOTracker.trackers,2)
        if MOTracker.trackers{i}.activated == 1
            trackerInfo=[trackerInfo;i, MOTracker.trackers{i}.confidence]; %#ok<*AGROW>           
        else
            res = [res;saveTrajectory(MOTracker, MOTracker.trackers{i})];
            dead_trackers=[dead_trackers,i];
        end
    end

    fprintf(['length of MOTracker:' num2str(size(MOTracker.trackers))]);
    disp('=======')
    [~, argsort] = sort(trackerInfo(:,2), 'descend');% sort according the trajectory confidence
    searchTrackerIds = trackerInfo(argsort,1);       % tracker with higher confidence first
    for i = 1:size(searchTrackerIds,1)
        % find possible dets in cur trackers neighborhood
        tid = searchTrackerIds(i);
        
        ref =MOTracker.trackers{tid}.region;        % row vector
        [nbh_dets, nbh_dets_index] = findNeighborhood(ref, dets, det_flags);
        if isempty(nbh_dets)
            cand_dets = [];
        else
            cand_dets = nbh_dets(:,3:6);
        end
        [MOTracker.trackers{tid}, idx] = MOTracker.trackers{tid}.SOTracker_Update(featmap, ...
                                                        cand_dets, frameId, nbh_dets(:,7));
        if ~isempty(idx)
            det_flags(nbh_dets_index(idx))=0;
        end    
    end
    % assign new SOTrackers for not assigned detections
    MOTracker.trackers(dead_trackers)=[];
    dets = dets(boolean(det_flags),:);
    MOTracker = AddTracker(MOTracker, dets, featmap, frameId);
end
function [dets, index] = findNeighborhood(ref, dets, det_flags)
    if isempty(dets)
        dets=[];
        index=[];
    else
        cx = ref(1)+ref(3)/2.;
        cy = ref(2)+ref(4)/2.;
        delta_h = mean(dets(:,6))*0.6;

        delta_w = mean(dets(:,5))*1;
        temp = dets(:,3:4) + dets(:,5:6)/2.;
        index = find(temp(:,1)> cx-delta_w & temp(:,1)< cx+delta_w & ...
                    temp(:,2) > cy-delta_h & temp(:,2)< cy+delta_h & ...
                    det_flags(:)==1);
        dets = dets(index,:);   
    end
end
    
function res = saveTrajectory(MOTracker, tracker)
    res=[];
    min_length = MOTracker.min_length; % minimum length of tracjectory
    max_miss_ratio = MOTracker.max_miss_ratio; % maximum ratio of miss to length
    miss_ratio = tracker.miss/tracker.length;
    avg_confidence = tracker.det_confidence/(tracker.length-tracker.miss);
    if tracker.length > min_length && miss_ratio < max_miss_ratio && avg_confidence > MOTracker.trk_confidence_th
        res = tracker.records;
    end
end

function saveTracker(MOTracker, dataset_name)
    min_length = MOTracker.min_length; % minimum length of tracjectory
    max_miss_ratio = MOTracker.max_miss_ratio; % maximum ratio of miss to length   
    res = [];
    for i=1:MOTracker.trackerNum
        miss_ratio = MOTracker.trackers{i}.miss/MOTracker.trackers{i}.length;
        avg_confidence = MOTracker.trackers{i}.det_confidence/...
                            (MOTracker.trackers{i}.length-MOTracker.trackers{i}.miss);
        if (MOTracker.trackers{i}.length > min_length && miss_ratio < max_miss_ratio)&& avg_confidence>MOTracker.trk_confidence_th
            res =[res; MOTracker.trackers{i}.records];
        end
    end
    dlmwrite(['results/', dataset_name, '.txt'], res);
end