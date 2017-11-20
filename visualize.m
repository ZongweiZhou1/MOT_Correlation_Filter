% display the results in sequence and stored in video
function visualize(img_path, dataset_type, dataset_name, res_path)
    results = dlmread([res_path '/' dataset_name,'.txt'],',');
    img_path = [img_path, '/', dataset_type, '/', dataset_name,'/img1/'];
    imgfile = dir([img_path, '*.jpg']);
    imgfile = fullfile(img_path, {imgfile.name});
    trackIds = unique(results(:,2));
    colors = rand(size(trackIds,1),3); % colors for different tracjectory
    videoObj = VideoWriter([res_path,dataset_name]);
    videoObj.FrameRate = 25;
    open(videoObj);
    figure('visible','off');
    for i=1:size(imgfile,2)
        frame = imread(imgfile{i});
        cur_boxes = results(results(:,1)==i,2:6);
        imshow(frame)
        hold on
        for j=1:size(cur_boxes,1)
            rectangle('Position', cur_boxes(j,2:5), 'LineWidth',2, 'EdgeColor', colors(trackIds==cur_boxes(j,1),:));    
        end
        hold off
        F=getframe;
        writeVideo(videoObj, F);
        if mod(i,20)
            disp(['====> frames to :  ', num2str(i), ' have been stored!']);
        end
    end
    disp('======== DONE =============');
    close(videoObj);
end





