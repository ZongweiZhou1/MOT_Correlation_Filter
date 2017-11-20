% CFTracker
classdef SOTracker
    properties   
        records=[];             % records of the tracjectory [frameId,trackId,0,0,0,0, 1,-1,-1,-1];   
        gpu;                    % whether gpu is used
        det_confidence = 0;     % confidence of all the dets in this trajectory
        activated = 1;          % denote whether this tracker is activated
        trackId;                % track id
        region;                 % current box in current frame
        frameId;
        
        affinity = 0;           % sum affinity along the time of this tracjectory
        confidence;             % confidence of current track
        length = 1;             % length of this tracjectory
        miss = 0;               % loss frames in recent time window
        consec_miss =0;         % consective miss tracked number, 
                                % this tracker will be destroyed if this number larger than a threshold
                                
        miss_th;                % miss threshold
        conf_th;                % tracker's confidence threshold
        consec_miss_length;     % maximum lost detected frames
        response_th = 0.5;      % resoonse threshold          
        overlap_th = 0.3;       % iou thresh
        w1 = 0.5;               % location
        w2 = 1.5;               % size
        w3 = 1.2;               % length weight for track's confidence
      
        
        output_sigma_factor = 0.1;  % variance factor of gaussian distribution
        lambda;                 % regulatize param for regression weight
        padding;                % padding for height and width 2x1
        interp_factor;          % learning factor
        num_scale;              % scale num
        scale_step;             % 
        scale_penalty;          % base penalty
        scale_factor;           %
        scale_penalties;        % array penalties get from scale_penalty
        net_input_size;         % predifined patch size
        yf;                     %
        cos_window;             % cosine mask
        yyxx;                   % corresponding to patch size for bilinearinterpolation
        pos;                    % lastest target's center location
        target_sz;              % target size
        min_sz;                 % minimum target size in next frame
        max_sz;                 % maximum target size in next frame
        numel_xf;               % channels number of deep features
        model_alphaf;       
        model_xf;
        
    end
    methods
        function obj = SOTracker(featmap, param)  
            % params:  featmap, the feature map of frame
                                                         
            obj = vl_argparse(obj, param);
            obj.det_confidence = obj.confidence;
            % corresponding to h, w
            min_scale_factor = [0.8;0.5];
            max_scale_factor = [1.5;1.5]; 
            obj.scale_factor = obj.scale_step.^((1:obj.num_scale)-ceil(obj.num_scale/2)); % scale_factor
            obj.scale_penalties = ones(1,obj.num_scale);
            obj.scale_penalties((1:obj.num_scale)~=ceil(obj.num_scale/2)) = obj.scale_penalty;

            % gaussian variance
            output_sigma = sqrt(prod(obj.net_input_size./(1+obj.padding)))*obj.output_sigma_factor;            
            obj.yf = single(fft2(obj.gaussian_shaped_labels(output_sigma, obj.net_input_size)));
            % cosine weight mask
            obj.cos_window = single(hann(size(obj.yf,1)) * hann(size(obj.yf,2))');
            
            yi = linspace(-1, 1, obj.net_input_size(1));
            xi = linspace(-1, 1, obj.net_input_size(2));
            [xx,yy] = meshgrid(xi,yi);
            obj.yyxx = single([yy(:), xx(:)]') ; % 2xM, for image crop used bilinearinterpolation 
            
            if obj.gpu %gpuSupport
                obj.yyxx = gpuArray(obj.yyxx);
                obj.yf = gpuArray(obj.yf);
                obj.cos_window = gpuArray(obj.cos_window);
            end
                      
            obj.pos = obj.region([2,1])+obj.region([4,3])/2;% center location of target (y,x)
            obj.pos=obj.pos';
            obj.target_sz = obj.region([4,3])'; % size of target
            obj.min_sz = max([19;9],min_scale_factor.*obj.target_sz);
            [im_h,im_w,~] = size(featmap);
            obj.max_sz = min([im_h;im_w],max_scale_factor.*obj.target_sz);

            window_sz = obj.target_sz.*(1+obj.padding); % size of padding window
            res = obj.imcrop_multiscale(featmap, obj.pos, window_sz, obj.net_input_size, obj.yyxx); % feature of cropped region
            
            x = bsxfun(@times, res, obj.cos_window); % add the cosine mask
            xf = fft2(x);
            obj.numel_xf = numel(xf); % channel number
            kf = sum(xf.*conj(xf),3)/obj.numel_xf; % accumulate different channel feature
            obj.model_alphaf = obj.yf ./ (kf + obj.lambda); % update model regression w in a dynamic mean way, reference Eq.13
            obj.model_xf = xf; % current crop's Fourier feature 
            
            obj.records = [obj.frameId, obj.trackId, obj.region, 1, -1,-1,-1];          
        end
        
        function labels = gaussian_shaped_labels(~, sigma, sz)
            [rs,cs] = ndgrid((1:sz(1)) - floor(sz(1)/2), (1:sz(2)) - floor(sz(2)/2));
            labels = exp(-0.5 / sigma^2 * (rs.^2 + cs.^2));
            labels = circshift(labels, -floor(sz(1:2) / 2) + 1);
            assert(labels(1,1) == 1)
        end
        function img_crop = imcrop_multiscale(~, img, pos, sz, output_sz, yyxx)
            [h,w,c,~] = size(img);
            if c==1, img = repmat(img, [1,1,3,1]); end
            
            pos = gather(pos);
            sz = gather(sz);
            h = gather(h);
            w = gather(w);
            
            cy_t = (pos(1)*2/(h-1))-1; % keep center location
            cx_t = (pos(2)*2/(w-1))-1;
            h_s = sz(1,:)/(h-1); % normalize the h and w on img size
            w_s = sz(2,:)/(w-1);
            
            s = reshape([h_s; w_s], 2,1,[]); % scale 
            t = [cy_t;cx_t]; % translation 
            
            g = bsxfun(@times, yyxx, s); % scale
            g = bsxfun(@plus, g, t); % translate
            g = reshape(g, 2, output_sz(1), output_sz(2), []);
            
            img_crop = vl_nnbilinearsampler(img, g); % this can generate some cropped imgs with size 'output_sz'
        end
        
        function [response,box] = predict(obj, featmap)
            window_sz = bsxfun(@times, obj.target_sz.*obj.padding, obj.scale_factor);
            patch = obj.imcrop_multiscale(featmap, obj.pos, window_sz, obj.net_input_size, obj.yyxx);
            z = bsxfun(@times, patch, obj.cos_window);
            zf = fft2(z);
            kzf = sum(bsxfun(@times, zf, conj(obj.model_xf)),3)/obj.numel_xf;
            response = squeeze(real(ifft2(bsxfun(@times, obj.model_alphaf, kzf)))); % get the response of CF
            [max_response, max_index] = max(reshape(response,[],obj.num_scale)); % find maximum response in each scale
            max_response = gather(max_response);
            max_index = gather(max_index);
            [response,scale_delta] = max(max_response.*obj.scale_penalties);
            [vert_delta, horiz_delta] = ind2sub(obj.net_input_size, max_index(scale_delta));
                    
            if vert_delta > obj.net_input_size(1) / 2  %wrap around to negative half-space of vertical axis
                vert_delta = vert_delta - obj.net_input_size(1);
            end
            if horiz_delta > obj.net_input_size(2) / 2  %same for horizontal axis
                horiz_delta = horiz_delta - obj.net_input_size(2);
            end       
            window_sz = window_sz(:,scale_delta);
            cpos = obj.pos + [vert_delta - 1, horiz_delta - 1]'.*...
                            window_sz./obj.net_input_size;
            ctarget_sz = min(max(window_sz./(1+obj.padding), obj.min_sz), obj.max_sz);
            box = [cpos([2,1])' - ctarget_sz([2,1])'/2, ctarget_sz([2,1])'];
        end 
        
        function ious = getIOUs(~, box, regions)
            if isempty(regions)
                ious=[0];
            else
                x1 = regions(:,1);
                y1 = regions(:,2);
                x2 = regions(:,3)+x1;
                y2 = regions(:,4)+y1;
                ref = repmat([box(1:2), box(1:2)+box(3:4)],size(regions,1),1);
                xx1 = max(ref(:,1), x1);
                yy1 = max(ref(:,2), y1);
                xx2 = min(ref(:,3), x2);
                yy2 = min(ref(:,4), y2);
                w = max(0.0, xx2-xx1+1);
                h = max(0.0, yy2-yy1+1);
                inter = w.*h;
                ious = inter./ (box(3)*box(4)+(x2-x1+1).*(y2-y1+1) - inter);
            end            
        end
        
        function obj = update_confidence(obj, max_response, box)
            cregion = obj.region;
            aff_mot = exp(-obj.w1 *((cregion(1)-box(1))^2/(box(1)^2) + (cregion(2)-box(2))^2/(box(2)^2))); 
            aff_shp = exp(-obj.w2*(abs(cregion(3)-box(3))/(cregion(3)+box(3)) + abs(cregion(4)-box(4))/(cregion(4)+box(4))));
            caffinity = (max_response + aff_mot +aff_shp)/3;
            obj.affinity = obj.affinity + caffinity;
            obj.confidence = obj.affinity/obj.length;%*(1-exp(-obj.w3*sqrt(obj.length)));           
        end
        
        function [obj, region_index] = SOTracker_Update(obj, featmap, regions, frameId, det_confidences)
            % params: regions, condidated patches
            if obj.gpu, featmap = gpuArray(featmap); end
            [max_response, box] = obj.predict(featmap);
            % calculate the ious of box and regions
            ious = obj.getIOUs(box, regions);
            [max_iou, max_iou_arg]=max(ious(:));        
            if max_iou > obj.overlap_th
                % find then update
                region_index=max_iou_arg;
                obj = obj.update_confidence(max_response, regions(max_iou_arg,:));
                
                obj.pos = regions(max_iou_arg,[2,1])+regions(max_iou_arg,[4,3])/2;
                obj.pos = obj.pos';
                obj.target_sz = regions(max_iou_arg,[4,3])';
                window_sz = obj.target_sz.*(1+obj.padding);
                res = obj.imcrop_multiscale(featmap, obj.pos, window_sz, obj.net_input_size, obj.yyxx); % feature of cropped region
            
                x = bsxfun(@times, res, obj.cos_window); % add the cosine mask
                xf = fft2(x);
                kf = sum(xf.*conj(xf),3)/obj.numel_xf; % accumulate different channel feature
                alphaf = obj.yf ./ (kf + obj.lambda);
                obj.model_alphaf = (1 - obj.interp_factor) * obj.model_alphaf + obj.interp_factor * alphaf; % update model regression w in a dynamic mean way, reference Eq.13
                obj.model_xf = (1-obj.interp_factor)*obj.model_xf + obj.interp_factor*xf;
                box = [obj.pos([2,1])- obj.target_sz([2,1])/2; obj.target_sz([2,1])]';
                location = double(gather(box));
                obj.records = [obj.records;frameId, obj.trackId, location, 1, -1,-1,-1];
                obj.region = location;
                obj.length = obj.length + 1;
                obj.consec_miss = 0;    
                obj.det_confidence = obj.det_confidence + det_confidences(max_iou_arg);
            else
                % not find, update but the confidence is decrease hugely.
                region_index = [];
                if max_response > obj.response_th
                    
                    obj = obj.update_confidence(max_response,box);
                    obj.pos = box([2,1])+box([4,3])/2.;
                    obj.pos = obj.pos';
                    
                    obj.target_sz = box([4,3])';

                    %window_sz = obj.target_sz.*(1+obj.padding);
                    %res = obj.imcrop_multiscale(featmap, obj.pos, window_sz, obj.net_input_size, obj.yyxx); % feature of cropped region

                    %x = bsxfun(@times, res, obj.cos_window); % add the cosine mask
                    %xf = fft2(x);
                    %kf = sum(xf.*conj(xf),3)/obj.numel_xf; % accumulate different channel feature
                    %alphaf = obj.yf ./ (kf + obj.lambda);
                    %obj.model_alphaf = (1 - obj.interp_factor) * obj.model_alphaf + obj.interp_factor * alphaf; % update model regression w in a dynamic mean way, reference Eq.13
                    %obj.model_xf = (1-obj.interp_factor)*obj.model_xf + obj.interp_factor*xf;
                    box = [obj.pos([2,1])- obj.target_sz([2,1])/2; obj.target_sz([2,1])]';
                    location = double(gather(box));
                    obj.region=location;
                    obj.records = [obj.records;frameId, obj.trackId, location, 1, -1,-1,-1];
                else
                    obj.consec_miss = obj.consec_miss+4;
                end
                obj.length = obj.length + 1;
                obj.consec_miss = obj.consec_miss + 1;
                obj.miss = obj.miss + 1;
                if obj.length<3
                    obj.activated =0;
                end
                
            end
            if  obj.consec_miss > obj.consec_miss_length || obj.confidence < obj.conf_th 
                obj.activated=0;
            end

        end        
    end
end