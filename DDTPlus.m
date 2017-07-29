% Copyright (C) 2017 Xiu-Shen Wei, Chen-Lin Zhang, Yao Li, Chen-Wei Xie, Jianxin Wu, Chunhua Shen and Zhi-Hua Zhou
% All rights reserved.
%
% This file is part of our DDT+ method.

clear;
clc;

% Experimental settings
opt.layer = 37; % Specify which layer is used in vgg19, for example, "37" indicating the relu5_4 layer
opt.secondlayer = 35; % Specify which layer is used in vgg19, for example, "35" indicating the relu5_3 layer
opts.gpu = 5; % Specify which gpu to use
gpuDevice(opts.gpu);

% Run matconvnet setup, please change the location as your own
run('./matconvnet/matlab/vl_setupnn.m');

% Choose the models like VGG19 or VGG16
opt.model = 'imagenet-vgg-verydeep-19';
net = load(['./' opt.model '.mat']);

% Remove FC layers and transfer to GPU
net.layers(38:end)=[];
net = vl_simplenn_tidy(net);
net = vl_simplenn_move(net, 'gpu') ;
disp('CNN model is ready ...');

% We suppose that each class is under the path variable and have a separate folder
path = './data/';
raw_path = dir(path);

% Remove "." and ".." structure
raw_path = raw_path(3:end);
	
% Get num_class
num_class = size(raw_path, 1);
for times = 1 : num_class
    % Load IMDB
    now_class_name = [path raw_path(times).name];
    now_dir = dir(now_class_name);
    % Remove "." and ".." structure
    now_dir = now_dir(3:end);
    % Prepare 
    averageImage = net.meta.normalization.averageImage;
    num_tr = size(now_dir,1);
    train_bag = {};
    train_bag_L2 = {};
    for i = 1 : num_tr
        % Read raw image
        im = imread([now_class_name '/' now_dir(i).name]);
        im_ = single(im);
        % if image is too large, resize images
        Is_resize = false;
        if min(size(im, 1), size(im, 2))>800
            rate = 800/min(size(im, 1), size(im, 2));
            im_ = imresize(im_, [rate*size(im, 1), rate*size(im, 2)]);
            Is_resize = true;
        end
        [h,w,c] = size(im_);
        % Subtract averageImage
        if  c > 2
            im_ = im_ - imresize(averageImage,[h,w]) ;
        % for gray-scale images,
        else    
            im_ = bsxfun(@minus,im_,imresize(averageImage,[h,w])) ;
        end
        res = vl_simplenn(net, gpuArray(im_));
        % Get feature map
        tmp_featmap = gather(res(opt.layer).x);
        tmp_featmap_L2 = gather(res(opt.secondlayer).x);
        train_bag{end+1,1} = reshape(tmp_featmap,[size(tmp_featmap,1) * size(tmp_featmap,2) 512]);
        train_bag_L2{end+1,1} = reshape(tmp_featmap,[size(tmp_featmap_L2,1) * size(tmp_featmap_L2,2) 512]);
        disp([num2str(i) ' images extract success!']);
    end
    % Doing PCA transformation
    train_sum = cell2mat(train_bag);
    train_sum_L2 = cell2mat(train_bag_L2);
    train_mean = mean(train_sum,1);
    train_mean_L2 = mean(train_sum_L2,1);
    new_train_mean = zeros(1,1,512);
    new_train_mean(1,1,:) = mean(train_sum,1);
    new_train_mean_L2 = zeros(1,1,512);
    new_train_mean_L2(1,1,:) = mean(train_sum_L2,1);
    train_sum = train_sum - repmat(train_mean,[size(train_sum,1) 1]);
    train_sum_L2 = train_sum_L2 - repmat(train_mean_L2,[size(train_sum_L2,1) 1]);
    %btic
    [coeff,~,~] = pca(train_sum);
    [coeff_L2,~,~] = pca(train_sum_L2);
    %bnow_pca_time = toc
    trans_matrix = coeff(:,1);
    trans_matrix_L2 = coeff_L2(:,1);
    num_te = num_tr;
    % Testing process
    for i = 1 : num_te
        tic
        % Image preprocessing
        im = imread([now_class_name '/' now_dir(i).name]);
        im_ = single(im);
        Is_resize = false;
        if min(size(im, 1), size(im, 2))>800
            rate = 800/min(size(im, 1), size(im, 2));
            im_ = imresize(im_, [rate*size(im, 1), rate*size(im, 2)]);
            Is_resize = true;
        end
        [h,w,c] = size(im_);
        if  c > 2
            im_ = im_ - imresize(averageImage,[h,w]) ;
        else    
            im_ = bsxfun(@minus,im_,imresize(averageImage,[h,w])) ;
        end
        res = vl_simplenn(net, gpuArray(im_));
        tmp_featmap = gather(res(opt.layer).x);
        % Doing transforming
        tmp_featmap = tmp_featmap - repmat(new_train_mean, [size(tmp_featmap,1) size(tmp_featmap,2) 1]);
        tmp_featmap_sum = zeros(size(tmp_featmap,1),size(tmp_featmap,2));
        for he = 1 : size(tmp_featmap,1)
           for we = 1 : size(tmp_featmap,2)
               tmp_featmap_sum(he,we) = squeeze(tmp_featmap(he,we,:))' * trans_matrix;
           end
        end
        % Second layer process
        tmp_featmap_L2 = gather(res(opt.secondlayer).x);
        tmp_featmap_L2 = tmp_featmap_L2 - repmat(new_train_mean_L2, [size(tmp_featmap_L2,1) size(tmp_featmap_L2,2) 1]);
        tmp_featmap_sum_L2 = zeros(size(tmp_featmap_L2,1),size(tmp_featmap_L2,2));
        for he = 1 : size(tmp_featmap_L2,1)
           for we = 1 : size(tmp_featmap_L2,2)
               tmp_featmap_sum_L2(he,we) = squeeze(tmp_featmap_L2(he,we,:))' * trans_matrix_L2;
           end
        end
        highlight = zeros(size(tmp_featmap_sum));
        highlight(tmp_featmap_sum>0) = 1;    
        cc = bwconncomp(highlight);
        numPixel = cellfun(@numel,cc.PixelIdxList);
        [~,conn_idx] = max(numPixel);
        
        highlight_conn = zeros(size(highlight));
        highlight_conn(cc.PixelIdxList{conn_idx}) = 1;
        % Interact with previous layer result
        highlight_L2 = zeros(size(tmp_featmap_sum_L2));
        highlight_L2(tmp_featmap_sum_L2 > 0) = 1;
        highlight_raw = highlight_conn;
        highlight_conn = highlight_conn & highlight_L2;
        % Visualizing process
        highlight_big = imresize(highlight_conn, [h w], 'nearest');
        highlight_3 = cat(3, highlight_big*255 , zeros([h w]));
        highlight_3 = cat(3, highlight_3, zeros([h w]));
        im_ori = imread([now_class_name '/' now_dir(i).name]);
        im_ori_highlight = imresize(im_ori, [h w])+uint8(highlight_3);

        b = regionprops(double(highlight_big), 'BoundingBox');
        
        figure(1);
        % Original image
        imshow(im_ori);
        figure(2);
        % Image with the highlighted region
        imshow(im_ori_highlight);
        figure(3);
        % Image with the bounding box
        imshow(imresize(im,[h w]));
        xmin = b.BoundingBox(1,1);
        ymin = b.BoundingBox(1,2);
        xmax = b.BoundingBox(1,1) + b.BoundingBox(1,3);
        ymax = b.BoundingBox(1,2) + b.BoundingBox(1,4);
        linewidth = 5;
        line([xmin xmin],[ymin ymax],'Color','r','Linewidth',linewidth);
	
	% Draw right line
        line([xmax xmax],[ymin ymax],'Color','r','Linewidth',linewidth);
	
	% Draw top line
        line([xmin xmax],[ymin ymin],'Color','r','Linewidth',linewidth);
													   
	% Draw bottom line
        line([xmin xmax],[ymax ymax],'Color','r','Linewidth',linewidth);
        %}        
        eclipse_time = toc;
        disp(['i=' num2str(i) ' ,num_te=' num2str(num_te) ' success!']);
        disp('Press Any Key to Continue!');
        pause;
        
    end
end
