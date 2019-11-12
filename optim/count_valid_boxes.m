function [ count, iou_score ] = count_valid_boxes(proposals, bboxes, x, threshold)
% COUNT_VALID_BOXES
% [ count, iou ] = count_valid_boxes(proposals, bbox, x, threshold)
%
% Count the number of proposals retained in 'x' that have the IoU with one of  
% the ground truth bounding boxes greater than or equal to 'threshold'.
%
% Parameters:
%
%     proposals: (n x 1) cell, proposals{i} contains proposals for image i.
%
%     bboxes: (n x 1) cell, bboxes{i} contains ground truth bboxes in image i.
%
%     x: (n x 1) cell, x{i} contains indices of chosen regions in image i.
%
%     threshold: double, IoU threshold.
%
% Returns:
%
%     count: (n x 1) array, the number of positive proposals in each image.
%
%     iou_score: (n x 1) cell, iou_score{i} contains iou scores of chosen regions of 
%                image i.
%

n = size(x, 1);
count = zeros(n,1);
iou_score = cell(n,1);
for i = 1:n
	positive_boxes = proposals{i}(x{i},:);
    iou = [];
    for box_idx = 1:size(bboxes{i},1)
    	iou = [iou, bbox_iou(positive_boxes, bboxes{i}(box_idx,:))];
    end
    iou_score{i} = iou';
    count(i) = sum(max(iou, [], 2) >= threshold);
end
end