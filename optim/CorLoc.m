function [ corloc, iou_score ] = CorLoc(proposals, bboxes, x, threshold)
% CORLOC
% [ corloc, iou_matrix ] = CorLoc(proposals, bbox, x, thresh)
%
% Compute CorLoc.
%
% Parameters:
%
%   	proposals: (n x 1) cell, proposals{i} contains the proposals of image i.
%
%     bboxes: (n x 1) cell, ground truth boxes.
%
%     x: (n x 1) cell, x{i} contains indices of chosen regions for image i.
%
%     threshold: double, IoU threshold.
%
%
% Returns:
%
%     corloc: double, CorLoc.
%
%     iou_score: (n x 1) cell, iou_score{i} contains iou scores of chosen regions of 
%				         image i.
%

n = size(x, 1);
iou_score = cell(n,1);
count = 0;
for i = 1:n
    positive_boxes = proposals{i}(x{i},:);
    iou = [];
    for box_idx = 1:size(bboxes{i},1)
    	iou = [iou, bbox_iou(positive_boxes, bboxes{i}(box_idx,:))];
    end
    iou_score{i} = iou';
    if sum(max(iou, [], 1) >= threshold) > 0
        count = count+1;
    end
end
corloc = count/n;
end

