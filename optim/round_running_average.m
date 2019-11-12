function [x_s, e_s] = round_running_average( data_loader, x_run, e_run, e_candidate, nu, tau, row_order)
% ROUND_RUNNING_AVERAGE
% [x_s, e_s] = round_running_average( data_loader, x_run, e_run, e_candidate, nu, tau, row_order)
% 
% Enforce constraints by greedy block ascend.
%
% Parameters:
%
%   data_loader: an instance of the DataLoader class.
%
%   x_run: (n x 1) cell, the continuous solution x.
%
%   e_run: (n x n) matrix, the continuous function e.
%
%   e_candidate: (n x 1) cell, e_candidate{i} is the array of indices
%                of neighbor candidates of image i.
%
%   nu: int, maximum number of proposals retained in each image.
%
%		tau: int, maximum number of neighbors of each image.
%
%		row_order: (1 x n) array, order in which rows of x are processed.
%
% Returns:
%
%       x_s: (n x 1) cell, x_s{i} is a subset of indices of proposals in image i.
%   
%       e_s: (n x 1) cell, e{i} represents neighbors of image i.

if ~exist('row_order', 'var')
	row_order = 1:size(x_run, 1);
end

x_s = round_x(data_loader, x_run, e_run, e_candidate, nu, row_order);
e_s = ascent_e(data_loader, x_s, e_candidate, tau);

end
