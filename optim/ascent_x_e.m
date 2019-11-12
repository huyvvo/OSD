function [x,e] = ascent_x_e_dev(data_loader, x, e, e_candidate, ...
								nu, tau, num_regions, num_iter)
% ASCENT_X_E
% [x,e] = ascent_x_e(data_loader, x, e, e_candidate, ...
%						 nu, tau, num_regions, num_iter)
%
% Perform greedy gradient ascent on (x,e).
%
% Parameters:
%
%   	data_loader: instance of class DataLoader.
%
%   	x: (n x 1) cell, x{i} is a subset of indices of proposals in image i.
%
%     e: (n x 1) cell, e{i} represents current neighbors of image i.
%
%		  e_candidate: (n x 1) cell, e_candidate{i} is the array of indices
%                    of neighbor candidates of image i.
%
%     nu: int, maximum number of proposals retained in each image.
%
%		  tau: int, the maximum number of neighbors of each image.
%
%   	num_regions: (n x 1) array, number of proposals in each image.
%
%     num_iter: int, number of iterations.
%
%
% Returns:
%
%     (x,e) after the greedy gradient ascent.
%

n = size(x, 1);
for i = 1:num_iter
  x = ascent_x(data_loader, x, e, e_candidate, ...
               nu, num_regions, randperm(n,n));
  e = ascent_e(data_loader, x, e_candidate, tau);
end
