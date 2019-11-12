function [S] = compute_objective_cont( data_loader, x, e, e_candidate)
% COMPUTE_OBJECTIVE_CONT
% [S] = compute_objective_cont( data_loader, x, e, e_candidate)
%
% Compute value of the CONTINUOUS objective function.
%
% Parameters:
%
%       data_loader: an instance of the DataLoader class.
%
%       x: (n x 1) cell, x{i} is an array in [0,1]^p_i where p_i is 
%          the number of proposals in image i.
%
%       e: (n x n) matrix, elements are in [0,1].
%
%       e_candidate: (n x 1) cell, contains neighbor candidates of images.
%
% Returns:
%
%       S: the value of the objective function at the point (x,e).
%

S = 0;
n = size(x, 1);
for i = 1:n
    for j = e_candidate{i}
        current_S = get_S(data_loader, i, j);
        p_i = numel(x{i});
        p_j = numel(x{j});
        % computation for the pair (i,j)
        % compute the (p_i x p_j) matrix min(x_i^k, x_j^l);
        min_ij = min(repmat(transpose(x{i}), [1,p_j]), ...
                     repmat(x{j}, [p_i,1]));
        min_e_x = min(e(i,j), min_ij); 
        S = S + sum(sum(current_S .* min_e_x));
    end
end

