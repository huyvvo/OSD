function [sparse_S] = sparsify_matrix(S, num_keep)
% SPARSIFY_MATRIX
%
% Get a new sparse matrix with only 'num_keep' top elements from 'S'.
%
% Paramters:
%
%   S: a matrix.
%
%   num_keep: int, number of top elements to be kept.
%
% Returns:
%
% A sparse matrix containing only top elements in 'S'.

sparse_S = sparse(size(S,1), size(S,2));
[~,max_idx] = sort(S(:), 'descend');
sparse_S(max_idx(1:min(num_keep, numel(S)))) = ...
                              S(max_idx(1:min(num_keep, numel(S))));

end