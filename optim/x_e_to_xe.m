function [ xe ] = x_e_to_xe(x,e)
% X_E_TO_XE
% [xe] = x_e_to_xe(x,e)
%
% Transform and combine x and e into a vector of size 
% (sum(num_regions) + n*(n-1)) x 1.
%
% Parameters:
%
%     x: (n x 1) cell, x{i} is an array in [0,1]^p_i where p_i
%        is the number of proposals in image i.
%
%     e: (n x n) matrix, elements are in [0,1].
%
% Returns:
%
%     xe: [(sum(num_regions) + n*(n-1)) x 1] matrix, the concatenation
%         of vectorized x and e.
%

n = size(x, 1);
num_regions = cellfun(@numel, x);

xe = [];
for i = 1:n
	xe = [xe; transpose(x{i})];
end

e = transpose(e);
e = e(tril(ones(n),-1) + tril(ones(n),-1)' == 1);
xe = [xe; e];

end
