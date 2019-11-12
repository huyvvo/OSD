function [ x, e ] = get_x_e( xe, n, num_regions)
% GET_X_E
% [ x, e ] = get_x_e(xe, n, num_regions)
%
% get x and e from the array xe of size (sum(num_regions) + n(n-1) x 1).
%
% Parameters:
%
%     xe: (sum(num_regions) + n(n-1) x 1) array, the concatenation of the 
%         vectorized x and e.
%
%     n: int, number of images.
%
%     num_regions: (n x 1), number of proposals in images.
%
% Returns:
%
%     x: (n x 1) cell.
%
%     e: (n x n) matrix.
%

assert(length(num_regions) == n);
assert(numel(xe) == sum(num_regions) + n*(n-1));

x = mat2cell(xe(1:sum(num_regions))', [1], num_regions)';

e = zeros(n);
e(tril(ones(n),-1) + tril(ones(n),-1)' == 1) = xe(sum(num_regions)+1:end);
e = e'; % transpose to get the e matrix

end

