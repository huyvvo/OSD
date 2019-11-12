function [sum_row] = sum_row_k(x, k)
% SUM_ROW_K
% [sum_row] = sum_row_k(x, k)
%
% Get sum of k greatest elements in each row of matrix x.
% Parameters:
%
%		x: 2-dimensional matrix.
%
%		k: int.
%

assert(numel(size(x)) == 2);

[m, n] = size(x);
sum_row = zeros(m, 1);
for i = 1:m
	[~, idx] = sort(x(i,:), 'descend');
	sum_row(i) = sum(x(i,idx(1:min(k,n))));
end

end