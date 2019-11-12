function [x_dense] = sparse_to_dense_x(x, num_regions)
% SPARSE_TO_DENSE_X

n = numel(num_regions);
x_dense = cell(n, 1);
for i = 1:n
	x_dense{i} = zeros(1, num_regions(i));
	x_dense{i}(x{i}) = 1;
end

end