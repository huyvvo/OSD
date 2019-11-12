function [e_dense] = sparse_to_dense_e(e)
% SPARSE_TO_DENSE_E
n = size(e,1);
e_dense = zeros(n);
for i = 1:n
	e_dense(i,e{i}) = 1.0;
end

end