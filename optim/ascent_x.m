function [x] = ascent_x( data_loader, x, e, e_candidate, nu, num_regions, row_order)
% ASCENT_X
% [x] = ascent_x( data_loader, x, e, e_candidate, nu, num_regions, row_order)
% 
% Perform coordinate ascent on x.
%
% Parameters:
%
%       data_loader: an instance of DataLoader class.
%
%       x: (n x 1) cell, x{i} is a subset of indices of proposals in image i.
%
%       e: (n x 1) cell, e{i} represents current neighbors of image i.
%
%       e_candidate: (n x 1) cell, e_candidate{i} is the array of indices
%                    of neighbor candidates of image i.
%
%       nu: int, maximum number of proposals retained in each image.
%
%       num_regions: (n x 1) cell, number of proposals in each image.
%
%       row_order: (1 x n) matrix, the order in which rows of x
%                   are processed.
%
% Returns:
%
%       x: (n x 1) cell, x after performing coordinate ascent.
%

n = size(x,1);
if ~exist('row_order', 'var')
    row_order = 1:n;
end
% sanity check on e_candidate
for i = 1:n
  if ismember(i, e_candidate{i})
    error('i must not be in e_candidate{i}');
  end
end
assert(size(row_order, 2) == n & size(row_order, 1) == 1);
% perform gradient ascent on x
for i = row_order
  Sx_sum = zeros(num_regions(i),1);
  for j = e_candidate{i}
    linked_level = sum(ismember(i, e{j})) + sum(ismember(j, e{i}));
    if i ~= j & linked_level > 0
      current_S = get_S(data_loader, i, j);
      Sx_sum = Sx_sum + linked_level*sum(current_S(:, x{j}), 2); 
    end    
  end
  [~, idx_top] = sort(transpose(Sx_sum), 'descend');
  x{i} = idx_top(1:nu);
end
