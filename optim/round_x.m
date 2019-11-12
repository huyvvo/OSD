function [x_s] = round_x(data_loader, x, e, e_candidate, nu, row_order)
% ROUND_X
%
% [x_s] = round_x(data_loader, x, e, e_candidate, nu, row_order)
% 
% Rounding continuous x to get discrete x_s.
%
% Parameters:
%
%       data_loader: an instance of the DataLoader class.
%
%       x: (n x 1) cell, continuous solution x.
%
%       e: (n x n) matrix, continuous solution e.
%
%       e_candidate: (n x 1) cell, e_candidate{i} is the array of indices
%                    of neighbor candidates of image i.
%
%       nu: int, the maximum number of proposals retained in each image.
%
%       row_order: (1 x n) matrix, the order in which rows of x
%                   are processed.
%
% Returns:
%
%       x_s: (n x 1) cell, x after block coordinate ascent.
%

n = size(x,1);
num_regions = cellfun(@numel, x);
x_s = cell(n,1);
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

processed = zeros(1, n); % mark rows of x that have been rounded for faster computation
for i = row_order
  Sx_sum = zeros(num_regions(i),1);
  for j = e_candidate{i}
    current_S = get_S(data_loader, i, j);
    if processed(j) == 0
        Sx_sum = Sx_sum + (e(i,j)+e(j,i))*current_S*transpose(x{j});
    else
        Sx_sum = Sx_sum + (e(i,j)+e(j,i))*sum(current_S(:, x_s{j}), 2);
    end      
  end
  [~, idx_top] = sort(Sx_sum, 'descend');
  x_s{i} = transpose(idx_top(1:nu));
  processed(i) = 1;
end
