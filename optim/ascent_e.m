function [e] = ascend_e(data_loader, x, e_candidate, tau)
% ASCENT_E
% [e] = ascend_e(data_loader, x, e_candidate, tau)
% 
% Perform coordinate ascent on e keeping x fixed with pre-filtered
% neighbors.
%
% Parameters:
%
%       data_loader: an instance of DataLoader class containing
%                    scores.
%
%       x: (n x 1) cell. If x is discrete, x{i} is a subset of 
%          indices in [1..p_i] where p_i is the number of proposals
%          in image i. Otherwise, x{i} is a vector in [0,1]^p_i.
%
%       e_candidate: (n x 1) cell, e_candidate{i} is the array of indices
%                    of neighbor candidates of image i.
%
%       tau: int, the maximum number of neighbors of each image.
%
% Returns:
%
%       e: (n x 1) cell, e after performing coordinate ascent.
%

n = size(x,1);
e = cell(n,1);
% sanity check on e_candidate
for i = 1:n
  if ismember(i, e_candidate{i})
    error('i must not be in e_candidate{i}');
  end
end
% compute similarity between pairs of images
A = zeros(n, n);
for i = 1:n
  for j = e_candidate{i}
    if A(j,i) ~= 0
      A(i,j) = A(j,i);
    else
      current_S = get_S(data_loader, i, j);
      try
        A(i,j) = sum(sum(current_S(x{i}, x{j})));
      catch exception
        A(i,j) = x{i}*current_S*transpose(x{j});
      end
    end
  end
end
% gradient ascent on e
for i = 1:n
  line_weight = A(i,:);
  line_weight(e_candidate{i}) = line_weight(e_candidate{i}) + 1;
  [ ~, idx_top ] = sort(line_weight, 'descend');
  e{i} = idx_top(1:tau);
  if ismember(i,e{i})
    e{i} = [setdiff(e{i},i) idx_top(tau+1)];
  end
end
