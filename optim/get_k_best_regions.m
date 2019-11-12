function [x_opt] = get_k_best_regions(data_loader, x, e, k, max_num_neighbors, ...
                                      max_num_proposals, proposal_weights)
% GET_K_BEST_REGIONS
% [x_opt] = get_k_best_regions(data_loader, x, e, k, max_num_neighbors, ...
%                              max_num_proposals, proposal_weights)
%
% Parameters:
%
%     data_loader: an instance of DataLoader class containing scores.
%
%     x: 
%
%     e:
%
%     k:
%
%     max_num_neighbors: 
%
%     max_num_proposals:
%
%     proposal_weights:
%
% Returns:
%
%     x_opt:
%
%

n = size(x, 1);

if ~exist('max_num_neighbors', 'var')
    max_num_neighbors = Inf;
end

if ~exist('max_num_proposals', 'var')
    max_num_proposals = Inf;
end

if ~exist('proposal_weights', 'var')
    proposal_weights = cell(n, 1);
    for i = 1:n
        proposal_weights{i} = ones(1, numel(x{i}));
    end
end

for i = 1:n
    assert(numel(proposal_weights{i}) == numel(x{i}));
end


x_opt = cell(n, 1);
for i = 1:n
  confidence_with_neighbors = zeros(numel(x{i}), numel(e{i}));
  for j = 1:numel(e{i})
    current_S = get_S(data_loader, i, e{i}(j));
    confidence_with_neighbors(:,j) = sum_row_k(...
      (transpose(proposal_weights{i}) * proposal_weights{e{i}(j)}) .* ...
       current_S(x{i}, x{e{i}(j)}), max_num_proposals);
  end
  line_confidence = zeros(numel(x{i}), 1);
  for j = 1:size(confidence_with_neighbors, 1)
    [~,idx] = sort(confidence_with_neighbors(j,:), 'descend');
    chosen_idx = idx(1:min(max_num_neighbors, numel(e{i})));
    line_confidence(j) = sum(confidence_with_neighbors(j,chosen_idx));
  end
  [~, idx_top] = sort(line_confidence, 'descend');
  x_opt{i} = x{i}(idx_top(1:min(k, numel(x{i}))));
end


end