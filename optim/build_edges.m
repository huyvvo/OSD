function [ main_edges, cubic_info ] = build_edges(data_loader, n, num_regions, neighbors)
% BUILD_EDGES
% [ main_edges, cubic_info ] = build_edges(data_loader, n, num_regions, neighbors)
%
% Get edges of the bipartite graph from similarity matrices S_ij given by 'data_loader'. 
% Only cubic terms with corresponding S{i,j} positive are included in the graph.
%
% Each node in the graph has a unique id. Nodes representing linear terms have ids 
% in the range [1, sum(num_regions) + n*(n-1)]. Nodes representing cubic terms have 
% ids greater than sum(num_regions) + n*(n-1).
%
% Parameters:
%
%   data_loader: instance of DataLoader class.
%
%   n: int, number of images.
%
%   num_regions: (n x 1) matrix, number of regions in each image.
%
%   neighbors: (n x 1) cell, neighbors{i} is the array of indices
%              of neighbor candidates of image i.
%
% Returns:
%
%   main_edges: (Nx3) matrix, each line represents an edges (From, To, weight) where
%				        'From' represents a node in the left side, which corresponds to a 
%				        linear term; 'To' represents a node in the right side, which 
%				        corresponds to a cubic term; weight is always set to 'Inf'.
%
%		cubic_info: (Nx2) matrix, each line represents information of a cubic term. The
%					      first column contains the ids of cubic terms and the second column
%					      contains weights of cubic terms

%--------------------------------------------
% SANITY CHECK

% Check that all cell of neighbors are row vectors
for i = 1:n 
  assert(size(neighbors{i}, 1) == 1);
end

%--------------------------------------------

disp('Building main edges of the bipartite graph ...')
tic;
% the cell 3*idx+1, 3*idx+2, and 3*idx+3 contain edges induced by regions in the image
% pair of index 'idx'. The first cell is for x_i, the second for x_j and the third for e_ij. 
main_edges = cell(3*n*(n-1), 1);
% the cell idx contains information of cubic terms in the image pair of index 'idx'
cubic_info = cell(n*(n-1), 1);
% offset to recover index of cubic terms
cubic_offset = sum(num_regions) + n*(n-1); 

for i = 1:n
  for j = neighbors{i}
    % get index of the image pair
    if i < j
	   current_S_index = (n-1)*(i-1) + j-1;
    elseif i > j
        current_S_index = (n-1)*(i-1) + j;
    else
        continue;
    end
    % get score matrix
    current_S = get_S(data_loader, i, j);
    num_rows = num_regions(i);
    num_cols = num_regions(j);
    % get indices of non-zero terms in cur_S
    IND = find(current_S > 0);
    if isempty(IND)
        continue;
    end
    % get indices of proposals in the image pair which correspond to positive scores
    [kk, ll] = ind2sub([num_rows, num_cols], IND);
    % re-sort kk and ll prioritizing the order in 'kk'
    [~, idx] = sortrows([kk,ll]);
    kk = kk(idx); ll = ll(idx); 
    IND = sub2ind([num_rows,num_cols],kk,ll);
    % get indices of nodes
    x_i_nodes = sum(num_regions(1:i-1)) + kk; % find the index of the node x_{ik} for k in kk
    x_j_nodes = sum(num_regions(1:j-1)) + ll; % find the index of the node x_{jl} for k in ll
    e_ij_node = sum(num_regions) + current_S_index;
    S_ijkl_nodes = cubic_offset + num_cols*(kk-1) + ll;
    cubic_offset = cubic_offset + num_rows * num_cols;
    % build edges
    main_edges{3*(current_S_index-1) + 1} = [x_i_nodes, S_ijkl_nodes, Inf * ones(numel(x_i_nodes), 1)];
    main_edges{3*(current_S_index-1) + 2} = [x_j_nodes, S_ijkl_nodes, Inf * ones(numel(x_j_nodes), 1)];
    main_edges{3*(current_S_index-1) + 3} = [e_ij_node*ones(numel(IND), 1), S_ijkl_nodes, Inf * ones(numel(IND), 1)];
    % collect cubic weights
    cubic_info{current_S_index} = [S_ijkl_nodes, full(double(current_S(IND)))];
  end
end

main_edges = cell2mat(main_edges);
cubic_info = cell2mat(cubic_info);
disp(sprintf('Number of lines in cubic info %d', size(cubic_info, 1)));
disp(sprintf('Number of unique lines in cubic info %d', size(unique(cubic_info, 'rows'), 1)));
disp(sprintf('Main edges built in %.2f seconds', toc));

end