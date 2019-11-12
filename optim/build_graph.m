function [G, linear_unique, cubic_unique] = build_graph
% BUILD_GRAPH
% [G, linear_unique, cubic_unique] = build_graph
% 
% Build a bipartite graph from infomation given by 'main_edges' and 'cubic_info' which 
% saved as two global variables.
%
% Parameters: Requires two global variables 'main_edges' and 'cubic_info'.
%
%		main_edges: (Nx3) matrix, each line represents an edges (From, To, weight) where
%               'From' represents a node in the left side, which corresponds to a 
%               linear term; 'To' represents a node in the right side, which 
%               corresponds to a cubic term; weight is always set to 'Inf'.
%
%		cubic_info: (Nx2) matrix, each line represents information of a cubic term. The
%               first column contains the ids of cubic terms and the second column
%               contains weights of cubic terms.
%
%
% Returns:
%
%   G: A digraph object correspond to the bipartite graph.
%
%   linear_unique: a colum containing indices of unique linear node in the graph.
%
%   cubic_unique: a column containing indices of unique cubic node in the graph.
%

% --------------------------------------------------------------------------------------------------
% GET THE ID OF LEFT NODES AND RIGHT NODES, COMPUTE NUMBER OF NODES IN THE GRAPH

% SANITY CHECK
assert(sum(cellfun(@(x) isequal(x, 'main_edges'), who('global'))) == 1);
assert(sum(cellfun(@(x) isequal(x, 'cubic_info'), who('global'))) == 1);
global main_edges cubic_info

% GET NODE INFO
linear_unique = unique(main_edges(:,1));
num_left = numel(linear_unique);
left_coefs = zeros(num_left, 1);

cubic_unique = cubic_info(:,1); % do not use unique here because unique() sorts elements
right_coefs = cubic_info(:,2);
num_right = numel(cubic_unique);

num_nodes = num_left + num_right; 
source = num_nodes + 1;
sink = num_nodes + 2;
disp(sprintf('Total number of nodes: %d, left nodes: %d, right nodes: %d\n', num_nodes, num_left, num_right));

% -------------------------------------------------------------------------------------------------
% BUILD TERMINAL EDGES

source_edges = [ones(num_left,1)*source,  (1:num_left)', left_coefs];
sink_edges = [(num_left+1:num_nodes)', ones(num_right,1)*sink, right_coefs];

% replace original node number with continues node number. The build-in changem is too slow.
main_edges(:,1:2) = changEM(main_edges(:,1:2), 1:num_nodes, [linear_unique' cubic_unique']); 
edges = [source_edges; main_edges; sink_edges];

% ------------------------------------------------------------------------------------------------
% BUILD GRAPH FOR MAX FLOW

disp('Building graph...');
tic
G = digraph(edges(:,1), edges(:,2), edges(:,3));
disp(sprintf('BuildGraph Time: %.2f', toc));
