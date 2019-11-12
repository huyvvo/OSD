function [xe, L, sum_cubic_terms, sum_linear_terms, unlabeled] = solve_max_flow(linear_unique, cubic_unique, n, ...
                                                                                num_regions, lambda, mu, nu, tau)
% SOLVE_MAX_FLOW
% [xe, L, sum_cubic_terms, sum_linear_terms, unlabeled] = solve_max_flow(linear_unique, cubic_unique n, ...
%                                                                        num_regions, lambda, mu, nu, tau)
% Find optimal value of (x,e) given the current value of \lambda and \mu by solving the maximum stable set in the induced 
% bipartite graph. The graph represents the supermodular function:
%                   f(S, lambda, mu) = \sum_{1 <= i \neq j <= n} e_{ij}*x{i}*S_{ij} x{j}' 
%                                    + \sum_{1 <= i <= n} \lambda_i \sum_{1 <= k <= num_regions[i]} (1 - x_i^k) 
%                                    + \sum_{1 <= i <= n} \mu_i \sum_{j \neq i} (1 - e_{ij})
%
% To save computations and memory, the structure of the graph is precomputed in 'G', 'main_edges' and 'cubic_info' and saved 
% into three corresponding global variables.
%
%
% Parameters: Requires three global variables 'G', 'main_edges', 'cubic_info'.
%
%           G: digraph, bipartite graph created by build_graph.
%
%           main_edges: (Nx3) matrix, each line represents an edges (From, To, weight) where
%                       'From' represents a node in the left side, which corresponds to a 
%                       linear term; 'To' represents a node in the right side, which 
%                       corresponds to a cubic term; weight is always set to 'Inf'. Nodes 
%                       represented in main_edges are continuous in the range [1..num_nodes]
%                       where num_nodes is the sum of number of elements of linear_unique 
%                       and cubic_unique.
%
%           cubic_info: (Nx2) matrix, each line represents information of a cubic term. The
%                       first column contains the ids of cubic terms and the second column
%                       contains weights of cubic terms.
%	
%           linear_unique: (N x 1) matrix, id of linear nodes before mapping to a contiguous
%                       range. Nodes in linear_unique correspond to nodes [1..num_left] in 
%                       main_edges where num_left is the number of elements in linear_unique.
%
%           cubic_unique: (N x 1) matrix, id of cubic nodes before mapping to a contiguous
%                       range. Nodes in cubic_unique correspond to nodes [1..num_right] in 
%                       main_edges where num_right is the number of elements in cubic_unique.
%
%           n: int, number of images.
%
%           num_regions: (n x 1) matrix, number of regions in images.
%
%           lambda: n x 1 array, the current value of \lambda.
%
%           mu: n x 1 array, the current value of \mu.
%
%           nu: int, the upper bound for number of object regions in an image.
%
%           tau: the upper bound for number of neighbors of each image.
%
%
% Returns:
%           xe: (n*p+n*(n-1)) x 1 array, the optimal values of (x,e), x is the first n*p components, 
%               e is the last n*(n-1) components.
%
%           L: double, value of the dual function or the Langrangian at the maximum point (x,e).
%
%           sum_cubic_terms: double, value given by the sum of cubic terms.
%
%           sum_linear_terms: double, value given by the sum of linear terms.
%
%           unlabled: array, nodes that are not decided by the maxflow problem (those e_{ij} or x_{ik}
%                     having no positive weight cubic term associated).
%

% --------------------------------------------------------------------------------------------------
% SANITY CHECK
assert(sum(cellfun(@(x) isequal(x, 'G'), who('global'))) == 1);
assert(sum(cellfun(@(x) isequal(x, 'main_edges'), who('global'))) == 1);
assert(sum(cellfun(@(x) isequal(x, 'cubic_info'), who('global'))) == 1);
global G main_edges cubic_info
assert(numel(num_regions) == n);
assert(numel(lambda) == n);
assert(numel(mu) == n);

% --------------------------------------------------------------------------------------------------
% BUILD THE WEIGHTS FOR NODE IN THE LEFT SIDE OF THE BIPARTITE GRAPH

lambda_vec = [];
for i = 1:n
    lambda_vec = [lambda_vec; lambda(i)*ones(num_regions(i), 1)];
end

mu = mu';
mu_mat = repmat(mu, [n,1]);
mu_vec = mu_mat(tril(ones(n),-1) + tril(ones(n), -1)' == 1);

linear_coef = [lambda_vec; mu_vec];
xe = -ones(sum(num_regions) + n*(n-1), 1); % init all x and e to be -1, means unassigned
assert(numel(linear_coef) == numel(xe));

left_coefs = linear_coef(linear_unique);

% ------------------------------------------------------------------------------------------------
% GET GRAPH INFO

num_left = numel(linear_unique);
num_right = numel(cubic_unique);
num_nodes = num_left + num_right;
source = num_nodes + 1;
sink = num_nodes + 2;
disp(sprintf('Total number of nodes: %d, left nodes: %d, right nodes: %d\n', num_nodes, num_left, num_right));

% ------------------------------------------------------------------------------------------------
% UPDATE WEIGHTS OF LINEAR NODES IN THE GRAPH FOR MAX FLOW

tic
G.Edges.Weight(G.Edges.EndNodes(:,1) == source) = left_coefs;
disp(sprintf('Update graph time: %.2f', toc));

% -------------------------------------------------------------------------------------------------
% SOLVE MAX FLOW PROBLEM

tic
[mf,~,cs,ct] = maxflow(G, source, sink, 'searchtrees');
disp(sprintf('MaxFlow Time: %.2f', toc));

% -------------------------------------------------------------------------------------------------
% GET STABLE SET

A = cs(1:end-1); % exclude source node, we are sure that source > all left nodes
A_bar = ct(1:end-1); % exclude sink node, we are sure that sink > all right nodes
stable_set_linear = intersect(A, (1:num_left)');
stable_set_cubic = intersect(A_bar, (num_left+1:num_nodes)');
stable_set = [stable_set_linear; stable_set_cubic];

% disp(sprintf('Size of stable set: %d\n', length(stable_set)));

% -------------------------------------------------------------------------------------------------
% SET TRUE VALUE FOR X AND E

left_assigment = -ones(num_left, 1); % left_assigment stores the result linear nodes, initialized to -1, means unassigned
if ~isempty(stable_set_linear)
    left_assigment(stable_set_linear) = 0; % x_bar = 1, i.e., x = 0;
end
% assign linear terms connected to stable cubic terms to 1.
if ~isempty(stable_set_cubic)
    linear_linked = main_edges(ismember(main_edges(:,2), stable_set_cubic), 1);
    left_assigment(linear_linked) = 1;
end
% convert the index of stable set back to the original index
% we can aligned linear_unique with nodes in the left side since linear_unique is in increasing order 
xe(linear_unique) = left_assigment;

% identify unassigned x and e terms
% unidentified x and e terms do not appear in the objective function, therefore should be assigned to 0 to maximize the Lagrangian
disp(sprintf('Number of linear terms unassigned is %d!', sum(xe < 0)))
unlabeled = (xe == -1);
xe(unlabeled) = 0; % set the unlabeled node to be 0

% -------------------------------------------------------------------------------------------------
% COMPUTE STATISTICS 

stable_set_cubic_original = changEM(stable_set_cubic, cubic_unique, num_left+1:num_nodes); 
is_stable_cubic = ismember(cubic_info(:, 1), stable_set_cubic_original);
sum_cubic_terms = is_stable_cubic'*cubic_info(:, 2);  

sum_linear_terms = linear_coef'*(1-xe);
L = sum_cubic_terms + sum_linear_terms - dot(lambda, (num_regions - nu)) - (n-1-tau)*sum(mu);

end