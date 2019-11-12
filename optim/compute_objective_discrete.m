function [S] = compute_objective_discrete( data_loader, x, e)
% COMPUTE_OBJECTIVE_DISCRETE
% [S] = compute_objective_discrete(data_loader, x, e)
%
% Compute value of the discrete objective function
% Note that this function is only used when x and e are discrete.
%
% Parameters:
%
%     data_loader: an instance of the DataLoader class.
%
%     x: (nx1) cell, each cell contains indices of active regions.
%
%     e: (nx1) cell, each cell contains indices of neighbors.
%
% Returns:
%
%     S: the value of the objective function at the point (x,e).
%

n = size(x, 1);
S = 0;
for i = 1:n
    for j = e{i}
		if i == j
      error('An image can not be a neighbor of itself');
    end
      current_S = get_S(data_loader, i, j);
      S = S + sum(sum(current_S(x{i}, x{j})));
    end
end

end

