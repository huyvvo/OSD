function [ x_opt, e_opt ] = force_constraints_real_xe(x_run, e_run, nu, tau)
% FORCE_CONSTRAINTS_REAL_XE
% [ x_opt, e_opt ] = force_constraints_real_xe(x_run, e_run, nu, tau)
%
% Get a feasible (x,e) form the running average (x_run, e_run) by linearly scaling infeasible row. 
%
% Parameters:
%
%       x_run: (n x 1) cell, the running average x.
%
%       e_run: (n x n) matrix, the running average e.
%
%       nu: int, constraint for each line of x.
%
%       tau: int, constraint for each line (and column) of e.
%
% Returns:
%
%       x_opt: (n x 1) cell, a feasible solution close to x_run.
%
%       e_opt: (nxn) matrix, a feasible solution close to e_run.
%


n = size(x_run, 1);
num_regions = cellfun(@numel, x_run);

% forcing that the sum of elements of each line of x to be exactly \nu
x_opt = x_run;
line_sum_x = cellfun(@sum, x_opt);
for i = 1:n
	x_opt{i} = x_opt{i} / line_sum_x(i) * nu;
end
assert(sum(cellfun(@sum, x_opt) <= nu+0.00001) == n);

% forcing that the sum of elements of each line of e to be exactly \tau
e_opt = e_run;
line_sum_e = sum(e_opt, 2);
e_opt(line_sum_e > 0,:) = e_opt(line_sum_e > 0,:) .* (tau ./ line_sum_e(line_sum_e > 0));
assert(sum(sum(e_opt, 2) <= tau + 0.00001) == n);

end

