function [] = optim
% optim
% [] = optim
%
% Run optimization.
%
% Parameters: Requires global struct 'opt' with the following fields
%       
%   opt.imdb: string or struct, path to a structure contains the 
%              dataset for optimization or the dataset itself. The 
%              dataset must contain 3 cells 'proposals', 'bboxes' and 'S'.
%
%   opt.neighbors: (n x 1) cell, each contains a list of 
%                   neighbor candidates of the corresponding image.
%
%   opt.nu: int, the maximum number of proposals retained in each image.
%
%   opt.tau: int, the maximum neighbors of an image.
%
%   opt.learning_rate: float, initial stepsize for updating 
%                      \lambda and \mu.
%
%   opt.min_learning_rate: float, minimum learning rate during optim.
%
%   opt.max_iter: int, maximum number of iterations to run.
%
%   opt.threshold: float, iou threshold to compute CorLoc
%
%   opt.save_path: string, path to the folder to save outputs.
%
%   opt.dual_primal_save_step: int, the gap between iterations in which the
%                              the statistics of dual and primal values are 
%                              saved.
%
%   opt.result_save_step: int, the gap between iterations in which the
%                         the result is saved.
%

% ----------------------- CHECK AND SET DEFAULT PARAMETERS ---------------------
assert(sum(cellfun(@(x) isequal(x, 'opt'), who('global'))) == 1);
global opt

if ~isfield(opt, 'imdb') | ~isfield(opt, 'nu') | ~isfield(opt, 'tau') | ...
        ~isfield(opt, 'save_path')
   error(['All of the fields "imdb", "nu", "tau" and "save_path" ', ...
          'must be in "opt"!']);
end

if ~isfield(opt, 'learning_rate') | ~isfield(opt, 'min_learning_rate')
   error('"learning_rate" and "min_learning_rate" must be in opt!"');
end

if ~isfield(opt, 'max_iter')
    opt.max_iter = 100;
end

if ~isfield(opt, 'threshold')
    opt.threshold = 0.5;
end

if ~isfield(opt, 'dual_primal_save_step')
  opt.dual_primal_save_step = 50;
end

if ~isfield(opt, 'result_save_step')
  opt.result_save_step = 50;
end

imdb = opt.imdb;
num_neighbors = opt.num_neighbors;
neighbors = opt.neighbors;
nu = opt.nu;
tau = opt.tau;
learning_rate = opt.learning_rate;
min_learning_rate = opt.min_learning_rate;
max_iter = opt.max_iter;
threshold = opt.threshold;
save_path = opt.save_path;
dual_primal_save_step = opt.dual_primal_save_step;
result_save_step = opt.result_save_step;
clear opt; 

% ----------------------------- LOADING DATA -----------------------------------
disp('Loading data...');
if strcmp(class(imdb), 'char')
    imdb = load(imdb);
end
n = numel(imdb.proposals);
num_regions = cellfun(@(x) size(x,1), imdb.proposals);
data_loader = DataLoader(imdb.S);

% -------------------------- INITIALIZE VARIABLES ------------------------------

% running average
xe_run = zeros(sum(num_regions)+n*(n-1), 1);
% norm of the dual variable
dual_norm_stat = [];
% norm of violated constraints
violated_constraints_norm = [];
% the value of dual function computed using (x_k,e_k)
dual_stat = [];
% the value of primal function using infeasible (x_run, e_run)
S_run_stat = [];
% the value of primal function using feasible (x_run, e_run) (by scaling)
primal_stat = [];
% the value of the primal function using (x_s, e_s) computed from 
% infeasible (x_run, e_run)
objective_from_infeasible = [];
% the value of the primal function using (x_s, e_s) computed from 
% feasible (x_run, e_run)
objective_from_feasible = [];
% CorLoc score given by (x_s, e_s) computed from infeasible (x_run, e_run)
corloc_from_infeasible = [];
% CorLoc score given by (x_s, e_s) computed from feasible (x_run, e_run)
corloc_from_feasible = [];

lambda = zeros(n,1);
mu = zeros(n,1);

iter = 0;
lr_updater = LinearLRUpdater(learning_rate, min_learning_rate, max_iter);

% ---------------------- MAIN LOOP OF THE ALGORITHM ----------------------------
global main_edges cubic_info G
[main_edges, cubic_info] = build_edges(data_loader, n, num_regions, neighbors);
[G, linear_unique, cubic_unique] = build_graph;

iter_to_compute_primal = [50:50:(1000-50) 1000:100:(max_iter-100) max_iter];
begin_time = tic;
while true
    iter = iter+1;
    if iter > max_iter
        break;
    end
    fprintf('Iteration %d........................................\n', iter);

    % ----------- SOLVE STABLE SET PROBLEM AND ADD NEW XE TO HISTORY -----------
    fprintf('Graph solver starting...\n');
    [xe, L, sum_cubic, sum_linear, unlabled] = solve_max_flow(...
                              linear_unique, cubic_unique, ...
                              n, num_regions, lambda, mu, nu, tau ...
                            );
    fprintf('Graph solver stopping...\n')
    xe_run = (xe_run * (iter - 1) + xe)/iter;
    
    % -------------- GET (x_k, e_k) AND COMPUTE VIOLATED CONSTRAINTS -----------
    [ x_k, e_k ] = get_x_e(xe, n, num_regions);
    assert(~any(arrayfun(@(ii) sum(e_k(ii, setdiff(1:n, neighbors{ii}))), 1:n)));
    constraint_values = [cellfun(@sum, x_k) - nu; sum(e_k, 2) - tau];
    fprintf('X_k - nu: \n'); fprintf('%d ', constraint_values(1:n)'); fprintf('\n');
    fprintf('E_k - tau: \n'); fprintf('%d ', constraint_values(n+1:end)'); fprintf('\n');

    % ------------------- COMPUTE STATISTICS -----------------------------

    % compute norm of the dual parameter
    dual_norm_stat = [dual_norm_stat, norm([lambda; mu])];
    fprintf('L2 norm of dual parameter: %.2f\n', dual_norm_stat(end));
    % compute running averages
    [ x_run, e_run ] = get_x_e(xe_run, n, num_regions);
    % scaling (x_run, e_run) to get a feasible solution
    [ x_run_feasible, e_run_feasible ] = force_constraints_real_xe(...
                                            x_run, e_run, nu, tau);
    % compute (x_s, e_s)
    [ x_s_infeasible, e_s_infeasible ] = round_running_average(...
                    data_loader, x_run, e_run, neighbors, nu, tau ...
                  );
    [ x_s_feasible, e_s_feasible ] = round_running_average(...
                    data_loader, x_run_feasible, e_run_feasible, ...
                    neighbors, nu, tau ...
                  );
    [x_s_infeasible, e_s_infeasible] = ascent_x_e(...
                    data_loader, x_s_infeasible, e_s_infeasible, ...
                    neighbors, nu, tau, num_regions, 10 ...
                  );
    [x_s_feasible, e_s_feasible] = ascent_x_e(...
                    data_loader, x_s_feasible, e_s_feasible, ...
                    neighbors, nu, tau, num_regions, 10 ...
                  );

    % compute values with the feasible solution
    fprintf('Computing objective...\n');
    tic;
    objective_from_infeasible = [objective_from_infeasible, ...
                                 compute_objective_discrete(data_loader, ...
                                            x_s_infeasible, e_s_infeasible)];
    objective_from_feasible = [objective_from_feasible, ...
                               compute_objective_discrete(data_loader, ...
                                            x_s_feasible, e_s_feasible)];
    toc;
    % compute primal value
    if iter >= iter_to_compute_primal(1)
      fprintf('Computing approximate primal and S_run...\n');
      tic;
      S_run_stat = [S_run_stat, compute_objective_cont(...
                                    data_loader, x_run, e_run, neighbors)];
      primal_stat = [primal_stat, compute_objective_cont(...
                                    data_loader, x_run_feasible, e_run_feasible, neighbors)];
      toc;
      while ~isempty(iter_to_compute_primal) & iter >= iter_to_compute_primal(1)
        iter_to_compute_primal(1) = [];
      end
    elseif iter == 1
      S_run_stat = [0];
      primal_stat = [0];
    else
      S_run_stat = [S_run_stat, S_run_stat(end)];
      primal_stat = [primal_stat, primal_stat(end)];
    end
    % compute dual value
    dual_stat = [dual_stat, L];
    % norm of constraint violation vector
    violated_constraints_norm = [violated_constraints_norm, ...
                                 norm((cellfun(@sum, x_run) - nu) .* ...
                                      (cellfun(@sum, x_run) - nu > 0)) + ...
                                 norm((sum(e_run, 2) - tau) .* ...
                                      (sum(e_run, 2) - tau > 0))];
    fprintf('Dual: %.3f, primal: %.3f\n', L, primal_stat(end));
    fprintf('Objective from infeasible %.2f\n', objective_from_infeasible(end));
    fprintf('Objective from feasible %.2f\n', objective_from_feasible(end));  
    fprintf('Norm of violated contraints %.2f\n', violated_constraints_norm(end));

    corloc_from_infeasible = [corloc_from_infeasible, CorLoc(imdb.proposals, ...
                                                             imdb.bboxes, ...
                                                             x_s_infeasible, ...
                                                             threshold)];
    corloc_from_feasible = [corloc_from_feasible, CorLoc(imdb.proposals, ...
                                                         imdb.bboxes, ...
                                                         x_s_feasible, ...
                                                         threshold)];
    fprintf('Corloc of (x_s,e_s) from infeasible (x_run,e_run) is %.5f\n', ...
            corloc_from_infeasible(end));
    fprintf('Corloc of (x_s,e_s) from feasible (x_run, e_run) is %.5f\n', ...
            corloc_from_feasible(end));

    
    lr = get_lr(lr_updater, iter);
    fprintf('Learning rate for iteration %d is %f\n', iter, lr);
    % save current result
    save_struct.lr = lr;
    save_struct.lambda = lambda; 
    save_struct.mu = mu;
    save_struct.xe_run = xe_run;
    save_struct.xe = xe;
    save_struct.dual = dual_stat(end);
    save_struct.primal = primal_stat(end);
    save_struct.ob_feas = objective_from_feasible(end);
    save_struct.ob_infeas = objective_from_infeasible(end);
    save_struct.violated_norm = violated_constraints_norm(end);

    if iter == max_iter | mod(iter, result_save_step) == 0
      save(fullfile(save_path, sprintf('result_%d.mat', iter)), '-struct', ...
           'save_struct');
    end
    
    if iter == max_iter | mod(iter, dual_primal_save_step) == 0
      save(fullfile(save_path, 'dual_primal.mat'), 'dual_stat', 'primal_stat');
    end

    % update lambda and mu
    [ lambda, mu ] = update_dual(lambda, mu, x_k, e_k, nu, tau, lr);
    fprintf('Time elapsed: %.2f\n', toc(begin_time));
end
    

