%----------------------------------------------------------------
% SET PARAMETERS
args.imgset = 'vocx';
root = fullfile(ROOT, args.imgset);
classes = get_classes(args.imgset);
%-----------------------------------
args.nu = 5;
args.tau = 10;
%----------------------------------
args.save_result = false;
%----------------------------------
args.iterations = 1;
args.ensemble_size = 500;
args.num_neighbor_for_one_regions = 1;
%----------------------------------
args.normalized_score = true;
args.rounding = '1000';
args.iterating ='1000';
args.initialization = 'one';
args.res_path_file = '~/code/releases/UODOptim/vocx/cont_solution_path.mat';
%----------------------------------
args.prefilter_neighbors = false;
args.num_neighbor_rounding = 20;
args.num_neighbor_iterating = 20;
args.neighbor_root = {fullfile(ROOT, 'vocx')};
args.neighbor_type = {'neighbor_gist'};
%----------------------------------
args.lite_imdb = true;
args.score_type = 'standout'; % 'confidence' or 'standout'
args.score_name = 'who_v4_05';
%-----------------------------------
args = argument_reader(args);
fprintf('%s\n', argument_checker(args));
args

%----------------------------------------------------------------
final_corloc_nu = [];
final_corloc_1 = [];
for idx = 1:12
  clname = classes{idx};
  fprintf('Processing for class %s\n', clname);
  %------------------------------
  sol_path = get_sol_path(args, root, clname);
  save_path = get_save_path(args, root, clname);
  score_path = get_score_path(args, root, clname);
  fprintf('Solution path: %s\n', sol_path);
  fprintf('Save path: %s\n', save_path);
  fprintf('Score path: %s\n', score_path);
  %------------------------------
  fprintf('Loading imdb and reading class info ...\n');
  [proposals, bboxes, imdb_path] = load_imdb(args, root, clname);
  n = size(proposals, 1);
  num_regions = cellfun(@(x) size(x,1), proposals);
  %----------------------------------------------------
  fprintf('Creating data loaders and loading scores ...\n');
  scores = getfield(load(score_path), 'S');
  DL = DataLoader(scores);
  clear scores;
  %----------------------------------------------------
  x_opt_to_save.x_opt_nu = {};
  x_opt_to_save.x_opt_1 = {};
  x_opt_to_save.x_combined = {};
  corloc_nu = [];
  corloc_1 = [];
  
  solutions = get_first_solutions(sol_path);
  for iter = args.iterations
    disp(sprintf('Iterations %d', iter));
    x_list = cell(n, args.ensemble_size);
    e_list = cell(n, args.ensemble_size);
    %-----------------------------
    l = (iter-1)*args.ensemble_size+1;
    r = iter*args.ensemble_size;
    solutions = get_solutions(sol_path, solutions, l, r);
    x_list = horzcat(solutions.x_s{l:r});
    e_list = horzcat(solutions.e_s{l:r});
    solutions.x_s(l:r) = cell(1,r-l+1);
    solutions.e_s(l:r) = cell(1,r-l+1);
    %------------------------------
    % COMBINE SOLUTIONS
    x_combined = cell(n,1);
    e_combined = cell(n,1);
    x_weight = cell(n,1);
    for i =1:n
      x_combined{i} = unique(cell2mat(x_list(i,:)));
      x_weight{i} = ones(1, numel(x_combined{i}));
      e_combined{i} = unique(cell2mat(e_list(i,:)));
    end
    fprintf('Average number of retained proposals: %.2f\n', ...
            mean(cellfun(@numel, x_combined)));
    fprintf('Average number of neighbors: %.2f\n', ...
                      mean(cellfun(@numel, e_combined)));
    %------------------------------
    % ENSEMBLE METHOD
    x_opt_nu = get_k_best_regions(DL, x_combined, e_combined, ...
                                  args.nu, args.tau, args.nu, x_weight);
    corloc_nu = [corloc_nu, CorLoc(proposals, bboxes, x_opt_nu, 0.5)];
    x_opt_1 = get_k_best_regions(DL, x_combined, e_combined, ...
                                 1, args.tau, args.num_neighbor_for_one_regions, x_weight);
    corloc_1 = [corloc_1, CorLoc(proposals, bboxes, x_opt_1, 0.5)];
    fprintf('corloc_nu: %.2f, corloc_1: %.2f\n', ...
            100*corloc_nu(end), 100*corloc_1(end));
    x_opt_to_save.x_opt_nu{iter} = x_opt_nu;
    x_opt_to_save.x_opt_1{iter} = x_opt_1;
    x_opt_to_save.x_combined{iter} = x_combined;
  end
  fprintf('Mean CorLoc_nu: %.2f / Std CorLoc_nu: %.2f\n', ...
          100*mean(corloc_nu), 100*std(corloc_nu));
  fprintf('Mean CorLoc_1: %.2f / Std CorLoc_1: %.2f\n', ...
          100*mean(corloc_1), 100*std(corloc_1));
  final_corloc_nu = [final_corloc_nu; corloc_nu];
  final_corloc_1 = [final_corloc_1; corloc_1];
  % save results
  if args.save_result
    save_result(args, save_path, corloc_nu, corloc_1, x_opt_to_save);
  end
end

fprintf('CorLoc with nu regions retained: %.2f\n', 100*mean(mean(final_corloc_nu)));
fprintf('CorLoc with 1 regions retained: %.2f\n', 100*mean(mean(final_corloc_1)));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LOCAL FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [args] = argument_reader(args)
  assert(all(isfield(args, {'nu', 'tau', 'save_result', 'lite_imdb', ...
                            'score_type', 'score_name', 'normalized_score', ...
                            'rounding', 'iterating', 'initialization', 'res_path_file', ...
                            'prefilter_neighbors', 'num_neighbor_rounding', 'num_neighbor_iterating', ...
                            'neighbor_root', 'neighbor_type', ...
                            })));
  if args.normalized_score
    args.sol_home = sprintf('solutions/norm_%d', args.nu);
    args.corloc_home = sprintf('corloc/norm_%d', args.nu);
  else 
    args.sol_home = sprintf('solutions/unnorm_%d', args.nu);
    args.corloc_home = sprintf('corloc/unnorm_%d', args.nu);
  end

  if ~args.prefilter_neighbors
    rmfield(args, 'num_neighbor_rounding');
    rmfield(args, 'num_neighbor_iterating');
    rmfield(args, 'neighbor_root');
    rmfield(args, 'neighbor_type');
    args.neighbor_text = 'full_neighbors';
  else 
    assert(numel(args.neighbor_type) == numel(args.neighbor_root));
    args.neighbor_text = join(args.neighbor_type, '_');
    args.neighbor_text = args.neighbor_text{1};
    args.neighbor_text = sprintf('%s_%d_%d', args.neighbor_text, ...
                                             args.num_neighbor_rounding, ...
                                             args.num_neighbor_iterating);
  end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function msg = argument_checker(args)
  msg = 'Message from checker: ';
  if ~any(cellfun(@(x) isequal(args.initialization, x), {'one', 'infeasible', 'feasible'}))
    msg = sprintf('%s\n\t%s', msg, 'initialization must be in {"one", "infeasible", "feasible"}');
  end

  if ~any(cellfun(@(x) isequal(args.score_type, x), {'confidence', 'standout'}))
    msg = sprintf('%s\n\t%s', msg, 'score_type must be in {"confidence", "standout"}');
  end

  msg = sprintf('%s\nEnd message.\n', msg);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [proposals, bboxes, imdb_path] = load_imdb(args, root, clname)
  if args.lite_imdb
    imdb_path = fullfile(root, clname, [clname, '_lite.mat']);
  else 
    imdb_path = fullfile(root, clname, [clname, '.mat']);
  end
  imdb = load(imdb_path);
  proposals = imdb.proposals;
  bboxes = imdb.bboxes;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [score_path] = get_score_path(args, root, clname)
  if args.normalized_score
    score_path = fullfile(root, clname, [args.score_type, '_imdb'], ...
                          sprintf('%s_normalized_%s.mat', ...
                                  args.score_name, args.iterating));
  else
    score_path = fullfile(root, clname, [args.score_type, '_imdb'], ...
                          sprintf('%s_%s.mat', args.score_name, args.iterating)); 
  end
  if exist(score_path, 'file') ~= 2
    error(sprintf('Score file %s does not exist', score_path));
  end 
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [e_rounding, e_iterating] = load_neighbors_single(args, clname, idx)
  neighbor_path_rounding = fullfile(args.neighbor_root{idx}, args.neighbor_type{idx}, clname, ...
                           sprintf('%d.mat', args.num_neighbor_rounding));
  e_rounding = getfield(load(neighbor_path_rounding), 'e');

  neighbor_path_iterating = fullfile(args.neighbor_root{idx}, args.neighbor_type{idx}, clname, ...
                           sprintf('%d.mat', args.num_neighbor_iterating));
  e_iterating = getfield(load(neighbor_path_iterating), 'e');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [e_rounding, e_iterating] = load_neighbors_multiple(args, clname, ids)
  if exist('ids', 'var') == 0
    ids = 1:numel(args.neighbor_root);
  end
  [e_rounding, e_iterating] = load_neighbors_single(args, clname, ids(1));
  for idx = ids(2:end)
    [e_r, e_i] = load_neighbors_single(args, clname, idx);
    e_rounding = arrayfun(@(i) [e_rounding{i}, e_r{i}], [1:size(e_r,1)]', 'Uni', false);
    e_iterating = arrayfun(@(i) [e_iterating{i}, e_i{i}], [1:size(e_i,1)]', 'Uni', false);
  end 
  e_rounding = cellfun(@(x) unique(x), e_rounding, 'Uni', false);
  e_iterating = cellfun(@(x) unique(x), e_iterating, 'Uni', false);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function sol_path = get_sol_path(args, root, clname)
  if strcmp(args.score_type, 'confidence')
    sol_path = fullfile(root, args.sol_home, clname, args.initialization, ...
                        sprintf('%s_confidence/%s_%s_%s', ...
                                 args.neighbor_text, args.score_name, ...
                                 args.rounding, args.iterating));
  else 
    sol_path = fullfile(root, args.sol_home, clname, args.initialization, ...
                        sprintf('%s/%s_%s_%s', ...
                                 args.neighbor_text, args.score_name, ...
                                 args.rounding, args.iterating));
  end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [save_path] = get_save_path(args, root, clname)
  if strcmp(args.score_type, 'confidence')
    save_path = fullfile(root, clname, args.corloc_home, args.initialization, ...
                         sprintf('%s_confidence/%s_%s_%s', ...
                                 args.neighbor_text, args.score_name, ...
                                 args.rounding, args.iterating));
  else 
    save_path = fullfile(root, clname, args.corloc_home, args.initialization, ...
                         sprintf('%s/%s_%s_%s', ...
                                 args.neighbor_text, args.score_name, ...
                                 args.rounding, args.iterating));
  end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [sorted_files] = sort_files(files)
  % All files are named as 'm_n.mat' where m and n are integers.
  begin_points = [];
  end_points = [];
  for i = 1:numel(files)
    filename = files{i};
    indices = cellfun(@str2num, strsplit(filename(1:end-4), '_'));
    begin_points = [begin_points indices(1)];
    end_points = [end_points indices(2)]; 
  end
  assert(numel(unique(begin_points)) == numel(begin_points));
  assert(numel(unique(end_points)) == numel(end_points));
  [~, min_idx] = sort(begin_points, 'ascend');
  sorted_files = files(min_idx);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [files] = get_solution_files(sol_path)
  files = dir(fullfile(sol_path, '*_*.mat'));
  files = {files.name};
  files = sort_files(files);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [sol] = get_first_solutions(sol_path)
  files = get_solution_files(sol_path);
  sol = load(fullfile(sol_path, files{1}));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [sol] = get_solutions(sol_path, sol, l, r)
  % get solutions from files in sol_path and update 'sol'.
  % The funtion loads the needed file(s) such that the solutions
  % indexed from 'l' to 'r' are in 'sol' at the end.
  files = get_solution_files(sol_path);
  if numel(sol.x_s) < r | isempty(sol.x_s{l}) | isempty(sol.x_s{r})
    for i = 1:numel(files)
      indices = cellfun(@str2num, strsplit(files{i}(1:end-4), '_'));
      if numel(intersect([l:r], [indices(1):indices(2)])) > 0
        current_sol = load(fullfile(sol_path, files{i}));
        sol.x_s(indices(1):indices(2)) = current_sol.x_s(indices(1):indices(2));
        sol.e_s(indices(1):indices(2)) = current_sol.e_s(indices(1):indices(2));
      end
    end
  end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [] = save_result(args, save_path, corloc_nu, corloc_1, x_opt_to_save)
  fprintf('Saving result ... ');
  mkdir(save_path);
  save(fullfile(save_path, ...
       sprintf('ensemble_%d_iterations_%d_to_%d_nnbfor_%d.mat', ...
       args.ensemble_size, args.iterations(1), args.iterations(end), ...
       args.num_neighbor_for_one_regions)), ...
       'corloc_nu', 'corloc_1');
  save(fullfile(save_path, ...
       sprintf('x_opt_ensemble_%d_iterations_%d_to_%d_nnbfor_%d.mat', ...
       args.ensemble_size, args.iterations(1), args.iterations(end), ...
       args.num_neighbor_for_one_regions)), ...
       '-struct', 'x_opt_to_save');
  fprintf('DONE!\n');
end