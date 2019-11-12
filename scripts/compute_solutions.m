set_path;
%--------------------------------------------------------------
args.imgset = 'vocx';
root = fullfile(ROOT, args.imgset);
classes = get_classes(args.imgset);
num_classes = size(classes, 1);
%----------------------------------------
% SET PARAMETERS
args.nu = 5;
args.tau = 10;
%---------------------------
args.save_result = true;
args.solution_indices = 1:500;
%---------------------------
args.normalized_score = true;
args.rounding = '1000';
args.iterating = '1000';
args.initialization = 'one';
args.res_path_file = fullfile(ROOT, 'vocx/cont_solution_path.mat');
%---------------------------
args.prefilter_neighbors = false;
args.num_neighbor_rounding = 20;
args.num_neighbor_iterating = 20;
args.neighbor_root = {fullfile(ROOT, 'vocx')};
args.neighbor_type = {'neighbor_gist'};
%---------------------------
args.lite_imdb = true;
args.score_type = 'standout'; % 'confidence' or 'standout'
args.score_name = 'who_v4_05';
%-------------------------------
args = argument_reader(args);
fprintf('%s\n', argument_checker(args));

%---------------------------------------------------------------
corloc_nu_all = [];
corloc_1_all = [];
for cl = 1:12
  clname = classes{cl};
  fprintf('Computing solutions for class %s\n', clname);
  % load imdb
  [proposals, bboxes, imdb_path] = load_imdb(args, root, clname);
  n = size(proposals, 1);
  num_regions = cellfun(@(x) size(x,1), proposals);
  % get score path
  [score_path_rounding, score_path_iterating] = get_score_paths(args, root, clname);
  % loading scores
  fprintf('Loading scores and creating data loaders\n');
  if strcmp(score_path_rounding, score_path_iterating) == 1
    DL = get_loader(score_path_rounding); % use only one DataLoader to save memory
  else 
    rounding_DL = get_loader(score_path_rounding);
    iterating_DL = get_loader(score_path_iterating);
  end
  % load neighbors
  if args.prefilter_neighbors
    [e_rounding, e_iterating] = load_neighbors_multiple(args, clname);
  else
    e_rounding = arrayfun(@(i) {setdiff([1:n], i)}, [1:n]');
    e_iterating = arrayfun(@(i) {setdiff([1:n], i)}, [1:n]'); 
  end
  % initialize (x,e)
  [x,e] = initialize_xe(args, n, num_regions, cl);
  %--------------------------------
  % CREATE SOLUTIONS
  corloc_nu_class = {};
  corloc_1_class = {};
  sol_x = {};
  sol_e = {};
  DL_exists = exist('DL', 'var');
  parfor i = args.solution_indices
    if mod(i, 100) == 1
      fprintf('Computing solution %d\n', i);
    end
    if DL_exists == 1
      [x_s, e_s] = round_running_average(DL, x, e, e_rounding, args.nu, args.tau, randperm(n,n));
      [x_s, e_s] = ascent_x_e(DL, x_s, e_s, e_iterating, args.nu, args.tau, num_regions, 10);
      x_opt_1 = get_k_best_regions(DL, x_s, e_s, 1, args.tau, 1);
    else 
      [x_s, e_s] = round_running_average(rounding_DL, x, e, e_rounding, args.nu, args.tau, randperm(n,n));
      [x_s, e_s] = ascent_x_e(iterating_DL, x_s, e_s, e_iterating, args.nu, args.tau, num_regions, 10);
      x_opt_1 = get_k_best_regions(iterating_DL, x_s, e_s, 1, args.tau, 1);
    end 
    sol_x{i} = x_s;
    sol_e{i} = e_s;
    corloc_1_class{i} = CorLoc(proposals, bboxes, x_opt_1, 0.5);
    corloc_nu_class{i} = CorLoc(proposals, bboxes, x_s, 0.5);
    fprintf('CorLoc nu/1: %.4f/%.4f\n', corloc_nu_class{i}, corloc_1_class{i});
  end
  fprintf('\n');
  corloc_1_class = cell2mat(corloc_1_class);
  corloc_nu_class = cell2mat(corloc_nu_class);
  fprintf('CorLoc_1: %.2f +/- %.2f\n', 100*mean(corloc_1_class), 100*std(corloc_1_class));
  corloc_nu_all = [corloc_nu_all; corloc_nu_class];
  corloc_1_all = [corloc_1_all; corloc_1_class];
  %----------------------------------
  % SAVE RESULTS
  save_path = get_save_path(args, root, clname);
  fprintf('Results will be saved to %s\n', save_path);
  if args.save_result
    sol.x_s = sol_x;
    sol.e_s = sol_e;
    save_result(args, save_path, sol, args.solution_indices);
  end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LOCAL FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [args] = argument_reader(args)
  assert(all(isfield(args, {'nu', 'tau', 'save_result', ...
                            'score_type', 'score_name', 'normalized_score', ...
                            'rounding', 'iterating', 'initialization', 'res_path_file', ...
                            'prefilter_neighbors', 'num_neighbor_rounding', 'num_neighbor_iterating', ...
                            'neighbor_root', 'neighbor_type', ...
                            })));
  if args.normalized_score
    args.sol_home = sprintf('solutions/norm_%d', args.nu);
  else 
    args.sol_home = sprintf('solutions/unnorm_%d', args.nu);
  end

  if strcmp(args.initialization, 'infeasible') | strcmp(args.initialization, 'feasible')
    args.res_paths = getfield(load(args.res_path_file), 'data');
  else 
    args = rmfield(args, 'res_path_file');
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

function [score_path_rounding, score_path_iterating] = get_score_paths(args, root, clname)
  if args.normalized_score
    score_path_rounding = fullfile(root, clname, [args.score_type, '_imdb'], ...
                                   sprintf('%s_normalized_%s.mat', ...
                                           args.score_name, args.rounding));
    score_path_iterating = fullfile(root, clname, [args.score_type, '_imdb'], ...
                                    sprintf('%s_normalized_%s.mat', ...
                                            args.score_name, args.iterating));
  else
    score_path_rounding = fullfile(root, clname, [args.score_type, '_imdb'], ...
                                   sprintf('%s_%s.mat', args.score_name, args.rounding));
    score_path_iterating = fullfile(root, clname, [args.score_type, '_imdb'], ...
                                    sprintf('%s_%s.mat', args.score_name, args.iterating)); 
  end
  if exist(score_path_rounding, 'file') ~= 2
    error(sprintf('Rounding score file %s does not exist', score_path_rounding));
  end
  if exist(score_path_iterating, 'file') ~= 2
    error(sprintf('Iterating score file %s does not exist', score_path_iterating));
  end 
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [loader] = get_loader(score_path)
  score = getfield(load(score_path), 'S');
  loader = DataLoader(score);
  clear score;
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

function [save_path] = get_save_path(args, root, clname)
  if strcmp(args.score_type, 'confidence')
    save_path = fullfile(root, args.sol_home, clname, args.initialization, ...
                         sprintf('%s_confidence/%s_%s_%s', ...
                                 args.neighbor_text, args.score_name, ...
                                 args.rounding, args.iterating));
  else 
    save_path = fullfile(root, args.sol_home, clname, args.initialization, ...
                         sprintf('%s/%s_%s_%s', ...
                                 args.neighbor_text, args.score_name, ...
                                 args.rounding, args.iterating));
  end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [x,e] = initialize_xe(args, n, num_regions, cl)
  if strcmp(args.initialization, 'one') == 1
    x = mat2cell(ones(1, sum(num_regions)), [1], num_regions)';
    e = ones(n);
    e(sub2ind([n,n], 1:n, 1:n)) = 0;
  else
    result = load(args.res_paths{cl});
    xe = result.xe_run;
    [x,e] = get_x_e(xe, n, num_regions);
    if strcmp(args.initialization, 'feasible') == 1
      [x,e] = force_constraints_real_xe_2(x, e, args.nu, args.tau);
    end
  end
end 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [] = save_result(args, save_path, solutions, solution_indices)
  fprintf('Saving results ... ');
  mkdir(save_path);
  save(fullfile(save_path, sprintf('%d_%d.mat', ...
                                 solution_indices(1), ...
                                 solution_indices(end))), ...
     '-struct', 'solutions');
  fprintf('DONE!\n');
end 