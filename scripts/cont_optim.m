set_path;

args.imgset = 'vocx';
root = fullfile(ROOT, args.imgset);
classes = get_classes(args.imgset);
%---------------------------------
args.nu = 5; 
args.tau = 10;
%---------------------------------
args.num_entries_text = '1000';
%---------------------------------
args.prefilter_neighbors = false;
args.num_neighbors = 20;
args.neighbor_root = {fullfile(ROOT, 'vocx')};
args.neighbor_type = {'neighbor_gist'};
%---------------------------------
args.lite_imdb = true;
args.normalized_score = true;
args.score_type = 'standout'; % 'confidence' or 'standout'
args.score_name = 'who_v4_05';
%---------------------------------
args = argument_reader(args);
fprintf('%s\n', argument_checker(args));
args

%------------------------------------------------------------

for cl = 1:12
  clname = classes{cl};
  fprintf('Running optimization for class %s\n', clname);
  imdb = load_imdb(args, root, clname);
  n = size(imdb.proposals, 1);
  %--------------------------------
  score_path = get_score_path(args, root, clname);
  fprintf('Score path is %s\n', score_path);
  fprintf('Loading score ...\n');
  S = getfield(load(score_path), 'S');
  if args.prefilter_neighbors
    e = load_neighbors_multiple(args, clname);
  end
  %--------------------------------
  imdb.S = S;
  clear S;
  num_regions = cellfun(@(x) size(x,1), imdb.proposals);
  %--------------------------------
  fprintf('Beginning optimization ...\n');
  global opt
  opt.imdb = imdb; clear imdb;
  if args.prefilter_neighbors
    opt.num_neighbors = args.num_neighbors;
    opt.neighbors = e;
  else
    opt.num_neighbors = n - 1;
    opt.neighbors = arrayfun(@(i) {setdiff(1:n,i)}, [1:n]'); 
  end
  opt.nu = args.nu;
  opt.tau = args.tau;
  opt.learning_rate = 10000;
  opt.min_learning_rate = 100;
  opt.max_iter = 10000;
  opt.threshold = 0.5;
  opt.dual_primal_save_step = 50;
  opt.result_save_step = 50;
  opt.save_path = fullfile(root, 'cont_optim', clname);
  mkdir(opt.save_path);
  disp(opt);
  %--------------------------------
  optim;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LOCAL FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [args] = argument_reader(args)
  assert(all(isfield(args, {'nu', 'tau', 'lite_imdb', 'score_type', 'score_name', ...
                            'prefilter_neighbors', 'num_neighbors', 'neighbor_root', 'neighbor_type', ...
                            })));
  % clean parameters to limited neighbors if necessary
  if ~args.prefilter_neighbors
    args = rmfield(args, {'neighbor_root', 'neighbor_type', 'num_neighbors'});
  end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function msg = argument_checker(args)
  msg = 'Message from checker: ';
  msg = sprintf('%s\nEnd message.\n', msg);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [imdb] = load_imdb(args, root, clname)
  if args.lite_imdb 
    imdb_path = fullfile(root, clname, [clname, '_lite.mat']);
  else 
    imdb_path = fullfile(root, clname, [clname, '.mat']);
  end 
  imdb = load(imdb_path);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [score_path] = get_score_path(args, root, clname)
  if args.normalized_score
    score_path = fullfile(root, clname, [args.score_type, '_imdb'], ...
                          sprintf('%s_normalized_%s.mat', ...
                                  args.score_name, args.num_entries_text));
  else
    score_path = fullfile(root, clname, [args.score_type, '_imdb'], ...
                          sprintf('%s_%s.mat', args.score_name, args.num_entries_text));
  end
  if exist(score_path, 'file') ~= 2
    error(sprintf('Score file %s does not exist', score_path));
  end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [e] = load_neighbors_single(args, clname, idx)
  neighbor_path = fullfile(args.neighbor_root{idx}, args.neighbor_type{idx}, clname, ...
                           sprintf('%d.mat', args.num_neighbors));
  e = getfield(load(neighbor_path), 'e');

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [e] = load_neighbors_multiple(args, clname, ids)
  if exist('ids', 'var') == 0
    ids = 1:numel(args.neighbor_root);
  end
  e = load_neighbors_single(args, clname, ids(1));
  for idx = ids(2:end)
    current_e = load_neighbors_single(args, clname, idx);
    e = arrayfun(@(i) [e{i}, current_e{i}], [1:size(current_e,1)]', 'Uni', false);
  end 
  e = cellfun(@(x) unique(x), e, 'Uni', false);
end

