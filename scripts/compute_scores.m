set_path;

% class_indices and row_indices can be used to split the computation into multiple cpus
class_indices = 1:12;
row_indices = {1:903, 1:741, 1:1128, 1:1225, 1:946, 1:903, 1:210, 1:253, 1:1128, 1:1035, 1:741, 1:561};

imgset = 'vocx';
root = fullfile(ROOT, imgset);
classes = get_classes(imgset);

%-------------------------------
% SET PARAMETERS
% PHM function
args.PHM_func = @PHM_lite;
% maximum number of nonzero entries retained in the score 
% matrices. If not 'Inf', only biggest 'max_num_entries' are 
% are retained, others are set to 0.
args.max_num_entries = 1000;
args.max_num_entries_text = '1000';
% Parameters of the function computing the standout score.
args.standout_version = 4;
args.max_iteration = 10000;
args.area_ratio = 0.5;
args.area_ratio_text = '05';
% Whether to overwrite existing standout files.
args.overwrite = true;
args.save_confidence = true; % Whether to save confidence score
args.save_standout = true; % Whether to save standout score
% feature info
args.feat_type = 'who';

%-------------------------------
args = argument_reader(args);
fprintf('%s\n', argument_checker(args));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% MAIN CODE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for cl = class_indices
  clname = classes{cl};
  fprintf('Processing for class %s\n', clname);

  % load imdb
  [images_size, proposals, bboxes] = load_imdb(args, root, clname);
  n = size(bboxes, 1);
  % get list of files to compute
  indices = build_indices(args, clname, n);
  % fprintf('%d ', size(indices, 1)); continue;
  % get paths
  feat_path = fullfile(root, clname, 'features/proposals', args.feat_type);
  [save_path_confidence, save_path_standout] = get_save_paths(args, root, clname);          
  %------------------------------------------------------------------------
  % COMPUTE SCORES
  S_confidence = {};
  S_standout = {};
  parfor row = row_indices{cl}
    %----------------------------------------
    % load and process features
    i = indices(row, 1); j = indices(row, 2);
    [feati, featj] = read_feat(args, feat_path, i, j);
    %------------------------------------------
    % compute confidence score
    begin_time_confidence = tic;
    
    score1 = PHM_confidence_lite(args.PHM_func, images_size{i}, images_size{j}, ...
                            proposals{i}, proposals{j}, ...
                            feati, featj, 'RAW');
    score2 = PHM_confidence_lite(args.PHM_func, images_size{j}, images_size{i}, ...
                            proposals{j}, proposals{i}, ...
                            featj, feati, 'RAW');
    score = max(score1, 0) + max(score2', 0);
    
    size_c = size(score);
    score = score/(size_c(1)*size_c(2))*1e6;
    if args.save_confidence
      if args.max_num_entries ~= Inf
        S_confidence{row} = sparsify_matrix(score, args.max_num_entries);
      else 
        S_confidence{row} = sparse(double(score));
      end
    end
    %---------------------------------------
    % compute standout score
    if args.save_standout
      score = args.standout_func(transpose(proposals{i}), ...
                                 transpose(proposals{j}), ...
                                 score, args.max_iteration, args.area_ratio);
      if args.max_num_entries ~= Inf
        S_standout{row} = sparsify_matrix(score, args.max_num_entries);
      else 
        S_standout{row} = sparse(double(score));
      end
    end
  end

  % save result
  save_results(args, row_indices{cl}, indices, S_confidence, S_standout, ...
               save_path_confidence, save_path_standout);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LOCAL FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [args] = argument_reader(args)
  assert(all(isfield(args, {'PHM_func', 'max_num_entries', 'max_num_entries_text', ...
                            'standout_version', 'max_iteration', 'area_ratio', 'area_ratio_text', ...
                            'overwrite', 'save_confidence', 'save_standout', 'feat_type' ...
                            })));
  % get standout function or remove relating parameters if not necessary
  if args.save_standout
    assert(isfield(args, 'standout_version'))
    if args.standout_version == 4
      args.standout_func = @standout_box_pair_v4;
    elseif args.standout_version == 5
      args.standout_func = @standout_box_pair_v5;
    else 
      error('Version of standout function not supported!');
    end
  else 
    args = rmfield(args, {'standout_version', 'max_iteration', 'area_ratio', 'area_ratio_text'});
  end

  args.score_name = args.feat_type;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function msg = argument_checker(args)
  msg = 'Message from checker: ';
  if (args.max_num_entries == Inf & ~strcmp(args.max_num_entries_text, 'full')) | ...
     (args.max_num_entries < Inf & args.max_num_entries ~= str2double(args.max_num_entries_text))
    msg = sprintf('%s\n\t%s', msg, 'max_num_entries and max_num_entries_text are not compatible');
  end

  msg = sprintf('%s\nEnd message.\n', msg);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [images_size, proposals, bboxes] = load_imdb(args, root, clname)
  imdb = load(fullfile(root, clname, [clname, '_lite.mat']));
  images_size = imdb.images_size;
  proposals = imdb.proposals;
  bboxes = imdb.bboxes;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [indices] = build_indices(args, clname, n)
  indices = [repelem([1:n]',n,1) repmat([1:n]',n,1)];
  indices = indices(indices(:,1) < indices(:,2), :);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [save_path_confidence, save_path_standout] = get_save_paths(args, root, clname)
  if args.save_confidence
    save_path_confidence = fullfile(root, clname, 'confidence', ...
                          sprintf('%s_normalized_%s', ...
                                  args.score_name, args.max_num_entries_text));
  else 
    save_path_confidence = '/tmp';
  end
  if args.save_standout
    save_path_standout = fullfile(root, clname, 'standout', ...
                          sprintf('%s_v%d_%s_normalized_%s', ...
                                  args.score_name, args.standout_version, ...
                                  args.area_ratio_text, args.max_num_entries_text)); 
  else 
    save_path_standout = '/tmp';
  end 
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [feati, featj] = read_feat(args, feat_path, i, j)
  feati = getfield(load(fullfile(feat_path, sprintf('%d.mat', i))), ...
                   'data');
  featj = getfield(load(fullfile(feat_path, sprintf('%d.mat', j))), ...
                   'data');
  assert(numel(size(feati)) == 2);
  assert(numel(size(featj)) == 2);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [] = save_results(args, row_indices, indices, ...
                           S_confidence, S_standout, ...
                           save_path_confidence, save_path_standout)
  if args.save_confidence
    mkdir(save_path_confidence);
    if exist(fullfile(save_path_confidence, 'indices.mat'), 'file') ~= 2
      save(fullfile(save_path_confidence, 'indices.mat'), 'indices');
    else 
      fprintf('Indices.mat in confidence save path has been created!\n');
    end
    fprintf('Saving confidence score ...\n');
    data = S_confidence;
    save(fullfile(save_path_confidence, sprintf('%d_%d.mat', ...
                  min(row_indices), max(row_indices))), ...
        'data', '-v7.3');
  end

  if args.save_standout
    mkdir(save_path_standout);
    if exist(fullfile(save_path_standout, 'indices.mat'), 'file') ~= 2
      save(fullfile(save_path_standout, 'indices.mat'), 'indices');
    else 
      fprintf('Indices.mat in standout save path has been created!\n');
    end
    fprintf('Saving standout score ...\n');
    data = S_standout;
    save(fullfile(save_path_standout, sprintf('%d_%d.mat', ...
                  min(row_indices), max(row_indices))), ...
        'data', '-v7.3');
  end
end
