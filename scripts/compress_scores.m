args.imgset = 'vocx';
root = fullfile(ROOT, args.imgset);
classes = get_classes(args.imgset);
%-------------------------------
% SET PARAMETERS
% 'confidence' or 'standout' 
args.score_type = 'standout';
args.score_name = 'who_v4_05_normalized_1000';
% number of top entries to keep in each score matrix
args.num_keep = Inf;
args.num_keep_text = '1000';
%-------------------------------
args = argument_reader(args);
fprintf('%s\n', argument_checker(args));
args

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% MAIN CODE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('Compress %s for class ', args.score_type);
for cl = 1:12
  clname = classes{cl};
  fprintf('%s ', clname);
  
  % get score files
  score_path = fullfile(root, clname, args.score_type, args.score_name);
  % get only score files, not indices.mat
  files = dir(fullfile(score_path, '*_*.mat')); 
  files = {files.name};
  num_files = numel(files);
  % indices.mat contains corresponding subscripts of score matrices 
  indices = getfield(load(fullfile(score_path, 'indices.mat')), 'indices');
  % get number of images
  imdb = load(fullfile(root, clname, [clname, '_lite.mat']));
  n = size(imdb.proposals, 1);
  clear imdb;

  % begin collecting scores
  S = cell(n,n);
  for i = 1:num_files
    % read indices from filenames
    filename = files{i};
    end_points = cellfun(@str2num, ...
                         strsplit(filename(1:end-4), '_'));
    rows = end_points(1):end_points(2);
    % load scores
    scores = getfield(load(fullfile(score_path, filename)), 'data');
    % sparsify score matrices if necessary (num_keep < Inf)
    if args.num_keep ~= Inf
      for row = rows 
        scores{row} = sparsify_matrix(scores{row}, args.num_keep);
      end
    end
    S(sub2ind([n,n], indices(rows,1), ...
                     indices(rows,2))) = scores(rows);
  end
  % save result
  mkdir(fullfile(root, clname, [args.score_type, '_imdb']));
  save(fullfile(root, clname, [args.score_type, '_imdb'], ...
                args.score_name_to_save), 'S', '-v7.3');
end 
fprintf('\n');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LOCAL FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [args] = argument_reader(args)
  assert(all(isfield(args, {'score_type', 'score_name', 'num_keep', })));
  pos = strfind(args.score_name, '_');
  pos = pos(end);
  args.score_name_to_save = sprintf('%s_%s.mat', args.score_name(1:pos-1), args.num_keep_text);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function msg = argument_checker(args)
  msg = 'Message from checker: ';
  if strcmp(args.score_type, 'confidence') == 0 && strcmp(args.score_type, 'standout') == 0
      msg = sprintf('%s\n\t%s', msg, 'score_type must be "confidence" or "standout"');
  end
  msg = sprintf('%s\nEnd message.\n', msg);
end
