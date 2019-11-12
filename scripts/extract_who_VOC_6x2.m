imgset = 'vocx';
root = fullfile(ROOT, imgset);
classes = get_classes(imgset);

load('bgwho.mat'); % load a struct 'bg' into memory;

fprintf('Processing for class ');
for cl = 1:12
  clname = classes{cl};
  fprintf('%s ', clname);

  save_path_who = fullfile(root, clname, 'features/proposals/who');  
  mkdir(save_path_who);
  
  imdb = load(fullfile(root, clname, [clname, '.mat']));  
  n = size(imdb.images, 1);
  parfor i = 1:n
    feat_hog = extract_segfeat_hog(imdb.images{i}, struct('coords', imdb.proposals{i}));
    feat_hog = feat_hog.hist;
    feat_who = (feat_hog - repmat(bg.mu_bg', size(feat_hog, 1), 1))*bg.inv_sig;
    save_par(fullfile(save_path_who, sprintf('%d.mat', i)), feat_who);
  end
end 
fprintf('\n');

