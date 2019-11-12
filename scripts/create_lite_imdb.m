set_path;
imgsets = {'vocx'};
for imgset_idx = 1:numel(imgsets)
  imgset = imgsets{imgset_idx};
  root = fullfile(ROOT, imgset);
  classes = get_classes(imgset);
  num_classes = numel(classes);
  for cl = 1:12
    clname = classes{cl};
    fprintf('%s ', clname);
    imdb_path = fullfile(root, clname, [clname, '.mat']);
    save_path = fullfile(root, clname, [clname, '_lite.mat']);
    imdb = load(imdb_path);
    n = size(imdb.images, 1);
    images_size = cellfun(@(x) [size(x,1), size(x,2)], imdb.images, 'UniformOutput', false);
    imdb = rmfield(imdb, 'images');
    imdb.images_size = images_size;
    save(save_path, '-struct', 'imdb');
  end
end
