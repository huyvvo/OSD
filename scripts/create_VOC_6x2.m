% set paths to VOC2007 dataset
voc_root = '/meleze/data0/datasets/VOC2007/';
image_path = fullfile(voc_root, 'JPEGImages');
anno_path = fullfile(voc_root, 'Annotations');
addpath(fullfile(voc_root, 'VOCdevkit/VOCcode'));
imglist_path = fullfile(ROOT, 'VOC2007_6x2_imglist'); % ROOT is set in set_path
% set path to save created datasets
save_path = fullfile(ROOT, 'vocx');
mkdir(save_path);

% parameters for the randomized Prim's algorithm
rp_params = LoadConfigFile('config/rp_4segs.mat');


classes = get_classes('vocx');
fprintf('Creating class ');
for cl = 1:12
  clname = classes{cl};
  clname_parts = split(clname, '_');
  clname_prefix = clname_parts{1};
  fprintf('%s ', clname);
  imglist = getfield(load(fullfile(imglist_path, [clname, '.mat'])), 'id');
  
  n = numel(imglist);
  images = cell(n,1);
  bboxes = cell(n,1);
  proposals = cell(n,1);
  
  parfor i = 1:n
    images{i} = imread(fullfile(image_path, [imglist{i}, '.jpg']));
    anno = PASreadrecord(fullfile(anno_path, [imglist{i}, '.xml']));
    valid_box_ids = cellfun(@(x) isequal(clname_prefix, x), {anno.objects.class});
    bboxes{i} = reshape([anno.objects(valid_box_ids).bbox], 4, [])';
    proposals{i} = RP(images{i}, rp_params);
  end
  proposals = eliminate_border_boxes(proposals, images);

  mkdir(fullfile(save_path, clname));
  save(fullfile(save_path, clname, [clname, '.mat']), 'images', 'bboxes', 'proposals');
end
fprintf('\n');

rmpath(fullfile(voc_root, 'VOCdevkit/VOCcode'));
