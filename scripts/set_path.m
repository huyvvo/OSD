ROOT='~/data/releases/UODOptim';
% VLFEAT path
addpath(genpath(fullfile(ROOT, 'cho/feature')));
addpath(genpath(fullfile(ROOT, 'cho/commonFunctions')));
vlf_path = fullfile(ROOT, 'cho/vlfeat-0.9.20/toolbox/vl_setup');
run(vlf_path);

% Randomized Prim's algo path
addpath(genpath(fullfile(ROOT, 'rp')));

% tools for playing with boxes
addpath(genpath(fullfile(ROOT, 'cho/bbox_tool')));

% others
addpath(genpath(fullfile(ROOT, 'utils')));
addpath(fullfile(ROOT, 'optim'));
addpath(fullfile(ROOT, 'scripts'));

% remove path overridden by Huy
rmpath(fullfile(ROOT, 'cho/feature/who2/misc'));
rmpath(fullfile(ROOT, 'rp/evaluation/xml_toolbox'));



