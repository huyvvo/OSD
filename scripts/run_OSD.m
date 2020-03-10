
% Change the path of ROOT in set_path because executing
set_path;
% Initialize parallel pool with 12 workers, comment the following line and replace
% all 'parfor' in the scripts by 'for' for using only 1 cpu.
pp = parpool('local', 12);
% Change the path to VOC 2007 datasets before executing this line
fprintf('Creating imdbs ...\n');
create_VOC_6x2;
% create a lite version of imdbs, useful for computing scores
create_lite_imdb;
% extract WHO for proposals
fprintf('Extracting WHO features for proposals ...\n');
extract_who_VOC_6x2;
% compute confidence and standout scores
fprintf('Computing scores ...\n');
compute_scores;
compress_scores;

% uncomment the following line to run continuous optimization
% it takes ~ 1 hour for 1000 iterations of CO for each class
% the number of iterations can be set by modifying 'opt.max_iter' variable
% for speed, run cont_optim on multiple cpus with different value of 'cl'
% cont_optim;

% change 'initialization' to 'infeasible' to get solutions with CO, 
% modify also the file '../vocx/cont_solution_path.mat' for the correct
% paths to continuous solutions
fprintf('Computing solutions ...\n');
compute_solutions;
% change 'initialization' to 'infeasible' to post process solutions with CO
fprintf('Post processing solutions ...\n');
postproc_solutions;
% Close the parallel session
delete(gcp('nocreate'));

