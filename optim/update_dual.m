function [lambda, mu] = update_dual(lambda, mu, x_k, e_k, nu, tau, lr)
% UPDATE_DUAL
% [lambda, mu] = update_dual(lambda, mu, x_k, e_k, nu, tau, lr)
%
% Update dual parameters lambda and mu.
%
% Parameters:
%
%       lambda: (n x 1) array, dual parameters corresponding to constraints
%               on x.
%
%       mu: (n x 1) array, dual parameters corresponding to constraints
%           on e.
%
%       x_k: (n x 1) cell, x{i} is in [0,1]^p_i where p_i is the number 
%            of proposals in image i. 
%
%       e_k: (n x n) matrix, elements are in [0,1].
%
%       nu: int, maximum number of regions retained in each image.
%
%       tau: int, maximum number of neighbors of each image.
%
%       lr: float, learning rate.
%
% Returns:
%
%       lambda: (n x 1) array, updated value of lambda.
%
%       mu: (n x 1) array, updated value of mu.
%

n = size(e_k, 1);

lambda_gradient = cellfun(@sum, x_k) - nu;
lambda = lambda + lr * lambda_gradient;
lambda = lambda .* (lambda > 0);

mu_gradient = sum(e_k, 2) - tau;
mu = mu + lr * mu_gradient;
mu = mu .* (mu > 0);

