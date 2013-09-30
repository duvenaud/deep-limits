function samples = stable_mvnrnd( mu, sigma, n_samples)
%
% A version of mvnrnd that doesn't needlessly complain about non-psd
% matrices.
%
% David Duvenaud
% Sept 2013

if nargin < 3; n_samples = 1; end

%L = chol(sigma);

N = numel(mu);

[reg_vectors, reg_values] = eig(sigma);
reg_values(reg_values < 0) = 0;

% Throw away 
reg_values = real(diag(reg_values));
reg_vectors = real(reg_vectors);

randvals = rand(N, n_samples); %mvnrnd(zeros(N,1), eye(N), n_samples);
samples = (reg_vectors*(randvals.*sqrt(reg_values)))' + repmat(mu', n_samples, 1);

%samples = (L'*randvals)' + repmat(mu', n_samples, 1);
%exp_samples = exp(samples);
