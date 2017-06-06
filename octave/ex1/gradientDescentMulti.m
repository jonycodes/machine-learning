function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
errors = zeros(size(X,1), size(theta,1));
for iter = 1:num_iters

    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    pred = X * theta;

    % calculating error using matrix and vector mulitplication 
    % here theta0 and theta1 are being calculating separately but simultaneously
    % since errors is a 1x2 matrix  
    errors =  (pred - y) .* X;

    % theta0 and theta1 are being calculated sepeartely using vectors
    theta = theta - ((alpha / m) * sum(errors))';
    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
