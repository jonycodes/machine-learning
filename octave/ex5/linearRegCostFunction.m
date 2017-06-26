function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% making prediction
pred = X * theta;
% calculating prediction error
pred_error = pred - y;

% excluding the theta0 term from regularization
temp_theta = theta(2:end);

% calculating cost function
J = (1/(2*m)) * sum(pred_error .** 2) + (lambda/(2*m)) * sum(temp_theta .** 2);

% calculating regularization term except for theta0
regul = [0 ((lambda/m) * temp_theta)'];
 
% unregularized gradient
unreg_grad = (1/m) * sum(pred_error .* X);

% calculating regularized gradient
grad = unreg_grad + regul;

grad = grad(:);

end
