function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% Instructions: Compute the cost of a particular choice of theta.
%               Set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


% excluding the theta0 term as convention when doing regularization
temp_theta = theta(2:end);

% calculating z to feed to the sigmoid function
z = X * theta;
% size(z)
% making prediction with logistic function (sigmoid function)
pred = sigmoid(z);
% size(pred)
% cost function (maximum likelihood estimation) plus regularization term with lambda
J = ((1/m) * sum((-y.*log(pred))-(1 - y).*log(1 - pred))) + (lambda/(2*m)) * sum(temp_theta .** 2); 

% gradient to feed to the optimzed conjugate gradient
% regularization for gradient
reg = (lambda / m) * temp_theta;

% unregularized gradient
unreg_grad = (1/m) * sum((pred - y).* X);

% adding unregularized term theta0
grad = [unreg_grad(1) (unreg_grad(2:end) + reg')];










% =============================================================

grad = grad(:);

end
