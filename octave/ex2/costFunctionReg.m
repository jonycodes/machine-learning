function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% excluding the theta0 term as convention when doing regularization

temp_theta = theta(2:end);

% size(temp_theta)
z = X * theta;

% making prediction with logistic function (sigmoid function)
pred = sigmoid(z);

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

end
