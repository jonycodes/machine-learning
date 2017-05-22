function [theta] = normalEqn(X, y)
%NORMALEQN Computes the closed-form solution to linear regression 
%   NORMALEQN(X,y) computes the closed-form solution to linear 
%   regression using the normal equations.

theta = zeros(size(X, 2), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the code to compute the closed form solution
%               to linear regression and put the result in theta.
%

% ---------------------- Sample Solution ----------------------

% more efficient way to compute inverse of matrix instead of pinv,
% use the \ operator which is optimized to use gradient decent to find x on the equality: 
% Ax = b (x = pinv(A)*b) instead of manually inversing the matrix 
theta = (X'*X) \ X'*y;


end
