function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% store hisotry into J variable
J = 0;

% calculating prediction
pred = X * theta;

% calculating error using mean square cost function
% divived by two to make gradient decent easier
error = sum((pred .- y) .** 2);
J = (1 / (2 * m)) * error ;

end
