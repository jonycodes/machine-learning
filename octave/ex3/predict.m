function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Instructions: implement forward propagation neural networks with weigths theta1 and theta2
%
%       The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% adding bias term of 1
X = [ones(m, 1), X];
size(X);
size(Theta1);

% applying input to first layer  
% X is 5000 by 401 Theta1 is 25 by 401
z = X * Theta1';
% adding intercept term
g = [ones(m, 1), sigmoid(z)];


% applying input to second layer
% g is 5000 by 26 Theta2 is 10 by 26
z = g * Theta2';

% the g below is 5000 by 10
g = sigmoid(z);

% output 5000 by 1 matrix
[v p] = max(g, [], 2);






% =========================================================================


end
