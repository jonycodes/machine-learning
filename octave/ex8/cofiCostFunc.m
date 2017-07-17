function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%


% making prediction
pred = X * Theta';

% calculating cost function without regularization for the the movies that have been rated only
J = ( 1 / 2) * sum ( (pred(R==1) .-  Y(R==1)) .^ 2 );


% for-loop iterating over the movies, especially the X matrix with row-i
% the inner loop is vectorized 
for i=1:num_movies
    % finding all users that have rated the i-th movie
    rated = R(i, :) == 1;
    % getting the theta parameters of users who have rated the i-th movie
    temp_theta = Theta(rated, :); 
    % getting the expected value of users who have rated the i-th movie
    temp_y = Y(i, rated);    
    % making prediciton 
    pred = X(i, :) * temp_theta';
    % updating gradient for optmization function, plus regularization term lambda * x(i, :), i-th movie features
    X_grad(i, :) = ((pred - temp_y ) * temp_theta) + lambda * X(i, :);    
endfor

% for-loop iterating over the users, especially the Theta matrix with row-i
for j=1:num_users
    % getting the movie ratings that user j-th has rated
    rated = R(:, j) == 1;
    % getting the movie paramters user-j has rated
    temp_X = X(rated, :);
    % getting the expected outputs for movies user-j has rated
    temp_y = Y(rated, j);    
    % making predictions
    pred = temp_X * Theta(j, :)';
    % calculating gradient with regularization term lambda * theta(J, :), j-th user parameters
    Theta_grad(j, :) = ( pred - temp_y )' * temp_X + lambda * Theta(j, :);     
endfor

% adding regularization term to the cost function. Regularization for Theta and X
J = J + (lambda/2) * sum(  sum ( Theta .^ 2 )) + (lambda/2) * sum(  sum(  X .^ 2  )  );













% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
