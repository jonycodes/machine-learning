function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

% creating all_y_vector variable is a m by num_labels matrix 
% the output of each training sample is stored in a row vector 
% with a 1 on the right answer e.g if right ouput is 6 then [0 0 0 0 0 1 0 0 0 0]
all_y_vector = zeros(size(m), size(num_labels + 1));
for i=1:m
    all_y_vector(i, y(i)) = 1;   
endfor

size(Theta1);

% adding bias term
a1 = [ones(m ,1), X];

% calculating z1 
z = a1 * Theta1';

% using sigmoid function to calculate output of the first layer and adding bias term
a2 = [ones(m, 1), sigmoid(z)];

% calculating z2
z = a2 * Theta2';

% calculating final output
a3 = sigmoid(z);

% removing bias term to calculate regulatization
temp_theta1 = Theta1(:,2:end);
temp_theta2 = Theta2(:,2:end);


% maximum likelihood estimation cost function 
% modified for neural networks:
% sum all the k number of outputs of the neural network:  
% 5000 (training size) by 10 (k outputs) 
% Summing all nodes gives a 5000 by 1 vector
% Then we sum that vector
% added with regularization for theta value
% for regularization we sum the features first:
% the number of activation units for theta
% then we sum the number of output units 
% e.g theta is 25 (outputs) by 400 (inputs) we sum all 400 parameters for each 25
% then sum those 25 parameters
% we do this for each single theta that belongs to a layer in the network
% then we add all these results and then multiply them by lambda/2m 
% surprisinly we can do all of this in ONE line of code ^_^ thanks to octave O:
J = (1/m * sum(sum(-all_y_vector.*log(a3)-(1-all_y_vector).*log(1 - a3)))) + (lambda/(2*m)) * (sum(sum(temp_theta1 .** 2), 2) + sum(sum(temp_theta2 .** 2), 2));

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

%  initialzing delta values that hold
%  the accumuladors for all training samples
delta1 = zeros(size(Theta1));
delta2 = zeros(size(Theta2));

% One of the trickiest part of back propagation
% we loop through each single training input
% perform feedforward first then back prop
for i=1:m 
    % we get the training sample as a row vector
    a1 = X(i, :)(:);
    % we add the bias unit
    a1 = [1; a1];
    % we compute z for layer one
    z2 = Theta1 * a1;
    % we compute a2, the output of layer one and input of layer three
    a2 = [1; sigmoid(z2)];
    % we compute z3, the pre-final output for our three layer NN
    z3 = Theta2 * a2;
    % we computer a3 the final output of our network
    a3 = sigmoid(z3);
    % we get the y values for this training set
    % the y value is a row vector of either 0 or 1, 
    % where the one is at the position for the right output
    y = all_y_vector(i, :)(:);
    % we compute the small delta ("error" of the final layer)
    % which is just the prediction minus our output 
    % d3 is used to calculate the delta of the previous layer
    d3  = a3 - y;
    % then we compute d2 (error of the second to final layer)
    % by using the d3 the "error" of the final layer
    % we also calculate the sigmoid gradient of z2 (not a2!, interesting...) 
    % the formula for sigmoid gradient is sigmoid(z)*(1 - sigmoid(z))
    % during this process we also remove the bias unit
    % since sigmoid gradient does not has a bia unit 
    % because we computed it using z2  
    % we use d2 to calculus the delta of the previous layer
    d2  = (Theta2' * d3)(2:end) .* sigmoidGradient(z2);

    % we update our accumulator for the second layer delta2
    % simply by adding our current delta 
    % to the previews small delta times our activation input a2
    delta2 = delta2 + d3*a2';
    
    % we do the same for the layer before the last,
    % which is the first layer
    % same formula but in this case we use d2 
    % which comes from the next layer after this
    % and a1, our activation input for this layer
    delta1 = delta1 + d2*a1';
endfor

% then we calculate the gradient by simply multiplying (1/m) to our accumulator delta
Theta1_grad = (1/m)*delta1;
Theta2_grad = (1/m)*delta2;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.

% to add regularization we simply add the values of our previews gradient 
% to the regularization term which is (lambda/m)*theta excludid
% theta0 the first parameter in theta
% since our theta is matrix we exclude the whole first column of theta by adding 0's instead
Theta1_grad =  Theta1_grad + [zeros(size(Theta1_grad, 1), 1)  (lambda/m)*temp_theta1];
Theta2_grad =  Theta2_grad + [zeros(size(Theta2_grad, 1), 1)  (lambda/m)*temp_theta2];

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
