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

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
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
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

h1 = sigmoid([ones(m, 1) X] * Theta1');
h2 = sigmoid([ones(m, 1) h1] * Theta2');
[dummy, p] = max(h2, [], 2);

y_new = zeros(m,rows(unique(y)));
for i = 1:m
  y_new(i, y(i)) = 1;
endfor

left = -y_new .* log(h2) ;
right = (1-y_new) .* log(1 - h2);
sums = sum(sum(left .- right, 1),2);
J = sums / m;
theta1_tmp = Theta1(:,[2:size(Theta1, 2)]);
theta2_tmp = Theta2(:,[2:size(Theta2, 2)]);
J = (sums / m) + ( sum(sum(theta1_tmp .* theta1_tmp,1), 2) + sum(sum(theta2_tmp .* theta2_tmp, 1), 2) ) * lambda / ( 2 * m);

Del_1 = 0;
Del_2 = 0;

% Back Propogation
for t = 1:m
% Step 1
X_t = X(t, :);
h1_t = sigmoid([1 X_t] * Theta1');
h2_t = sigmoid([1 h1_t] * Theta2');

% Step 2 

del_3 = h2_t - y_new(t,:);

% Step 3

del_2 =  del_3 * Theta2 .* [1 sigmoidGradient(h1_t)];

% Step 4
del_2 = del_2(2:end);

Del_1 = Del_1 + del_2 * h1_t';
Del_2 = Del_2 + del_3 * h2_t';

endfor

Theta1_grad = Del_1/m;
Theta2_grad = Del_2/m;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
