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




%------------------------------- Part I ----------------------------------
% Unregularized cost function

% Input layer
% Adding bias unit to a1(X), and computing z2.
a1 = [ones(size(X,1),1), X];
z2 = a1 * Theta1';

% Hidden layer.
% Calling sigmoid function to compute a2.
% Adding bias unit to a2, and computing z3. 
a2 = sigmoid(z2);
a2 = [ones(size(a2,1),1), a2];
z3 = a2 * Theta2';

% Output layer.
% Calling sigmoid function to compute a3(h). 
a3 = sigmoid(z3);
h = a3;

% Turning y from the number of vector(1,2,...,10) to the biany matrix(0,1).
y_matrix = eye(num_labels)(y,:);

% Transposing y_martix and using trace()to compute the sum of diagonal elements.
% Or using double-sum() after element-wise muliplying to get the summation.
% J will be just one number.

%% NOTE: Using double-sum() after transposing y_matrix will get an incorrect result.

J = trace((-y_matrix' * log(h) - (1-y_matrix)' * log(1-h))) / m;
% J = sum(sum(-y_matrix .* log(h) - (1-y_matrix) .* log(1-h))) / m;


% Regularized cost function
% Excluding the bias unit in Theta1 and Theta2.(should separte the frist column from others.)
Theta1_regu = sum(sum(Theta1(:,2:end) .^2));
Theta2_regu = sum(sum(Theta2(:,2:end) .^2));

% Computing regularized and addded to J.
regu = (lambda / (2 * m)) * (Theta1_regu + Theta2_regu);
J = J + regu; 


%----------------------------- Part II ----------------------------------
% Computing delta3 and delta2.
% Theta2 must be exculding bias unit before used to compute dleta2, since the bias unit of Theta2 has no contribute to the previous layer.
d3 = a3 - y_matrix;
d2 = d3 * Theta2(:,2:end) .* sigmoidGradient(z2);

% Computing partial derivative Delta, matrix muliply to do the automatic accumulation.
% If we use for-loop for computing, the formula chanded.(Delta(l) = Delta(l) + d(l+1) * a(l))
Delta1 =  d2' * a1;
Delta2 =  d3' * a2;



%----------------------------- Part III ----------------------------------
% Set the bias units of Theta1 and Theta2 to 0, so we can keep the matrix size also separate first column from others.
Theta1(:,1) = 0; Theta2(:,1) = 0;

% Reularized gradient.
Theta1_regu = (lambda / m) * Theta1;
Theta2_regu = (lambda / m) * Theta2;


Theta1_grad = (Delta1 / m) + Theta1_regu;
Theta2_grad = (Delta2 / m) + Theta2_regu;



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
