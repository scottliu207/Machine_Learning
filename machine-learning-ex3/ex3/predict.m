function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Add bias units to each feature.
a1 = [ones(size(X, 1), 1), X];

% Compute hidden units(a2).
z2 = a1 *Theta1';
a2 = sigmoid(z2);

% Add bias units each feature of a2.
a2 = [ones(size(a2,1),1), a2];

% Compute output layer units.
z3 = a2 * Theta2';
a3 = sigmoid(z3);

% Finding out the maximun probablity, and locate it from its index.
[max_val, p] =max(a3,[],2); 





% =========================================================================


end
