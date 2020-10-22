function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).
%
%   In this function, you will compute the train and test errors for
%   dataset sizes from 1 up to m. In practice, when working with larger
%   datasets, you might want to do this in larger intervals.
%

% Number of training examples
m = size(X, 1);
r = size(Xval,1);
% You need to return these values correctly
error_train = zeros(m, 1);
error_val   = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in 
%               error_train and the cross validation errors in error_val. 
%               i.e., error_train(i) and 
%               error_val(i) should give you the errors
%               obtained after training on i examples.
%
% Note: You should evaluate the training error on the first i training
%       examples (i.e., X(1:i, :) and y(1:i)).
%
%       For the cross-validation error, you should instead evaluate on
%       the _entire_ cross validation set (Xval and yval).
%
% Note: If you are using your cost function (linearRegCostFunction)
%       to compute the training and cross validation error, you should 
%       call the function with the lambda argument set to 0. 
%       Do note that you will still need to use lambda when running
%       the training to obtain the theta parameters.
%
% Hint: You can loop over the examples with the following:
%
%       for i = 1:m
%           % Compute train/cross validation errors using training examples 
%           % X(1:i, :) and y(1:i), storing the result in 
%           % error_train(i) and error_val(i)
%           ....
%           
%       end
%

% ---------------------- Sample Solution ----------------------
% Since we want to compute error from 1 to m examples every time, we need to set a for-loop to do that. 
% Selecing the number of examples(train_X, train_y), and feed them to trainLinearReg function each time to get the theta vector for current size of trainning set.
% Feeding the theta vector we just get, and feed them to linearRegCostFunction to get the error for training set and cross validation.

for i = 1 : m;
	for j = 1 :50;
		rNumT = randperm(m,i);
		rNumV = randperm(r,i);
		train_X = X(rNumT,:);
		train_y = y(rNumT,:);
		thetaVec = trainLinearReg(train_X, train_y, lambda);
		J_train(j) = linearRegCostFunction(train_X, train_y, thetaVec, 0);
		J_val(j) = linearRegCostFunction(Xval(rNumV,:), yval(rNumV,:), thetaVec, 0);
	end;
		

	error_train(i) = mean(J_train);
	error_val(i) = mean(J_val);
	

end;



% -------------------------------------------------------------

% =========================================================================

end