function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
% Create lists for C and sigma, which contain values we want to try.
C_list = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
s_list = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];

% Create a matrix that contains three columns for C, sigma, error.
results = ones(length(C_list) * length(s_list),3);

% Use nest-loop to loop over each C and sigma.
% Max row will be length(C_list) * length(s_list).
% Prediction should give us a vector which contains {0,1} compare with yval.
% We can use mean to caculate the percentage of our prediction. (double means double-precision).

row = 1;
for i = 1:length(C_list);
	for j = 1: length(s_list);

		model = svmTrain(X, y, C_list(i), @(x1, x2) gaussianKernel(x1,x2,s_list(j)));
		prediction = svmPredict(model, Xval);
		error = mean(double(prediction != yval));
		results(row,:) = [C_list(i) s_list(j) error];
		row += 1;
	end;
end;

[minErrorVal P] = min(results(:,3));

C = results(P,1);
sigma = results(P,2);





% =========================================================================

end
