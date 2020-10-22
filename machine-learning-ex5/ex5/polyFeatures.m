function [X_poly] = polyFeatures(X, p)
%POLYFEATURES Maps X (1D vector) into the p-th power
%   [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x 1) and
%   maps each example into its polynomial features where
%   X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];
%


% You need to return the following variables correctly.
X_poly = zeros(numel(X), p);

% ====================== YOUR CODE HERE ======================
% Instructions: Given a vector X, return a matrix X_poly where the p-th 
%               column of X contains the values of X to the p-th power.
%
% 

% ---------------------- For-loop method ----------------------
% Set number of rows as m.
%m = size(X,1);

% Loop over for each row and each column.
%for i = 1:m;
%	for j = 1:p;
%		X_poly(i,j) = X(i) .^ (j);
%	end;
%end;

% ---------------------- bsxfun fuction method ----------------------
% Collecting p value to a 1 to p row_vector, so that we can use p-th power to broadcast to each column.
p_rowV = [1:p];
X_poly = bsxfun(@power, X, p_rowV);


% =========================================================================

end
