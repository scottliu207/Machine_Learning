function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

% Use for-loop to loop over each centroid.(1 to K)
% Inside the loop, finds out which example was given to current centroid.
% By finding their index first and feed it to X, so that we will have only the examples which we were given to current centroid.
% Computing the average of the examples which we were given to current centroid, and store it to "centroids".
for i = 1:K;
	u_idx = find(idx ==i);
	centroids(i,:) = sum(X(u_idx,:)) / size(X(u_idx,:),1);

end;

% =============================================================


end

