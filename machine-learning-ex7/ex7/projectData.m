function Z = projectData(X, U, K)
%PROJECTDATA Computes the reduced data representation when projecting only 
%on to the top k eigenvectors
%   Z = projectData(X, U, K) computes the projection of 
%   the normalized inputs X into the reduced dimensional space spanned by
%   the first K columns of U. It returns the projected examples in Z.
%

% You need to return the following variables correctly.
Z = zeros(size(X, 1), K);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the projection of the data using only the top K 
%               eigenvectors in U (first K columns). 
%               For the i-th example X(i,:), the projection on to the k-th 
%               eigenvector is given as follows:
%                    x = X(i, :)';
%                    projection_k = x' * U(:, k);
%

%  m is the size of examples.
% Setting Ureduce by K. (n * K) dimension.
% Loop over each example, compute Z by X i-th example time Ureduce. Z will be m * K matrix. 
m = size(X,1);
Ureduce = U(:,1:K);
for i = 1:m;
	Z(i,:) = X(i,:) * Ureduce;
end;

% =============================================================

end
