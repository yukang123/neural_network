function p = predict_regression(Theta1, Theta2, X)
% Useful values
m = size(X, 1);
% You need to return the following variables correctly 
h1 = sigmoid([ones(m, 1) X] * Theta1');
p = sigmoid([ones(m, 1) h1] * Theta2');
% =========================================================================
end
