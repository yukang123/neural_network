clear
filename='Wisconsin_Cancer_dataset.txt';
[X,y]=load_cancer_data(filename);
X_val=X(:,2:end);
[X, mu, sigma] = featureNormalize(X_val);
rate=0.7;
[X_train,X_test,y_train,y_test]=split_data(X_val,y,rate);
input_layer_size  = 9;  
hidden_layer_size = 25;   
num_labels = 2;         
m = size(X_train, 1);
%% ================  Initializing Parameters ================
fprintf('\nInitializing Neural Network Parameters ...\n')
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];
%% ===================  Training NN ===================
fprintf('\nTraining Neural Network... \n')

%  After you have completed the assignment, change the MaxIter to a larger
%  value to see how more training helps.
options = optimset('MaxIter', 50);
%  You should also try different values of lambda
lambda = 1;
% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X_train, y_train, lambda);
% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================= Implement Predict =================
pred_test = predict(Theta1, Theta2, X_test);
fprintf('\nTest Set Accuracy: %f\n', mean(double(pred_test == y_test)) * 100);


