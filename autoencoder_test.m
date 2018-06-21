
%% Initialization
clear ; close all; clc

% Setup the parameters you will use for this exercise
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size =   300;   % 25 hidden units
output_layer_size  = 400;  %same size as the input

%% =========== Loading and Visualizing Data =============
% Load Training Data
fprintf('Loading and Visualizing Data ...\n')
load('ex4data1.mat');
m = size(X, 1);
% Randomly select 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:100);
displayData(X(sel, :));
fprintf('Program paused. Press enter to continue.\n');
[X, mu, sigma] = featureNormalize(X);
rate=0.7;
rank=randperm(m);
X_train=X(rank(1:floor(rate*m)),:);
X_test=X(rank(floor(rate*m)+1:end),:);
y_train=X_train;
y_test=X_test;
%% ================ Initializing Parameters ================

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, output_layer_size );
%initial_Theta2 = initial_Theta1';
% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

%% =================== Training NN ===================
%  You have now implemented all the code necessary to train a neural 
%  network. To train your neural network, we will now use "fmincg", which
%  is a function which works similarly to "fminunc". Recall that these
%  advanced optimizers are able to train our cost functions efficiently as
%  long as we provide them with the gradient computations.
%
fprintf('\nTraining Neural Network... \n')
lambda = 0;
% alpha=0.02;
% iter_times=4000;
% nn_params=initial_nn_params;
% J=zeros(iter_times,1);
% for i=1:iter_times
%     [J(i) ,grad]= nnCostFunction_regression(nn_params, ...
%         input_layer_size, ...
%         hidden_layer_size, ...
%         output_layer_size, X_train, y_train, lambda);
%     nn_params=nn_params-alpha*grad;
% end
% plot(J);

%  After you have completed the assignment, change the MaxIter to a larger
%  value to see how more training helps.
options = optimset('MaxIter', 200);
%  You should also try different values of lambda
lambda = 1;
% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction_regression(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   output_layer_size, X_train, y_train, lambda);
% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 output_layer_size, (hidden_layer_size + 1));

fprintf('Program paused. Press enter to continue.\n');
pause;


% %% ================= Part 9: Visualize Weights =================
% %  You can now "visualize" what the neural network is learning by 
% %  displaying the hidden units to see what features they are capturing in 
% %  the data.
% 
% fprintf('\nVisualizing Neural Network... \n')
% 
% displayData(Theta1(:, 2:end));
% 
% fprintf('\nProgram paused. Press enter to continue.\n');
% pause;

% %% ================= Part 10: Implement Predict =================
% %  After training the neural network, we would like to use it to predict
% %  the labels. You will now implement the "predict" function to use the
% %  neural network to predict the labels of the training set. This lets
% %  you compute the training set accuracy.
% 
% pred_test = predict_regression(Theta1, Theta2, X_test);
% 
% fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred_test == y_test)) * 100);
% 
% 
