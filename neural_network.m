clear
filename='Iris_dataset.txt';
[X,y]=load_irs_data(filename);
[X, mu, sigma] = featureNormalize(X);
rate=0.7;
[X_train,X_test,y_train,y_test]=split_data(X,y,rate);
%%
input_layer_size  = 4;  
num_labels = 3;         
m = size(X_train, 1);
%% ================  Initializing Parameters ================
hidden_layer_size = 25;  
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
lambda =1;
alpha=0.2;
iter_times=200;
nn_params=initial_nn_params;
% J=zeros(iter_times,1);
% for i=1:iter_times
%     [J(i) ,grad]=nnCostFunction(nn_params, ...
%         input_layer_size, ...
%         hidden_layer_size, ...
%         num_labels, X_train, y_train, lambda);
%     J(i)
%     nn_params=nn_params-alpha*grad;
% end
%  plot(J);

J=[];
iter_time_all=1;
J_before=0;
tic
while(1)
    [J(iter_time_all) ,grad]=nnCostFunction(nn_params, ...
        input_layer_size, ...
        hidden_layer_size, ...
        num_labels, X_train, y_train, lambda);
    if abs(J(iter_time_all)-J_before)<=0.000001
        break
    end
    nn_params=nn_params-alpha*grad;
    J_before=J(iter_time_all);
    iter_time_all=iter_time_all+1;
end
toc
plot(J);xlim([0,iter_time_all]);ylabel('J');xlabel('iteration times');
% % Create "short hand" for the cost function to be minimized
% costFunction = @(p) nnCostFunction(p, ...
%                                    input_layer_size, ...
%                                    hidden_layer_size, ...
%                                    num_labels, X_train, y_train, lambda);
% % Now, costFunction is a function that takes in only one argument (the
% % neural network parameters)
% [nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
% % Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
fprintf('Program paused. Press enter to continue.');
pause;
% ================= Implement Predict =================
pred_test = predict(Theta1, Theta2, X_test);
fprintf('\nTest Set Accuracy: %f\n', mean(double(pred_test == y_test)) * 100);
fprintf('\niteration times: %d\n', iter_time_all);
%
pred_train = predict(Theta1, Theta2, X_train);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred_train == y_train)) * 100);
fprintf('\niteration times: %d\n', iter_time_all);
