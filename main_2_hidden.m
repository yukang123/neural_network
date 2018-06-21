clear
filename='Iris_dataset.txt';
[X,y]=load_irs_data(filename);
[X, mu, sigma] = featureNormalize(X);
rate=0.7;
[X_train,X_test,y_train,y_test]=split_data(X,y,rate);
%%
input_layer_size  = 4;  
hidden_layer_size_1 = 25; 
hidden_layer_size_2 = 25; 
num_labels = 3;         
m = size(X_train, 1);
%% ================  Initializing Parameters ================
fprintf('\nInitializing Neural Network Parameters ...\n')
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size_1);
initial_Theta2 = randInitializeWeights(hidden_layer_size_1, hidden_layer_size_2);
initial_Theta3 = randInitializeWeights(hidden_layer_size_2, num_labels);
% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:);initial_Theta3(:)];
index=[length(initial_Theta1(:)) ,length(initial_Theta2(:)),length(initial_Theta3(:))];
fprintf('\nTraining Neural Network... \n')

%  After you have completed the assignment, change the MaxIter to a larger
%  value to see how more training helps.
options = optimset('MaxIter', 200);
%  You should also try different values of lambda
lambda =0;
alpha=2;
iter_times=200;
nn_params=initial_nn_params;
% J=zeros(iter_times,1);
% tic
% for i=1:iter_times
%     [J(i) ,grad]=nnCostFunction_2_hidden(nn_params, ...
%         input_layer_size, ...
%         hidden_layer_size_1, ...
%         hidden_layer_size_2,...
%         num_labels, X_train, y_train, lambda,index);
%     J(i)
%     nn_params=nn_params-alpha*grad;
% end
% toc
J=[];
iter_time_all=1;
J_before=0;
tic
while(1)
    [J(iter_time_all) ,grad]=nnCostFunction_2_hidden(nn_params, ...
        input_layer_size, ...
        hidden_layer_size_1, ...
        hidden_layer_size_2,...
        num_labels, X_train, y_train, lambda,index);
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
% costFunction = @(p) nnCostFunction_2_hidden(p, ...
%                                    input_layer_size, ...
%                                    hidden_layer_size_1, ...
%                                    hidden_layer_size_2,... 
%                                    num_labels, X_train, y_train, lambda,index);
% % Now, costFunction is a function that takes in only one argument (the
% % neural network parameters)
% [nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:index(1)), ...
   hidden_layer_size_1, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + index(1)):index(1)+index(2)), ...
    hidden_layer_size_2, (hidden_layer_size_1 + 1));
Theta3 = reshape(nn_params(1+index(1)+index(2):end), ...
    num_labels, (hidden_layer_size_2 + 1));
% fprintf('Program paused. Press enter to continue.\n');
% pause;
%% ================= Implement Predict =================
pred_test = predict_2_hidden(Theta1, Theta2, Theta3,X_test);
fprintf('\nTest Set Accuracy: %f\n', mean(double(pred_test == y_test)) * 100);
pred_train = predict_2_hidden(Theta1, Theta2, Theta3,X_train);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred_train == y_train)) * 100);
fprintf('\niteration times: %d\n', iter_time_all);

%%

