function [J ,grad] = nnCostFunction_2_hidden(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size_1, ...
                                   hidden_layer_size_2, ...
                                   num_labels, ...
                                   X, y, lambda,index)
%NNCOSTFUNCTION 
% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:index(1)), ...
                 hidden_layer_size_1, (input_layer_size + 1));

Theta2 = reshape(nn_params(1 + index(1) :index(1)+index(2)), ...
                 hidden_layer_size_2, (hidden_layer_size_1 + 1));
             
Theta3 = reshape(nn_params(1+index(1)+index(2):end), ...
                 num_labels, (hidden_layer_size_2 + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
% J = 0;
% Theta1_grad = zeros(size(Theta1));
% Theta2_grad = zeros(size(Theta2));
% Theta3_grad = zeros(size(Theta3));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
%%
ynew=zeros(m, num_labels);
for i=1:num_labels
    ynew(find(y==i),i)=1;
end
%% 
a1=[ones(m,1) X];
z2=a1*Theta1';
a2=sigmoid(z2);
a2=[ones(m,1) a2];
z3= a2*Theta2';
a3=sigmoid(z3);
a3=[ones(m,1) a3];
z4=a3*Theta3';
hypo=sigmoid(z4);
cost=ynew.*log(hypo)+(1-ynew).*log(1-hypo);
J=-sum(sum(cost'))/m;
s1=sum(Theta1.^2);
s2=sum(Theta2.^2);
s3=sum(Theta3.^2);
J=J+lambda*(sum(s1(1,2:end))+sum(s2(1,2:end))+sum(s3(1,2:end)))/(2*m);
delta4=(hypo-ynew)';
delta3=(Theta3'*delta4).*sigmoidGradient([ones(1,m); z3']);
delta2=(Theta2'*delta3(2:end,:)).*sigmoidGradient([ones(1,m); z2']);
Theta1_grad =(delta2(2:end,:)*a1+lambda*Theta1)/m;
Theta1_grad(:,1)=Theta1_grad(:,1)-lambda*Theta1(:,1)/m;
Theta2_grad =(delta3(2:end,:)*a2+lambda*Theta2)/m;
Theta2_grad(:,1)=Theta2_grad(:,1)-lambda*Theta2(:,1)/m;
Theta3_grad =(delta4*a3+lambda*Theta3)/m;
Theta3_grad(:,1)=Theta3_grad(:,1)-lambda*Theta3(:,1)/m;







    


















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:);Theta3_grad(:)];


end