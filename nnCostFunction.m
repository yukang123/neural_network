function [J ,grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

ynew=zeros(m, num_labels);
for i=1:num_labels
    ynew(find(y==i),i)=1;
end
a1=[ones(m,1) X];
z2=a1*Theta1';
a2=sigmoid(z2);
%a2=relu(z2);
%a2=tanh(z2);
%a2=leakyrelu(z2);
a2=[ones(m,1) a2];
z3= a2*Theta2';
hypo=sigmoid(z3);
cost=ynew.*log(hypo)+(1-ynew).*log(1-hypo);
J=-sum(sum(cost'))/m;
s1=sum(Theta1.^2);
s2=sum(Theta2.^2);
J=J+lambda*(sum(s1(1,2:end))+sum(s2(1,2:end)))/(2*m);
delta3=(hypo-ynew)';
delta2=(Theta2'*delta3).*sigmoidGradient([ones(1,m); z2']);
%delta2=(Theta2'*delta3).*dleakyrelu([ones(1,m); z2']);
%delta2=(Theta2'*delta3).*drelu([ones(1,m); z2']);
%delta2=(Theta2'*delta3).*dtanh([ones(1,m); z2']);
Theta1_grad =(delta2(2:end,:)*a1+lambda*Theta1)/m;
Theta1_grad(:,1)=Theta1_grad(:,1)-lambda*Theta1(:,1)/m;
Theta2_grad =(delta3*a2+lambda*Theta2)/m;
Theta2_grad(:,1)=Theta2_grad(:,1)-lambda*Theta2(:,1)/m;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
