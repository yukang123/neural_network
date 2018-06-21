function [J ,grad] = nnCostFunction_regression(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

% Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
%                  hidden_layer_size, (input_layer_size + 1));
             Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size)), ...
                 hidden_layer_size, (input_layer_size));

%  Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
%                  num_labels, (hidden_layer_size + 1));
Theta2=Theta1';
% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
ynew=y;
%a1=[ones(m,1) X];
a1= X;
z2=a1*Theta1';
a2=sigmoid(z2);
%a2=[ones(m,1) a2];
z3= a2*Theta2';
hypo=sigmoid(z3);
%hypo=z3;
cost=(ynew-hypo).^2;
J=sum(sum(cost,2))/(2*m);
% cost=ynew-hypo;
% J=-sum(sum(cost'))/m;
s1=sum(Theta1.^2);
s2=sum(Theta2.^2);
%J=J+lambda*(sum(s1(1,2:end))+sum(s2(1,2:end)))/(2*m);
J=J+lambda*(sum(s1(1,1:end))+sum(s2(1,1:end)))/(2*m);
dW=zeros(hidden_layer_size,input_layer_size);
for h=1:hidden_layer_size
    for i=1:input_layer_size
        %for k=1:m
            for o=1:num_labels
                %dW(h,i)=dW(h,i)-(X(k,o)-hypo(k,o))*1*Theta1(h,o)*a2(k,h)*(1-a2(k,h))*X(k,i); %输入—隐含层之间的梯度
                dW(h,i)=dW(h,i)-((X(:,o)-hypo(:,o))*1*Theta1(h,o))'*(a2(:,h).*(1-a2(:,h)).*X(:,i));
            end;
            %dW(h,i)=dW(h,i)-(X(k,i)-hypo(k,i))*1*a2(k,h); %加上隐含—输出层之间的梯度
            dW(h,i)=dW(h,i)-(X(:,i)-hypo(:,i))'*1*a2(:,h); %加上隐含—输出层之间的梯度
       % end
    end;
end;
dW=dW/m;
% %delta3=(hypo-ynew)';
% delta3=(hypo-ynew)'.*sigmoidGradient(hypo');
% %delta2=(Theta2'*delta3).*sigmoidGradient([ones(1,m); z2']);
% delta2=(Theta2'*delta3).*sigmoidGradient(z2');
% %Theta1_grad =(delta2(2:end,:)*a1+lambda*Theta1)/m;
% Theta1_grad =(delta2*a1+lambda*Theta1)/m;
% %Theta1_grad(:,1)=Theta1_grad(:,1)-lambda*Theta1(:,1)/m;
% Theta2_grad =(delta3*a2+lambda*Theta2)/m;
% %Theta2_grad(:,1)=Theta2_grad(:,1)-lambda*Theta2(:,1)/m;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
%grad = [Theta1_grad(:) ; Theta2_grad(:)];
grad = dW(:);

end
