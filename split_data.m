function [X_train,X_test,y_train,y_test]=split_data(X,y,rate)
%multi classes
X_train=[];
X_test=[];
y_train=[];
y_test=[];
num_label=length(unique(y));
for i=1:num_label
    X_s=X(find(y==i),:);
    num_s=length(X_s);
    rank=randperm(num_s);
    X_train=[X_train;X_s(rank(1:floor(rate*num_s)),:)];    
    y_train=[y_train;i*ones(floor(rate*num_s),1)];
    X_test=[X_test;X_s(rank(floor(rate*num_s)+1:end),:)];
    y_test=[y_test;i*ones(num_s-floor(rate*num_s),1)];
end
end