function result=leakyrelu(z)
result=zeros(size(z));
index=find(z>0);
index_1=find(z<=0);
if ~isempty(index)
    result(index)=z(index);
end
if ~isempty(index_1)
    result(index_1)=0.1*z(index_1);
end
end