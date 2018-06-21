function [X,label]=load_cancer_data(filename)
fidin=fopen(filename);
C=textscan(fidin,'%s%s%s%s%s%s%s%s%s%s%s','delimiter',',');
N=length(C{1});
X=zeros(N,10);
for i=1:10
    abs=C{i};
    for j=1:N
        if strcmp(abs{j},'?')
             X(j,i)=0;
        else
            X(j,i)=str2num(abs{j});
        end
    end
end
num=C{end};
label=zeros(N,1);
for i=1:N
    switch num{i}
        case '2'
            label(i)=-1;
        case '4'
            label(i)=1;    
    end    
end
for i=1:10
    list=X(:,i);
    index_los=find(list==0);
    if ~isempty(index_los)
        for j=1:length(index_los)
            samp_lab=label(index_los(j));
            valid_index=(label==samp_lab)&(list~=0);
            X(index_los(j),i)=mean(list(valid_index));
        end
    end
end
end
