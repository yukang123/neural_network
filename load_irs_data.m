function [X,label]=load_irs_data(filename)
fidin=fopen(filename);
%C=textscan(fidin,'%3.1f%3.1f%3.1f%3.1f%s','delimiter',',');
C=textscan(fidin,'%f%f%f%f%s','delimiter',',');
num=C{end};
N=length(num);
label=zeros(N,1);
for i=1:N
    switch num{i}
        case 'Iris-virginica'
            label(i)=3;
        case 'Iris-setosa'
            label(i)=1;
        case 'Iris-versicolor'
            label(i)=2;        
    end    
end
%X=zeros(N,4);
X=[C{1},C{2},C{3},C{4}];
end
