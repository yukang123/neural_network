function dre= drelu(x)
dre=zeros(size(x));
dre(x>0)=1;
dre(x<=0)=0;%�����Բ�Ҫ
end