function dre= dleakyrelu(x)
dre=zeros(size(x));
dre(x>0)=1;
dre(x<=0)=0.1;%这句可以不要
end