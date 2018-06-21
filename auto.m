clear
input=rand(20,100);
%************************************************%
I=100;
H=5;
O=100;
W=2*rand(H,I)-1; 
vh=2*rand(H,1)-1; 
vo=2*rand(I,1)-1;
W_1=W;
W_2=W;
n=0.5; 
a=0; 
DA=1000; 
datasize=20; 
error=zeros(DA,1);
for m=1:1:DA
    error(m)=0;
    for t=1:datasize
         Hinput=W*(input(t,:)')-vh;
         Houtput=1./(exp(-Hinput)+1); 
         Oinput=W'*Houtput-vo;
         output=Oinput; 
         for  o=1:1:O
              error(m)=error(m)+0.5*(input(t,o)-output(o))^2;
         end;
         dW=zeros(H,I);
         for h=1:1:H
              for i=1:I
                   for o=1:1:O
                        dW(h,i)=dW(h,i)-(input(t,o)-output(o))*1*W(h,o)*Houtput(h)*(1-Houtput(h))*input(t,i); 
                   end;
                   dW(h,i)=dW(h,i)-(input(t,i)-output(i))*1*Houtput(h); 
              end;
         end;
          W=W_1-n*dW+a*(W_1-W_2); 
          W_2=W_1;
          W_1=W;
    end;
    error(m)=error(m)/datasize;
   if m>=2&&error(m)>=error(m-1)
         n=n*exp(-0.1);
   end;
end;
for t=1:datasize
         Hinput=W*(input(t,:)')-vh;
          Houtput=1./(exp(-Hinput)+1);
          Oinput=W'*Houtput-vo;
         output(:,t)=Oinput;
end;
input-output'
plot(2:DA,error(2:end),'-');