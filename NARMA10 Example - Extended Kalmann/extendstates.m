function statenetext = extendstates(statenet, combin, Nnet,len)

dimen = combin+3*Nnet+nchoosek(Nnet,3)+factorial(Nnet)/factorial(Nnet-2);
%dimen = combin+2*Nnet;

statenetext = zeros(int16(dimen),len);
% Order 1 states
statenetext(1:Nnet,:) = statenet;
% Order 2 product states
i = Nnet + 1;
for j = 1:Nnet
    for k=j+1:Nnet
        statenetext(i,:) = statenetext(j,:).*statenetext(k,:);
        i = i+1;
    end
end

% Order 2 squared states
for j = 1:Nnet
    statenetext(i,:) = statenetext(j,:).*statenetext(j,:);
    i=i+1;
end

% Order 3 product states
for j = 1:Nnet
    for k = j+1:Nnet
        for p = k+1:Nnet
            statenetext(i,:) = statenetext(j,:).*statenetext(k,:).*statenetext(p,:);
            i = i+1;
        end
    end
end
% Order 3 product squared states
for j = 1:Nnet
    for k = 1:Nnet
        if j~=k
            statenetext(i,:) = statenetext(j,:).*statenetext(j,:).*statenetext(k,:);
            i = i+1;
        end
    end
end
% Order 3 cube states
for j = 1:Nnet
    statenetext(i,:) = statenetext(j,:).*statenetext(j,:).*statenetext(j,:);
    i = i+1;
end
i = i-1;
end