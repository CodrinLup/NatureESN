function statepred = statepredgen(input,leninput,lensys,A,B)

statepred = zeros(lensys,leninput);
statepred(:,1) = B*input(:,1);
for i = 2:leninput
    statepred(:,i) = A*statepred(:,i-1) + B*input(:,i);
end
end