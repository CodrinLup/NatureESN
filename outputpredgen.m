function outputpred = outputpredgen(weightstepp,indices,statepred)

[xlen, len] = size(statepred);
lenstepp = length(weightstepp);
outputpred = zeros(len,1);
for i=2:len
    j = 1;
    while(j<=length(indices)&&indices(j))
        outputpred(i) = outputpred(i) + weightstepp(j)*statepred(indices(j),i);
        j = j+1;
    end
end
end