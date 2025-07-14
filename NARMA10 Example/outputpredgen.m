function outputpred = outputpredgen(weightstepp,indices,statepred)

[xlen, len] = size(statepred);
lenstepp = length(weightstepp);
outputpred = zeros(len,1);
for i=2:len
        for j=1:lenstepp
                outputpred(i) = outputpred(i) + weightstepp(j)*statepred(indices(j),i);
        end
end
end