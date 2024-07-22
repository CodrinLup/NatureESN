function outputpred = foutputpredgen(Cfnetwork,statepred)

[xlen, len] = size(statepred);
outputpred = zeros(len,1);
for i=2:len
    outputpred(i) = Cfnetwork*statepred(:,i);
end
end