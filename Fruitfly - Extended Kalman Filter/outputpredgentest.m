function outputpred = outputpredgentest(weightstepp,indices,statepred,output_testing,mean_output_training)

[xlen, len] = size(statepred);
lenstepp = length(weightstepp);
outputpred = zeros(len,1);
mean_pred = 0;
mean_test = 0;
count = 0;
for i=2:len
    count = count+1;
    for j=1:lenstepp
        outputpred(i) = outputpred(i) + weightstepp(j)*statepred(indices(j),i) ;
    end
      mean_test = mean_test + output_testing (:,count);
%      mean_pred = mean_pred + outputpred(i);
%     % V2
%     outputpred(i) = outputpred(i) + (mean_test)/count - mean_output_training;
end
%V1
mean_test = (mean_test)/count - mean_output_training;
mean_test = 0;
outputpred = outputpred + mean_test;
end