function nrmse = FORCEp (Nnet, Anetwork, Bnetwork, input_training, output_training, input_testing, output_testing, input_validation, output_validation, rho, mean_output_training, tranz, trans)

state = zeros(Nnet, length(input_training));
orig_output_training = output_training;
% Computing the readout

state(:,1) = Bnetwork*input_training(:,1);
for i = 2:length(input_training)
    state(:,i) = Anetwork*state(:,i-1) + Bnetwork*input_training(:,i);
end

combin = nchoosek(Nnet,2);
% Choosing the dimension of the 3rd order extended state vector;
dimen = combin+3*Nnet+nchoosek(Nnet,3)+factorial(Nnet)/factorial(Nnet-2);
%dimen = combin+2*Nnet;

%combin = 0;
statenetext = extendstates(state,combin,Nnet,length(input_training));

% Orthogonal states

maxerr = 0;                             % Maximum error for the current step
selected_regressors = zeros(dimen,length(input_training)); % Selected regresors
regressors_step_p = statenetext;        % All regresors
errorsind = zeros(dimen,1);     % Maximum error vector corresponding to each computational step
indices = zeros(dimen,1);       % Indices of the selected regresors

% Optimum parameters for the validation

minerr = inf;    % Minimum error for the validation
optweight = 0;   % Optimum weights
optstateind = 0; % Optimum regresor indices
optstateext = 0; % Optimum extended state vector
optoutput = 0;   % Optimum output

% Initialization step

for i=1:dimen
    % Finding the best regresor
    currerr = errors(statenetext(i,:),output_training);
    if(currerr>maxerr)
        ind = i;
        maxerr = currerr;
    end
end

% First regresor

selected_regressors(1,:) = statenetext(ind,:);
indices(1) = ind;
errorsind(1,:) = maxerr;
weightstepp = orig_output_training*pinv(selected_regressors(1,:));

% For loop

for p=2:dimen
    maxerr = 0;
    ind =0;
    currerr=0;
    for i=1:dimen
        % Testing to see if the regresor has not been selected
        flag = 0;
        for j=1:p-1
            if i==indices(j)
                flag = 1;
            end
        end
        % Orthogonalize the rest of the unselected regresors
        if flag == 0
            regressors_step_p(i,:) = regressors_step_p(i,:) - wei(regressors_step_p(i,:),regressors_step_p(indices(p-1),:))*regressors_step_p(indices(p-1),:);
            %regressors_step_p(i,:) = regressors_step_p(i,:) - wei(regressors_step_p(i,:), selected_regressors(p-1,:))* selected_regressors(p-1,:);
        end
    end
    for i=1:dimen
        % Testing to see if the regresor has not been selected
        flag = 0;
        for j=1:p-1
            if i==indices(j)
                flag = 1;
            end
        end
        % Find the best unselected regresor
        if flag == 0
            currerr = errorsp(regressors_step_p(i,:),orig_output_training,output_training);
            if(currerr > maxerr)
                ind = i;
                maxerr = currerr;
            end
        end
    end
    
    % Regresor p
    if ind == 0
        break;
    end
    selected_regressors(p,:) = statenetext(ind,:);
    indices(p) = ind;
    errorsind(p,:) = maxerr;
    
    % Least Squares for the weights of the selected regressors
    
    weightstepp = orig_output_training*pinv(selected_regressors(1:p,:));
    
    % Substract the selected regressor from the training output data
    % Computing the validation error to see the optimum
    % Extended state generation for the testing data
    
    statepred = statepredgen(input_validation,length(input_validation),Nnet,Anetwork,Bnetwork);
    statepredext = extendstates(statepred,combin,Nnet,length(input_validation));
    
    % Output generation for the testing data using the
    % identified parameters at step p
    
    outputpred = outputpredgen(weightstepp,indices,statepredext);
    outputpred = outputpred + mean_output_training;
   
    outerr = sum((outputpred - output_validation').^2)/sum((output_validation - mean(output_validation)).^2);
    % Test the current error vs the optimum error
    if(outerr < minerr)
        % Save this step as the new optimum
        minerr = outerr;
        optweight = weightstepp;
        optstateind = indices;
        optstateext = statepredext;
        optoutput = outputpred;
    end
    if(1-sum(errorsind)<rho)
        break;
    end
end

statepred = statepredgen(input_testing,length(input_testing),Nnet,Anetwork,Bnetwork);
statepredext = extendstates(statepred,combin,Nnet,length(input_testing));
outputpred = outputpredgen(optweight,optstateind,statepredext);
outputpred = outputpred + mean_output_training;

% Compute the error on the testing data
trans = length(input_testing)/4;
tnet = 1:length(input_testing)-trans-1;
output_testing_tranz = output_testing(:,trans+1:length(output_testing)-1);
output_tranz = outputpred(trans+1:length(outputpred)-1,:);

nrmse = sum((output_tranz-output_testing_tranz').^2)/sum((output_testing_tranz-mean(output_testing_tranz)).^2)

% Plot the response of the network with the testing data

figure(11)
plot(tnet, output_testing_tranz, '-');             % Test signal
hold on
plot(tnet,  output_tranz, '--o');                 % ARLS signal
xlabel('KT','fontsize', 18)
ylabel('y(KT)','fontsize', 18)
set(gca,'FontSize',16)

end