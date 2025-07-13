clear all
close all
clc

% Input signal generation and flags initialization
trans = 0;
Ts = 1;
N = 40;                                                  % System dimension
L = 550;                                                % Signal Length
notests = 1;                                            % Number of tests
noise = rand(1,L);                                       % White Noise
contorr = 0;
avgynet = 0;
avgyfnet = 0;
% % Filter design
% Ts = 1;                                            % Sampling time
% Fs = 1/Ts;                                         % Sampling Frequency (Hz)
% Fn = Fs/2;                                         % Nyquist Frequency (Hz)
% Fco = Fs/10;                                       % Cutoff Frequency (Hz)
% Wp = Fco/Fn;                                       % Normalised Cutoff Frequency (rad)
% Ws = 1.6*Wp;                                       % Stopband Frequency (rad)
% Rp = 3;                                            % Passband Ripple (dB)
% Rs = 20;                                           % Stopband Ripple (dB)
% [n,Wn]  = buttord(Wp,Ws,Rp,Rs);                    % Calculate Filter Order
% [b,a]   = butter(n,Wn);                            % Calculate Filter Coefficients
% [sos,g] = tf2sos(b,a);                             % Convert To Second Order Section Representation
% figure(1)
% freqz(sos, 1024, Fs)                               % Plot Filter Response (Bode Plot)
% flago = 0;
% % Filtering the input to the dynamics of the system
%
% noise = filter(b,a,noise);
% noise = noise - mean(noise);
% noise = 2 * noise;
inoise = load('noise.mat');
noise = inoise.noise;
noise = noise-mean(noise);
noise = 1*noise;
rho = 0.01;
immsemed = 0;                                            % Average IMMSE for ARLS
immsemedf = 0;                                           % Average IMMSE for FORCE

% Target system
matices = load('matrices.mat');
% Aclosedtarget = zeros(N,N);
% for j=1:N/2
%     a1=2*rand-1;
%     b1=2*rand-1;
%     while(abs(a1+i*b1)>=0.99)
%         a1=2*rand-1;
%         b1=2*rand-1;
%     end
%     Aclosedtarget(2*j-1,2*j-1) = a1;
%     Aclosedtarget(2*j,2*j) = a1;
%     Aclosedtarget(2*j-1,2*j) = -b1;
%     Aclosedtarget(2*j,2*j-1) = b1;
% end
% if(mod(Nnet,2))
%     Aclosedtarget(Nnet,Nnet) = 2*rand-1;
% end
% P = rand(N,N);
% P = P/norm(P);
% Aclosedtarget = P*Aclosedtarget*inv(P);
% Btarget = 2*rand(N,1)-1;
% Ctarget = 2*rand(1,N)-1;

Aclosedtarget = matices.Amtarget;                          % Closed-loop system matrix
Btarget = matices.Btarget;                                 % Input matrix
Ctarget = matices.Ctarget;                                 % Output matrix
Dtarget = 0;                                               % Feedthrough matrix
% Aclosedtarget = matices.Amtarget;                  
tranz = floor(L/11);

CO = ctrb(Aclosedtarget, Btarget);                       % Controllability of the target system
OB = obsv(Aclosedtarget, Ctarget);                       % Observability of the target system
rOB = rank(OB);                                          % Controllability rank
rCO = 0;                                          % Observability rank

% Target output generation

[y,state_training] = affinetarget(length(Aclosedtarget(1,:)), Aclosedtarget, Btarget, Ctarget, L, noise);
t = 0:1:L-1;

% % Can add post-synaptic noise to the output
SNR = 200;
y=awgn(y,SNR);

% Input generation plot

figure(2)
subplot(2,1,1)
plot(t,noise,'k','Linewidth',2);
xlabel('Time(s)','fontsize', 20)
ylabel('u(t)','fontsize', 20)
set(gca,'FontSize',20)
title ('a','fontsize', 20)
ylim([-0.5 1.5])
xlim([0 550]) ;
subplot(2,1,2)

plot(t,y,'k','Linewidth',2);
xlabel('Time(s)','fontsize', 20)
ylabel('y^*(t)','fontsize', 20)
set(gca,'FontSize',20)
xlim([0 550]);
title ('b','fontsize', 20)
output(1,1:1:L) = y;
input(1,1:1:L) = noise;
save('points.mat','input','output');

% Training and test data generation

input_training = input(1,1:1:L/2);
input_validation = input(1,L/2+1:1:3*L/4);
input_testing = input(1,3*L/4+1:1:L);
output_training = output(1,1:1:L/2);
output_validation = output(1,L/2+1:1:3*L/4);
output_testing = output(1,3*L/4+1:1:L);

ooutput_training = output_training;
mean_output_training = mean(output_training);
%mean_output_training = 0;
output_training = ooutput_training - mean_output_training;
%output_validation = output_validation - mean_output_training;
%output_testing = output_testing - mean_output_training;
orig_output_training = output_training;
save('training.mat','input_training','output_training');
save('testing.mat','input_testing','output_testing');

Nnet = N/2;    % Network dimensionality

% Billings linear-block identification
% With tfest

dat = iddata(input_training',output_training',Ts);                      % Training data for the identification of the linear block
tfesttarget = tfest(dat,Nnet,'Ts',Ts);                           % Transfer function estimation
MSEtfesttarget = tfesttarget.Report.Fit.MSE/sum((ooutput_training-mean(ooutput_training)).^2);     % NRMSE estimation error for the linear block
[numtarg, dentarg, Tstarg] = tfdata(tfesttarget);                           % Extract nominator, denominator and sampling time

numtarg = cell2mat(numtarg);
dentarg = cell2mat(dentarg);
[Att, Btt, Ctt, Dtt] = tf2ss(numtarg,dentarg);                      % State-space representation from the transfer function

Btarg = Btt;                                                        % Target input matrix
Atarg = Att;                                                    % Target system matrix

%Target state generation from the determined matrices
statedata = targetstategen(Nnet, L, noise, Atarg, Btarg);
statedata = statedata' - mean(statedata');
statedata = statedata';
state_training = statedata(:,trans+1:1:floor(L/2)+trans);
for count=1:notests

    % Intiliazing the networks parameters

    while(1)
        Anetwork = zeros(Nnet,Nnet);
        for j=1:Nnet/2
            a1=2*rand-1;
            b1=rand;
            while(abs(a1+i*b1)>=1)
                a1=2*rand-1;
                b1=rand;
            end
            Anetwork(2*j-1,2*j-1) = a1;
            Anetwork(2*j,2*j) = a1;
            Anetwork(2*j-1,2*j) = -b1;
            Anetwork(2*j,2*j-1) = b1;
        end
        if(mod(Nnet,2))
            Anetwork(Nnet,Nnet) = 2*rand-1;
        end
        P = rand(Nnet,Nnet);
        P = P/norm(P);

        Anetwork = P*Anetwork*inv(P);
        Bnetwork = (2*rand(Nnet,1)-1);                            % Network input matrix
        Dnetwork = Dtarget;
        Cnetwork = zeros(1,Nnet);                           % Network output matrix
        Knetwork = rand(1,Nnet);                           % Network feedback matrix

        Initstate = zeros(3*length(Anetwork(1,:)),1);
        InitState(length(Anetwork(1,:))+1:2*length(Anetwork(1,:)),1) = Bnetwork;
        InitState(2*length(Anetwork(1,:))+1:3*length(Anetwork(1,:)),1) = Knetwork;
        InitState(:,1) = mean(InitState(:,1));
        Acl = eye(3*length(Anetwork(1,:)), 3*length(Anetwork(1,:)));
        Acl(1:length(Anetwork(1,:)),1:length(Anetwork(1,:))) = Anetwork;
        B = zeros(3*length(Anetwork(1,:)),1);
        B(1:length(Anetwork(1,:))) = Bnetwork;
        K = zeros(1,3*length(Anetwork(1,:)));
        K(1:length(Anetwork(1,:))) = Knetwork;

        SF = @(x,u)((Acl-[x(length(Acl)/3+1:2*length(Acl)/3); zeros(2*length(Acl)/3,1)]*[x(2*length(Acl)/3+1:length(Acl))' zeros(1,2*length(Acl)/3)])*x+[x(length(Acl)/3+1:2*length(Acl)/3); zeros(2*length(Acl)/3,1)]*u) - [x(length(Acl)/3+1:2*length(Acl)/3); zeros(2*length(Acl)/3,1)]*x'*[x(2*length(Acl)/3+1:length(Acl))' zeros(1,2*length(Acl)/3)]' ;
        % SF = @(x,u)((Acl-B*K)*x+B*u)-B*x'*K';
        MF = @(x)(ones(1,length(x))*x);
        obj = extendedKalmanFilter(SF,MF,InitState);
        for ll = 1:length(input_training)
            statec = correct(obj,output_training(ll));
            statep = predict(obj,input_training(ll));
            B(1:length(Anetwork(1,:))) = statep(length(Anetwork(1,:))+1:2*length(Anetwork(1,:)));
            K(1,1:length(Anetwork(1,:))) = statep(2*length(Anetwork(1,:))+1:3*length(Anetwork(1,:)));
        end
        Bnetwork = B(1:length(Anetwork(1,:)));
        Knetwork = K(1,1:length(Anetwork(1,:)));
        Anetworkclosed = (Anetwork-Bnetwork*Knetwork);
        if (max(abs(eig(Anetworkclosed)))<=1)
            break
        end

    end

    statenet = statepredgen(input_training,length(input_training),Nnet,Anetworkclosed,Bnetwork);

    % Extending the state vector for the 3rd order polynomial
    combin = nchoosek(Nnet,2);
    dimen = combin+3*Nnet+nchoosek(Nnet,3)+factorial(Nnet)/factorial(Nnet-2);
    statenetext = extendstates(statenet,combin,Nnet,length(input_training));
    % Extended state generation for the validation data

    statepred = statepredgen(input_validation,length(input_validation),Nnet,Anetworkclosed,Bnetwork);
    statepredext = extendstates(statepred,combin,Nnet,length(input_validation));
    errvect = zeros(dimen,1);                   % Validation error vector for each step
    % Orthogonal states

    maxerr = 0;                             % Maximum error for the current step
    selected_regressors = zeros(dimen,length(input_training)); % Selected regresors
    regressors_step_p = statenetext;                           % All regresors
    errorsind = zeros(dimen,1);     % Maximum error vector corresponding to each computational step with regards to the training output
    indices = zeros(dimen,1);       % Indices of the selected regresors

    % Optimum parameters for the validation

    minerr = inf;    % Minimum error for the validation
    optweight = 0;   % Optimum weights
    optstateind = 0; % Optimum regresor indices
    optstateext = 0; % Optimum extended state vector
    optoutput = 0;   % Optimum output

    % Initialization step
    initial_errs = zeros(dimen,1);
    for i=1:dimen
        % Finding the best regresor
        currerr = errors(statenetext(i,:),output_training);
        initial_errs(i,1) = currerr;
        if(currerr>maxerr)
            ind = i;
            maxerr = currerr;
        end
    end

    % First regresor selection and weight computation

    selected_regressors(1,:) = statenetext(ind,:);
    indices(1) = ind;
    errorsind(1,:) = maxerr;

    % Weight computation of first regressor
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
            % with the last selected regressor
            if flag == 0
                regressors_step_p(i,:) = regressors_step_p(i,:) - wei(regressors_step_p(i,:),regressors_step_p(indices(p-1),:))*regressors_step_p(indices(p-1),:);
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

        % Regresor p selection and weight computation
        if ind == 0
            break;
        end
        selected_regressors(p,:) = statenetext(ind,:);

        indices(p) = ind;
        errorsind(p,:) = maxerr;

        % Least Squares for the weights of the p selected regressors

        weightstepp = orig_output_training*pinv(selected_regressors(1:p,:));

        % Computing the validation error to see the optimum

        % Output generation for the testing data using the
        % identified parameters at step p

        outputpred = outputpredgen(weightstepp,indices,statepredext);
        outputpred = outputpred + mean_output_training;
        output_validationn = output_validation(1,1:length(output_validation));
        outputpred = outputpred(1:length(outputpred));
        outerr = sum((outputpred - output_validationn').^2)/sum((output_validationn-mean(output_validationn)).^2);
        errvect(p) = outerr*100;
        % Test the current error vs the optimum error
        if(outerr < minerr)
            % Save this step as the new optimum
            minerr = outerr;
            optweight = weightstepp;
            optstateind = indices;
            optstateext = statepredext;
            optoutput = outputpred;
            flago = 0;
        end
        if(1-sum(errorsind)<rho)
            break;
        end
        if(flago>100)
            break;
        end
        flago = flago+1;
    end

    %transit = floor(length(output_testing)/4);
    transit = 0;
    tnet = 0:Ts:length(output)*Ts-Ts-transit*Ts;
    statepred = statepredgen(input,length(input),Nnet,Anetworkclosed,Bnetwork);
    statepredext = extendstates(statepred,combin,Nnet,length(input));
    outputpred = outputpredgen(optweight,optstateind,statepredext);
    outputpred = (outputpred + mean_output_training);
    % Compute the error on the testing data
    outputpred = outputpred(transit+1:length(output));
    output_testingg = output(transit+1:length(output));
    outerr = sum((outputpred - output_testingg').^2)/sum((output_testingg-mean(output_testingg)).^2);

    err = outerr
    ynet = outputpred;

    % FORCE Algorithm

    Afnetwork = Anetwork;                                       % Network system matrix
    Bfnetwork = Bnetwork;                                       % Network input matrix
    Dfnetwork = Dtarget;
    Cfnetwork = zeros(1,Nnet);                                  % Network output matrix
    Kfnetwork = zeros(1,Nnet);                                  % Network feedback matrix
    % Training the output

    [errf,outputpred] = FORCEpp(Nnet, Afnetwork, Bfnetwork, input_training, orig_output_training, input, output, input_validation, output_validation, rho, mean_output_training, tranz, trans,Ts);
    yfnet = outputpred;


    % Plot of the error between the responses of the output of the target
    % system and the trained system

    figure(5)
    plot(abs(ynet-output_testingg'));    % ARLS point-wise error
    hold on
    plot(abs(yfnet-output_testingg));   % FORCE point-wise error
    title('Error between the methodologies responses')
    xlabel('Time(s)','fontsize', 16)
    ylabel('y(t)','fontsize', 16)

    immsemed = immsemed + err;
    avgynet = avgynet + ynet;
    avgyfnet = avgyfnet + yfnet;

end
avgynet = avgynet / notests;
figure(3)
subplot(3,1,1)
plot(t,noise,'k','Linewidth',2);
xlabel('Time(s)','fontsize', 20)
ylabel('u(t)','fontsize', 20)
set(gca,'FontSize',20)
title ('a','fontsize', 20)
xlim([0 L]) ;
ylim([-0.5 1.5])
subplot(3,1,2)

plot(t,y,'k','Linewidth',2);
xlabel('Time(s)','fontsize', 20)
ylabel('y(t)','fontsize', 20)
set(gca,'FontSize',20)
xlim([0 L]);
%ylim([-5 15])
title ('b','fontsize', 20)

subplot(3,1,3)
plot(tnet, output,'k', 'Linewidth', 3);     % Test signal
hold on
plot(tnet, avgynet,'r', 'Linewidth', 2);    % AFRICO signal
hold on
plot(tnet, avgyfnet,'b', 'LineWidth', 2);   % FORCE signal

xlim([0 L]) ;
xlabel('Time(s)','fontsize', 20)
ylabel('y(t)','fontsize', 20)
set(gca,'FontSize',20)
title ('c','fontsize', 20)
immsemed = immsemed/notests
immsemedf = immsemedf/notests