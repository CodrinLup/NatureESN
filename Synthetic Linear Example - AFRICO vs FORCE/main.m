clear all
close all

%% Input signal generation and flags initialization
trans = 0;                                              % Transitory regime for AFRICO signal
Ts = 1;
N = 40;                                                 % System dimension
L = 4*550;                                              % Signal Length
notests = 1;                                            % Number of tests
noise = rand(1,L);                                      % White Noise
contorr = 0;
avgynet = 0;                                            % AFRICO output average over notests
avgyfnet = 0;                                           % FORCE output average over notests

rho = 0.01;
immsemed = 0;                                            % Average IMMSE for AFRICO
immsemedf = 0;                                           % Average IMMSE for FORCE
Nnet = N;                                                % Network dimensionality       

%% Target system
matices = load('matrices.mat');
P = rand(N,N);                                          % Similarity transformation matrix
Pm = inv(P);                                            % Inverse of the similarity transformation matrix

Antarget = zeros(N,N);
for j=1:N/2
    a1=2*rand-1;
    b1=rand;
    while(abs(a1+1i*b1)>0.2)
        a1=2*rand-1;
        b1=rand;
    end
    Antarget(2*j-1,2*j-1) = a1;
    Antarget(2*j,2*j) = a1;
    Antarget(2*j-1,2*j) = -b1;
    Antarget(2*j,2*j-1) = b1;
end
if(mod(Nnet,2))
    Anetwork(Nnet,Nnet) = 2*rand-1;
end
Amtarget = P*Antarget*Pm;
Btarget = rand(N,1);                                       % Input matrix
Ctarget = rand(1,N);                                       % Output matrix
Dtarget = 0;                                               % Feedthrough matrix
Aclosedtarget = Amtarget;                                  % Closed-loop system matrix
transf = floor(L/11);                                      % Transitory regime for FORCE signal

CO = ctrb(Aclosedtarget, Btarget);                       % Controllability of the target system
OB = obsv(Aclosedtarget, Ctarget);                       % Observability of the target system
rOB = rank(OB);                                          % Controllability rank
rCO = 0;                                                 % Observability rank

%% Target output generation

[y,state_training] = affinetarget(length(Aclosedtarget(1,:)), Aclosedtarget, Btarget, Ctarget, L, noise);
t = 0:1:L-1;

% % Can add post-synaptic noise to the output
 SNR = 50;
 y=awgn(y,SNR);

% Input generation plot

figure(1)
subplot(2,1,1)
plot(t,noise,'k','Linewidth',2);
xlabel('Time(s)','fontsize', 20)
ylabel('u(t)','fontsize', 20)
set(gca,'FontSize',20)
title ('a','fontsize', 20)
xlim([0 L]) ;
ylim([-0.5 1.5])
subplot(2,1,2)

plot(t,y,'k','Linewidth',2);
xlabel('Time(s)','fontsize', 20)
ylabel('y^*(t)','fontsize', 20)
set(gca,'FontSize',20)
xlim([0 L]);
%ylim([-5 15])
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
output_training = ooutput_training - mean_output_training;
%output_validation = output_validation - mean_output_training;
%output_testing = output_testing - mean_output_training;
orig_output_training = output_training;

save('training.mat','input_training','output_training');
save('testing.mat','input_testing','output_testing');

%% STAGE 1 - EKF estimation of input and state-feedback weights
% Billings linear-block identification
% With tfest

dat = iddata(output_training',input_training',Ts);                      % Training data for the identification of the linear block
tfesttarget = oe(dat,[Nnet,Nnet,1]);
%tfesttarget = tfest(dat,Nnet,'Ts',Ts);                                     % Transfer function estimation
MSEtfesttarget = tfesttarget.Report.Fit.MSE;                                % NRMSE estimation error for the linear block
[numtarg, dentarg, Tstarg] = tfdata(tfesttarget);                           % Extract nominator, denominator and sampling time

numtarg = cell2mat(numtarg);
dentarg = cell2mat(dentarg);
[Att, Btt, Ctt, Dtt] = tf2ss(numtarg,dentarg);                      % State-space representation from the transfer function

Btarg = Btt;                                                        % Target input matrix
Atarg = (Att);                                                      % Target system matrix
P = rand(Nnet,Nnet);
P = P/norm(P);
Atarg = P*Atarg*inv(P);

%% Target state generation from the determined matrices
% AFRICO and FORCE algorithms testing
statedata = targetstategen(Nnet, L, noise, Atarg, Btarg);
state_training = statedata(:,trans+1:1:floor(L/2)+trans);

for count=1:notests
    % Intiliazing the networks parameters
    while(1)
        Anetwork = zeros(Nnet,Nnet);
        for j=1:Nnet/2
            a1=2*rand-1;
            b1=rand;
            while(abs(a1+1i*b1)<0.7 || abs(a1+1i*b1)>0.8)
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
        Bnetwork = 2*rand(Nnet,1)-1;                            % Network input matrix
        Dnetwork = Dtarget;
        Cnetwork = zeros(1,Nnet);                               % Network output matrix
        Knetwork = 2*rand(1,Nnet)-1;                                % Network feedback matrix
        InitState = zeros(3*length(Anetwork(1,:)),1);
%         InitState(length(Anetwork(1,:))+1:2*length(Anetwork(1,:)),1) = Bnetwork;
%         InitState(2*length(Anetwork(1,:))+1:3*length(Anetwork(1,:)),1) = Knetwork;
%         InitState(:,1) = mean(InitState(:,1));
        Acl = eye(3*length(Anetwork(1,:)), 3*length(Anetwork(1,:)));
        Acl(1:length(Anetwork(1,:)),1:length(Anetwork(1,:))) = Anetwork;
        B = zeros(3*length(Anetwork(1,:)),1);
        B(1:length(Anetwork(1,:))) = Bnetwork;
        K = zeros(1,3*length(Anetwork(1,:)));
        K(1:length(Anetwork(1,:))) = Knetwork;
        SF = @(x,u)((Acl-[x(length(Acl)/3+1:2*length(Acl)/3); zeros(2*length(Acl)/3,1)]*[x(2*length(Acl)/3+1:length(Acl))' zeros(1,2*length(Acl)/3)])*x+[x(length(Acl)/3+1:2*length(Acl)/3); zeros(2*length(Acl)/3,1)]*u) - [x(length(Acl)/3+1:2*length(Acl)/3); zeros(2*length(Acl)/3,1)]*x'*[x(2*length(Acl)/3+1:length(Acl))' zeros(1,2*length(Acl)/3)]' ;
        %SF = @(x,u)((Acl-B*K)*x+B*u)-B*x'*K';
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
        %Anetworkclosed = Anetwork;
        if (max(abs(eig(Anetworkclosed)))<=1)
            break
        end
    end

    statenet = statepredgen(input_training,length(input_training),Nnet,Anetworkclosed,Bnetwork);
%% STAGE 2 - Linear Readout Estimation
    % Extending the state vector for the 3rd order polynomial
    Cnetwork = orig_output_training*pinv(statenet);
    % transit = floor(length(output_testing)/4);
    transit = 0;
    tnet = 0:Ts:length(output)*Ts-Ts-transit*Ts;
    statepred = statepredgen(input,length(input),Nnet,Anetworkclosed,Bnetwork);
    outputpred = Cnetwork*statepred;
    outputpred = (outputpred + mean_output_training);
    % Compute the error on the testing data
    outputpred = outputpred(transit+1:length(output));
    output_testingg = output(transit+1:length(output));
    outerr = sum((outputpred' - output_testingg').^2)/sum((output_testingg-mean(output_testingg)).^2);

    err = outerr
    ynet = outputpred;

%% FORCE Algorithm

    Afnetwork = Anetwork;                                       % Network system matrix
    Bfnetwork = Bnetwork;                                       % Network input matrix
    Dfnetwork = Dtarget;
    Cfnetwork = zeros(1,Nnet);                                  % Network output matrix
    Kfnetwork = 2*rand(Nnet,1)-1;                   % Network feedback matrix

    [errf,outputpred,Cfnetwork] = FORCEpp(Nnet, Anetwork, Bnetwork, input_training, orig_output_training, input, output, input_validation, output_validation, rho, mean_output_training, transf, trans,Ts,Kfnetwork);
    yfnet = outputpred;

    syss = ss(Anetworkclosed,Bnetwork,Cnetwork,Dnetwork,Ts);
    
    immsemed = immsemed + err;
    immsemedf = immsemedf + errf;
    avgynet = avgynet + ynet;

end
%% RESULTS PLOTS
% Plot of the response of the target system vs the response of the trained
% system on the training data

tsyss = ss(Aclosedtarget,Btarget,Ctarget,Dtarget,Ts);
fsyss = ss(Anetwork+Kfnetwork*Cfnetwork,Bnetwork,Cfnetwork,Dtarget,Ts);
avgynet = avgynet / notests;
avgyfnet = avgyfnet / contorr;
figure (3)
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
plot(tnet, output_testingg,'k', 'Linewidth', 3);    % Test signal
hold on
plot(tnet, avgynet,'r', 'Linewidth', 2);            % AFRICO signal
xlim([0 L]) ;
xlabel('Time(s)','fontsize', 20)
ylabel('y(t)','fontsize', 20)
set(gca,'FontSize',20)
title ('c','fontsize', 20)

%% Frequency response of AFRICO and FORCE

figure (4)
P = bodeoptions;
P.Xlim = [0.01 4];
P.Title.String = 'd';
P.Title.FontSize = 20;
P.Xlabel.FontSize = 20;
P.Ylabel.FontSize = 20;
bodeplot(tsyss, 'k', syss, 'r', P)
hold on
bodeplot(fsyss, 'b', syss, 'r', P)
% Fh = gcf;                                                   % Handle To Current Figure
% Kids = Fh.Children;                                         % Children
% AxAll = findobj(Kids,'Type','Axes');                        % Handles To Axes
% Ax1 = AxAll(1);                                             % First Set Of Axes
% LinesAx1 = findobj(Ax1,'Type','Line');                      % Handle To Lines
% LinesAx1(2).LineWidth = 2.5;                                % Set .LineWidth5
% LinesAx1(3).LineWidth = 2;                                  % Set 3LineWidth.
% %set(gcf,'FontSize',20);
% Ax2 = AxAll(2);                                             % Second Set Of Axes
% LinesAx2 = findobj(Ax2,'Type','Line');                      % Handle To Lines
% LinesAx2(2).LineWidth = 2.5;                                % Set .LineWidth5
% LinesAx2(3).LineWidth = 2;                                  % Set 3LineWidth1
%set(gca,'FontSize',20);
hold on

immsemed = immsemed/notests
immsemedf = immsemedf/notests
