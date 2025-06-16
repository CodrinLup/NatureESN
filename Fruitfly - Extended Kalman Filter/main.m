clc
clear all
close all
format longg

%% Input signal generation and flags initialization
Ts = 0.0025;                                            % Sampling time
Ltot = 80000;                                           % Total Signal Length
L = 8000;                                               % Level signal Length
L_stationary = 6400;                                    % Number of stationary points (points that are after the transitory points)
trans = 1600;                                           % Number of transitory points (first points of the signal)
notests = 1;                                           % Number of tests
rho = 0.000048;                                         % Stopping criteria for the OFR algorithm

MSEtf = 1;                                              % Stopping criteria for the TF estimation

% Network parameters cross-validation initialization

minnet = 12;                                            % Minimum number of neurons
maxnnet = 12;                                           % Maximum number of neurons
minlags = 5;                                            % Minimum number of lags
maxlags = 5;                                            % Maximum number of lags

% Error parameters for all algorithms

mean_outerr = zeros(maxnnet-minnet+1,maxlags-minlags+1);                    % Mean error of the AFRICO algorithm
mean_nrmse_force_poly = zeros(maxnnet-minnet+1,maxlags-minlags+1);          % Mean error of the FORCE polynomial algorithm
mean_nrmse_force_lin = zeros(maxnnet-minnet+1,maxlags-minlags+1);           % Mean error of the FORCE linear algorithm

for Nnet = minnet:maxnnet
    for Nolags = minlags:maxlags
        Npoles = Nnet;                                              % Number of desired poles
        for count=1:notests
            close all
            
%% TARGET PLOTS

            % Importing total data

            data = importdata('170_PreProcessed.mat');
            input_total = data.u;
            output_total = data.y;

            % Level 1: 1-3*L/2
            % Level 2: 3-5*L/2
            % Level 3: 5-7*L/2
            % Level 4: 7-9*L/2
            % Level 5: 9-11*L/2
            % Selecting the L2 level

            %     u_trans(1,:) = input_total(3*L/2+1:5*L/2,:);
            %     y_trans(1,:) = output_total(3*L/2+1:5*L/2,:);

            % Selecting the L3 level

            u_trans(1,:) = input_total(3*L/2+1:5*L/2,:);
            y_trans(1,:) = output_total(3*L/2+1:5*L/2,:);

            % Eliminating the first transitory points

            u(1,:) = u_trans(1,trans+1:L);
            y(1,:) = y_trans(1,trans+1:L);
            % Input generation plot

            t = 0:Ts:L_stationary*Ts-Ts;

            figure(1)
            subplot(2,1,1)
            plot(t,u,'k','Linewidth',1);
            xlabel('Time(s)','fontsize', 20)
            ylabel('Amplitude','fontsize', 20)
            set(gca,'FontSize',20)
            subplot(2,1,2)
            plot(t,y,'k','Linewidth',1);
            xlabel('Time(s)','fontsize', 20)
            ylabel('Amplitude','fontsize', 20)
            set(gca,'FontSize',20)
            output(1,1:1:L_stationary) = y;
            input(1,1:1:L_stationary) = u;

            output(1,1:1:L_stationary) = y;
            input(1,1:1:L_stationary) = u;
            save('points.mat','input','output');

            % Training and test data generation
            iinput_training = input(1,1:1:600);
            for j = 2:5
                iinput_training = [iinput_training input(1,(j-1)*800+1:1:j*800-200)];
            end

            input_training = zeros(Nolags, length(iinput_training));
            % Adding the input lags from the original input signal
            for i=1:Nolags
                input_training(i,:) = [zeros(1,i-1) iinput_training(1:1:length(iinput_training)-i+1)];
            end

            iinput_validation = input(1,601:1:800);
            for j=2:3
                iinput_validation = [iinput_validation input(1,j*800-199:1:j*800)];
            end

            input_validation = zeros(Nolags, length(iinput_validation));
            for i=1:Nolags
                input_validation(i,:) = [zeros(1,i-1) iinput_validation(1:1:length(iinput_validation)-i+1)];
            end

            iinput_testing = input(1,floor(3*L_stationary/4)+1:1:L_stationary);
            input_testing = zeros(Nolags, length(iinput_testing));
            for i=1:Nolags
                input_testing(i,:) = [zeros(1,i-1) iinput_testing(1:1:L_stationary-floor(3*L_stationary/4)-i+1)];
            end

            ooutput_training = output(1,1:1:600);
            for j = 2:5
                ooutput_training = [ooutput_training output(1,(j-1)*800+1:1:j*800-200)];
            end

            output_validation = output(1,601:1:800);
            for j=2:3
                output_validation = [output_validation output(1,j*800-199:1:j*800)];
            end

            output_testing = output(1,floor(3*L_stationary/4)+1:1:L_stationary);

            % Substracting the training output mean
            mean_output_training = mean(ooutput_training);
            output_training = ooutput_training - mean_output_training;
            output_validation = output_validation + mean_output_training - mean(output_validation);
            output_testing = output_testing + mean_output_training - mean(output_testing);
            orig_output_training = output_training;
            tranz = floor(L_stationary/11);

            save('training.mat','input_training','output_training');
            save('testing.mat','input_validation','output_validation');

 %% STAGE 1 - EKF Estimation of the input and state-feedback weights         
            
            facto = 0;
            while(1)
               Anetwork = zeros(Nnet,Nnet);
               for j=1:Nnet/2
                   a1=2*rand-1;
                   b1=rand;

                   %while(abs(a1+1i*b1)<0.7 || abs(a1+1i*b1)>0.8)
                   while(abs(a1+1i*b1)>=0.99)
                       a1=2*rand-1;
                       b1=rand;
                   end
                   Anetwork(2*j-1,2*j-1) = a1;
                   Anetwork(2*j,2*j) = a1;
                   Anetwork(2*j-1,2*j) = -b1;
                   Anetwork(2*j,2*j-1) = b1;
               end
               P = rand(Nnet,Nnet);
               P = P/norm(P);
               Anetwork = P*Anetwork*inv(P);
               Bnetwork = 2*rand(Nnet,Nolags)-1;                            % Network input matrix
               Dnetwork = 0;
               Cnetwork = zeros(1,Nnet);                               % Network output matrix
               Knetwork = rand(1,Nnet); 
               factooo = 0;

               InitState = zeros((2*(Nolags))*length(Anetwork(1,:))+length(Anetwork(1,:)),1);
               for ll = 1:Nolags
                    InitState(ll*length(Anetwork(1,:))+1:(ll+1)*length(Anetwork(1,:)),1) = Bnetwork(:,ll);
               end
               InitState((ll+1)*length(Anetwork(1,:))+1:(ll+2)*length(Anetwork(1,:)),1) = Knetwork;
               InitState(:,1) = mean(InitState(:,1));
               Acl = eye((2*(Nolags))*length(Anetwork(1,:))+length(Anetwork(1,:)), (2*(Nolags))*length(Anetwork(1,:))+length(Anetwork(1,:)));
               Acl(1:length(Anetwork(1,:)),1:length(Anetwork(1,:))) = Anetwork;
              
              for ll = 1:length(input_training)
%                 SF = @(x,u)((Acl-[x(length(Acl)/3+1:2*length(Acl)/3); zeros(2*length(Acl)/3,1)]*[x(2*length(Acl)/3+1:length(Acl))' zeros(1,2*length(Acl)/3)])*x+[x(length(Acl)/3+1:2*length(Acl)/3); zeros(2*length(Acl)/3,1)]*u) - [x(length(Acl)/3+1:2*length(Acl)/3); zeros(2*length(Acl)/3,1)]*x'*[x(2*length(Acl)/3+1:length(Acl))' zeros(1,2*length(Acl)/3)]' ;
                %SF = @(x,u)((Acl-B*K)*x+B*u)-B*x'*K';
                MF = @(x)(ones(1,length(x))*x);

                obj = extendedKalmanFilter(@SF,MF,InitState);
                statec = correct(obj,output_training(ll));
                statep = predict(obj,input_training(:,ll),Anetwork,Nolags,Nnet);
                B = zeros(Nnet,Nolags);
                for lll = 1:Nolags
                    B(:,lll) = statep(Nnet+1+(lll-1)*Nnet:Nnet+lll*Nnet);
                end
                K = zeros(Nolags,Nnet);
                for lll = 1:Nolags
                    K(lll,:) = statep(Nnet+Nnet*Nolags+1+(lll-1)*Nnet:Nnet+Nnet*Nolags+(lll)*Nnet)';
                end
               
            end
            Bnetwork = B;
            Knetwork = K;
            Anetworkclosed = (Anetwork-Bnetwork*Knetwork);
               if (max(abs(eig(Anetworkclosed)))<=1)
                   break
               end

           end

%% STAGE 2 - OFR Readout Estimation

            % Training the output using the OFR algorithms.
            % Reservoir state generation using the training data
            statenet = statepredgen(input_training,length(input_training),Nnet,Anetworkclosed,Bnetwork);
            % Extending the state vector for the 3rd order polynomial
            combin = nchoosek(Nnet,2);
            dimen = combin+3*Nnet+nchoosek(Nnet,3)+factorial(Nnet)/factorial(Nnet-2);
            statenetext = extendstates(statenet,combin,Nnet,length(input_training));

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
                % Extended state generation for the testing data

                statepred = statepredgen(input_validation,length(input_validation),Nnet,Anetworkclosed,Bnetwork);
                statepredext = extendstates(statepred,combin,Nnet,length(input_validation));

                % Output generation for the testing data using the
                % identified parameters at step p

                outputpred = outputpredgen(weightstepp,indices,statepredext);
                outputpred = outputpred + mean_output_training;
                outerr = sum((outputpred - output_validation').^2)/sum((output_validation-mean(output_validation)).^2);
                errvect(p) = outerr*100;
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

            % Run the network on the testing data

            transit = length(output_testing)/4;
            tnet = 0:Ts:length(output_testing)*Ts-Ts-transit*Ts;

            statepred = statepredgen(input_testing,length(input_testing),Nnet,Anetworkclosed,Bnetwork);
            statepredext = extendstates(statepred,combin,Nnet,length(input_testing));
            outputpred = outputpredgen(optweight,optstateind,statepredext);
            outputpred = (outputpred + mean_output_training);
            % Compute the error on the testing data
            outputpred = outputpred(transit+1:length(output_testing));
            output_testingg = output_testing(transit+1:length(output_testing));
            outerr = sum((outputpred - output_testingg').^2)/sum((output_testingg-mean(output_testingg)).^2);

%% RESULTS PLOTS            
            % Plot the response of the network with the testing data

            figure(2)
            plot(tnet, output_testingg, 'k', 'Linewidth', 2);             % Test signal
            hold on
            plot(tnet,  outputpred, 'r', 'Linewidth', 1);                 % AFRICO signal
            xlabel('Time(s)','fontsize', 20)
            ylabel('Amplitude','fontsize', 20)
            set(gca,'FontSize',20)
            title ('d','fontsize', 20)

            % Plot of the error between the responses of the output of the target
            % system and the trained system

            % Plot each regression error

            figure(3)
            plot(errorsind);
            xlabel('Polynomial term - in order of states','fontsize',18);
            ylabel('Relative error to the training output','fontsize',18);
            set(gca, 'FontSize', 16);
           
            mean_outerr(Nnet-minnet+1,Nolags-minlags+1) = mean_outerr(Nnet-minnet+1,Nolags-minlags+1) + outerr;
          
        end

        mean_outerr(Nnet-minnet+1,Nolags-minlags+1) = mean_outerr(Nnet-minnet+1,Nolags-minlags+1) / notests
        

        figure(4)
        subplot(3,1,1)
        plot(t,u,'k','Linewidth',2);
        xlabel('Time(s)','fontsize', 20)
        ylabel('u(t)','fontsize', 20)
        set(gca,'FontSize',20)
        title ('a','fontsize', 20)
        ylim([-0.5 1.5])
        subplot(3,1,2)

        plot(t,y,'k','Linewidth',2);
        xlabel('Time(s)','fontsize', 20)
        ylabel('y^*(t)','fontsize', 20)
        set(gca,'FontSize',20)
        title ('b','fontsize', 20)

        subplot(3,1,3)
        plot(tnet, output_testingg,'k', 'Linewidth', 3);  % Test signal
        hold on
        plot(tnet, outputpred,'r', 'Linewidth', 2);            % AFRICO signal
        hold on
        xlabel('Time(s)','fontsize', 20)
        ylabel('Amplitude','fontsize', 20)
        set(gca,'FontSize',20)
        title ('c','fontsize', 20)
    end
end

figure(5)
surf(minnet:maxnnet,minlags:maxlags,mean_outerr)
xlabel('System dimensionality','fontsize',18);
ylabel('No. of lags','fontsize',18);
zlabel('NRMSE [%]','fontsize', 18);
