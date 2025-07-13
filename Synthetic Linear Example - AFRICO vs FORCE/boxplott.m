clc
clear all
close all

exAaF = [0.0489 , 0.0532, 0.0512; 0.0817, 0.1025, 0.0867; 0.0669, 0.0301, 0.0505]';
exAF = [0.4473, 0.4920, 0.4516; 0.5280, 0.5203, 0.2589; 0.1481, 0.1805, 0.2329]';
exBaF = [0.0392, 0.2721, 0.0433; 0.0142 0.02722, 0.0148; 0.0601, 0.01799, 0.0705]';
exBF = [0.6513, 0.6240, 0.6490; 0.4997, 0.4757, 0.4021; 0.2159, 0.2664, 0.2162]';
label = ["N=20","N=40","N=80"];
figure (1)
subplot(1,2,1)
boxplot(exAaF, 'Labels',label);
set(findobj(gca,'type','line'),'linew',2)
set(gca,'FontSize',20)
ylabel('NMSE','FontSize',20)
title ('a', 'fontsize', 20)
subplot(1,2,2);
boxplot (exAF, 'Labels',label);
set(findobj(gca,'type','line'),'linew',2)
set(gca,'FontSize',20)
ylabel('NMSE','FontSize',20)
title ('b', 'fontsize', 20)
figure (2)
subplot(1,2,1)
boxplot(exBaF, 'Labels',label);
set(findobj(gca,'type','line'),'linew',2)
set(gca,'FontSize',20)
ylabel('NMSE','FontSize',20)
title ('a', 'fontsize', 20)
subplot(1,2,2);
boxplot (exBF, 'Labels',label);
set(findobj(gca,'type','line'),'linew',2)
set(gca,'FontSize',20)
ylabel('NMSE','FontSize',20)
title ('b', 'fontsize', 20)