# Adaptive State-feedback Echo State Networks for Temporal Sequence Learning

This is the Matlab code for the results and figures in our paper "Adaptive State-feedback Echo State Networks for Temporal Sequence Learning".

The main program can be launched by running main.m in Matlab. Main steps in the implementation are commented in the main program.

The minimum required version is R2021b Update 7 (9.11.0.2358333). 

The programs are organised in 3 folders, highlighting the breakdown of the main advantages brought by AFRICO:
  
  ---Synthetic Linear Example - AFRICO vs FORCE - is Example A
  
  ---Synthetic Non-linear Example - AFRICO vs FORCE - is Example B
  
  ---NARMA10 Example - is Example D

Each folder contains the main program main.m accompanied by some functions files:

  ---errors.m - computes the Error Reduction Ratio for the OFR algorithm
  
  ---errorsp.m - computes the Error Reduction Ratio for the unselected regressors
  
  ---extendstates.m - extends the states of the ESN for the polynomial readout
  
  ---outputpredgen.m - computes the output of the ESN using the polynomial readout
  
  ---SF.m - represents the state update function for the EKF algorithm
  
  ---statepredgen - computes the states of the ESN for a specific input
  
  ---targetstategen - computes the target states corresponding to the target system generated in Examples A and B
  
  ---FORCEpp.m - computes the output-feedback ESN trained using FORCE algorithm

  ---wei.m - computes the orthogonalisation term used for orthogonalising the unselected regressors
