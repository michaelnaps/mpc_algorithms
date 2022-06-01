# mpc_algorithms
Repository for the Model Predictive Control (MPC) architecture. 
Designed and tested by Michael Napoli as a part of the Ohio State University Cyberbotics Lab. 
Continuation of the **michaelnaps/ipm.git** repository (last developed 2/2022).

To use the most up-to-date system, see the */pyth* folder which has been generalized for use with
theoretically any dynamics system. The *mpc.py* folder contains the MPC class type which will have 
access to two built-in optimization algorithms (*nno*, and *ngd*).

The functions found in the */mlab* folder will eventually be generalized to work with a given
configuration space and simulator, but are currently written for the use of an n-link pendulum
model. The */pyth* folder is an improved version of the MATLAB system written in Python.
