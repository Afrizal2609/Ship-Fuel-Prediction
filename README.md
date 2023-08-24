# Ship-Fuel-Prediction

In this project I created a Support Vector Regression (SVR) model to predict the use of fuel needed by ships in one trip from Surabaya to Lembar and vice versa. I compared 4 SVR kernels namely Linear, Polynomial, RBF and Sigmoid. Parameters for each kernel are obtained by optimizing using the Grid Search Optimization algorithm. The data used for prediction are route, vessel speed, wind speed, current speed, swell, and wave.  The result is that the SVR model with Sigmoid Kernel has the best accuracy. I made a simple website with the flask framework using the SVR model and Sigmoid Kernel which can be accessed at the following link

http://webshippred.pythonanywhere.com/
