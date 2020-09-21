function [y,y_deriv] = sigmoid(x)
y = tanh(x);
y_deriv = 1 - tanh(x)^2;