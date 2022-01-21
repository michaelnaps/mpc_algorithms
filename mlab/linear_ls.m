%% linear_ls() function
% Created by: Michael Napoli
% Created on: 2/13/2020
% Last modified on: 2/13/2020

% Function that takes two arrays and calculates the
% least squares regression line.
% Output:
% 'a1' - slope of the regression line
% 'a0' - y-intercept value for line
% 'Er' - average error of line comapared to points
function [a0, a1, Err] = linear_ls(x, y)
    % check that x and y arrays are the same length
    if (length(x) ~= length(y))
        fprintf("Arrays are not balanced. \n")
        return;
    else
        % initializations for sum values
        Sy = sum(y);
        Sx = sum(x);
        Sxy = 0;
        Sxx = 0;
        Err = 0;

        % for the data a1 value
        for i = 1:length(x)
            Sxy = Sxy + (x(i) * y(i));
            Sxx = Sxx + x(i)^2;
        end

        % calculate a0
        a0 = (Sxx * Sy - Sxy * Sx) / (length(x) * Sxx - Sx^2);

        % calulate a1
        a1 = (length(x) * Sxy - Sx * Sy) / (length(x) * Sxx - Sx^2);

        % calculate Error
        for i = 1:length(x)
            Err = Err + (y(i) - (a1 * x(i) + a0))^2;
        end
    end
end