%% Function: least_squares()
% Created by: Michael Napoli
% Created on: 2/13/2020
% Last modified on: 2/13/2020

% Purpose: Function that takes x and y data points and computes
%   the best fit polynomial of the maximum degree given.

% Input: 'x' - array of x-coordinate values
%   'y' - array of y-coordinate values
%   'm' - maximum polynomial power
% Output: 'a' - array of coefficient values for the polynomial equation
function [a] = polynomial_fit(x, y, m)
    % if the lengths of the arrays are not equal
    if (length(x) ~= length(y))
        return;  % return nothing
    else
        % initialize S, Sx and Sy to matrix/arrays of 0
        S = zeros((m + 1), (m + 1));  % matrix for the a coefficients
        Sx = zeros(1, (2 * m));  % array for the sum of x values raised
                                 % to the appropriate power
        Sy = zeros((m + 1), 1);  % array for the 'b' matrix
        
        % for all values of the 'Sx' array
        for i = 1:(2 * m)
            for k = 1:length(x)
                % sum the x-values raised to their index in the array
                Sx(i) = Sx(i) + x(k)^i;
            end
        end
        
        % create 'S' matrix using the 'Sx' values calculated
        for i = 0:m
            for k = 0:m
                % evaluate position and place the necessary 'Sx' value
                if (i == 0 && k == 0)
                    S((i + 1), (k + 1)) = length(x);
                else
                    S((i + 1), (k + 1)) = Sx(i + k);
                end
            end                
        end
        
        % for all values of the 'Sy' array
        for i = 1:(m + 1)
            for k = 1:length(y)
                % sum the values of x and y combinations appropriately
                Sy(i) = Sy(i) + (y(k) * x(k)^(i - 1));
            end
        end
        
        % solve for 'a' in the matrix format
        a = S\Sy;
        
        return;
    end
    
end