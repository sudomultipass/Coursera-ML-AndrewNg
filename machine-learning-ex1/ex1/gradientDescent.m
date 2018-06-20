function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
% GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); 						% number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    % Vectorized in parts.  Could be a one-liner, but this is clear to me.
    h = X * theta;						% m-dimensional vector of values of h(x)
    err = h - y;						% error vector (h(x) - y)
    delta = (1/m) * sum(Err .* X)';		% 2-dim vector (same as theta) 
    theta = theta - alpha * delta;		% alpha = learning rate
    									% 2-dim vector theta (update theta)


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta); 	% J_history is a iter-length vector

end

end
