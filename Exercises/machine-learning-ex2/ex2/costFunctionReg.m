function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
reg = zeros(size(theta(2:end)));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
% Note: grad should have the same dimensions as theta
%
% First, get sigmoid of (theta' X) for h(x)
% h(x) = g(theta' * X) or g(X*theta);
h = sigmoid(X * theta);					 

% ------------------- Regularized Cost Function J ---------------
% Compute cost function J = -y * log(h) - (1-y) * log(1-h)
J = (1/m) * sum(-y .* log(h) - (1-y) .* log(1-h));		% Regular Logistic Cost Function
regTermCost = (lambda/(2*m)) * sum(theta(2:end).^2);	% Regularization Term
J += regTermCost;										% Add to Cost

% ----------------------- Gradient Descent ----------------------
% Now, we minimize cost function J by gradient descent for Logistic Regression:
% grad = partial derivative of J(theta j)/theta j
% Vectorized in parts.  Could be a one-liner, but this is clear to me.
    err = h - y;									% error vector (h(x) - y)
    grad = (1/m) * sum(err .* X)';					% 2-dim vector (same as theta) 
    reg = (lambda / m) * theta(2:end);				% Regularization Term
    grad(2:end) .+= reg;





% =============================================================

end
