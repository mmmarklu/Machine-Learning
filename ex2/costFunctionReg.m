function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = size(theta,1);

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

summation = 0;
for i = 1:m;
	otx = sum(theta' .* X(i,:));
	hx = 1 / (1 + (e^-otx));
	loopsum = (-y(i) * log (hx)) - ((1-y(i)) * log((1-hx)));
	summation = summation + loopsum;
end;
J = summation / m;

summation = 0;
for i = 2:n;
	summation = summation + (theta(i)^2);
end;
J = J + ((lambda*summation)/(2*m));

for h = 1: n;
	summation = 0;
	for i = 1:m;
		otx = sum(theta' .* X(i,:));
		hx = 1 / (1 + (e^-otx));
		loopsum = (hx - y(i)) * X(i,h);
		summation = summation + loopsum;
	end;
	
grad(h) = summation / m;

if (h >1)
grad(h) = grad(h) + ((lambda/m)*theta(h));
endif;
end;


% =============================================================

end
