function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Recode the y vector into a boolean matrix where each sample has
% a vector of solutions (one hot encoding)
% (from https://stackoverflow.com/questions/38947948/ ...
% how-can-i-hot-one-encode-in-matlab)
y_onehot = y==1:max(y);

% a1 = x with bias
a1 = [ones(m, 1) X];

z2 = a1*transpose(Theta1);
a2 = sigmoid(z2);                       % calculate g(z1)
a2 = [ones(size(a2, 1) ,1) a2];         % add bias

z3 = a2*transpose(Theta2);
a3 = sigmoid(z3);                       % calculate g(z2) 


% double sum for cost function as in notes
J = 1/m * sum(sum(-y_onehot.*log(a3) ...    % y_onehot is in the correct shape
        - (1-y_onehot).*log(1-a3)));        % elementwise multiplying....

% regularizer term, assume only two layers so can hardcode t1 t2. 
% a small error on the dimensionality of theta, as it was inverted to the 
% previous exercise. 
regularizer = lambda/(2*m)*(    sum(sum(Theta1(:, 2:end).^2, 2),1) ...
                            +   sum(sum(Theta2(:, 2:end).^2, 2),1));

J = J + regularizer;

for i=1:m
    % if you transpose the a1i value its less transposes in the next steps
    a1i = a1(i,:)';
    z2i = z2(i, :); 
    a2i = a2(i, :);
    z3i = z3(i, :);
    a3i = a3(i, :); 
    yi = y_onehot(i, :); 
    
    d3i = a3i - yi; % loss on classification

    d2i = d3i*Theta2.* [1 sigmoidGradient(z2i)];  % loss in second layer
    d2i = d2i(2:end); % remove d2
    Theta1_grad = Theta1_grad + transpose(a1i*d2i); %a1i [401 1] d2i[1 25]
    Theta2_grad = Theta2_grad + transpose(transpose(a2i)*d3i); %a2i [1 26] d3i[1 10]
end

Theta1_grad = 1/m * Theta1_grad;  
Theta2_grad = 1/m * Theta2_grad;  

% elementwise because its a scalar :) 
Theta1_regularizer = lambda/m .* Theta1; 
Theta2_regularizer = lambda/m .* Theta2; 

Theta1_grad(:, 2:end) += Theta1_regularizer(:, 2:end); 
Theta2_grad(:, 2:end) += Theta2_regularizer(:, 2:end); 

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
