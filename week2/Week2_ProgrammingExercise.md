# Programming exercise 1

## plotData.m

```matlab
plot(x,y, 'rx', 'MarkerSize', 10); 
xlabel("Population of City in 10 000s");
ylabel("Profit in $10 000s");
```

## computeCost.m

Implementing using a vectorized approach from [here (Cost function)](https://www.coursera.org/learn/machine-learning/resources/QQx8l)

Calculating the hypothesis as a column vector of size (mx1) with: 

$$ h_{\t{}}(X) = X\t{}$$

Calculating the cost in a vectorized form: 

$$ J(\t{}) = \frac{1}{2m}(X\t{} - \vec{y})^T(X\t{} - \vec{y})$$

```matlab
% size(X) = 97x2 , size_y = 97x1, size_theta=2x1

hx = X*theta;

J = 1/(2*m)*transpose((hx - y))*(hx - y);
```

## gradientDescent.m

The vectorized GD is:

$$\t{} := \t{} - \frac{\alpha}{m}X^T(X\t{} - \vec{y})$$

```matlab
theta = theta - (alpha/m)*transpose(X)*(X*theta - y);
```

## featureNormalize.m

```matlab
% Original solution
%mu = mean(X, 1);
%sigma = std(X, 1);
%for feat=1:size(X,2)
%    X_norm(:, feat) = (X(:, feat) - mu(feat)) / sigma(feat);
%end

% more vectorized
mu = mean(X, 1).*ones(size(X));
sigma = std(X, 1).*ones(size(X));

X_norm = (X-mu)./sigma;
```

## gradientDescentMulti.m && computeCostMulti.m

Are already implemented above.

## normalEqn.m

The generic normal equation is

$$\normalEqGeneric$$

