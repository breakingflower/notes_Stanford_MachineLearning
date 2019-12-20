# Week 2

More general notes can be found [on the coursera website](https://www.coursera.org/learn/machine-learning/resources/QQx8l)

## Multiple features

Instead of only using the housing price, also include other variables such as number of bedrooms, ...

### Notations

* $n$ = number of features.
* $x^{(i)}$ = input features of the $i^{th}$ training example.
* $x^{(i)}_j$ = value of the feature $j$ in $i^{th}$ training example ($j=1,2, ...m$). For convenience, at $j=0$ every value of $x^{(i)}$ is one.

### Hypothesis

Standard hypothesis updated to be

$$\mvlrHypotSum$$

where $x_0 = 1$. This means that x becomes an n-dimensional feature vector: 

$$ x = \mat{x_0 \\ x_1 \\ x_2 \\ ... \\ x_n} \text{in } \RR^{n+1} \qquad \t{} = \mat{\t{0} \\ \t{1} \\ \t{2} \\ ... \\ \t{n}} \text{in } \RR^{n+1}$$

This means that the hypothesis becomes the inner product between our parameter vector $\t{}$ and our feature vector $x$

$$\mvlrHypot = \mat{\t{0} & \t{1} & \t{2} & ... & \t{n}} \mat{x_0 \\ x_1 \\ x_2 \\ ... \\ x_n} = \t{}^Tx$$

Which is also called **multivariate linear regression**.

### Gradient Descent for Multiple variables

Hypothesis:

$$\mvlrHypot$$

Cost function:

$$\mvlrCost$$

Partial derivative:

* new algorithm: $n \ge 1$

Repeat until convergence {\
    $\qquad \mvlrGDDeriv$\
} (simultaneous update $\t{j}$ for $j=0,...,n$)

### Gradient Descent in practice

#### Feature scaling

Make sure that the features you are using are on SIMILAR scale.

* $x_1$ = size (0-5000m)
* $x_2$ = number of rooms (1-10)

If you try this, the result is an ellipse, and it will take a long time to reach the minimum. In this case, GD will have a much harder time to find the global minimum.

In these settings it's good to scale the features, this is also called **feature normalization**.

The way to do this is to get every feature into *approximately* the scale $-1 \leq x_i \leq 1$. Approximately means $-3$ to $+3$ is ok, $\frac{-1}{3}$ to $\frac{1}{3}$ is ok, but -0.000001 to 0.00001 is NOT ok.

##### Mean Normalization

Replace $x_i$ with $x_i - \mu_i$, with $\mu_i$ being the average of the variable (do not apply to $x_0$).
For example:

* $$x_1 = \frac{size - 1000}{2000}$$
* $$x_2 = \frac{num\_bedrooms - 2}{5}$$

More generally, with $\mu_i$=avg value and $s_i = (max-min)$ or stdev:

$$x_1 \leftarrow \frac{x_i - \mu_i}{s_i}$$

`NOTE: Quizzes use (max-min) for s_i but programming exercises use stdev!`

##### Learning rate

Assert that GD is working correctly! You can do this by plotting the cost $J(\t{})$, it should *decrease* after every iteration. At some point, it should flatten out and *converge*.

Declare convergence if $J(\t{})$ decreases by less than $10^{-3}$ in one iteration.

If $J(\t{})$ is **increasing**, GD is not working, you probably need to **lower** $\alpha$, because it keeps *overshooting* the minimum of the cost function. And ofcourse make sure you dont have bugs :).

If $J(\t{})$ is **going up and down and up and down ... **, also use a smaller $\alpha$.

For sufficiently small $\alpha$, $J(\t{})$ should decrease every iteration. If $\alpha$ is too small, it may be slow to converge.

To choose $\alpha$, try $0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, ...$ (i.e. 3-fold) and plot $J(\t{})$. Pick the one that looks the best.

### Features and Polynomial Regression

**Example**: house price prediction. $x_1$ is frontage (front size of the house), $x_2$ is depth of the house. One could create a new feature $x_3$ for the area which is $x_1 \cdot x_2$. Sometimes by defining new features you might get a better model.

In Polynomial Regression, we take can set the variables to match for example a quadratic/cubic or order function (a), or a

$$\mat{x_1 = (size) \\ x_2 = (size)^2 \\ x_3 = (size)^3 \\ ...}$$ (a)
$$\mat{x_1 = (size) \\ x_2 = \sqrt{size} \\ x_3 = \sqrt[3]{size} \\ ...}$$ (b)

Be sure to use feature scaling/normalization for this, as the ranges will be the relation you impose to the features, if $size$ has a range of 1-$10^3$, $size^2$ has a range of 1-$10^6$ etc.

In the case of (b), given that the $size$ variable ranges from 1-1000, using feature scaling the correct scaling becomes $x_1 = \frac{size}{1000}$, $x_2 = \frac{\sqrt{size}}{\sqrt{1000}} = \frac{\sqrt{size}}{32}$.

### Normal Equation

Up to this point we used GD, which uses a cost function $J(\t{})$ and steps down in the loss curve.

The normal equation is a method to solve for $\t{}$ analytically and allows you to find the minimum in one step.

Example:
> Intuition: If 1D ($\t{} \in \RR$) \
> $J(\t{}) = a\t{}^2 + b\t{} + c$. Solving the partial derivative by setting it equal to zero will minimize $J$.

> In the case of $\t{} \in \RR^{n+1}$, we need to do it for each of the partial derivatives at the same time. We will minimize $J$ by explicitely taking its derivatives with respect to $\t{j}$ and setting them to zero. This allows us to find the optimum $\t{}$ without iteration.

Example: $m =$ number of samples $= 4$, $n$ = number of features, where $x_0$ is always equal to one. $X$ is also called the *design matrix*.

$$X = \underbrace{\overbrace{\mat{1 & 2104 & 5 & 1 & 45 \\ 1 & 1416 & 3 & 2 & 40 \\ 1 & 1534 & 3 & 2 & 30 \\ 1 & 852 & 2 & 1 & 36}}^{x_0 ... x_n (x_0\textnormal{ is always 1})}}_{m \textnormal{ by } (n+1) \textnormal{-dimensional}} \qquad y=\underbrace{\mat{460\\232\\315\\178}}_{m\textnormal{-dimensional}}$$

If you now use the following, this will give you the value of $\t{}$ that minimizes your cost function.

$$\normalEqGeneric$$ (1)

Where
* The dimension of $\t{}$ is equal to the amount of features $n+1$.
* $(X^TX)^{-1}$ is the inverse of matrix $X^TX$.

Solving the normal equation (1) is done in octave by:
```matlab
% pinv is the inverse function. X' is the transpose of X.
pinv(X'*X)*X'*y
```

If you **use the normal equation method, there is no need to do feature scaling.**

Use case: $m$ training examples, $n$ features. GD versus Normal Equation

| Gradient Descent          | Normal Equation   |
| --------------------------|-------------------|
| need to choose $\alpha$                  | no need to choose $\alpha$     |
| needs many iterations        | don't need to iterate        |
| works well even if $n$ is large. Complexity $\mathcal{O}(kn^2)$           | need to compute $(X^TX)^{-1}$, which is dimension nxn. Complexity $\mathcal{O}(n^3)$           |
| | slow if $n$ is very large. $n < 10^3$ normal is probably faster and better, but $n > 10^3$ choose GD.

#### Normal Equation Noninvertibility.

`A phenomenom you may run into.`

Given the normal equation

$$\normalEqGeneric$$

In octave, *pinv* is the **pseudo-inverse**, and *inv* is the inverse. *pinv* should give you the right solution.

What if $X^TX$ is non-invertible? (singular / degenerate matrix?) **This should happen very rarely.**

* Reduntant features (linearly dependent). e.g.

    * $x_1$ = size in $feet^2$
    * $x_2$ = size in $m^2$
    * means that $x_1 = (3.28^2)x_2$ $\rightarrow$ makes $X^T$ non-invertible.

* Too many features (e.g. $m \leq n$)

    * so you try to fit for example 100 parameters with 10 samples.
    * delete some features or use **regularization**.

### Octave exercises

```matlab
Your functions must handle the general case. This means:

- You should avoid using hard-coded array indexes.

- You should avoid having fixed-length arrays and matrices. 

Debugging:

If your code runs but gives the wrong answers, you can insert a "keyboard" command in your script, just before the function ends. This will cause the program to exit to the debugger, so you can inspect all your variables from the command line. This often is very helpful in analysing math errors, or trying out what commands to use to implement your function.
```

## Vectorized implementations

Calculating the hypothesis as a column vector of size (mx1) with:

$$ h_{\t{}}(X) = X\t{}$$

Calculating the cost in a vectorized form

$$ J(\t{}) = \frac{1}{2m}(X\t{} - \vec{y})^T(X\t{} - \vec{y})$$

Gradient descent rule can be expressed as

$$ \t{} := \t{} - \alpha\nabla J(\t{}) $$

Where $\nabla J(\t{})$ is a column vector of the form

$$ \nabla J(\t{}) = \mat{\frac{\delta J(\t{})}{\delta\t{O}}\\\frac{\delta J(\t{})}{\delta\t{1}}\\\frac{\delta J(\t{})}{\delta\t{2}}\\...\\\frac{\delta J(\t{})}{\delta\t{n}}}$$

Finally, the vectorized GD is:

$$\t{} := \t{} - \frac{\alpha}{m}X^T(X\t{} - \vec{y})$$
