# Week 2

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

**Example**: house price prediction. $x_1$ is frontage (front size of the house), $x_2$ is depth of the house. One could create a new feature $x_3$ for the area which is $x_1 * x_2$. Sometimes by defining new features you might get a better model.

In Polynomial Regression, we take can set the variables to match for example a quadratic/cubic or order function (a), or a 

$$\mat{x_1 = (size) \\ x_2 = (size)^2 \\ x_3 = (size)^3 \\ ...}$$ (a)
$$\mat{x_1 = (size) \\ x_2 = \sqrt{size} \\ x_3 = \sqrt[3]{size} \\ ...}$$ (b)

Be sure to use feature scaling/normalization for this, as the ranges will be quadratic to eachother. 

In the case of (b), given that the $size$ variable ranges from 1-1000, using feature scaling the correct scaling becomes $x_1 = \frac{size}{1000}$, $x_2 = \frac{\sqrt{size}}{\sqrt{1000}} = \frac{\sqrt{size}}{32}$.

