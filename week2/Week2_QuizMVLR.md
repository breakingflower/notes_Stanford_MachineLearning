# Quiz: Linear Regression with multiple variables

## Question 1

Suppose $m=4$ students have taken some class, and the class had a midterm exam and a final exam. You have collected a dataset of their scores on the two exams, which is as follows:

![q1 example table](q1_exampletable.png)

You'd like to use polynomial regression to predict a student's final exam score from their midterm exam score. Concretely, suppose you want to fit a model of the form $\mvlrHypotSum$ (with $n=2$)​, where $x_1$ is the midterm score and $x_2$ is (midterm score)$^2$. Further, you plan to use both feature scaling (dividing by the "max-min", or range, of a feature) and mean normalization.

What is the normalized feature $x_2^{(2)}$? (Hint: midterm = 72, final = 74 is training example 2.) Please round off your answer to two decimal places and enter in the text box below.

Normalized feature:

$$x_1 \leftarrow \frac{x_i - \mu_i}{s_i}$$

Which equates to

$$x_2^{(2)} \leftarrow \frac{\overbrace{5184}^{x_i} - \overbrace{(7921 + 5184 + 8836 + 4761)/4}^{\mu_i = \textnormal{average}}}{\underbrace{8836 - 4761}_{s_i}} = -0.36601 = -0.37$$

## Question 2

You run gradient descent for 15 iterations with $\alpha=0.3$ and compute $J(\t{})$ after each iteration. You find that the value of $J(\t{})$ increases over time. Based on this, which of the following conclusions seems most plausible?

* [ ] $\alpha=0.3$ is an effective choice of learning rate.

* [ ]Rather than use the current value of $\alpha$, it'd be more promising to try a larger value of $\alpha$ (say $\alpha$=1.0).

* [x] Rather than use the current value of $\alpha$, it'd be more promising to try a smaller value of $\alpha$ (say $\alpha$=0.1).

## Question 3

Suppose you have $m=14$ training examples with $n=3$ features (excluding the additional all-ones feature for the intercept term, which you should add). The normal equation is $\normalEqGeneric$. For the given values of $m$ and $n$, what are the dimensions of $\t{}$, $X$ and $y$ in this equation?

Rationale: 

$X$ is an mx(n+1) matrix (including the intercept term). $y$ is an m-dimensional vector. Using a random example in octave, we get: 

```matlab
>> X = rand(14,4)
X =

   0.855058   0.706476   0.519163   0.472514
   0.496810   0.555345   0.232275   0.081717
   0.266226   0.666397   0.196394   0.832171
   0.084183   0.156168   0.293238   0.146048
   0.504255   0.289975   0.471725   0.551899
   0.212692   0.281623   0.480850   0.107059
   0.864698   0.964447   0.184047   0.107120
   0.872396   0.348710   0.871252   0.262403
   0.976972   0.063229   0.845351   0.430062
   0.122115   0.563635   0.459052   0.360703
   0.419255   0.745681   0.835356   0.312995
   0.841649   0.267023   0.134920   0.956378
   0.796072   0.857152   0.586249   0.846780
   0.595125   0.425078   0.146681   0.389626

>> X(:,1) = 1
X =

   1.000000   0.706476   0.519163   0.472514
   1.000000   0.555345   0.232275   0.081717
   1.000000   0.666397   0.196394   0.832171
   1.000000   0.156168   0.293238   0.146048
   1.000000   0.289975   0.471725   0.551899
   1.000000   0.281623   0.480850   0.107059
   1.000000   0.964447   0.184047   0.107120
   1.000000   0.348710   0.871252   0.262403
   1.000000   0.063229   0.845351   0.430062
   1.000000   0.563635   0.459052   0.360703
   1.000000   0.745681   0.835356   0.312995
   1.000000   0.267023   0.134920   0.956378
   1.000000   0.857152   0.586249   0.846780
   1.000000   0.425078   0.146681   0.389626

>> y = rand(14,1)
y =

   0.44435
   0.20773
   0.76097
   0.84853
   0.15835
   0.93535
   0.99959
   0.28343
   0.31620
   0.80491
   0.75140
   0.86375
   0.37022
   0.56993

>> t = pinv(X'*X)*X'*y
t =

   0.78850
   0.13572
  -0.44478
  -0.14969
```

Which shows us that the output $\t{}$ is of dimension $(n+1)$x1. 

In total

* dim X = 14 x 3+1 = 14x4
* dim y = 14x1
* dim $\t{}$ = 4x1

## Question 4

Suppose you have a dataset with $m=10^5$ examples and $n=2*10^4$ features for each example. You want to use multivariate linear regression to fit the parameters $\t{}$ to our data. Should you prefer gradient descent or the normal equation?

Rationale: 
If $n<10^3$ the normal equation is preferred. Here, $n$ is sufficiently large to use GD.

* [ ] Gradient descent, since it will always converge to the optimal $\t{}$.
* [ ] The normal equation, since it provides an efficient way to directly find the solution.
* [ ] The normal equation, since gradient descent might be unable to find the optimal $\t{}$.
* [x] Gradient descent, since $(X^TX)^{-1}$ will be very slow to compute in the normal equation.

## Question 5

Which of the following are reasons for using feature scaling?

* [ ] It prevents the matrix $(X^TX)^{-1}$ (used in the normal equation) from being non-invertable (singular/degenerate).
* [x] It speeds up gradient descent by making it require fewer iterations to get to a good solution.
* [ ] It speeds up gradient descent by making each iteration of gradient descent less expensive to compute.
* [ ] It is necessary to prevent the normal equation from getting stuck in local optima.
