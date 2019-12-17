# Linear Regression with One Variable

Question 1
----------
> Consider the problem of predicting how well a student does in her second year of college/university, given how well she did in her first year.
> Specifically, let x be equal to the number of "A" grades (including A-. A and A+ grades) that a student receives in their first year of college (freshmen year). We would like to predict the value of y, which we define as the number of "A" grades they get in their second year (sophomore year).
> Here each row is one training example. Recall that in linear regression, our hypothesis is $\lrHypot$, and we use *m* to denote the number of training examples.

`Note: this table changes in every quiz`
> ![Quiz table 1](01_quiz_q1.png)


> For the training set given above (note that this training set may also be referenced in other questions in this quiz), what is the value of *m*? In the box below, please enter your answer (which should be a number between 0 and 10).

```
4
```

Question 2 
----------
> Many substances that can burn (such as gasoline and alcohol) have a chemical structure based on carbon atoms; for this reason they are called hydrocarbons. A chemist wants to understand how the number of carbon atoms in a molecule affects how much energy is released when that molecule combusts (meaning that it is burned). The chemist obtains the dataset below. In the column on the right, “kJ/mol” is the unit measuring the amount of energy released.

> ![Quiz table 2](01_quiz_q2.png)

> You would like to use linear regression ($\lrHypot$) to estimate the amount of energy released (y) as a function of the number of carbon atoms (x). Which of the following do you think will be the values you obtain for $\t{0}$ and $\t{1}$​? You should be able to select the right answer without actually implementing linear regression. 


$\t{0} = -569.6, \t{1} = -530.9$ \
Rationale: The value is always negative and sort of increases by around 500 per x++. 


Question 3
----------
> Suppose we set $\t{0} = -1, \t{1} = 0.5$. What is $h_\theta(4)$?
$h_\theta(4) = -1 + 0.5 * 4 = 1$ 

Question 4
----------

`This answer was not correct (- - - x), second try (x x - x) was also incorrect ` 
> Let $\mathit{f}$ be some function so that $\mathit{f}(\t{0}, \t{1})$ outputs a number. For this problem, $\mathit{f}$ is some arbitrary/unknown smooth function (not necessarily the cost function of linear regression, so $\mathit{f}$ may have local optima).
> Suppose we use gradient descent to try to minimize $\mathit{f}(\t{0}, \t{1})$ as a function of $\t{0}$ and $\t{1}$​. Which of the following statements are true? (Check all that apply.)

- [x] if $\t{0}$ and $\t{1}$ are initialized at a local minimum, then one iteration will not change their values. $\ra$ `The derivative is 0, so an update will not change the values`
- [ ] even if the learning rate $\alpha$ is very large, every iteration of gradient descent will decrease the value of $\mathit{f}(\t{0}, \t{1})$. 
- [ ] if $\t{0}$ and $\t{1}$ are initialized so that $\t{0} = \t{1}$, then by symmetry (because we do simultaneous updates to the two parameters), after one iteration of gradient descent, we will still have $\t{0} = \t{1}$.
- [x] if the learning rate is too small, then gradient descent may take a very long time to converge. 


Question 5
----------

> Suppose that for some linear regression problem (say, predicting housing prices as in the lecture), we have some training set, and for our training set we managed to find some $\t{0}, \t{1}$ such that $J(\t{0}, \t{1}) = 0$. Which of the statements below must then be true? (Check all that apply.)

- [x] Our training set can be fit perfectly by a straight line, i.e. all of our training examples lie perfectly on some straight line. 
- [ ] for this to be true, we must have $\t{0} = 0$ and $\t{1} = 0$, so that $h_\theta(x) = 0$. 
- [ ] for this to be true, we must have $y^{(i)} = 0$ for every value of $i = 1,2, ...., m$.
- [ ] Gradient descent is likely to get stuck at a local minimum and fail to find the global minimum. 

Question 6
----------

`This original answer was wrong, as highlighted in the "and not" ` 
> For this question, assume that we are using the training set from Q1. Recall our definition of the cost function was $\lrLoss$. What is $J(0, 1)$?  In the box below, please enter your answer (Simplify fractions to decimals when entering answer, and '.' as the decimal delimiter e.g., 1.5).

Rationale:

Given the loss function and the values of $\t{0}, \t{1}$, the hypothesis is:

$$\lrHypot = 0 + 1$$ 

`and not `

$$0 + 1*(x^{(i)})$$

This means: 

$$\lrLoss = \frac{1}{2*4} \sum_{1}^{4} (1*x^{(i)} - y^{(i)})^2) $$ 

$$ = \frac{1}{8} [(3 - 2)^2) + (1 - 2)^2) + (0 - 1)^2 + (4 - 3)^2)]$$

$$ = \frac{1}{8} [1 + 1 + 1 + 1] = \frac{1}{8}[4] = 0.5$$


Question 7 
----------

> Suppose we set $\theta_0 = -1$, $\theta_1=2$ in the linear regression hypothesis from Q1. What is $h_\theta(6)$?

Rationale: 

The hypothesis is $\lrHypot = -1 + 2x^{(i)} = -1 + 2*6^2 = 11$

> Suppose we set $\theta_0 = 0$, $\theta_1=1.5$ in the linear regression hypothesis from Q1. What is $h_\theta(2)$?

Rationale: 

The hypothesis is $\lrHypot = 0 + 1.5*2 = 3$



Question 8
----------

Consider the linear regression model $\lrHypot$. What are the values of $\t{0}$ and $\t{1}$​ that you would expect to obtain upon running gradient descent on this model? (Linear regression will be able to fit this data perfectly.)

Rationale: the model will be able to fit the data perfectly, so $|y-x| = 0$. In the first case, $x=1$, $y=0.5$, therefore $\t{0}$ = 0 and $\t{1}=0.5$, which makes the hypothesis $\lrHypot = 0 + 0.5$.