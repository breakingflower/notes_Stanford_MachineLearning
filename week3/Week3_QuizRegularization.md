# Quiz Regularization

## Q1:  You are training a classification model with logistic regression. Which of the following statements are true? Check all that apply
`incorrect`

* [x] Adding many new features to the model makes it more likely to overfit the training set.
* [ ]  Adding many new features to the model helps prevent overfitting on the training set.
* [ ] Introducing regularization to the model always results in equal or better performance on examples not in the training set.
* [ ] Introducing regularization to the model always results in equal or better performance on the training set.
* [x] Adding a new feature to the model always results in equal or better performance on examples not in the training set.


## Q2: Suppose you ran logistic regression twice, once with $\lambda = 0$, and once with $\lambda = 1$. One of the times, you got parameters $\t{} = \mat{23.4 \\ 37.0}$, and the other time you got $\t{} = \mat{1.03 \\ 0.28}$. However, you forgot which value of $\lambda$ corresponds to which value of $\t{}$. Which one do you think corresponds to $\lambda=1$?

* [ ] $\t{} = \mat{23.4 \\ 37.0}$
* [x] $\t{} = \mat{1.03 \\ 0.28}$

`smallest one`

## Q3: Which of the following statements about regularization are true? Check all that apply.
`incorrect`

* [ ] Because logistic regression outputs values $0 \leq \htx \leq 1$, its range of output values can only be "shrunk" slightly by regularization anyway, so regularization is generally not helpful for it. `not true` 

* [ ] Using a very large value of $\lambda$ cannot hurt the performance of your hypothesis; the only reason we do not set $\lambda$ to be too large is to avoid numerical problems.

* [x] Using too large a value of $\lambda$ can cause your hypothesis to overfit the data; this can be avoided by reducing $\lambda$.

* [ ] Consider a classification problem. Adding regularization may cause your classifier to incorrectly classify some training examples (which it had correctly classified when not using regularization, i.e. when $\lambda=0$).

* [ ]  Because regularization causes $J(\t{})$ to no longer be convex, gradient descent may not always converge to the global minimum (when $\lambda \gt 0$, and when using an appropriate learning rate $\alpha$). `has nothing to do with making convex`

* [x]  Using too large a value of $\lambda$ can cause your hypothesis to underfit the data. 

## Q4: select the figure that overfits... clear

## Q5 : slect the figure that underfits .. .claer