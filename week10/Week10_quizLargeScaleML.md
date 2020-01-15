# week 10 quiz Large scale ml

## Q1

Suppose you are training a logistic regression classifier using stochastic gradient descent. You find that the cost (say, $\textnormal{cost}(\t{}, (\ssbi{x}, \ssbi{y}))$, averaged over the last 500 examples), plotted as a function of the number of iterations, is slowly increasing over time. Which of the following changes are likely to help?

* [ ] Use fewer examples from your training set.

* [x] Try halving (decreasing) the learning rate $\alpha$, and see if that causes the cost to now consistently go down; and if not, keep halving it until it does.

* [ ] Try averaging the cost over a smaller number of examples (say 250 examples instead of 500) in the plot.

* [ ] This is not possible with stochastic gradient descent, as it is guaranteed to converge to the optimal parameters $\theta$.

## q2

Which of the following statements about stochastic gradient descent are true? Check all that apply.

* [ ] Suppose you are using stochastic gradient descent to train a linear regression classifier. The cost function ... is guaranteed to decrease after every iteration of the stochastic gradient descent algorithm. `no`

* [ ] Stochastic gradient descent is particularly well suited to problems with small training set sizes; in these problems, stochastic gradient descent is often preferred to batch gradient descent. `inverse`

* [x] In each iteration of stochastic gradient descent, the algorithm needs to examine/use only one training example.

* [x] One of the advantages of stochastic gradient descent is that it can start progress in improving the parameters $\t{}$ after looking at just a single training example; in contrast, batch gradient descent needs to take a pass over the entire training set before it starts to make progress in improving the parameters' values.

## Q3

Which of the following statements about online learning are true? Check all that apply.

* [ ] When using online learning, you must save every new training example you get, as you will need to reuse past examples to re-train the model even after you get new training examples in the future. `no the advantage is that you dont`

* [x] One of the advantages of online learning is that if the function we're modeling changes over time (such as if we are modeling the probability of users clicking on different URLs, and user tastes/preferences are changing over time), the online learning algorithm will automatically adapt to these changes.

* [ ] Online learning algorithms are most appropriate when we have a fixed training set of size mmm that we want to train on. `no`

* [x] Online learning algorithms are usually best suited to problems were we have a continuous/non-stop stream of data that we want to learn from.

## Q4

Assuming that you have a very large training set, which of the following algorithms do you think can be parallelized using
map-reduce and splitting the training set across different machines? Check all that apply.

* [ ] Logistic regression trained using stochastic gradient descent. `you cant because there's no sum`

* [ ] An online learning setting, where you repeatedly get a single example $(x,y)$, and want to learn from that single example before moving on. `cant because theres no sum`

* [x] A neural network trained using batch gradient descent.

* [x] Linear regression trained using batch gradient descent.

## Q5

Which of the following statements about map-reduce are true? Check all that apply.

* [x] Because of network latency and other overhead associated with map-reduce, if we run map-reduce using $N$ computers, we might get less than an $N$-fold speedup compared to using 1 computer.

* [x] When using map-reduce with gradient descent, we usually use a single machine that accumulates the gradients from each of the map-reduce machines, in order to compute the parameter update for that iteration. `the master`

* [ ] Linear regression and logistic regression can be parallelized using map-reduce, but not neural network training. `yes it can`

* [x] If you have only 1 computer with 1 computing core, then map-reduce is unlikely to help. 