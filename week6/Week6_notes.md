# Week 6 notes

More general notes can be found [on the coursera website](https://www.coursera.org/learn/machine-learning/resources/LIZza)

## Deciding what to try next

* Imagine you have an algo to predict housing prizes but you suck on real data, you can solve it by maybe trying one of these
    * get more data (can fail)
    * try a smaller set of features (reduce redundancy to prevent overfitting)
    * try get additional features
    * add polynomial features
    * decreasing / increasing $\lambda$ (regularization parameter)

Most ppl will go by "gut feeling".

However, theres a simple technique that you can use to rule out many of these options, Machine Learning Diagnostics. Diagnostics can take time to implement, but can be very useful because they can often save you from months of experimentation.

## Evaluating a hypothesis

* Fails to generalize to examples not in training set
* imagine $x_1, x_2 , ... x_{100}$
    * split dataset into train / test set 70-30 ($m_{test}$ = num of test examples)
    * apply random shuffle
* Learn parameters $\t{}$ from training data (ergo $\undersetnorm{min}{J_{train}}$)
* Compute the test set error by calculating the cost on the test examples $x_m$. 
* A different error could be the misclassification error (you get an example right / you get an example wrong). Imagine 10 classes, if you get the right number you get 1, else you get 0. Take the sum of this and divide by the number of test samples $m_{test}$.

## Model selection and Train/Val/Test sets

* The problem of overfitting has been established.
* Training set error is not a good metric to evaluate quality of the system.

### Model selection

* what degree ($d$) of polynomial do you want to pick (i.e. $d=1,2,...10$)
* You could fit each of the models and evaluate all of the test errors, choosing the one that has the lowest test set error. However, this cost is probably an optimistic estimate.
* Introduce a new part: a (cross)validation set. Now we have a training set, validation set and test set.
* Now use the validation set error to select the model. Finally, once chosen, evaluate error on test set.

## Diagnosing Bias vs Variance

* Underfitting (bias) vs overfitting (variance)

Hypotheses

* too simple hypothesis -> underfit (high bias)
* too complex hypothesis -> overfit (high variance)

Bias/variance

* Remember training error / validation error.
* Plot the training error/validation error versus the degree of the chosen polynomial of the hypothesis.
* For the training set, the more complex the hypothesis the lower the error will be.
* For the validation set, we have a minimum somewhere (optimum).
 
![bias_variance_polynomial.png](bias_variance_polynomial.png)

A high bias problem is where $d$ is too small for the dataset, and so the validation error and training error will be high ($J_{val} ~= J_{train}$)

A high variance problem is where $d$ is too large for the dataset, and so the training error will be very low but the validation error is very high ($J_{val} >> J_{train}$).

## Regularization and bias/variance

* Suppose we fit a high order polynomial, but to prevent overfitting we add a regularization term.
* If our regularization is too large ($\lambda$ >>>), we will get a high bias because all of the parameters are nearly 0.
* If our regularization is too small ($\lambda$ <<<<), we will get a high variance because theres actually minimal / no regularization.
* Good lambda makes a big difference.

How do we choose a good value for the regularization parameter $\lambda$? 

* Try a range of lambda i.e. $\lambda = 0, 0.1, 0.02, 0.04, 0.08, ..., 10$). You get a bunch of models, and for each of these evaluate the costs $J_{val}, J_{train}$, next choose the lowest validation error. Finally, evaluate on the test set.

![bias_variance_lambda.png](bias_variance_lambda.png)

* For small values of lambda, you dont have regularization, so you will fit the train set well, but val set error will be big. If you have a large value of lambda, you will not fit your training set well and your val set will be similar because all of the params are 0.

## Learning curves

Imagine 

$$\Jtrain$$ 
$$\Jval$$

and plot the error in function of the value $m$, i.e. training set size, but LIMIT yourself to a subset of the training set.

* If the training set size is small, the training error will be really small too.
* If the training size grows, the training set error will grow with $m$. (Intuition: if you only have 1 or 2 training samples, it will be easy to get 0 error, but if you increase the size of the set you will get to a point where you can no longer ensure 0 error.)

![learning_curves_intuition.png](learning_curves_intuition.png)

Now imagine alot more training examples. If the hypothesis is too simple, the validation set error will remain constant (plateau) pretty soon. The training set error will start off small, and the training error will end up close to the validation error. (high bias) This is very problematic if the error is actually high.

`If a learning algo is suffering from high bias, getting more training data will not help much!`

`If a learning algo is suffering from high variance, the difference between training error and val error will be big (a gap), and getting more training data will likely improve the model.`

## Deciding what to do next revisited

Debugging a learning algo:

* get more training examples --> `helps to fix high variance, but not in high bias!`
* try smaller set of features --> `helps to fix high variance, but dont even try this if you have a high bias problem`
* try getting additional features --> `helps to fix high bias problems, as your current hypothesis is too simple.`
* try adding polynomial features --> `helps to fix high bias problems, ssimilar to above`
* decreasing / increasing lambda --> `decreasing lambda helps to fix high bias, increasing lambda helps to fix high variance.`

Neural networks and overfitting:

* small neural network: few parameters, more prone to underfitting ,computationally cheap
* large neural network: more parameters, more prone to overfitting, computationally expensive. Use regularization ($\lambda$) to address overfitting. `often the larger the better`. Usually, using one hidden layer is default & ok, but you can try more to see how it performs on the validation set.

`To evaluate the different lambda parameters, use lambda to train the network but during inference leave out this parameter (ie set it to 0)`

## Machine Learning System Design

### Spam Classification

Building a spam classifier

* spam (1)/ no spam(0)
* supervised learning
* $x$ = features in the emai
    * words like deal buy discount ,... (1) or non spam words such as my name , ...
* labels $y \in \{0,1\}$.
* Define a feature vector $x = \mat{0 \\ 1 \\ 0 \\ 1 \\ 0}$ that contains 0 or one if an occurence of this word appears in the email.
* If the represnetation has length $x=100$ the size of the vector will be 100x1, with $x=1$ if the word appears in the email and $x=0$ if it does not appear.
* `Note, in practice we take the most frequently occuring words (10k - 50k) in training set rather than manually picking them.`

How do I best use my time to build this:

* collect lots of data --> "honeypot" project: create fake email adresses to collect spam email
* develop features based on email routing from the email header
* develop more sophisticated features for message body, e.g. punctuation ,. ...
* develop algorithm to detect misspellings (m0rtgage , w4tches ,...).

`Dont fixate on one thing!! Keep looking at your errors to assert you're not wasting time; become more systematic.`

### Recommended Approach

1) Start with a simple algo that you can implement quickly. Spend at most 24hours to get a quick-and-dirty run implementation and test it on your validation data.
2) Once you have done this, plot learning curves of the training and test errors to figure out if you are facing high bias/ high variance. `avoid pre-mature optimization! Let evidence guide your decisions`
3) Error analysis: manually examine examples that your algo made errors on. See if you spot a systematic trend.

### Error analysis

* $m_{val} = 500$ samples in your validation set
* algo misclassifies 100 emails
* manually examine 100 errors, categorize them based on
    * type of email (pharma, fake, phising...) --> count these mistakes, and figure out that it performs the worst on "xx" --> focus on this first.
    * features that are good / bad (misspellings, email routing, 
    punctuations..) --> focus on the one that performs the worst.

Numerical evaluation -> shuold discount /discounts /... be treated as the same word? use stemming software for this (Porter stemmer); this lets u treat all of these words as the same word. This can also hurt performance; universe and university can be the same thing, bc they start with the same word part. `Just try and see if it works. Look at validation errors with / without stemming.` Also consider upper / lower case, ...

`A single value error metric is easy to evaluate if a design choice was right or not.`

### Error metrics for skewed classes

#### Cancer classification example

Train $\htx$ where y=1 if cancer, y=0 otherwise. We find 1% error. However, only 0.50% of patients have cancer.

If you just predict 0 always you get less error!

`When facing skewed classes, use precision / recall.`

![precision vs recall](precision_recall.png)

### Trading off precision and recall

$$ \textnormal{Precision} = \frac{\textnormal{TP}}{\textnormal{TP+FP}}$$

$$ \textnormal{Recall} = \frac{\textnormal{TP}}{\textnormal{TP+FN}}$$

Logistic regression $0 \leq \htx \leq 1$, predict $y=1$ if $\htx > 0.5$. You can up the trheshold value to have higher certainty on the result, for example $\htx \geq 0.9$. You get higher precision, but lower recall.

Suppose you want to avoid missing too many cases of cancer (avoid FN). You can set for example $\htx \geq 0.3$.Now we get a higher recall classifier, but lower precision.

In other words, there's a trade-off. In general, we predict $y=1$ if $\htx \geq$ threshold. You can plot a curve:

![prec_recall_tradeoff.png](prec_recall_tradeoff.png)

`The shape of the PR curve is not always like this`

### $F_1$ score

Takes the precision and recall and creates a single real number evaluation metric.

$F_1$ score combines precision and recall as follows: 

$$F_1 = 2\frac{PR}{P+R}$$

### Data for machine learning

`It's not who has the best algo that wins, it's who has the most data.`

Large data rationale: assume $x \in \RR^{n+1}$ has enough info to predict $y$ correctly.

Given input $x$, can a human expert confidently predict $y$? If yes, increasing amount of data will probably make the prediction better.

Lets use a learnign algo with `alot of parameters` (`low bias` algorithms; neural networks with many hidden units), so they can fit very complex functions. $J_{train}$ will be small. Use a very `large training` set so the algo is unlikely to overfit on the dataset (`low variance` problem). $J_{train} ~= J_{test}$. 

1) can a human predict this accurately from this input
2) can we get a big training set

