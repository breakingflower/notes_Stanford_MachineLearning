# Quiz anomaly detection

## Q1

For which of the following problems would anomaly detection be a suitable algorithm?

* [ ] Given data from credit card transactions, classify each transaction according to type of purchase (for example: food, transportation, clothing).

* [x] In a computer chip fabrication plant, identify microchips that might be defective.

* [x] From a large set of primary care patient records, identify individuals who might have unusual health conditions.

* [ ] From a large set of hospital patient records, predict which patients have a particular disease (say, the flu).

## Q2

Suppose you have trained an anomaly detection system for fraud detection, and your system that flags anomalies when $p(x)$ is less than $\epsilon$, and you find on the cross-validation set that it is missing many fradulent transactions (i.e., failing to flag them as anomalies). What should you do?

* [ ] Decrease $\epsilon$

* [x] Increase $\epsilon$

## Q3

Suppose you are developing an anomaly detection system to catch manufacturing defects in airplane engines. You model uses

$$ p(x) = \prod_{j=1}^n p(x_j; \mu_j, \sigma_j^2) =  \prod_{j=1}^n  \frac{1}{\sqrt{2\pi} \sigma_j} exp\Bigg(-\frac{(x_j-\mu_j)^2}{2\sigma_j^2}\Bigg)$$

You have two features $x_1$​ = vibration intensity, and $x_2$​ ​ = heat generated. Both $x_1$ and $x_2$​​ take on values between 0 and 1 (and are strictly greater than 0), and for most "normal" engines you expect that $x_1 \approx x_2$​. One of the suspected anomalies is that a flawed engine may vibrate very intensely even without generating much heat (large $x_1$, small $x_2$​), even though the particular values of $x_1$and $x_2$​ may not fall outside their typical ranges of values. What additional feature $x_3$ should you create to capture these types of anomalies:

* [ ] $x_3 = x_1 + x_2$
* [ ] $x_3 = \frac{1}{x_1}$
* [x] $x_3 = \frac{x_1}{x_2}$ `relates both`
* [ ] $x_3 = \frac{1}{x_2}$



## Q4 

Which of the following are true? Check all that apply.

* [x] When developing an anomaly detection system, it is often useful to select an appropriate numerical performance metric to evaluate the effectiveness of the learning algorithm.

* [ ] In a typical anomaly detection setting, we have a large number of anomalous examples, and a relatively small number of normal/non-anomalous examples. `invertred`

* [x] In anomaly detection, we fit a model $p(x)$ to a set of negative ($y=0$) examples, without using any positive examples we may have collected of previously observed anomalies. `true`

* [ ] When evaluating an anomaly detection algorithm on the cross validation set (containing some positive and some negative examples), classification accuracy is usually a good evaluation metric to use. `no use recall/f1/prec`

* [ ] If you are developing an anomaly detection system, there is no way to make use of labeled data to improve your system.

* [x] When choosing features for an anomaly detection system, it is a good idea to look for features that take on unusually large or small values for (mainly the) anomalous examples.

* [x] If you do not have any labeled data (or if all your data has label $y=0$), then is is still possible to learn p(x), but it may be harder to evaluate the system or choose a good value of ϵ.

* [ ] If you have a large labeled training set with many positive examples and many negative examples, the anomaly detection algorithm will likely perform just as well as a supervised learning algorithm such as an SVM.

## q5
.You have a 1-D dataset $\trainingSetUnsupervised$ and you want to detect outliers in the dataset. You first plot the dataset and it looks like this:

(fig)

What are the parameters $\mu_1, \sigma_1^2$? 

`middle is ~-3, sigma about 2`

This was wrong, sigma shuold be 4 (wider)