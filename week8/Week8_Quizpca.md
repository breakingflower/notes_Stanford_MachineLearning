# Quiz week 8 - PCA

## Q1

![quiz pca q1](quiz_pca_q1.jpg)

Which of the following figures correspond to possible values that PCA may return for $\ssb{u}{1}$ (the first eigenvector / first principal component)? Check all that apply (you may have to check more than one figure).

`select the vectors that go in the "long" direction of the data, ie bottom left to top right`

## Q2

Which of the following is a reasonable way to select the number of principal components $k$?

(Recall that nnn is the dimensionality of the input data and $m$ is the number of input examples.)

* [ ] Choose the value of $k$ that minimizes the approximation error $\frac{1}{m} \sum_{i=1}^m ||\ssbi{x} - x_{approx}^{(i)}||^2$
.

* [ ] Choose $k$ to be the smallest value so that at least 1% of the variance is retained.

* [ ] Choose $k$ to be 99% of $n$ (i.e., k=0.99∗nk = 0.99*nk=0.99∗n, rounded to the nearest integer).

* [x] Choose $k$ to be the smallest value so that at least 99% of the variance is retained.

## Q3 

Suppose someone tells you that they ran PCA in such a way that "95% of the variance was retained." What is an equivalent statement to this?

`answer below`

$$ \frac{\frac{1}{m} \sum_{i=1}^m ||\ssbi{x} - x_{approx}^{(i)}||^2}{\frac{1}{m} \sum_{i=1}^m ||\ssbi{x} || ^2} \leq 0.05$$

## Q4

Which of the following statements are true? Check all that apply.

* [x] Given an input $x\in\RR^n$, PCA compresses it to a lower-dimensional vector $z\in\RR^k$. `maybe wording is not 100%, and we can also get z in R^n`

* [ ] Feature scaling is not useful for PCA, since the eigenvector calculation (such as using Octave's svd(Sigma) routine) takes care of this automatically. `always apply prior to svd`

* [x] If the input features are on very different scales, it is a good idea to perform feature scaling before applying PCA.

* [ ] PCA can be used only to reduce the dimensionality of data by 1 (such as 3D to 2D, or 2D to 1D)

## Q5

Which of the following are recommended applications of PCA? Select all that apply.

* [ ] Data visualization: To take 2D data, and find a different way of plotting it in 2D (using k=2). `fuck i misread it is already 2d...`

* [x] Data compression: Reduce the dimension of your input data $\ssbi{x}$, which will be used in a supervised learning algorithm (i.e., use PCA so that your supervised learning algorithm runs faster).

* [x] Data compression: Reduce the dimension of your data, so that it takes up less memory / disk space.

* [ ] As a replacement for (or alternative to) linear regression: For most learning applications, PCA and linear regression give substantially similar results.
1 point