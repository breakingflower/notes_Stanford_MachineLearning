# Linear Algebra

## Question 1

Let two matrices be $A = \mat{4 & 3 \\ 6 & 9}, \quad B = \mat{-2 & 9 \\ -5 & 2}$

What is $A+B$

$$ A + B = \mat{2 & 12 \\ 1 & 11}$$

## Question 2

$$ x = \mat{8 \\ 2 \\ 5 \\ 1}$$

$$ 2*x = \mat{16 \\ 4 \\ 10 \\ 2}$$

## Question 3

$$u = \mat{3 \\ 5 \\ 1}$$

$$u^T = \mat{3 & 5 & 1}$$

## Question 4

` The answer to this question was not correct, but I didn't check why.`

$$u = \mat{-3 \\ 4 \\ 3}, v = \mat{3 \\ 1 \\ 5}$$

$$ u^Tv = \mat{-3 & 4 & 3} \cdot \mat{3 \\ 1 \\ 5}$$

```matlab
>> transpose(u)*v
ans =

   -9   -3  -15
   12    4   20
    9    3   15
```

## Question 5

Let $A$ and $B$ be square matrices (3x3). Which of the following must necessarily hold true? Check all that apply.

* [ ] if $C = A \cdot B$, then $C$ is a 6x6 matrix.
* [x] $A + B = B + A$
* [x] if $A$ is the 3x3 identity matrix, then $A*B = B*A$
* [ ] $A*B = B*A$