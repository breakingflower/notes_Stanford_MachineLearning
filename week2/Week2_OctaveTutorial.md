# Octave Tutorial

## Basic operations

```matlab
%not equal
1 ~= 2

% logical
1 && 0
1 || 0
xor(1, 0)

% change octave prompt
PS1('<wht you want as prompt> ');
% eg.
PS1('<< ');

% semicolon suppresses
a = 2 % prints
a = 2; % doesnt print

% comparison
c = (3 >=1 ); %true

% display
>> a = 3.1416
a =  3.1416
>> disp(sprintf('2 decimals: %0.2f', a))
2 decimals: 3.14

% change display fromat
>> format long % more
>> format short

% matrices
>> A = [1 2; 3 4; 5 6]
A =

   1   2
   3   4
   5   6

% vector with steps specified
>> v = 1:0.1:2
v =

    1.0000    1.1000    1.2000    1.3000    1.4000    1.5000    1.6000    1.7000    1.8000    1.9000    2.0000

>> ones(2,3)
ans =

   1   1   1
   1   1   1

>> w = zeros(1,3)
w =

   0   0   0

% rnd from uniform distribution 0-1
>> z = rand(1,3)
z =

   0.674594   0.267522   0.014574

% rnd from gaussian distribution mean 0 std 1. change std by multiplying randn -> ie z = 5+ sqrt(5)randn(1,3) is stdev sqrt(5) 
>> z = randn(1,3)
z =

   0.29399  -0.62777   1.19401


>> help
..

```

## Move data around

```matlab
>> A = [1 2; 3 4 ; 5 6]
A =

   1   2
   3   4
   5   6

>> size(A)
ans =

   3   2

>> sz = size(A)
sz =

   3   2

>> size(sz)
ans =

   1   2

>> size(A,1)
ans =  3
>> size(A,2)
ans =  2
>> v = [ 1 2 3 4]
v =

   1   2   3   4

>> length(v)
ans =  4
>>
>> length(A) % returns longest size
ans =  3
>> length(A')
ans =  3

%% loading data
>> load ex1data1.txt
>> load ex1data2.txt

>> who
Variables in the current scope:

A         ans       ex1data1  ex1data2  sz        v

>> whos
Variables in the current scope:

   Attr Name          Size                     Bytes  Class
   ==== ====          ====                     =====  =====
        A             3x2                         48  double
        ans           1x2                         16  double
        ex1data1     97x2                       1552  double
        ex1data2     47x3                       1128  double
        sz            1x2                         16  double
        v             1x4                         32  double

Total is 349 elements using 2792 bytes

>> clear v % clears variable v from workspace

>> v = ex1data1(1:10)
v =

   6.1101   5.5277   8.5186   7.0032   5.8598   8.3829   7.4764   8.5781   6.4862   5.0546

>> save xxxx.mat v; % saves v in file xxxx.mat (binary)
>> load xxxx.mat; % loads variable v again
>> save hello.txt v -ascii; % saves v in file hello.txt as text file
```

### Indexing / changing data

```matlab
>> A
A =

   1   2
   3   4
   5   6

>> A(3,2)  % 3rd row second column
ans =  6

>> A(2, :) % : means every elementalong that row / column
ans =

   3   4

>> A([1 3], :) % all elemnets in A whos first index is 1 or 3 and all columns
ans =

   1   2
   5   6

>> A(:,2) = [10;11;12] % change values
A =

    1   10
    3   11
    5   12

>> A = [A, [100; 101; 101]]; % append another colun vector to the right
>> A
A =

     1    10   100
     3    11   101
     5    12   101

>> A(:) % put all elements of A into single vector
ans =

     1
     3
     5
    10
    11
    12
   100
   101
   101

>> A = [1 2; 3 4; 5 6]
A =

   1   2
   3   4
   5   6

>> B = [11 12; 13 14; 15 16]
B =

   11   12
   13   14
   15   16

>> C = [A B]
C =

    1    2   11   12
    3    4   13   14
    5    6   15   16

>> C = [A; B]
C =

    1    2
    3    4
    5    6
   11   12
   13   14
   15   16

```

## Computing on data

```matlab
>> A = [1 2; 3 4 ; 5 6]
A =

   1   2
   3   4
   5   6

>> B = [11 12; 13 14 ; 15 16]
B =

   11   12
   13   14
   15   16

>> C= [1 1; 2 2]
C =

   1   1
   2   2

>> A*C
ans =

    5    5
   11   11
   17   17

>> A.*B %elementwise multiplication
ans =

   11   24
   39   56
   75   96

>> A .^ 2 %elementwise squaring
ans =

    1    4
    9   16
   25   36

>> v = [1;2;3]
v =

   1
   2
   3

>> 1 ./ v % elementwise reciprocal over v
ans =

   1.00000
   0.50000
   0.33333

>> 1./A % elementwise inverse of A
ans =

   1.00000   0.50000
   0.33333   0.25000
   0.20000   0.16667

>> log(v) % elementwise logarithm
ans =

   0.00000
   0.69315
   1.09861

>> exp(v) % e ^ v
ans =

    2.7183
    7.3891
   20.0855

>> -v % -1 * v
ans =

  -1
  -2
  -3

>> A'
ans =

   1   3   5
   2   4   6

>> a = [1 15 2 0.5]
a =

    1.00000   15.00000    2.00000    0.50000

>> val = max(a)
val =  15
>> [val, ind] = max(a) % value and index
val =  15
ind =  2
>> max(A) % columnwise maximum
ans =

   5   6

>> a
a =

    1.00000   15.00000    2.00000    0.50000

>> a < 3
ans =

  1  0  1  1

>> a < 3  % elemntwise comparison
ans =

  1  0  1  1

>> find(a < 3) % find elemnts in a that are less than 3
ans =

   1   3   4

>> A = magic(3) % rows and cols and diag sum up to the same thing
A =

   8   1   6
   3   5   7
   4   9   2

>> help magic
'magic' is a function from the file /usr/share/octave/4.2.2/m/special-matrix/magic.m

 -- magic (N)

     Create an N-by-N magic square.

     A magic square is an arrangement of the integers '1:n^2' such that
     the row sums, column sums, and diagonal sums are all equal to the
     same value.

     Note: N must be a scalar greater than or equal to 3.  If you supply
     N less than 3, magic returns either a nonmagic square, or else the
     degenerate magic squares 1 and [].

Additional help for built-in functions and operators is
available in the online version of the manual.  Use the command
'doc <topic>' to search the manual index.

Help and information about Octave is also available on the WWW
at http://www.octave.org and via the help@octave.org
mailing list.

>> [r,c] = find(A >=7)
r =

   1
   3
   2

c =

   1
   2
   3

>> A(2,3)
ans =  7

>> a
a =

    1.00000   15.00000    2.00000    0.50000

>> sum(a)
ans =  18.500
>> prod(a)
ans =  15
>> floor(a)
ans =

    1   15    2    0

>> ceil(a)
ans =

    1   15    2    1
>> A
A =

   8   1   6
   3   5   7
   4   9   2

>> max(A, [], 1) % columnwise maximum
ans =

   8   9   7

>> max(A, [], 2) % row-wise maximum
ans =

   8
   7
   9

>> max(max(A)) % max of A
ans =  9
>> max(A(:)) % max of A
ans =  9

>> A = magic(9)
A =

   47   58   69   80    1   12   23   34   45
   57   68   79    9   11   22   33   44   46
   67   78    8   10   21   32   43   54   56
   77    7   18   20   31   42   53   55   66
    6   17   19   30   41   52   63   65   76
   16   27   29   40   51   62   64   75    5
   26   28   39   50   61   72   74    4   15
   36   38   49   60   71   73    3   14   25
   37   48   59   70   81    2   13   24   35

>> sum(A, 1)
ans =

   369   369   369   369   369   369   369   369   369

>> sum(A,2)
ans =

   369
   369
   369
   369
   369
   369
   369
   369
   369

>> A.*eye(9)
ans =

   47    0    0    0    0    0    0    0    0
    0   68    0    0    0    0    0    0    0
    0    0    8    0    0    0    0    0    0
    0    0    0   20    0    0    0    0    0
    0    0    0    0   41    0    0    0    0
    0    0    0    0    0   62    0    0    0
    0    0    0    0    0    0   74    0    0
    0    0    0    0    0    0    0   14    0
    0    0    0    0    0    0    0    0   35

>> sum(sum(A.*eye(9))
)
ans =  369

>> sum(sum(A.*flipud(eye(9)))
)
ans =  369

>> flipud(eye(9)) % flip up down
ans =

Permutation Matrix

   0   0   0   0   0   0   0   0   1
   0   0   0   0   0   0   0   1   0
   0   0   0   0   0   0   1   0   0
   0   0   0   0   0   1   0   0   0
   0   0   0   0   1   0   0   0   0
   0   0   0   1   0   0   0   0   0
   0   0   1   0   0   0   0   0   0
   0   1   0   0   0   0   0   0   0
   1   0   0   0   0   0   0   0   0

>> A = magic(3)
A =

   8   1   6
   3   5   7
   4   9   2

>> pinv(A)
ans =

   0.147222  -0.144444   0.063889
  -0.061111   0.022222   0.105556
  -0.019444   0.188889  -0.102778

>> temp = pinv(A)
temp =

   0.147222  -0.144444   0.063889
  -0.061111   0.022222   0.105556
  -0.019444   0.188889  -0.102778

>> temp * A
ans =

   1.00000   0.00000  -0.00000
  -0.00000   1.00000   0.00000
   0.00000   0.00000   1.00000
```

## Plotting data

```matlab
>> plot(t,y1) ; hold on; plot(t,y2) % holds on to first plot instead of replacing

>> print -dpng 'sometitle.png' % prints to png file

>> close % closes figure
>> figure(1) % starts fig 1 ...

>> subplot(1,2,1);% subdivides plot in 1x2 grid, access first element
>> subplot(1,4,3); %subdivide in 1x4 grid, access 3rd element.

>> axis([0.5 1 -1 1]); % [x0 x1 y0 y1]

>> clf % clears fig

>> imagesc(A), colorbar, colormap gray; % shows matrix with colorbar in grayscale.

>> a=1, b=2, c=3; % command chaining --> , does print ; does not print.

```

## Control statements

```matlab
>> v = zeros(10,1)
v =

   0
   0
   0
   0
   0
   0
   0
   0
   0
   0

>> for i=1:10
v(i) = 2^i;
end;
>> v
v =

      2
      4
      8
     16
     32
     64
    128
    256
    512
   1024
```

### Updating search path

```matlab
addpath('/tmp/some/folder')
```

### Functions

Can return multiple values

```matlab
function [y1, y2] = squareAndCube(x)

    y1=x^2;
    y2=x^3;

end
```

## Vectorization

See classic hypothesis in the non vectorized form

$$\mvlrHypotSum$$

Or in the vectorized form, which is the same as the above

$$\mvlrHypot$$

With

$$\t{} = \mat{\t{0} \\ \t{1} \\ \t{2}} \qquad X=\mat{x_0\\x_1\\x_2}$$

And in Octave this becomes

```matlab
% non vectorized
for j=1:n+1
    prediction = prediction + theta(j) * X(j)
end;

% vectorized
prediction = theta' * X
```

A more sophisticated example, consider below with $j \leq 2$.

$$\mvlrGDDeriv$$

Can be implemented vectorized by $\t{} := \t{} - \alpha\delta$

```matlab
theta = theta - alpha*delta
```
