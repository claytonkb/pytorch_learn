Matrix multiplication and the dot-product
=========================================

Consider the general product of two matrices:

    A * B

In general, this product is not commutative, that is:

    A * B =/= B * A

Established conventions from linear algebra govern how we arrange the resulting
matrix product. Suppose we define:

        [ 1 2 ]
    A = [ 3 4 ]
        [ 5 6 ]

    
    B = [ 1 2 3 ]
        [ 4 5 6 ]

A has 3 rows and 2 columns, so by convention it is a 3x2 matrix (rows x columns).
B, similarly, is a 2x3 matrix.

We can multiply any mxn matrix by any other nxp matrix and we will get an
mxp matrix as a result. To visualize this, imagine sliding the multiplier up
so that the two matrices define a rectangular area, as follows:

    A * B

    [ 1 2 ]   [ 1 2 3 ]
    [ 3 4 ] * [ 4 5 6 ]
    [ 5 6 ]

            [ 1 2 3 ]
            [ 4 5 6 ]

    [ 1 2 ]   o o o
    [ 3 4 ]   o o o
    [ 5 6 ]   o o o

The fields marked 'o' are the rectangular (square, in this case) area where
the results of the matrix multiplication will be written. Note that the number
of columns in the multiplicand (first matrix) must match the number of rows
in the multiplier (second matrix). These restrictions are depicted in the
diagram below:

           __[ 1 2 ... 3 ]
          / _[ 4 5 ... 6 ]
         / /
        / / 
       / /
    [ 1 2 ]    o o ... o
    [ 3 4 ]    o o ... o
    [ ... ]     .  .  .
    [ 5 6 ]    o o ... o

The dot-product is a product operation that operates on two vectors of the
same dimension and produces a scalar result. Each element of the result matrix
is actually a dot product of the corresponding column/row vector of the
matrices being multiplied. Let us give a name to each element of the result
matrix:
    
            [ 1 2 3 ]
            [ 4 5 6 ]

    [ 1 2 ]   a b c
    [ 3 4 ]   d e f
    [ 5 6 ]   g h i

We can visually locate the row/column vectors of the operand matrices involved
in calculating the element labeled 'f' as follows:

            [ # # 3 ]
            [ # # 6 ]

    [ # # ]   # # #
    [ 3 4 ]   # # f
    [ # # ]   # # #

Let us notate the dot-product operator as '.' Then:

    f = [ 3 4 ] . [ [3] [6] ]

... and since the dot-product is insensitive to a vector's orientation (row
or column):

    f = [ 3 4 ] . [ 3 6 ] = 3*3 + 4*6 = 9*24 = 216

This procedure generalizes and allows us to utilize a visual intuition in
understanding the matrix multiplication procedure for 2-dimensional matrices,
relieving us of the need to try to juggle indices, columns and rows in our 
memory.

Since the dimensions of A and B^T (transpose of B) are identical, we can
multiply B and A:

    B * A

    [ 1 2 3 ]   [ 1 2 ]
    [ 4 5 6 ] * [ 3 4 ]
                [ 5 6 ]

              [ 1 2 ]
              [ 3 4 ]
              [ 5 6 ]

    [ 1 2 3 ]   a b
    [ 4 5 6 ]   c d

This illustrates one reason why A\*B =/= B\*A, in general.

Kronecker product
-----------------

The Kronecker product (x) of two matrices A (mxn) and B (pxq) is an mpxnq 
matrix which is the scalar product of B and each element of A:

    [ a b ]     [ e f ]
    [ c d ] (x) [ g h ]

    [ ae af be bf ]
    [ ag ah bg bh ]
    [ ce cf de df ]
    [ cg ch dg dh ]

The pattern should be obvious from this example and it generalizes to matrices
of arbitrary dimensions.

Hadamard product
----------------

Any two matrices of the same dimension may be multiplied "element-wise" - this
is called the Hadamard product of two matrices:

    [ 1  2  ]   [ 7  8  ]   [ 7  16 ]
    [ 3  4  ] * [ 9  10 ] = [ 27 40 ]
    [ 5  6  ]   [ 11 12 ]   [ 55 72 ]

This matrix operations is commonly used in neural-nets implementations and
sometimes confusingly referred to as "matrix multiplication", even though it is
completely different than standard matrix multiplication (see above).

A note on subscripts
--------------------

The Mij subscript notation used frequently in linear algebra texts uses the
following conventions:

- The first subscript (i in this case) is the ROW number
- The second subscript (j in this case) is the COLUMN number

For small matrices (less than 10 columns and rows), this is notated by appending
the digits, e.g. M32 means "3rd row, 2nd column of matrix M".

