---
course: CS 189
semester: Fall 2025
type: pdf
title: sep22
source_type: pdf
source_file: sep22.pdf
processed_date: '2025-10-06'
processor: mathpix
slug: sep22-processed
---

# Matrices, ranks,... 

## Professor K. A. Ribet

September 22, 2025

## Office Hours

- 732 Evans Hall

Monday, 1:30-3 PM
Thursday, 10:30-noon
![The image depicts a group of students and instructors standing in a classroom with tables arranged i](https://cdn.mathpix.com/cropped/2025_10_06_243bfb33ad17c1c21559g-02.jpg?height=242&width=440&top_left_y=212&top_left_x=694)

**Image description:** The image depicts a group of students and instructors standing in a classroom with tables arranged in a U-shape, promoting interaction. The background features a colorful display of patterned geometric designs, likely intended to inspire creativity or relate to mathematical concepts. The classroom is set up for a collaborative learning environment, indicated by the presence of laptops and notebooks on the tables. Overall, the image serves to illustrate a dynamic academic setting.


Come to office hours, even only once. To introduce yourself. To meet other students and hear what they have to say.

## Lunches with Math Prof. Ken Ribet

![I'm unable to describe the content of the image directly. However, if you have specific information ](https://cdn.mathpix.com/cropped/2025_10_06_243bfb33ad17c1c21559g-03.jpg?height=299&width=309&top_left_y=188&top_left_x=98)

**Image description:** I'm unable to describe the content of the image directly. However, if you have specific information about diagrams, equations, or miscellaneous images to share, I can help you summarize or explain those elements in detail.


## Lunch with Math Prof. Ken Ribet

Location: Stern Courtyard, next to Foothill Dining Commons

Dates:
Time: $3 p m-4 p m$
Thurs. 9/18, Fri. 9/26, Fri. 10/3, Thurs. 10/9
At Foothill DC on September 18, September 26, October 3 and October 9. They're at 3 PM EDT, noon for us.
The next event in this series is on Friday at noon.
Lunch also on Wednesday (September 24) at 12:45 at Crossroads.

## Math Monday talk on Fermat's Last Theorem

After speaking with several of you in August, I volunteered to give a Math Mondays talk on Fermat's Last Theorem this semester.

Talk will be on October 20 in 1015 Evans from 5 to 6 PM.
Later that evening: Res Life Academic Empowerment Series event on office hours, 8-10 PM in Anchor House

## While you were sleeping on Friday morning. . .

If $\operatorname{dim} V=n$ and $\operatorname{dim} W=m$, then the choice of a basis of $V$ yields an isomorphism of vector spaces

$$
\mathcal{L}(V, W) \xrightarrow{\sim} W^{n} .
$$

This isomorphism depends on the chosen basis of $V$.
A numerical consequence:

$$
\operatorname{dim} \mathcal{L}(V, W)=\operatorname{dim}\left(W^{n}\right)=n \cdot \operatorname{dim} W .
$$

If $V=\mathbf{F}^{n}$ and we use the standard basis of $V$, then we obtain a "choice-free" isomorphism

$$
\mathcal{L}\left(\mathbf{F}^{n}, W\right) \xrightarrow{\sim} W^{n} .
$$

## Dimension of $\mathcal{L}(V, W)$

Alternative perspective: choose bases of both $V$ and $W$ and convince yourself that

$$
\mathcal{L}(V, W) \approx \mathcal{L}\left(\mathbf{F}^{n}, \mathbf{F}^{m}\right),
$$
if $m=\operatorname{dim} W$.
The space on the right is the space of $m \times n$ matrices, which has dimension $m n$, since an $m \times n$ matrix is secretly an $m n$-tuple written in two dimensions instead of along a line.

## Nitty gritty on matrices

Suppose that $v_{1}, \ldots, v_{n}$ and $w_{1}, \ldots, w_{m}$ are bases of $V$ and $W$ (respectively). The matrix corresponding to a linear map $T: V \rightarrow W$ is built as follows:
For $j=1, \ldots, n, T v_{j}$ gives rise to the $j$ column of the matrix $A$ corresponding to $T$. If

$$
T v_{j}=\sum_{i=1}^{m} a_{i j} w_{i}:=a_{1 j} w_{1}+\cdots+a_{m j} w_{m},
$$
the $j$ th column of $A$ is $\left(\begin{array}{c}a_{1 j} \\ a_{2 j} \\ \vdots \\ a_{m j}\end{array}\right)$ and $A=\left(\begin{array}{cccc}a_{11} & a_{12} & \cdots & a_{1 n} \\ a_{21} & a_{22} & \cdots & a_{2 n} \\ \vdots & \vdots & \vdots & \vdots \\ a_{m 1} & a_{m 2} & \cdots & a_{m n}\end{array}\right)$.

## Nitty gritty on matrices

## Formula

With $v_{1}, \ldots, v_{n}, w_{1}, \ldots, w_{m}$ and $T$ as before, suppose that $v \in V$ is the linear combination $x_{1} v_{1}+\cdots+x_{n} v_{n}$, and that $T v=y_{1} w_{1}+\cdots+y_{m} w_{m}$. If

$$
\begin{gathered}
T \longleftrightarrow A=\left(\begin{array}{cccc}
a_{11} & a_{12} & \cdots & a_{1 n} \\
a_{21} & a_{22} & \cdots & a_{2 n} \\
\vdots & \vdots & \vdots & \vdots \\
a_{m 1} & a_{m 2} & \cdots & a_{m n}
\end{array}\right), \\
\text { then }\left(\begin{array}{c}
y_{1} \\
y_{2} \\
\vdots \\
y_{m}
\end{array}\right)=A\left(\begin{array}{c}
x_{1} \\
x_{2} \\
\vdots \\
x_{n}
\end{array}\right) .
\end{gathered}
$$

## Axler's notation

If $A$ is a matrix, the entries of $A$ are called $A_{j, k}$ in LADR. I just call them $a_{j k}$.

## Verification of the formula

We use summation notation and the principle of changing the order of summation: if you're adding up a rectangular array of things that add, you can add up each column and sum the column totals, or else add up the rows and sum the row totals.
Remember also that the $\ell$ th entry of the matrix product $A\left(\begin{array}{c}x_{1} \\ x_{2} \\ \vdots \\ x_{n}\end{array}\right)$
is $\sum_{k} a_{\ell k} x_{k}$. The formula to be proved is

$$
T v=T\left(\sum_{k} x_{k} v_{k}\right) \stackrel{?}{=} \sum_{\ell}\left(\sum_{k} a_{\ell k} x_{k}\right) w_{\ell} .
$$

## Verification of the formula

$$
\begin{aligned}
T\left(\sum_{k} x_{k} v_{k}\right) & =\sum_{k} x_{k} T v_{k}=\sum_{k} x_{k}\left(\sum_{\ell} a_{\ell k} w_{\ell}\right) \\
& =\sum_{\ell}\left(\sum_{k} a_{\ell k} x_{k}\right) w_{\ell}
\end{aligned}
$$

## Axler's notation for the space of matrices

The space of $m \times n$ matrices with coefficients in $\mathbf{F}$ is $\mathbf{F}^{m, n}$ in LADR.

## Matrix multiplication

If $A$ is an $m \times n$ matrix and $B$ is an $n \times p$ matrix, we define the product $A B$ as an $m \times p$ matrix. The $(i, j)^{\text {th }}$ entry of the product (for $i=1, \ldots, m$ and $j=1, \ldots, p$ ) is

$$
a_{i 1} b_{1 j}+a_{i 2} b_{2 j}+\cdots+a_{i n} b_{n j}=\sum_{k=1}^{n} a_{i k} b_{k j} .
$$

Axler would never use $i$ as an index because it's $\sqrt{-1}$, and he'd use commas between indices $\left(a_{i, k}\right)$. He'd use capital letters for the entries. See page 73 of LADR.

## Matrix multiplication corresponds to composition

If $T: U \rightarrow V$ and $S: V \rightarrow W$ are linear maps, $S T=S \circ T$ is a linear map $U \rightarrow W$. The matrices corresponding to the maps $S, T, S T$ are denoted $\mathcal{M}(S), \mathcal{M}(T), \mathcal{M}(S T)$. Two years from now, you'll surely remember this result:

## Formula

$$
\mathcal{M}(S T)=\mathcal{M}(S) \mathcal{M}(T)
$$

Matrices depend on bases, though the bases do not appear in the simple notations $\mathcal{M}(S)$, etc.
In the formula, one has implictly chosen bases of $U, V$ and $W$. Each basis is needed twice in the formula. For example, the basis of $V$ is used in creating $\mathcal{M}(T)$ and $\mathcal{M}(S)$. The basis of $U$ is used in creating $\mathcal{M}(T)$ and $\mathcal{M}(S T)$. The basis of $W$ is used for $\mathcal{M}(S)$ and $\mathcal{M}(S T)$.

## Matrix multiplication corresponds to composition

A proof of the formula $\mathcal{M}(S T)=\mathcal{M}(S) \mathcal{M}(T)$ appears on p. 73 of LADR. I am unlikely to recite it in 155 Dwinelle.

## A computation

Recall $T: U \rightarrow V, S: V \rightarrow W, S T: U \rightarrow W$. Take bases $u_{1}, \ldots, u_{p}$ of $U, v_{1}, \ldots, v_{n}$ of $V, w_{1}, \ldots, w_{m}$ of $W$. Let $A=\mathcal{M}(S), B=\mathcal{M}(T)$; their sizes are $m \times n$ and $n \times p$. We verify the formula $\mathcal{M}(S T)=A B$ by showing that the $(i, j)^{\text {th }}$ entries of the two matrices are equal for each $i$ and $j$.
The $(i, j)^{\text {th }}$ entry of $\mathcal{M}(S T)$ is the coefficient of $w_{i}$ when $(S T) u_{j}$ is expressed in terms of the basis of $W$. This computation shows that it's equal to the $(i, j)^{\text {th }}$ entry of $A B$ :

$$
\begin{aligned}
(S T) u_{j} & =S\left(T u_{j}\right)=S\left(\sum_{k} b_{k j} v_{k}\right)=\sum_{k} b_{k j} S v_{k} \\
& =\sum_{k} b_{k j}\left(\sum_{i} a_{i k} w_{i}\right)=\sum_{i}\left(\sum_{k} a_{i k} b_{k j}\right) w_{i} .
\end{aligned}
$$

Everyone should study this computation carefully.

## Reflections on matrix multiplication

The $(i, j)^{\text {th }}$ entry of $A B$ is the sum $\sum_{k} a_{i k} b_{k j}$. This is the dot product of two elements of $\mathbf{F}^{n}$, namely the $i$ th row of $A$ and the $j$ th column of $B$. The dot product is the entry in the $i$ th row and $j$ th column of $A B$.

## Reflections on matrix multiplication

Left multiplication by $A$ sends columns of length $n$ to columns of length $m$. View $B$ as the concatenation of $p$ columns, each of length $n$, say $B=\left[c_{1} c_{2} \cdots c_{p}\right]$. Then we see that $A B$ is the concatenation of the $p$ columns

$$
A c_{1}, A c_{2}, \ldots, A c_{p}
$$
each of length $m$.

## Reflections on matrix multiplication

Consider $A=\left[d_{1} \cdots d_{n}\right]$ as the concatenation of $n$ columns of length $m$. Then

$$
A\left(\begin{array}{c}
x_{1} \\
x_{2} \\
\vdots \\
x_{n}
\end{array}\right)=x_{1} d_{1}+x_{2} d_{2}+\cdots+x_{n} d_{n}
$$

Thus the range of the is the span of the columns of $A$.

## Axler's notation

Let $A$ be an $m \times n$ matrix, with entries $A_{j, k}, 1 \leq j \leq m$, $1 \leq k \leq n$.

The symbol $A_{j}$, denotes the $j$ th row of $A$.
The symbol $A_{, k}$ denotes the $k$ th column of $A$.

## One more reflection, with new notation

We pivot notation from $A B$ to $C R$, where $C$ is an $m \times c$ matrix and $R$ is a $c \times n$ matrix. Thus $C R$ is an $m \times n$ matrix.

The $k$ th column of $C R$ is the product of $C$ with the $k$ th column of $R$. This product is a linear combination of the $c$ different columns of $C$; the scalars in the linear combination are the entries in the $k$ th column of $R$.

Analogously, the $j$ th row of $C R$ is the product of the $j$ th row of $C$ and the matrix $R$. It's a linear combination of the rows of $R$, the coefficients coming from the $j$ th row of $C$

## Column rank, row rank, transpose

The column rank of an $m \times n$ matrix $A$ is the dimension of the span of the $n$ columns of $A$. It's the rank of the linear map $\mathbf{F}^{n} \rightarrow \mathbf{F}^{m}$ defined by $A$.

The row rank of $A$ is the dimension of the span of the $m$ different rows of $A$, each row being in $\mathbf{F}^{n}$.

The transpose of an $m \times n$ matrix $A=\left(a_{j k}\right)$ is the $n \times m$ matrix $\left(a_{k j}\right)$. We exchange columns and rows to pass from $A$ to its transpose $A^{\mathrm{t}}$. Thus the column rank of $A$ is the row rank of $A^{\mathrm{t}}$, and vice versa.

## Theorem

The row and column ranks of a matrix are equal. A matrix and its transpose have equal column ranks.

## This is like a key result

## Theorem

The row and column ranks of a matrix are equal.
Three places to find proofs of this result in LADR:

- The theorem first appears as 3.57 on page 78.
- It's 3.133 on page 114.
- It's proved in Exercises 7 and 8 on page 239.

The proof that we're about to see uses column-row factorization (3.56), which I found to be hella opaque when I first encountered it. I provided an alternative explanation of the situation last semester; see the February 12 slides for spring, 2025 (available on bCourses).

## Column-row factorization

## Proposition

Let $A$ be an $m \times n$ matrix, and let $c$ be the column rank of $A$. If $c \geq 1$, then $A$ is the product $C R$, where $C$ is an $m \times c$ matrix and $R$ is a $c \times n$ matrix.

My proof of the proposition: The matrix $A$ is the matrix of the map $T=$ multiplication by $A$ from $\mathbf{F}^{n}$ to $\mathbf{F}^{m}$ (with respect to the standard bases of those spaces. Let $U \subseteq \mathbf{F}^{m}$ be the range of $A$. The dimension of $U$-call it $c$-is the column rank of $A$.

Let $\pi: \mathbf{F}^{n} \rightarrow U$ be $T$, thought of as taking values in $U$. (It's onto, by definition.) Let $\iota: U \rightarrow \mathbf{F}^{m}$ be the inclusion map. (It's 1-1.) By construction, $T=\iota \circ \pi$. Choose a basis of $U$ in order to represent $\pi$ and $\iota$ be matrices. Then

$$
A=\mathcal{M}(T)=\mathcal{M}(\pi) \mathcal{M}(\iota)
$$

The two right-hand matrices are respectively of sizes $m \times c$ and $c \times n$ if $c=\operatorname{dim} U=\operatorname{rank} T$.

## Column-row factorization

## Proposition

Let $A$ be an $m \times n$ matrix, and let $c$ be the column rank of $A$. If $c \geq 1$, then $A$ is the product $C R$, where $C$ is an $m \times c$ matrix and $R$ is a $c \times n$ matrix.

Axler's proof: The columns of $A$ are elements of $\mathbf{F}^{m}$. They span a subspace of $\mathbf{F}^{m}$ of dimension $c$. A basis of this subspace is obtained by pruning the list of columns until we have $c$ linearly independent columns. All columns of $A$ are then linear combinations of these $c$ columns. Let $C$ be the matrix consisting of these $c$ columns (in their original order, say). This is an $m \times c$ matrix. Each of the $n$ columns of $A$ is a linear combination of the $c$ columns. Let $R$ be the matrix of size $c \times n$ whose $k$ th column contains the coefficients of the linear combination expressing the $k$ th column of $A$ in terms of the $c$ different columns of $C$. Then $A=C R$ (as explained on the next slide).

## Column-row factorization

Another look at the proof: The columns of $A$ span a subspace of $\mathbf{F}^{m}$ of dimension $c$. A basis of this subspace (of length $c$ ) is obtained by pruning the list of columns. Each column of $A$ is some linear combination of these $c$ columns. The matrix $C$ consists of these $c$ columns. It has size $m \times c$.

Each column of $A$ is a linear combination of the $c$ columns of $C$. Let $R$ be the matrix of size $c \times n$ whose $k$ th column contains the coefficients of the linear combination expressing the $k$ th column of $A$ in terms of the columns of $C$. Then $A=C R$ :

The equation $A=C R$ results from our understanding of how $C R$ is computed: we compute column by column, multiplying $C$ by each column of $R$ in turn. If $x$ is one of those columns, then $C_{x}$ is a linear combination of the columns of $C$, the coefficients coming from the entries of $x$.

## An example

Start with the matrix $A=\left(\begin{array}{llll}1 & 2 & 3 & 4 \\ 1 & 5 & 5 & 0 \\ 0 & 1 & 1 & 0\end{array}\right)$. As is easy to guess, the column rank is 3 , and the first three columns are linearly independent elements of $\mathbf{F}^{3}$. Taking those columns gives $C=\left(\begin{array}{lll}1 & 2 & 3 \\ 1 & 5 & 5 \\ 0 & 1 & 1\end{array}\right)$. To get $R$, we express each of the four columns of $A$ as linear combinations of the three columns of $C$. The first column of $A$ is the first column of $C$ on the nose. The same goes for the second and third columns of $A$. The fourth column turns out to be 4 times the third column of $C$, minus 4 times the second column: $R=\left(\begin{array}{rrrr}1 & 0 & 0 & 0 \\ 0 & 1 & 0 & -4 \\ 0 & 0 & 1 & 4\end{array}\right)$. The great insight is that $A=C R$, which you can verify easily by hand.

## Column rank = row rank

## Theorem

The row and column ranks of a matrix are equal. A matrix and its transpose have equal column ranks.

Proof: Let $A$ be an $m \times n$ matrix. To prove that the row rank of $A$ is the column rank of $A$ is to prove that the column ranks of $A$ and $A^{\mathrm{t}}$ are equal. Let "c-rank" denote the column rank of a matrix. It suffices to prove

$$
\text { col-rank } A^{\mathrm{t}} \leq \text { col-rank } A
$$
because we can replace $A$ by its transpose and get
$$
\text { col-rank } A=\operatorname{col}-\operatorname{rank}\left(A^{\mathrm{t}}\right)^{\mathrm{t}} \leq \text { col-rank } A^{\mathrm{t}}
$$

and then conclude that the two ranks are equal.
If the column rank of $A$ is 0 , then $A$ is the 0 matrix, and its row rank is 0 as well. Thus we can and will assume that the column rank $c$ of $A$ is positive.

## Column rank = row rank

## Theorem

The row and column ranks of a matrix are equal. A matrix and its transpose have equal column ranks.

Because the column rank of $A$ is positive, we may write $A=C R$, where $C$ is an $m \times c$ matrix and $R$ is a $c \times n$ matrix.

We use the fact that the transpose of a product is the product of transposes in the opposite order.

$$
A^{\mathrm{t}}=R^{\mathrm{t}} C^{\mathrm{t}}, \quad T_{A^{\mathrm{t}}}=T_{R^{\mathrm{t}}} T_{C^{\mathrm{t}}}
$$

Here, " $T$ " refers to the linear map arising from multiplication by a matrix. The column rank of $A^{\mathrm{t}}$ is the dimension of the range of $T_{A^{\mathrm{t}}}$, which is contained in the range of $T_{R^{\mathrm{t}}}$. Since $T_{R^{\mathrm{t}}}$ is a linear map $\mathbf{F}^{c} \rightarrow \mathbf{F}^{n}$, its rank is at most $c$.
We have thus show:

$$
\operatorname{col}-\operatorname{rank} A^{\mathrm{t}}=\operatorname{rank} T_{A^{\mathrm{t}}} \leq \operatorname{rank} T_{R^{\mathrm{t}}} \leq c=\operatorname{col}-\operatorname{rank} A
$$

