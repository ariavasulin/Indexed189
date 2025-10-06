---
course: CS 189
semester: Fall 2025
type: pdf
title: sep24
source_type: pdf
source_file: sep24.pdf
processed_date: '2025-10-06'
processor: mathpix
slug: sep24-processed
---

# Col rank = row rank and more 

## Professor K. A. Ribet

![The image depicts the official seal of the University of California, Berkeley. It features an open b](https://cdn.mathpix.com/cropped/2025_10_06_1a6aea688e76fb3933f0g-01.jpg?height=218&width=219&top_left_y=419&top_left_x=519)

**Image description:** The image depicts the official seal of the University of California, Berkeley. It features an open book at the center, symbolizing knowledge, along with a radiant star above, referencing enlightenment. Surrounding the book are decorative elements, with the Latin phrase "Let There Be Light" inscribed. The year "1868" signifies the establishment date, and the outer circle is bordered with small dots, enhancing its formal design. This seal represents the university's heritage and mission in academia.


September 24, 2025

## Office Hours

- 732 Evans Hall Monday, 1:30-3 PM Thursday, 10:30-noon
![The image depicts a group of students and an instructor in an academic classroom setting. The partic](https://cdn.mathpix.com/cropped/2025_10_06_1a6aea688e76fb3933f0g-02.jpg?height=245&width=441&top_left_y=297&top_left_x=694)

**Image description:** The image depicts a group of students and an instructor in an academic classroom setting. The participants are arranged along a table facing the camera, with a green wall displaying geometric designs in colorful hexagonal patterns. The classroom has multiple tables and chairs set up. Miscellaneous items, such as notebooks and a laptop, are visible on the tables. The purpose of this image is likely to document a class session or group study, highlighting collaboration and interaction within the learning environment.



## Foothill lunches

Noon at Foothill this Friday, and also on October 3, October 9

Meet today at 12:45 at Crossroads

Thursday at 12:20 at the Faculty Club
![The image is a promotional poster for a lecture event featuring Professor Ken Ribet. It includes a h](https://cdn.mathpix.com/cropped/2025_10_06_1a6aea688e76fb3933f0g-03.jpg?height=613&width=478&top_left_y=167&top_left_x=612)

**Image description:** The image is a promotional poster for a lecture event featuring Professor Ken Ribet. It includes a headshot of Professor Ribet in a casual setting. The background is blue with white and yellow text detailing the event: "Lunch with Math Professor Ken Ribet" along with the date and time (September 18, September 26, October 3, October 9 from 12:00 PM to 1:00 PM) and location (Foothill Dining Hall). A QR code is included for additional information, encouraging attendee engagement. The layout is visually appealing, aimed at drawing attention to the event.


## Math Monday talk on Fermat's Last Theorem

By request: I arranged to give a Math Mondays talk on Fermat's Last Theorem this semester.

Talk on October 20 in 1015 Evans from 5 to 6 PM.
Later that evening: Res Life Academic Empowerment Series event on office hours, 8-10 PM in Anchor House

## Column rank, row rank, transpose

The column rank of an $m \times n$ matrix $A$ is the dimension of the span of the $n$ columns of $A$. It's the rank of the linear map $T_{A}: \mathbf{F}^{n} \rightarrow \mathbf{F}^{m}$ defined by $A$.
The row rank of $A$ is the dimension of the span of the $m$ different rows of $A$, each row being in $\mathbf{F}^{n}$.

The transpose of an $m \times n$ matrix $A=\left(a_{j k}\right)$ is the $n \times m$ matrix $\left(a_{k j}\right)$. We exchange columns and rows to pass from $A$ to its transpose $A^{\mathrm{t}}$. Thus the column rank of $A$ is the row rank of $A^{\mathrm{t}}$, and vice versa.

## Theorem

The row and column ranks of a matrix are equal. Equivalently, a matrix and its transpose have equal column ranks.

## Three proofs of the theorem in LADR

Theorem
Row rank = column rank.

- The theorem appears as 3.57 on page 78.
- It's 3.133 on page 114.
- It's proved in Exercises 7 and 8 on page 239.


## Column-row factorization

## Proposition

Let $A$ be an $m \times n$ matrix, and let $c$ be the column rank of $A$. If $c \geq 1$, then $A=C R$, where $C$ is an $m \times c$ matrix and $R$ is a $c \times n$ matrix.

Proof: The matrix $A$ is the matrix of the map $T_{A}=$ multiplication by $A$ from $\mathbf{F}^{n}$ to $\mathbf{F}^{m}$ with respect to the standard bases of those spaces. Let $U \subseteq \mathbf{F}^{m}$ be the range of $T_{A}$, so that $\mathcal{C}=\operatorname{dim} U$ is the column rank of $A$.

Let $\pi: \mathbf{F}^{n} \rightarrow U$ be $T$, thought of as taking values in $U$. Let $\iota: U \rightarrow \mathbf{F}^{m}$ be the inclusion map. By construction, $T=\iota \circ \pi$. Choose a basis of $U$ (needed to represent $\pi$ and $\iota$ by matrices). Then

$$
A=\mathcal{M}(T)=\mathcal{M}(\iota) \mathcal{M}(\pi)
$$

The right-hand matrices are respectively of sizes $m \times c$ and $c \times n$ (with $c=\operatorname{dim} U=\operatorname{rank} T$ ).

## A takeaway

We have written $T_{A}: \mathbf{F}^{n} \rightarrow \mathbf{F}^{m}$ as a surjection $\pi: \mathbf{F}^{n} \rightarrow U$ followed by an injection $\iota: U \rightarrow \mathbf{F}^{m}$. You can do this for every linear map $V \rightarrow W$ : just let $U$ be the range of the map and follow the same procedure.

You can do it for every function from a set to a set. In other words, this is a basic construction.

## Canonicity

The space $U \subseteq \mathbf{F}^{m}$ is the span of the $n$ columns of $A$. A choice-free way to find a basis of $U$ is to prune down the list of the $n$ columns by the method of 2.30: "Every spanning list in a vector space can be reduced to a basis of the vector space." Then the decomposition $A=C R$ becomes entirely canonical (choice free).

With this method, the $m \times c$ matrix is gotten from the $m \times n$ matrix $A$ by chucking away some of $A$ 's columns.

## Example

Take $A=\left(\begin{array}{llll}1 & 2 & 3 & 4 \\ 1 & 5 & 5 & 0 \\ 0 & 1 & 1 & 0\end{array}\right)$, which encodes our room and course numbers. The first three columns of $A$ form a linearly independent list (in $\mathbf{F}^{3}$ ), but the fourth column is 4 times the third column minus 4 times the second column. Thus the column rank is 3 and the first three columns of $A$ form a basis of $U$. The matrix of $\iota$ is the first chunk of $A$ : $C=\left(\begin{array}{lll}1 & 2 & 3 \\ 1 & 5 & 5 \\ 0 & 1 & 1\end{array}\right)$.
To calculate $R$, we express each of the four columns of $A$ as linear combinations of the three basis vectors of $U$ (i.e., columns of $C$ ). The first three columns of $A$ are the columns of $C$. As already mentioned, the fourth column of $A$ is 4 times the third column of $C$, minus 4 times the second column of $C$.
Thus $R=\left(\begin{array}{rrrr}1 & 0 & 0 & 0 \\ 0 & 1 & 0 & -4 \\ 0 & 0 & 1 & 4\end{array}\right)$, and $A=C R$ (check?).

## A somewhat silly example?

Let $A$ be an $m \times n$ matrix whose columns are linearly independent. This means that $T_{A}: \mathbf{F}^{n} \rightarrow \mathbf{F}^{m}$ is injective. Let $X=$ range $T_{A}$, as usual. The map $\pi: \mathbf{F}^{n} \rightarrow X$ is an isomorphism. The basis that we are using for $X$ is the list $A e_{1}, \ldots, A e_{n}$. Hence the matrix for $\pi$ is the $n \times n$ identity matrix. Meanwhile, $C$, which holds that pared down list of columns of $A$ in general, is the entire matrix $A$.

Hence the decomposition $A=C R$ becomes the equation $A=A I$, where $I$ is the $n \times n$ identity matrix.

Check that you really understand this and can reproduce it.

## Column rank = row rank

## Theorem

The row and column ranks of a matrix are equal.
Proof: Let $A$ be an $m \times n$ matrix. To prove that the row rank of $A$ is the column rank of $A$ is to prove that the column ranks of $A$ and $A^{\mathrm{t}}$ are equal. As explained on Monday, it suffices to prove

$$
\text { col-rank } A^{\mathrm{t}} \stackrel{?}{\leq} \text { col-rank } A,
$$
where "col-rank" stands for "column rank." Indeed, if we know this, we can replace $A$ by its transpose to get
$$
\text { col-rank } A=\operatorname{col}-\operatorname{rank}\left(A^{t}\right)^{t} \leq \operatorname{col}-\operatorname{rank} A^{t}
$$

which gives the equality of the two column ranks.

## Column rank = row rank

## Proposition

Let $A$ be a matrix. Then col-rank $A^{\mathrm{t}} \leq \operatorname{col}-\operatorname{rank} A$.
The proposition is true if the column rank of $A$ is 0 , since col-rank $A^{\mathrm{t}}$ is nonnegative.

Anyway, if the column rank of $A$ is $0, A$ is then the 0 matrix, and its row rank is clearly 0 as well. On the next slide, we can and will assume that the column rank $c$ of $A$ is positive.

## Column rank = row rank

## Proposition

Let $A$ be an $m \times n$ matrix. Then col-rank $A^{\mathrm{t}} \leq \operatorname{col}-\operatorname{rank} A$.
Because the column rank of $A$ is positive, we may write $A=C R$, where $C$ is an $m \times c$ matrix and $R$ is a $c \times n$ matrix.

As you agreed on Monday, the transpose of a product is the product of transposes in the opposite order:

$$
A^{\mathrm{t}}=R^{\mathrm{t}} C^{\mathrm{t}} \Longrightarrow T_{A^{\mathrm{t}}}=T_{R^{\mathrm{t}}} T_{C^{\mathrm{t}}} .
$$

One again, " $T$ " refers to the linear map arising from multiplication by a matrix. The column rank of $A^{t}$ is the dimension of the range of $T_{A^{\mathrm{t}}}$, which is contained in the range of $T_{R^{\mathrm{t}}}$. Since $T_{R^{\mathrm{t}}}$ is a linear map $\mathbf{F}^{c} \rightarrow \mathbf{F}^{n}$, its rank is at most $c$.

We have thus shown

$$
\operatorname{col}-\operatorname{rank} A^{\mathrm{t}}=\operatorname{rank} T_{A^{\mathrm{t}}} \leq \operatorname{rank} T_{R^{\mathrm{t}}} \leq c=\operatorname{col}-\operatorname{rank} A .
$$

## Rank of a matrix

If $A$ is a matrix, its column and row ranks are equal. The common value is called the rank of $A$.

## Stuff in Chapter 3

We have already discussed invertibility of linear maps. If $T: V \rightarrow W$ is a linear map between finite-dimensional vector spaces, we showed:

- $T$ injective $\Rightarrow \operatorname{dim} V \leq \operatorname{dim} W$.
- $T$ surjective $\Rightarrow \operatorname{dim} V \geq \operatorname{dim} W$.
- If $\operatorname{dim} V=\operatorname{dim} W$, then $T$ is surjective if and only if $T$ is injective.

If $f: A \rightarrow B$ is a function between finite sets:

- $f$ injective $\Rightarrow|A| \leq|B|$.
- $f$ surjective $\Rightarrow|A| V \geq|B|$.
- If $|\mathrm{A}|=|\mathrm{B}|$, then $f$ is surjective if and only if $f$ is injective.


## Change of basis

If $T: V \rightarrow W$ is a linear map between finite-dimensional vector spaces, we get a matrix $\mathcal{M}(T)$ by choosing bases for $V$ and $W$.

What happens if we change one of the bases? The three natural questions are:

- How does $\mathcal{M}(T)$ change if we change the basis of $V$ ?
- How does $\mathcal{M}(T)$ change if we change the basis of $W$ ?
- How does $\mathcal{M}(T)$ change if we change both bases?

If we know the answer to the first two questions, we get the answer to the third (combining the answers). If we know the answer to the first question, we're pretty close to knowing the answer to the second.

## Two bases of $V$

Suppose that $v_{1}, \ldots, v_{n}$ and $v_{1}^{\prime}, \ldots, v_{n}^{\prime}$ are bases for $V$. As an example, $V$ might be $\mathbf{F}^{n}$, the $v_{j}$ might be the standard basis vectors $e_{j}$, and the $v_{j}^{\prime}$ might be weird af.
It is natural to imagine writing the primed vectors $v_{j}^{\prime}$ in terms of the unprimed vectors $v_{j}$. Say

$$
v_{j}^{\prime}=c_{1 j} v_{1}+\cdots+c_{n j} v_{n} \text { for } \mathrm{j}=1, \ldots, \mathrm{n}
$$

The scalars $c_{i j}$ form an $n \times n$ matrix, whose $j$ th column pertains to $v_{j}^{\prime}$..
Insight of the moment: The matrix $C=\left(c_{i j}\right)$ is the matrix of the identity map $I: V \rightarrow V$ if we use $v_{1}^{\prime}, \ldots, v_{n}^{\prime}$ as the basis of the left-hand copy of $V$ and $v_{1}, \ldots, v_{n}$ for the right-hand copy of $V$. This just follows from the definition of the matrix of a linear map between spaces, each furnished with a basis.

## This will knock you out

Again, imagine $T: V \rightarrow W$ with $\mathcal{M}(T)$ made from $v_{1}, \ldots, v_{n}$ and some basis $w_{1}, \ldots, w_{m}$ of $W$. Let $\mathcal{M}^{\prime}(T)$ be the matrix of $T$ using the bases $v_{1}^{\prime}, \ldots, v_{n}^{\prime}$ and $w_{1}, \ldots, w_{m}$. View $T$ as the composite

$$
V \xrightarrow{l} V \xrightarrow{T} W,
$$
and use the bases
$$
v_{1}^{\prime}, \ldots, v_{n}^{\prime} ; \quad v_{1}, \ldots, v_{n} ; \quad w_{1}, \ldots, w_{m}
$$

on $V, V$ and $W$ (reading from left to right). Then with the right mindset

$$
\mathcal{M}^{\prime}(T)=\mathcal{M}(T \circ I)=\mathcal{M}(T) \mathcal{M}(I)=\mathcal{M}(T) C .
$$

The answer to the first question:
Moving from the first basis of $V$ to the second basis of $V$ multiplies $\mathcal{M}(T)$ on the right by the change of basis matrix $C$.

