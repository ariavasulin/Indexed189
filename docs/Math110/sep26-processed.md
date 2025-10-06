---
course: CS 189
semester: Fall 2025
type: pdf
title: sep26
source_type: pdf
source_file: sep26.pdf
processed_date: '2025-10-06'
processor: mathpix
slug: sep26-processed
---

# Change of basis 

## Professor K. A. Ribet

![The image depicts the official seal of the University of California, Berkeley. It features an open b](https://cdn.mathpix.com/cropped/2025_10_06_fad3d818a1696d9f1756g-01.jpg?height=213&width=218&top_left_y=427&top_left_x=519)

**Image description:** The image depicts the official seal of the University of California, Berkeley. It features an open book at the center, symbolizing knowledge, with the phrase "LET THERE BE LIGHT" inscribed below. Above the book, a star is placed, representing enlightenment. The seal is encircled by a gold border, adorned with dots, and includes the year "1868," marking the university's founding. This emblem serves to convey the institution's commitment to education and research in academia.


September 26, 2025

## Office Hours

- 732 Evans Hall Monday, 1:30-3 PM Thursday, 10:30-noon
![The image shows a group of students and an instructor standing in a classroom. The arrangement inclu](https://cdn.mathpix.com/cropped/2025_10_06_fad3d818a1696d9f1756g-02.jpg?height=245&width=441&top_left_y=297&top_left_x=694)

**Image description:** The image shows a group of students and an instructor standing in a classroom. The arrangement includes a long table with chairs, and some students are seated while others are standing. Behind them, there are geometric diagram posters on the walls, featuring colorful star-like shapes arranged in a symmetrical pattern. The classroom appears well-lit and organized, promoting an academic atmosphere. The image serves to illustrate a learning environment, perhaps showcasing a math or science course with interactive elements.



## Foothill lunches

Noon at Foothill today and also on October 3, October 9
![The slide features a portrait of Professor Ken Ribet, accompanied by the event details for "Lunch wi](https://cdn.mathpix.com/cropped/2025_10_06_fad3d818a1696d9f1756g-03.jpg?height=616&width=478&top_left_y=165&top_left_x=612)

**Image description:** The slide features a portrait of Professor Ken Ribet, accompanied by the event details for "Lunch with Math Professor Ken Ribet." The text includes the program's name, timing (12:00 PM - 1:00 PM), location (Foothill Dining Hall), and specific dates (September 18, September 26, October 3, October 9). A QR code is included for additional information. The design emphasizes accessibility, encouraging participation and collaboration among students and faculty.


## Two events on October 20

Math Mondays talk on Fermat's Last Theorem, 5:10-6 PM in 1015 Evans

Res Life Academic Empowerment Series event on office hours, 8-10 PM in Anchor House

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

The scalars $c_{i j}$ form an $n \times n$ matrix, whose $j$ th column pertains to $v_{j}^{\prime}$.
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

## Change basis of $W$

Now suppose that $v_{1}, \ldots, v_{n}$ stays invariant as the basis of $V$ but that we consider bases $w_{1}, \ldots, w_{m}$ and $w_{1}^{\prime}, \ldots, w_{m}^{\prime}$ of $W$. We think that the first bases as the basis that we started with and the second one as a variant or alternative. Then it is natural to write

$$
w_{j}^{\prime}=d_{1 j} w_{1}+\cdots+d_{m j} v_{m} \text { for } \mathrm{j}=1, \ldots, \mathrm{~m}
$$

The scalars $d_{i j}$ form an $m \times m$ matrix, whose $j$ th column pertains to $w_{j}^{\prime}$.
Insight of the moment: The matrix $D=\left(d_{i j}\right)$ is the matrix of the identity map $I: W \rightarrow W$ if we use $w_{1}^{\prime}, \ldots, w_{m}^{\prime}$ as the basis of the left-hand copy of $W$ and $w_{1}, \ldots, w_{m}$ for the right-hand copy of $W$.

## This will knock you out

Again, imagine $T: V \rightarrow W$ with $\mathcal{M}(T)$ made from $v_{1}, \ldots, v_{n}$ and some basis $w_{1}, \ldots, w_{m}$ of $W$. Let $\mathcal{M}^{\prime}(T)$ be the matrix of $T$ using the bases $v_{1}, \ldots, v_{n}$ and $w_{1}^{\prime}, \ldots, w_{m}^{\prime}$. View $T$ as the composite

$$
V \xrightarrow{T} W \xrightarrow{l} W,
$$
and use the bases
$$
v_{1}, \ldots, v_{n} ; \quad w_{1}^{\prime}, \ldots, w_{m}^{\prime} ; \quad w_{1}, \ldots, w_{m}
$$

on $V, W$ and $W$ (reading from left to right). Then with the right mindset

$$
\mathcal{M}(T)=\mathcal{M}(I \circ T)=\mathcal{M}(I) \mathcal{M}^{\prime}(T)=D \mathcal{M}^{\prime}(T)
$$

This gives the formula $\mathcal{M}^{\prime}(T)=D^{-1} \mathcal{M}(T)$. Moving from the first basis of $W$ to the second basis of $W$ multiplies $\mathcal{M}(T)$ on the left by the inverse of the change of basis matrix $D$.

## Changing both bases

If the matrix of $T: V \rightarrow W$ with respect to $v_{1}, \ldots, v_{n}$ and $w_{1}, \ldots, w_{m}$ is $A$, then the matrix of $T$ with respect to the bases $v_{1}^{\prime}, \ldots, v_{n}^{\prime}$ and $w_{1}^{\prime}, \ldots, w_{m}^{\prime}$ is $D^{-1} A C$, where $D$ is the change of basis matrix for $W$ and $C$

## One space, two bases

Suppose that $T: V \rightarrow V$ is an operator on $V$ (i.e., a linear map from a space to itself). If $A$ is the matrix of $T$ with respect to $v_{1}, \ldots, v_{n}$ (used on both copies of $V$ ), then the matrix of $T$ with respect to $v_{1}^{\prime}, \ldots, v_{n}^{\prime}$ (used on both copies of $V$ ) is $C^{-1} A C$, where the columns of $C$ express the basis $v_{1}^{\prime}, \ldots, v_{n}^{\prime}$ in terms of the vectors $v_{1}, \ldots, v_{n}$ (change of basis matrix).
This result is 3.84 on page 93 of the book.

## Quotient spaces

If $U$ is a subspace of $V$, we will construct a vector space $V / U$ together with a surjective linear map

$$
\pi: V \rightarrow V / U
$$
whose null space is $U$. If $V$ has finite dimension, then it follows from rank-nullity that
$$
\operatorname{dim} V / U=\operatorname{dim} V-\operatorname{dim} U .
$$

## Quotient spaces

If $V$ is an F -vector space and $U$ is a subspace of $V, V / U$ is the set of translates of $U$ by elements of $V$ :

$$
v+U:=\{v+u \mid u \in U\} .
$$

These are subsets of $V$ but typically not subspaces; for example, you can check that note that $v+U$ contains 0 if and only if $v$ is an element of $U$
To get a mental image of these subsets, take $V=\mathbf{R}^{2}$ and let $U$ be a 1 -dimensional subspace of $V$. Then $U$ is a line " $y=m x$ " through the origin. The translates of $U$ are the lines $y=m x+b$ as $b$ runs over $\mathbf{R}$. The whole plane is the union of these translated lines. The lines are parallel; two lines are either identical or have no point in common.

## Quotient spaces

## Proposition

For $v$ and $v^{\prime}$ in $V$ and $U$ a subspace of $V$,

$$
v+U=v^{\prime}+U \Longleftrightarrow v-v^{\prime} \in U .
$$

If you're a fan of Math 55 , you might prefer to introduce these translates as follows: Consider the equivalence relation on $V$ such that $v \sim v^{\prime}$ if and only if $v-v^{\prime} \in U$. Then $v+U$ is the equivalence class of $v$. As is true for equivalence relations in general, two translates (= equivalence classes) are either identical or disjoint.

## Addition of two translates

The set of all translates of $U$ is $V / U$ (by definition). We want to define an addition law and a scalar multiplication on this set.

Addition

$$
\left(v_{1}+U\right)+\left(v_{2}+U\right):=\left(v_{1}+v_{2}\right)+U .
$$

This seems straightforward, but we need to check that this addition is well defined. For this, imagine that $v_{1}+U$ is also $v_{1}^{\prime}+U$. Is it true that $\left(v_{1}+v_{2}\right)+U$ is also $\left(v_{1}^{\prime}+v_{2}\right)+U$ ? Yes because $v_{1}^{\prime}-v_{1}$ is in $U$, and therefore so is $\left(v_{1}^{\prime}+v_{2}\right)-\left(v_{1}+v_{2}\right)$.

## Scalar multiplication

We define $\lambda \cdot(v+U):=\lambda v+U$. This is once again well defined because if $v+U=v^{\prime}+U$, then $v^{\prime}-v$ is in $U$, so that $\lambda\left(v^{\prime}-v\right)=\lambda v^{\prime}-\lambda v$ is in $U$.

## Why are the axioms verified?

Basically because they're verified for addition of vectors together with scalar multiplication of vectors. The operations for $V / U$ are derived from the operations for $V$ by simple non-threatening formulas.

## What is $\pi$ ?

The function

$$
\pi: V \rightarrow V / U, \quad v \mapsto v+U
$$
is a linear map because of the vector space operations that we defined for $V / U$. It's null space is the set of vectors $v \in V$ such that $v+U=0+U$ (which is the 0 element of $V / U$, by the way). That set is $U$.

