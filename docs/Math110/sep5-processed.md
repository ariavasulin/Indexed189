---
course: CS 189
semester: Fall 2025
type: pdf
title: sep5
source_type: pdf
source_file: sep5.pdf
processed_date: '2025-10-06'
processor: mathpix
slug: sep5-processed
---

# Welcome again to Math 110 

## Professor K. A. Ribet

![The image is the official seal of the University of California, Berkeley. It features a central open](https://cdn.mathpix.com/cropped/2025_10_06_f8330a5833d2efa65b16g-01.jpg?height=213&width=219&top_left_y=427&top_left_x=519)

**Image description:** The image is the official seal of the University of California, Berkeley. It features a central open book with the phrase "LET THERE BE LIGHT," flanked by two crossed instruments, possibly symbolizing knowledge and discovery. The design is circular, with a gold and blue color scheme. The year "1868" is inscribed at the bottom, indicating the university's founding year. The seal serves to represent the institution's commitment to education and enlightenment. No diagrams or equations are present.


September 5, 2025

## Special office hour today

10:30 AM-noon today in 885 Evans

## Lunch schedule

I plan to come to the DCs at least once per week. There will be official Residential Life "Iunches with Professor Ribet" at noon at Foothil DC on September 18, September 26, October 3 and October 9. There will also be occasional lunch gatherings at the Faculty Club (none this week, though).
I will announce lunch meetings on class slides and also by email. (Send me email to get on the email list.)

The next lunch gathering will be at the Faculty Club at noon on Monday.

## Subspaces

The book introduces the notion of a subspace of a vector space on page 18. We mentioned subspaces briefly at the end of Wednesday's class.
A subspace of a vector space $V$ over $\mathbf{F}$ is a nonempty subset of $V$ that is stable under both addition and scalar multiplication.
If the subset is $U$, then the requirements are:

$$
u+u^{\prime} \in U \text { for all } u, u^{\prime} \in U, \quad \lambda u \in U \text { for all } u \in U, \lambda \in \mathbf{F} .
$$

Each subspace of $V$ is an F -vector space: we can use the addition inside $V$ to define an addition on the subspace and similarly use the scalar multiplication on $V$ to define a scalar multiplication on the subspace. The axioms were built exactly for that purpose.

## Examples of subspaces

For $m \geq 0, \mathcal{P}_{m}(\mathbf{F})$ is a subspace of $\mathcal{P}(\mathbf{F})$, and $\mathcal{P}(\mathbf{F})$ is a subspace of $\mathbf{F}^{\infty}$.

The space of continuous functions $\mathbf{R} \rightarrow \mathbf{R}$ is a subspace of the space of all functions $\mathbf{R} \rightarrow \mathbf{R}$. The space of differentiable functions $\mathbf{R} \rightarrow \mathbf{R}$ is a subspace of the space of continuous functions $\mathbf{R} \rightarrow \mathbf{R}$.

The null space of an $m \times n$ matrix is a subspace of $\mathbf{F}^{n}$.

## Subspaces of $\mathbf{F}^{2}$

The full space $V$ is a subspace of $V$. So is the singleton set $\{0\}$. Thus $\{0\}$ and $\mathbf{F}^{2}$ are subspaces of $\mathbf{F}^{2}$. Are there others?
Sure: take a vector $v \in \mathbf{F}^{2}$ and consider the set of its multiples; this set could be denoted $\mathbf{F} \cdot v$. If $v$ is nonzero, it's a line.

Is that it? Yup, but maybe it's better to see this after proving some theorems.

Informally at least: if $U \subseteq \mathbf{F}^{2}$ is a subspace, it could be $\{0\}$. If it isn't, it contains a nonzero vector $v$ and thus the line $\mathbf{F} \cdot v$. Is it $\mathbf{F} \cdot v$ ? Maybe, but if not, it also contains a vector $w \notin \mathbf{F} \cdot v$. It then contains all expressions $\lambda v+\mu w$ with $\lambda, \mu \in \mathbf{F}$. We can convince ourselves using Math 54 or whatever that all elements of $\mathbf{F}^{2}$ may be written as sums $\lambda, \mu \in \mathbf{F}$. Thus the subspace is all of $\mathbf{F}^{2}$.

## Smallest subspace containing a subset

If $S$ is a subset of $V$, then there is a subspace $U$ of $V$ such that $U$ contains $S$ and $U$ is contained in all subspaces of $V$ that contain $S$. Said otherwise, $U$ is the smallest subspace of $V$ containing $S$.

## Lemma

The subspace U consists of all "linear combinations"

$$
\lambda_{1} s_{1}+\cdots+\lambda_{m} s_{m}
$$
with $m \geq 0, s_{1}, \ldots, s_{m} \in S$ and $\lambda_{1}, \ldots, \lambda_{m} \in \mathbf{F}$. (The case $m=0$ corresponds to the empty sum, which is 0 .)

Proof: If $X$ be the set of linear combinations as in the lemma, then the desired equality $U=X$ is a consequence of:

- The set $X$ is a subspace of $V$.
- The set $X$ contains $S$.
- If $Y$ is a subspace of $V$ that contains $S$, then $Y$ contains $X$.


## The set $X$ is indeed a subspace of $V$.

For reference:
$X=\left\{\lambda_{1} s_{1}+\cdots+\lambda_{m} s_{m} \mid m \geq 0, s_{1}, \ldots, s_{m} \in S, \lambda_{1}, \ldots, \lambda_{m} \in \mathbf{F}\right\}$.
Multiply a sum $\lambda_{1} s_{1}+\cdots+\lambda_{m} s_{m}$ by a scalar $\mu$, and you get the analogous sum with each scalar $\lambda_{j}$ replaced by the product $\mu \lambda_{j}$.

Adding two linear combinations together gives a linear combination. (If the summands have $m$ and $n$ terms, respectively, addition yields a combination with $m+n$ terms.)
The set $X$ is nonempty because it contains the empty sum, which is 0 .

## The set $X$ contains $S$.

If $s$ is an element of $S$, the one-term sum $s$ is in $X$.

## If $Y$ is a subspace of $V$ that contains $S$, then $Y$ contains $X$.

Each element of $X$ is a sum $\lambda_{1} s_{1}+\cdots+\lambda_{m} s_{m}$. For each $j=1, \ldots, m, Y$ contains $s_{j}$ because $Y$ contains every element of $S$. Then $Y$ contains $\lambda_{j} s_{j}$ because $Y$, being a subspace, is closed under scalar multiplication. Finally, $Y$ contains the sum $\lambda_{1} s_{1}+\cdots+\lambda_{m} s_{m}$ because $Y$, being a subspace, is closed under addition.

## Connection with homework

If $S$ is a subset of $V$, consider the family of all subspaces $W$ of $V$ that contain $S$. This family includes $V$, for example. The intersection of all of these subspaces is again a subspace of $V$ : that's a homework problem.
The intersection is thus a subspace that contains $S$. It is contained in every $W$ as in the paragraph above because it's the intersection of all of the $W$. Hence it's the minimal, or smallest, subspace of $V$ that contains $S$. Namely, it's the $U$ of the previous slides.
I'd say that $U$ is the subspace of $V$ generated by $S$. Most people would say that $U$ is - wait for it - the span of $S$.

## Lists and sets

On page 5 of LADR, Axler defines' lists of vectors in $V$ : a list is a sequence

$$
v_{1}, v_{2}, \ldots, v_{\ell}, \quad v_{j} \in V \text { for all } j .
$$

The length of the list is $\ell$. The empty sequence

## (nothing here)

is the list of length 0 .
Lists are ordered, and repetition is allowed. For example,

$$
0,0,0,0, \ldots, 0
$$
is a list of length 155 if there are 1550 's in a row.
Every list $v_{1}, v_{2}, \ldots, v_{\ell}$ gives rise to a set $\left\{v_{1}, v_{2}, \ldots, v_{\ell}\right\}$, which will have fewer than $\ell$ elements if there is repetition. For example, $\{0,0,0,0, \ldots, 0\}=\{0\}$.

## Smallest subspace containing a list

If $v_{1}, v_{2}, \ldots, v_{\ell}$ is a list, then the set of linear combinations

$$
\lambda_{1} v_{1}+\cdots+\lambda_{\ell} v_{\ell}
$$
is a subspace that contains each vector $v_{j}$ that appears in the list. This subspace is the smallest subspace of $V$ containing each of the vectors. Indeed, if $U \subseteq V$ is a subspace that contains each $v_{j}$, then it contains each product $\lambda_{j} v_{j}$ and thus each linear combination $\lambda_{1} v_{1}+\cdots+\lambda_{\ell} v_{\ell}$.
A structural way to think about this is that the list defines a map
$$
T: \mathbf{F}^{\ell} \longrightarrow V, \quad\left(\lambda_{1}, \ldots, \lambda_{\ell}\right) \mapsto \lambda_{1} v_{1}+\cdots+\lambda_{\ell} v_{\ell} .
$$

The span of the list (i.e., the span of the set defined by the list) is the image of $T$.

## A linear map

Scrolling all the way down to page 52, you'll see that $T$ is a linear map from $\mathbf{F}^{\ell}$ to $V$. This means that the value of $T$ on the sum of two elements of $\mathbf{F}^{\ell}$ is the sum of the values of $T$ on the two elements and that the value of $T$ on a scalar multiple of a tuple ( $\lambda_{1}, \ldots, \lambda_{\ell}$ ) is the same scalar multiple of the value of $T$ on ( $\lambda_{1}, \ldots, \lambda_{\ell}$ ).
Mantra: lists define linear maps.
I wanted to write more about linear maps, and I did. But most of what I wrote is at the end of this slide deck and is not intended to be discussed this week.

## A linear map (said another way)

Let $v_{1}, v_{2}, \ldots, v_{\ell}$ be a list of vectors in $V$. Then

$$
T: \mathbf{F}^{\ell} \longrightarrow V, \quad\left(\lambda_{1}, \ldots, \lambda_{\ell}\right) \mapsto \lambda_{1} v_{1}+\cdots+\lambda_{\ell} v_{\ell}
$$
is a linear map that takes the standard basis vectors $e_{1}, e_{2}, \ldots, e_{\ell}$ of $\mathbf{F}^{\ell}$ to the list vectors $v_{1}, v_{2}, \ldots, v_{\ell}$ in $V$.
The function $T$ is the unique linear map taking $e_{j}$ to $v_{j}$ for each $j=1, \ldots, \ell$.
For each $j$,
$$
e_{j}=(0, \ldots, 0,1,0 \ldots, 0), \quad 1 \text { in the } j \text { th place. }
$$

Even though we say that the $e_{j}$ are standard basis vectors, we haven't yet said what a basis is.

## Coming attractions

If $v_{1}, \ldots, v_{\ell}$ is a list of vectors in $V$, the span of the list is the image of the map $T$ on the previous slide or two.
The list spans $V$ if the span of the list is all of $V$. This means that $T$ is onto (i.e., surjective).
A vector space is finite-dimensional if there is a (finite) list that spans the space. We will see later on that finite-dimensional vector spaces have a well-defined dimension.

Is there interest in lists for which $T$ is injective (1-1). You bet! We say that a list is linearly independent if $T$ is $1-1$.
We say that a list of vectors of $V$ is a basis of $V$ if $T$ is both 1-1 and onto.

The slides for each lecture are gotten by combining undiscussed slides from the previous two lectures. This is the Fibonacci method of class preparation.

## Going rogue with linear maps

I can't refrain from telling you about the functions that we study in linear algebra. The reference for this is the beginning of Chapter 3 of LADR (pp. 51-).
If $V$ and $W$ are $\mathbf{F}$-vector spaces, a function $T: V \rightarrow W$ is a linear map if it respects addition and scalar multiplication:

$$
\begin{aligned}
& T\left(v_{1}+v_{2}\right)=T v_{1}+T v_{2} \text { for all } v_{1}, v_{2} \in V \\
& T(\lambda v)=\lambda T v \text { for all } v \in V, \lambda \in \mathbf{F}
\end{aligned}
$$

## Familiar examples

Let $V$ be the space of differentiable functions on $\mathbf{R}$, and let $W$ be the space of all functions $\mathbf{R} \rightarrow \mathbf{R}$. The differentiation map $f \mapsto f^{\prime}$ is a linear map $V \rightarrow W$. (On this slide, $\mathbf{F}=\mathbf{R}$.)

Let $V$ be the space of integrable functions on $[0,1]$, and let $W=\mathbf{R}$. The association

$$
f \mapsto \int_{0}^{1} f(x) d x
$$
is a linear map $V \rightarrow W$.

## Matrix multiplication

Let $V=\mathbf{F}^{n}$ and $W=\mathbf{F}^{m}$, with the vectors in both spaces thought of as vertical tuples. Let $A$ be an $m \times n$ matrix of elements of $\mathbf{F}$. The map

$$
x \in \mathbf{F}^{n} \longmapsto A x \in \mathbf{F}^{m}
$$
is linear.

## More about the case $V=\mathbf{F}^{n}$

Let $T: \mathbf{F}^{n} \rightarrow W$ be a linear map. Recall the familiar "standard basis vectors" $e_{1}, e_{2}, \ldots, e_{n}$ from Math 54 :

$$
e_{j}=(0,0, \ldots, 0,1,0, \ldots 0), \text { the } 1 \text { in the } j \text { th place. }
$$

Then the vectors

$$
T e_{1}, T e_{2}, \ldots, T e_{n} \in W
$$
form a list of vectors of length $n$ in $W$.
Lists are introduced on page 5 of LADR:
A list of length $n$ is an ordered collection of $n$ elements. . . .
list of vectors in $W$ of length $n$ may be viewed as an element of $W^{n}=W \times \cdots \times W$ ( $n$ copies).

## The case $V=\mathbf{F}^{n}$

Thus we have an association

$$
\left\{\text { linear maps } \mathbf{F}^{n} \rightarrow W\right\} \longrightarrow W^{n}, \quad T \longmapsto\left(T e_{1}, \ldots, T e_{n}\right) .
$$

## Theorem (3.4 on page 54)

This association is a 1-1 correspondence between the set of linear maps $\mathbf{F}^{n} \rightarrow W$ and the set $W^{n}$.

The "association" is a function

$$
F:\left\{\text { linear maps } \mathbf{F}^{n} \rightarrow W\right\} \rightarrow W^{n} .
$$

One way to show that $F$ is a bijection is to exhibit a function

$$
G: W^{n} \rightarrow\left\{\text { linear maps } \mathbf{F}^{n} \rightarrow W\right\}
$$
such that $G \circ F$ is the identity map on the set of linear maps $F^{n} \rightarrow W$ and $F \circ G$ is the identity map on $W^{n}$.

## The case $V=\mathbf{F}^{n}$

The function $G$ from $W^{n}$ to the space of linear maps $\mathbf{F}^{n} \rightarrow W$ is defined by

$$
\begin{gathered}
\left(w_{1}, \ldots, w_{n}\right) \mapsto T: \mathbf{F}^{n} \rightarrow W, \\
T\left(a_{1}, \ldots, a_{n}\right):=a_{1} w_{1}+a_{2} w_{2}+\cdots+a_{n} w_{n} .
\end{gathered}
$$

The linearity of $T$ amounts to two compatibilities:

$$
\begin{aligned}
T\left(\left(a_{1}, \ldots, a_{n}\right)+\left(b_{1}, \ldots, b_{n}\right)\right) & \stackrel{?}{=} T\left(a_{1}, \ldots, a_{n}\right)+T\left(b_{1}, \ldots, b_{n}\right) \\
T\left(\lambda\left(a_{1}, \ldots, a_{n}\right)\right) & =\lambda T\left(a_{1}, \ldots, a_{n}\right)
\end{aligned}
$$

We'll check them live and in person in 155 Dwinelle.

