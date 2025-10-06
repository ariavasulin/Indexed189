---
course: CS 189
semester: Fall 2025
type: pdf
title: sep10
source_type: pdf
source_file: sep10.pdf
processed_date: '2025-10-06'
processor: mathpix
slug: sep10-processed
---

# We need to talk about lists, linear independence. . . 

## Professor K. A. Ribet

![The image depicts the seal of the University of California, Berkeley. It features an open book, symb](https://cdn.mathpix.com/cropped/2025_10_06_90fc71c525c6a8d6262fg-01.jpg?height=218&width=218&top_left_y=459&top_left_x=520)

**Image description:** The image depicts the seal of the University of California, Berkeley. It features an open book, symbolizing knowledge, with the phrase "LET THERE BE LIGHT" inscribed below. A star above the book signifies enlightenment. The seal is encircled by a blue ring with yellow dots, presenting the university's name and establishment year "1868" to highlight its heritage and academic tradition. There are no diagrams, graphs, or equations present; the image primarily serves as a formal representation of the university's identity.


September 10, 2025

## Office Hours

> 885 Evans Hall Mondays, 1:30-3 PM Thursdays, 10:30-noon.

Office full $\Longrightarrow$ possible move to a nearby classroom
Lots of people at the last two office hours. See you tomorrow at 10:30 AM

I plan to come to the DCs at least once per week. There will be official Residential Life "lunches with Professor Ribet" at noon at Foothill DC on September 18, September 26, October 3 and October 9. There will also be additional lunch gatherings at DCs and the Faculty Club.

Gatherings are optional and not part of Math 110, but I'll continue to list them on slides for those who are interested. Also, you can send me email to subscribe to email announcements.

- Lunch today at Crossroads at 11:45 AM
- Lunch after office hour on Thursday at Foothill DC (starting around 12:15
- Faculty Club lunch Monday, September 15 at noon PM)


## Monday

We discussed sums of subspaces, direct sums, criterion for the sum of two subspaces to be a direct sum.

Two subspaces $X$ and $Y$ of $V$ are complements of each other if $V=X \oplus Y$. I stated but did not prove that every subspace has at least one complement.
The subspace $\{0\}$ has $V$ as its complement, while $V$ has $\{0\}$ as its complement. If $\{0\} \subset X \subset V$, with $X \neq\{0\}, V$, then $X$ likely has a whole bunch of complements.

## Monday's last slide

Theorem (only implicit in LADR)
Let $V$ be an F -vector space, and let $X$ be a subspace of $V$.
Then $X$ has a complement in $V$. In other words, there is a
subspace $Y$ of $V$ so that $V=X \oplus Y$.
We will prove this theorem pretty soon.

## Wednesday

Yes, that is today. I hope to discuss:

- Lists
- Span of a List
- Linear maps (reminder)
- Linear map defined by a list
- Linear independence
- Lists that span
- Bases (certain lists)
- Finite-dimensional spaces


## Lists and sets

On page 5 of LADR, Axler defines lists of vectors in $V$ : a list is a sequence

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

## Lists

A length- $\ell$ list may be viewed as an element of

$$
\left.V^{\ell}=V \times V \times \cdots \times V \text { ( } \ell \text { factors }\right) .
$$

A list of length $\ell$ is also the same thing as a linear map $\mathbf{F}^{\ell} \rightarrow V$ : A list $v_{1}, v_{2}, \ldots, v_{\ell}$ defines the linear map

$$
\mathbf{F}^{\ell} \longrightarrow V, \quad\left(\lambda_{1}, \ldots, \lambda_{\ell}\right) \mapsto \lambda_{1} v_{1}+\cdots+\lambda_{\ell} v_{\ell} .
$$

In the other direction, a linear map $T: \mathbf{F}^{\ell} \rightarrow V$ yields the list

$$
T e_{1}, T e_{2}, \ldots, T e_{\ell} ; \quad e_{j}=(0,0, \ldots, 0,1,0, \ldots 0) .
$$
$\uparrow$
1 in jth place
The dictionary between lists and linear maps appears as 3.4 on p. 54 of LADR.

## Smallest subspace containing a list

If $v_{1}, v_{2}, \ldots, v_{\ell}$ is a list, then the set of linear combinations

$$
\lambda_{1} v_{1}+\cdots+\lambda_{\ell} v_{\ell}
$$
is a subspace that contains each vector $v_{j}$ that appears in the list. This subspace is the smallest subspace of $V$ containing each of the vectors. Indeed, if $U \subseteq V$ is a subspace that contains each $v_{j}$, then it contains each product $\lambda_{j} v_{j}$ and thus each linear combination $\lambda_{1} v_{1}+\cdots+\lambda_{\ell} v_{\ell}$.
As we've seen, the list defines a linear map
$$
\mathbf{F}^{\ell} \longrightarrow V, \quad\left(\lambda_{1}, \ldots, \lambda_{\ell}\right) \mapsto \lambda_{1} v_{1}+\cdots+\lambda_{\ell} v_{\ell}
$$

The span of the list (i.e., the span of the set defined by the list) is the image of this map.

## Linear maps

We saw the linear map

$$
S: U_{1} \times \cdots \times U_{m} \rightarrow V
$$
on Monday and a just now linear map
$$
\mathbf{F}^{\ell} \longrightarrow V .
$$

It might be helpful to have the definition of "linear map" on the screen: if $W$ and $V$ are $\mathbf{F}$-vector spaces, a function $T: W \rightarrow V$ is a linear map if it respects addition and scalar multiplication:

$$
\begin{aligned}
& T\left(w_{1}+w_{2}\right)=T w_{1}+T w_{2} \text { for all } w_{1}, w_{2} \in W ; \\
& T(\lambda W)=\lambda T W \text { for all } W \in W, \lambda \in \mathbf{F} .
\end{aligned}
$$

## Linear independence

A list $v_{1}, v_{2}, \ldots, v_{\ell}$ is linearly independent (page 31 ) if the map

$$
T: \mathbf{F}^{\ell} \longrightarrow V, \quad\left(\lambda_{1}, \ldots, \lambda_{\ell}\right) \mapsto \lambda_{1} v_{1}+\cdots+\lambda_{\ell} v_{\ell}
$$
is 1-1. In words: if a vector in $v$ can be expressed as a linear combination $\lambda_{1} v_{1}+\cdots+\lambda_{\ell} v_{\ell}$, then the scalars $\lambda_{1}, \ldots, \lambda_{\ell}$ are unique.

Note that $T$ is $1-1$ if and only if its null space is $\{0\}$. This means

$$
\lambda_{1} v_{1}+\cdots+\lambda_{\ell} v_{\ell}=0 \Longrightarrow \lambda_{1}=\lambda_{2}=\cdots=\lambda_{\ell}=0 .
$$

That's the usual definition of linear independence.

## A slide for nerds about the empty list

If $\ell=0$ (empty list), $T$ is a map $\mathbf{F}^{0} \rightarrow V$. Now $\mathbf{F}^{0}$ is the zero vector space (and has exactly one element). Hence the null space of $T$ is $\{0\}$ and the list is linearly independent.

## Lists of length 1

A list $v$ is linearly independent if and only if $v$ is nonzero.

## Lists of length 2

A list of length 2 is linearly independent if and only if neither vector is a scalar multiple of the other.

## Spanning lists

A list $v_{1}, v_{2}, \ldots, v_{\ell}$ spans $V$ if its span is all of $V$ (and not some smaller subspace).

The list spans if

$$
T: \mathbf{F}^{\ell} \longrightarrow V, \quad\left(\lambda_{1}, \ldots, \lambda_{\ell}\right) \mapsto \lambda_{1} v_{1}+\cdots+\lambda_{\ell} v_{\ell}
$$
is surjective (onto). This means that every vector in $V$ is some linear combination $\lambda_{1} v_{1}+\cdots+\lambda_{\ell} v_{\ell}$ of the vectors in the list.

## Basis

A list $v_{1}, v_{2}, \ldots, v_{\ell}$ is a basis of $V$ if it is both linear independent and a spanning list (page 39). The two conditions together mean that every vector in $V$ may be written uniquely as a linear combination $\lambda_{1} v_{1}+\cdots+\lambda_{\ell} v_{\ell}$ or (more formally) that $T$ is both onto and 1-1.

## Standard basis

If $V=\mathbf{F}^{n}$, the list $e_{1}, \ldots, e_{n}$ (defined before) is a basis of $\mathbf{F}^{n}$. People call it the standard basis.
Most vector spaces don't have specially defined bases. Nothing is standard in the abstract setting! Typical random vector spaces on the ground have infinitely many bases, all with their own special claim to fame.

## Finite-dimensionality

A vector space $V$ is finite-dimensional if there a list $v_{1}, v_{2}, \ldots, v_{\ell}$ (of finite length) that spans it.
An example of an $\mathbf{F}$-vector space that is not finite-dimensional is $\mathcal{P}(\mathbf{F})$, the space of polynomials with coefficients in $\mathbf{F}$.

## How come $\mathcal{P}(\mathbf{F})$ isn't finite-dimensional?

Let $p_{1}, \ldots, p_{\ell}$ be a list of polynomials. If $m$ is the largest degree of gthe polynomials $p_{j}$, then all $p_{j}$ are contained in $\mathcal{P}_{m}(\mathbf{F})$, so that the span of the list is contained in $\mathcal{P}_{m}(\mathbf{F})$. That's a proper subspace of $\mathcal{P}(\mathbf{F})$.

## An amazing theorem

## Theorem

All subspaces of a finite-dimensional F-vector space are finite-dimensional.

This is for the future, but in fact for the near future.

The slides for each lecture are gotten by combining undiscussed slides from the previous two lectures. This is the Fibonacci method of class preparation.

