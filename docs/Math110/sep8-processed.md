---
course: CS 189
semester: Fall 2025
type: pdf
title: sep8
source_type: pdf
source_file: sep8.pdf
processed_date: '2025-10-06'
processor: mathpix
slug: sep8-processed
---

## Sums of subspaces, some linear maps. . .

## Professor K. A. Ribet

![The image appears to show a notification from a device, indicating traffic conditions for a suggeste](https://cdn.mathpix.com/cropped/2025_10_06_ea6eca805de74dc87813g-01.jpg?height=77&width=212&top_left_y=488&top_left_x=523)

**Image description:** The image appears to show a notification from a device, indicating traffic conditions for a suggested route to a specific location (2227 Piedmont Ave, Berkeley, CA 94720). There are no diagrams, equations, or images of academic content. The notification suggests that traffic is light and estimates a travel time of 10 minutes. The focus seems to be on providing real-time navigation assistance rather than conveying academic material.


September 8, 2025

## Office Hours

> 885 Evans Hall Mondays, 1:30-3 PM Thursdays, 10:30-noon.

Office full $\Longrightarrow$ possible move to a nearby classroom
No more wrinkles involving Labor Day, Res Life symposium

## Lunch schedule

I plan to come to the DCs at least once per week. There will be official Residential Life "lunches with Professor Ribet" at noon at Foothil DC on September 18, September 26, October 3 and October 9. There will also be occasional totally optional lunch gatherings at the Faculty Club. Send me email to subscribe to announcements.

Faculty Club lunch today at noon for the curious. Don't all come at once. Please read description on bCourses,
Files -> Slides -> previews.pdf.

## Sums of subspaces

On Friday, we discussed the sum of subspaces $X$ and $Y$ of $V$. More generally ( p .19 of LADR), consider subspaces $U_{1}, \ldots, U_{m}$ of $V(m \geq 1)$. The sum of these subspaces is the image of the summation map

$$
S: U_{1} \times \cdots \times U_{m} \rightarrow V, \quad\left(u_{1}, \ldots, u_{m}\right) \mapsto u_{1}+u_{2}+\cdots+u_{m} .
$$

In general, this map is neither onto (surjective) nor 1-1 (injective).
If $m=1$, it's just the inclusion of a subspace into $V$.

## Sum of subspaces

## Observation

The sum of subspaces is the smallest subspace that contains their union.

How come: A subspace of $V$ that contains all the $U_{j}$ needs to contain all sums $u_{1}+u_{2}+\cdots+u_{m}$. The set of these sums is a subspace.

## Direct sums

The sum of $U_{1}, U_{2}, \ldots, U_{m}$ is denoted $U_{1}+\cdots+U_{m}$.
There is a special word and a special notation for the case where the summation map is 1-1: we say that the sum of the $U_{j}$ is a direct sum and write $U_{1} \oplus \cdots \oplus U_{m}$ for the sum.

As a first example, let $V=\mathbf{F}^{m}$ and let $U_{j}$ be the set of $m$-tuples whose entries are 0 except for the $j$ th place. Thus $U_{j}=\mathbf{F} \cdot e_{j}$, where $e_{j}$ has a 1 in the $j$ th place and 0 s elsewhere. Each $m$-tuple ( $a_{1}, \ldots, a_{m}$ ) is uniquely a linear combination of the standard basis vectors $e_{j}:\left(a_{1}, \ldots, a_{m}\right)=a_{1} e_{1}+\cdots+a_{m} e_{m}$. Equivalently, each $m$-tuple is uniquely a sum of elements of the various subspaces $U_{j}$. Hence the summation map is $1-1$ (and onto). The sum of the subspaces is direct, and in fact

$$
V=U_{1} \oplus \cdots \oplus U_{m} .
$$

## A two-dimensional example

Let $V=\mathbf{F}^{2}$, and let $X=\mathbf{F} \cdot(a, b), Y=\mathbf{F} \cdot(c, d)$, where the two vectors ( $a, b$ ) and ( $c, d$ ) are both nonzero. Then $X$ and $Y$ are lines in the plane.

If $X=Y$, the sum $X+Y$ is $X$ (or $Y$ ), and this sum is not direct. Indeed, we can write $(0,0) \in V$ as the sum of $(0,0) \in X$ and $(0,0) \in Y$, but also as $x-x$, where $x$ is a nonzero vector in $X=Y$.

If $X \neq Y$, then there is no nonzero vector that's a multiple of both ( $a, b$ ) and ( $c, d$ ). (This requires a sentence or two of explanation.) It follows that the sum is direct, as we'll see on the next slide(s).

## Injectivity of the summation map

For this slide, check out 1.45 of page 23 of LADR.
The summation map

$$
S: U_{1} \times \cdots \times U_{m} \longrightarrow V
$$
is a linear map. This means that $S$ of a sum is the sum of the $S$ 's and that $S$ of $\lambda$ times something is $\lambda$ times the value of $S$ on that something. More generally, if $W$ is an F -vector space, a function
$$
T: W \rightarrow V
$$

is linear if $T\left(w+w^{\prime}\right)=T w+T w^{\prime}$ for all $w, w^{\prime} \in W$ and $T(\lambda w)=\lambda T w$ for all $w \in W$ and $\lambda \in \mathbf{F}$.
The null space of $T$ is the set of $w \in W$ such that $T w=0$. It's a subspace of $W$.

## Lemma

The map $T$ is 1-1 if and only if the null space of $T$ is $\{0\}$.

## Proof of lemma

## Lemma

Let $T: W \rightarrow V$ be a linear map. Then $T$ is 1-1 if and only if the null space of $T$ is $\{0\}$.

Suppose that $T$ is $1-1$. If $w$ is in the null space of $T$, then

$$
0=T 0=T w \Longrightarrow 0=w .
$$

Hence the null space consists only of 0 .
Suppose that the null space of $T$ is 0 . To check that $T$ is $1-1$, we must show

$$
T w=T w^{\prime} \Longrightarrow w=w^{\prime}
$$
for $w, w^{\prime} \in W$. Assuming that $T w=T w^{\prime}$, we use the linearity of $T$ to write
$$
0=T w-T w^{\prime}=T\left(w-w^{\prime}\right)
$$

and conclude that $0=w-w^{\prime}$ because $w-w^{\prime}$ is in the null space of $T$. This conclusion is the statement that $w=w^{\prime}$. Hence $T$ is indeed 1-1.

## Consequence for sums

Because the summation map is a linear map, the lemma implies the following statement:

## Proposition

If $U_{1}, \ldots U_{m}$ are subspaces of $V$, then the sum $U_{1}+\cdots+U_{m}$ inside $V$ is a direct sum if and only if the null space of the summation map is $\{0\}$.

Concretely, the null space statement means this:
If $0=u_{1}+\cdots+u_{m}$ with $u_{j} \in U_{j}$ for all $j$, then each summand $u_{j}$ is 0 .

## Direct sum of two subspaces

## Proposition (1.46, p. 23)

If $X$ and $Y$ are subspaces of $V$, then the sum $X+Y$ is direct if and only if $X \cap Y=\{0\}$.

We have seen that $X+Y=X \oplus Y$ if and only if $x+y=0$ (for $x \in X, y \in Y)$ implies that $x=y=0$. We need to translate that equivalence into the statement of the proposition.
If $t \in X \cap Y$, then $(t,-t) \in X \times Y$ is in the null space of the summation map. Hence if the sum is direct, so that the null space is $\{0\}$, then $t=0$ for each $t \in X \cap Y$. Thus $X \cap Y=\{0\}$.
Conversely, suppose that $X \cap Y=\{0\}$ and that $(x, y)$ is in the null space of the sum map. Then $x+y=0$. Since $y=-x$, $y \in X$. Similarly, $x \in Y$. Thus $x$ and $y$ are in $X \cap Y$, which is $\{0\}$. Hence $(x, y)=(0,0)$.

## $V$ as direct sum of two subspaces

If $X$ and $Y$ are subspaces of $V$, we might have

$$
X+Y \stackrel{?}{=} V, \quad X+Y \stackrel{?}{=} X \oplus Y .
$$

The first equality states that $V$ is the sum of $X$ and $Y$. The second says that the sum $X+Y$ inside $V$ is direct.

If both statements are true, we say that $V$ is the direct sum of its subspaces $X$ and $Y$.

We say that $X$ and $Y$ are complementary subspaces of $V$ and that $Y$ is a complement of $X$ (and vice versa).

## Example, maybe for later

Let $V=\mathbf{F}^{2}$ and let $X=\{(x, 0) \mid x \in \mathbf{F}\}$. If $y$ is an element of $V$ that is not in $X$, the line $\mathbf{F} \cdot y$ is a complement of $X$ in $V$.

## A peek into the future

Once we know about dimensions, we'll be able to have the following discussion:
If $X$ and $Y$ are subspaces of $V$, we know that their sum is direct if $X \cap Y=\{0\}$. Suppose that this is the case.
Then the dimension of $X \oplus Y$ is $\operatorname{dim} X+\operatorname{dim} Y$. Further, $V=X \oplus Y$ if and only if

$$
\operatorname{dim} V=\operatorname{dim} X+\operatorname{dim} Y .
$$

In other words, complements inside $V$ are subspaces with trivial intersection whose dimensions are complementary relative to $\operatorname{dim} V$.

Big caution: we have no idea yet what dimension is, and we haven't yet imposed the hypothesis that all vector spaces in the book are going to be finite-dimensional.

## A main theorem of linear algebra

## Theorem

Let $V$ be an F -vector space, and let $X$ be a subspace of $V$. Then $X$ has a complement in $V$. In other words, there is a subspace $Y$ of $V$ so that $V=X \oplus Y$.

We will prove this theorem very soon if $V$ is a finite-dimensional vector space.
What does that mean? We'll say (on page 30) that $V$ is finite-dimensional if there is a finite subset of $V$ whose span is all of $V$.

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

