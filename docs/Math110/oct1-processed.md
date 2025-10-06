---
course: CS 189
semester: Fall 2025
type: pdf
title: oct1
source_type: pdf
source_file: oct1.pdf
processed_date: '2025-10-06'
processor: mathpix
slug: oct1-processed
---

## Quotients

## Professor K. A. Ribet

![The image depicts the seal of the University of California, Berkeley, featuring an open book with th](https://cdn.mathpix.com/cropped/2025_10_06_dc1ed2ba9166fad21bc6g-01.jpg?height=212&width=215&top_left_y=425&top_left_x=520)

**Image description:** The image depicts the seal of the University of California, Berkeley, featuring an open book with the Latin phrase "Let there be light" inscribed below. Surrounding the book are decorative elements, including a star at the top and a circular border with a double ring of dots. The year "1868" is prominently displayed, signifying the founding of the university. The use of gold and blue colors reinforces the university's branding. This seal typically serves to symbolize the institution's commitment to scholarship and knowledge.


October 1, 2025

## Homework \#5

The problems in §3D have been moved to HW \#6. Less to do over the next three days.

## Office hour

Tomorrow at 10:30 (as usual)

## Optional lunch meetings

Today at 11:45 at Café 3
Friday at noon at Foothill Dining (official Residential Life event)

## Quotient spaces

Let $U$ be a subspace of a vector space $V$. Last Friday, we described a set $V / U$, along with an addition law on this set. We will continue by completing the description of the vector space structure associated with $V / U$.
As mentioned on Friday, $V / U$ comes equipped with a surjective linear map

$$
\pi: V \rightarrow V / U
$$
whose null space is $U$. If $V$ has finite dimension, then rank-nullity implies that
$$
\operatorname{dim} V / U=\operatorname{dim} V-\operatorname{dim} U .
$$

## Quotient spaces

If $V$ is an F -vector space and $U$ is a subspace of $V, V / U$ is the set of translates of $U$ by elements of $V$ :

$$
v+U:=\{v+u \mid u \in U\} .
$$

These are subsets of $V$ but typically not subspaces; for example, $v+U$ contains 0 if and only if $v$ is an element of $U$

## Quotient spaces

We saw the proof of the following result on Friday of last week.

## Proposition

For $v$ and $v^{\prime}$ in $V$ and $U$ a subspace of $V$,

$$
v+U=v^{\prime}+U \Longleftrightarrow v-v^{\prime} \in U .
$$

Discrete math courses like Math 55 study equivalence relations. A natural equivalence relation on the set $V$ has $v \sim v^{\prime}$ if and only if $v-v^{\prime} \in U$.

## Proposition

The set $v+U$ is the equivalence class of $v$.
A vector $v^{\prime}$ is in $v+U$ if and only if it is $v+u$ for some $u \in U$, which is true if and only if $v^{\prime}-v$ is in $U$.

## Quotient spaces

As is true for equivalence relations in general, two translates (i.e., equivalence classes) are either identical or disjoint.

The equivalence classes fill up $V$ (because $v$ is in the translate $v+U)$.

Thus every vector in $V$ belongs to exactly one equivalence class.

The map $\pi: V \rightarrow V / U$ sends $v$ to the translate of $U$ that contains $v$ :

$$
\pi(v)=v+U \in V / U
$$

It's a surjective function because $V / U$ is the set of all $V+U$.

## Addition of two translates

Far, we have defined $V / U$ as a set. We wish to turn it into a vector space.

Addition (defined last Friday):

$$
\left(v_{1}+U\right)+\left(v_{2}+U\right):=\left(v_{1}+v_{2}\right)+U .
$$

A key point is that this addition is well defined. Indeed, imagine that $v_{1}+U$ is also $v_{1}^{\prime}+U$. Is it true that $\left(v_{1}+v_{2}\right)+U$ is also $\left(v_{1}^{\prime}+v_{2}\right)+U$ ? Yes because $v_{1}^{\prime}-v_{1}$ is in $U$, and therefore so is $\left(v_{1}^{\prime}+v_{2}\right)-\left(v_{1}+v_{2}\right)$.

## Scalar multiplication

Define $\lambda \cdot(v+U):=\lambda v+U$. This is again well defined because if $v+U=v^{\prime}+U$, then $v^{\prime}-v$ is in $U$, so that $\lambda\left(v^{\prime}-v\right)=\lambda v^{\prime}-\lambda v$ is in $U$.

## Are the axioms verified?

Yes because they're verified for addition of vectors together with scalar multiplication of vectors. The operations for $V / U$ are derived from the operations for $V$ by simple non-threatening formulas.

## More about $\pi$

The function

$$
\pi: V \rightarrow V / U, \quad v \mapsto v+U
$$
is a linear map because of the vector space operations that we defined for $V / U$. Its null space is the set of vectors $v \in V$ such that $v+U=0+U$ (which is the 0 element of $V / U$, by the way). That set is $U$.

## Dimensions

If $V$ has finite dimension, then

$$
\operatorname{dim} V=\operatorname{dim} \text { null } \pi+\operatorname{dim} \text { range } \pi=\operatorname{dim} U+\operatorname{dim} V / U .
$$

Thus

$$
\operatorname{dim} V / U=\operatorname{dim} V-\operatorname{dim} U .
$$

Another perspective: if $v_{1}+U, \ldots, v_{t}+U$ is a basis of $V / U$ and if $u_{1}, \ldots, u_{d}$ is a basis of $U$, then

$$
u_{1}, \ldots, u_{d} ; v_{1}, \ldots, v_{t}
$$
is a basis of $V$. (The semicolon is my way of emphasizing the separation between vectors that came from two different bins.)

## Relation to complements

Suppose that $U \subseteq V$ is a subspace and that $X \subseteq V$ is a vector space complement to $U$ in $V$ in the sense that $V=U \oplus X$. Then the restriction of $\pi$ to $X$ is an isomorphism

$$
X \xrightarrow{\sim} V / U .
$$

It's $1-1$ because its null space is $U \cap X$, which is $\{0\}$. It's onto because each $v \in V$ is a sum $u+x$ with $u \in U, x \in X$. With $v$ written this way, $\pi V=\pi u+\pi x=\pi x$.
Thus $V / U$ behaves like a choice-free complement to $U$ that lives externally to $U$ and $V$.

## Relation to linear maps that are 0 on $U$

Let $W$ be a vector space. If $S: V / / U \rightarrow W$ is a linear map, $S \circ \pi$ is a linear map $V \rightarrow W$ whose resriction to $U$ is 0 . View $S \mapsto S \circ \pi$ as a function

$$
\mathcal{L}(V / U, W) \xrightarrow{f} \mathcal{L}(V, W) .
$$

## Proposition

The function $f$ is an injective linear map $\mathcal{L}(V / U, W) \rightarrow \mathcal{L}(V, W)$ whose image is the set of linear maps $V \rightarrow W$ whose resriction to $U$ is 0 .

A linear map $V \rightarrow W$ of the form $S \circ \pi$ is said to factor through $\pi$. The proposition states that a linear map $V \rightarrow W$ factors through $\pi$ if and only if its null space contains $U$.

## Relation to linear maps that are 0 on $U$

## Proposition

The function $f$ is an injective linear map $\mathcal{L}(V / U, W) \rightarrow \mathcal{L}(V, W)$ whose image is the set of linear maps $V \rightarrow W$ whose null spaces contain $U$.

The linearity of $f$ just results from definitions. One of the two conditions for the linearity of $f$ is this: if $S_{1}$ and $S_{2}$ are two linear maps $V / U \rightarrow W$, then $\left(S_{1}+S_{2}\right) \circ \pi=S_{1} \circ \pi+S_{2} \circ \pi$.

The map $S \mapsto S \circ \pi$ is injective: if $S \circ \pi=0$, then $S(v+U)=0$ for all $v \in V$. But this equation just means that $S$ is 0 on all elements of $V / U$, so $S$ is the 0 map $V / U \rightarrow W$.

If $S: V / U \rightarrow W$ is a linear map, then $S \circ \pi$ is 0 on the subspace $U$ of $V$ because $\pi$ is 0 on $U$. Thus the null space of $S \circ \pi$ contains $U$. Said otherwise: the image of $f$ is contained in the set of linear maps $V \rightarrow W$ whose null spaces contain $U$.

## Relation to linear maps that are 0 on $U$

## Proposition

The function $f$ is an injective linear map $\mathcal{L}(V / U, W) \rightarrow \mathcal{L}(V, W)$ whose image is the set of linear maps $V \rightarrow W$ whose resrictions to $U$ are 0 .

We have seen that $f$ is a linear map whose image is contained in the set of linear maps $V \rightarrow W$ whose null spaces contain $U$. It remains to show that if $T: V \rightarrow W$ is a linear map whose restriction $T_{\mid U}$ to $U$ is 0 , then $T$ is in the image of $f$. This means that $T=S \circ \pi$ for some linear $S: V / U \rightarrow W$.
If $T$ is given with $T_{\mid U}=0$, we define $S: V / U \rightarrow W$ by $S(v+U)=T v$. This is a well defined linear map $V / U \rightarrow W$ : if $v+U=v^{\prime}+U$, then $T v^{\prime}=T\left(v^{\prime}-v\right)+T v=T v$ (since $v^{\prime}-v \in U$ is in the null space of $T$ ).

## The range and null space of $S$

Let $T: V \rightarrow W$ be a linear map and let $U \subseteq V$ be a subspace that is contained in null $T$. Let $S$ be the unique linear map $V / U \rightarrow W$ such that $T=S \circ \pi$.

## Proposition

The range of $S$ is the range of $T$.
The range of $S$ is the set of all $S(v+U)$, but $S(v+U)=T v$. Thus the range of $S$ consists of all vectors $T v \in W$ and is therefore the range of $T$.

## Proposition

The null space of $S$ is the quotient (null $T$ ) / $U$.
The null space of $S$ is the set of all $v+U \in V / U$ such that $S(v+U)=0$. This is the set of all $v+U$ for which $T v=0$, i.e., the set of all $v+U$ with $v \in$ null $T$. This is just the quotient $($ null $T) / U$ of the proposition.

## LADR's map $\tilde{T}$

A slightly different perspective. Start with a linear map $T: V \rightarrow W$, and let $U=$ null $T$. Then $T=S \circ \pi$ for some $S: V /(\operatorname{null} T) \rightarrow W$.
In LADR, the map $S$ in this situation is called $\tilde{T}$.

## Proposition (3.107)

If $U=$ null $T$, the map $\tilde{T}: V / U \rightarrow W$ is injective. Its range is the range of $T$.

This follows from our more general discussion, since the null space of $\tilde{T}$ is $($ null $T) / U=($ null $T) /($ null $T)=0$.

