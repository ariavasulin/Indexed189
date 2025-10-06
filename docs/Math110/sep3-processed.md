---
course: CS 189
semester: Fall 2025
type: pdf
title: sep3
source_type: pdf
source_file: sep3.pdf
processed_date: '2025-10-06'
processor: mathpix
slug: sep3-processed
---

# Welcome again to Math 110 

## Professor K. A. Ribet

![The image is the official seal of the University of California, Berkeley. It features an open book a](https://cdn.mathpix.com/cropped/2025_10_06_260078f9b700ff39b24ag-01.jpg?height=213&width=219&top_left_y=427&top_left_x=519)

**Image description:** The image is the official seal of the University of California, Berkeley. It features an open book at the center with the inscription "LET THERE BE LIGHT" above it. Surrounding the book are various emblematic elements, including a star and a laurel wreath. The year "1868" is displayed prominently at the bottom, indicating the establishment of the university. The outer circle is adorned with a decorative border typical of academic seals, reinforcing its official and institutional significance. The seal symbolizes knowledge and enlightenment, fundamental themes in academic discourse.


September 3, 2025

## This week

- Monday was Labor Day, but we have class today and on Friday
- Lunch today at noon, at Clark Kerr dining
- Faculty Research Discussion Panel 11 AM-1 PM at Unit 3 APR
- Special Friday office hour 10:30 AM-noon in 885 Evans

Lunches are optional no-host events. I plan to come to the DCs at least once per week. On many Fridays, there will be a lunch gathering at the Faculty Club (not this week, though).
I will announce lunch meetings on class slides and also by email. (Send me email to get on the email list.)

## Online discussion(s)

## There's a Math 110 Ed Discussion group and also a Discord for this class

## What's next in the course?

On bCourses, navigate to
Files -> Slides -> Previews

I posted a preview for this class way before posting the slides for today's class.

## Last Friday

We defined vector spaces over F and gave some examples, starting with

$$
\mathbf{F}^{n}, \quad \mathbf{F}^{\infty}, \quad \mathbf{F}^{S} .
$$

Recall that $\mathbf{F}^{\infty}$ is the space of sequences ( $a_{0}, a_{1}, \ldots$ ) or $\left(a_{1}, a_{2}, \ldots\right)$ with values in $\mathbf{F}$ and that $\mathbf{F}^{S}$ is the set of all functions $S \rightarrow \mathbf{F}$.
Note also that $\mathbf{F}^{n}$ may be viewed as the space of sequences ( $a_{1}, a_{2}, \ldots$ ) such that $a_{j}=0$ for all $j>n$.
Also, sequences with values in $\mathbf{F}$ are functions

$$
\mathbf{N} \rightarrow \mathbf{F}, \quad \mathbf{N}=\{0,1,2, \ldots\} .
$$

Inside $\mathbf{F}^{\infty}$, we see

$$
\mathbf{F}^{0} \subset \mathbf{F}^{1} \subset \mathbf{F}^{2} \subset \cdots .
$$

The union of the $\mathbf{F}^{n}$ inside $\mathbf{F}^{\infty}$ is the set of sequences that are 0 beyond some point.

## Polynomials

For $m \geq 0$, a polynomial of degree $m$ is an expression

$$
a_{0}+a_{1} z+\cdots+a_{m} z^{m}, \quad a_{m} \neq 0
$$
with the $a_{j}$ in $\mathbf{F}$. A polynomial of degree $\leq m$ is thus an expression $a_{0}+a_{1} z+\cdots+a_{m} z^{m}$ with the $a_{j}$ in $\mathbf{F}$ (and $a_{m}$ possibly 0 ).

## What's an expression?

Each polynomial defines a function $\mathbf{F} \rightarrow \mathbf{F}$; two polynomials give rise to the same function if and only if they're equal. In symbols, if

$$
a_{0}+a_{1} z+\cdots+a_{m} z^{m}=b_{0}+b_{1} z+\cdots+b_{n} z^{n}
$$
for all $z \in \mathbf{F}$, then $n=m$ and the $b_{j}$ are the $a_{j}$ for all $j=0, \ldots, m$.

## I don't care, do you?

Since two polynomial "expressions" are equal if and only if they are equal as functions, it's not necessary to distinguish between expressions and the functions they define.

There's an art in knowing when to distinguish two things.

![The image depicts a figure leaning against a curved concrete railing within a modern architectural s](https://cdn.mathpix.com/cropped/2025_10_06_260078f9b700ff39b24ag-13.jpg?height=525&width=700&top_left_y=321&top_left_x=279)

**Image description:** The image depicts a figure leaning against a curved concrete railing within a modern architectural setting. The background features vertical lines from a textured wall, which contrasts with the smooth curve of the staircase. The person is dressed in a coral-colored top and wears glasses, offering a contemplative expression. This visual serves to illustrate themes of modern design and human interaction within architectural spaces, emphasizing the interplay between structural elements and personal presence.

Eugenia Cheng, author of Unequal

## Down the rabbit hole

Mathematicians actually care about this stuff. If $\mathbf{F}=\{0,1\}$ were the field of numbers mod 2 , then the two polynomials $z$ and $z^{2}$ yield the same function $\mathbf{F} \rightarrow \mathbf{F}$ (namely, the identity function), but they're different formal expressions (one having degree 1 and the other of degree 2).

If $\mathbf{F}=\mathbf{R}$ or $\mathbf{F}=\mathbf{C}$, this distinction does not arise. Less for us to worry about.

## The same thing, from Friday's slides

Polynomials are defined relatively late in the book: you have to scroll down all the way to page 30.
There is some nuance about polynomials as "formal expressions" and polynomials as functions $\mathbf{F} \rightarrow \mathbf{F}$. Because $\mathbf{F}$ has infinitely many elements, the two points of view are the same. Namely, you may remember from high school or math 55 that a nonzero polynomial has no more roots than its degree allows. A quintic polynomial can have no more than five roots, for example.

If $p(z)$ and $q(z)$ are expressions as above that yield the same function, then the polynomial $p(z)-q(z)$ is identically 0 and thus has infinitely many roots. As a result, it can't be a nonzero polynomial.

## More words about degrees

A polynomial $a_{0}+a_{1} z+\cdots+a_{m} z^{m}$ has degree $m$ if $a_{m}$ is nonzero. A polynomial of degree 0 is a nonzero constant. A polynomial of degree 1 is an expression $a z+b$ with a nonzero. A polynomial of degree 2 is a quadratic $a z^{2}+b z+c$ with $a$ again nonzero.

The degree of the polynomial 0 is usually deemed to be $-\infty$ (whatever that means). If you don't like that, just say it's undefined.

The set of polynomials of degree $\leq m$ is denoted $\mathcal{P}_{m}(\mathbf{F})$. This set is a vector space under the natural addition and scalar multiplication of functions.
Note that
$\mathcal{P}_{m}(\mathbf{F}) \longleftrightarrow \mathbf{F}^{m+1}, \quad a_{0}+a_{1} z+\cdots+a_{m} z^{m} \longleftrightarrow\left(a_{0}, a_{1}, \ldots, a_{m}\right)$.
The union $\mathcal{P}(\mathbf{F}):=\bigcup_{m \geq 0} P_{m}(\mathbf{F})$ is the set of polynomials over $\mathbf{F}$ of all degrees. It's again a vector space over $\mathbf{F}$.

We can view $\mathcal{P}(\mathbf{F})$ as the set of sequences

$$
\left(a_{0}, a_{1}, a_{2}, \ldots,\right)
$$
of elements of $\mathbf{F}$ with the property that there's an $m \geq 0$ such that $a_{j}=0$ for $j>m$. These are the sequences with only a finite number of nonzero entries. These are the sequences that are "eventually 0 ."

## Comparing $\mathcal{P}(\mathbf{F})$ with $\mathbf{F}^{\infty}$

The space $\mathcal{P}(\mathbf{F})$ is the set of sequences that are eventually 0 . It doesn't matter whether we call the first entry $a_{0}$ or $a_{1}$. Every element of $\mathcal{P}(\mathbf{F})$ is also an element of $\mathbf{F}^{\infty}$ :

$$
\mathcal{P}(\mathbf{F}) \Longleftrightarrow \mathbf{F}^{\infty} .
$$

For example, the polynomial $1-x+x^{3}$ can be regarded as the sequence $(1,-1,0,1,0,0, \ldots, 0, \ldots)$, which is an element of $\mathbf{F}^{\infty}$.

After we define the notion of a subspace, you will agree that $\mathcal{P}(\mathbf{F})$ is a subspace of $\mathbf{F}^{\infty}$.

## Null spaces of matrices

Suppose that $A$ is an $m \times n$ matrix of elements of $\mathbf{F}$. Let

$$
V=\left\{x \in \mathbf{F}^{n} \mid A x=0\right\} .
$$

Thus $V$ is the set of solutions of $m$ homogeneous equations in $n$ unknowns, and we have

$$
V \hookrightarrow \mathbf{F}^{n} .
$$

After we define the notion of a subspace, you will agree enthusiastically that $V$ is a subspace of $\mathbf{F}^{n}$.

It's fine to think about $V$ as a set with an addition and scalar multiplication; it's a vector space over $\mathbf{F}$ in its own right (i.e., without thinking that it's living inside $\mathbf{F}^{\eta}$ ).

## Consequences of the axioms

The text (including the exercises) describes manifold consequences of the axioms. A sample:

- For $\lambda \in F$ and $v \in V$ : if $\lambda v=0$, then either $v=0$ or $\lambda=0$ (or both).
- For each $v \in V,-(-v)=v$.
- For $v \in V,(-1) \cdot v=-v$.

The first statement amounts to the implication
If $\lambda v=0$ and $\lambda$ is nonzero, then $v=0$.
To prove it, assume the hypothesis of the implication, namely that $\lambda v=0$ and $\lambda$ is nonzero. Then $\lambda$ is invertible in $\mathbf{F}$. Since $\lambda v=0,0=\frac{1}{\lambda}(\lambda v)$. By the associativity of multiplication,

$$
0=\left(\frac{1}{\lambda} \cdot \lambda\right) v=1 \cdot v=v
$$

## Subspaces

The book introduces the notion of a subspace of a vector space on page 18.
A subspace of a vector space $V$ over $\mathbf{F}$ is a nonempty subset of $V$ that is stable under both addition and scalar multiplication.
If the subset is called $U$, then the requirements are

$$
u+u^{\prime} \in U \text { for all } u, u^{\prime} \in U
$$
and
$$
\lambda u \in U \text { for all } u \in U, \lambda \in \mathbf{F} .
$$

If $U$ is a subspace of $V$, then $U$ is an F -vector space: we can use the addition inside $V$ to define an addition on $U$ and similarly use the scalar multiplication on $V$ to define a scalar multiplication on $U$. The axioms are built exactly for that purpose.

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

Suppose that $S$ is a subset of $V$. Then there is a subspace $U$ of $V$ that contains $S$ with the property that $U$ is contained in all subspaces of $V$ that contain $S$. Thus $U$ is the smallest subspace of $V$ containing $S$.
The description of $U$ is as follows: it consists of all sums

$$
\lambda_{1} s_{1}+\cdots+\lambda_{m} s_{m}
$$
with $m \geq 0, s_{1}, \ldots, s_{m} \in S$ and $\lambda_{1}, \ldots, \lambda_{m} \in \mathbf{F}$. The particular case $m=0$ corresponds to the empty sum, which is 0 .
Mathematicians might say that it is "clear" that $U$ is a subspace of $V$ that contains each element of $S$ and that all subspaces of $V$ that contain $S$ also contain $U$. It'll be on me to explain this to you.
Sums like $\lambda_{1} s_{1}+\cdots+\lambda_{m} s_{m}$ are referred to as linear combinations of the vectors $s_{j}$.

## Connection with homework

If $S$ is a subset of $V$, consider the family of all subspaces $W$ of $V$ that contain $S$. This family includes $V$, for example. The intersection of all of these subspaces is again a subspace of $V$ : that's a homework problem.
The intersection is thus a subspace that contains $S$. It is contained in every $W$ as in the paragraph above because it's the intersection of all of the $W$. Hence it's the minimal, or smallest, subspace of $V$ that contains $S$. Namely, it's the $U$ of the previous slide.
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

## A linear map (said another way)

Let $v_{1}, v_{2}, \ldots, v_{\ell}$ be a list of vectors in $V$. Then

$$
T: \mathbf{F}^{\ell} \longrightarrow V, \quad\left(\lambda_{1}, \ldots, \lambda_{\ell}\right) \mapsto \lambda_{1} v_{1}+\cdots+\lambda_{\ell} v_{\ell}
$$
is a linear map that takes the standard basis vectors $e_{1}, e_{2}, \ldots, e_{\ell}$ of $\mathbf{F}^{\ell}$ to the list vectors $v_{1}, v_{2}, \ldots, v_{\ell}$ in $V$.
The function $T$ is the unique linear map taking $e_{j}$ to $v_{j}$ for each $j=1, \ldots, \ell$.
To close this slide: for each $j$,
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

