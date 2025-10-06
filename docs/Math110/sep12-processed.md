---
course: CS 189
semester: Fall 2025
type: pdf
title: sep12
source_type: pdf
source_file: sep12.pdf
processed_date: '2025-10-06'
processor: mathpix
slug: sep12-processed
---

# Span, linear independence, dimension 

## Professor K. A. Ribet

![The image features the official seal of the University of California, Berkeley. It depicts an open b](https://cdn.mathpix.com/cropped/2025_10_06_98a6140aef1d069f0ab9g-01.jpg?height=213&width=221&top_left_y=427&top_left_x=517)

**Image description:** The image features the official seal of the University of California, Berkeley. It depicts an open book at the center, symbolizing knowledge, with the inscription "LET THERE BE LIGHT" on a ribbon beneath. Surrounding the book are an ornate design and a star above. The year "1868" is also included, marking the university's founding. The circular border enhances its ceremonial appearance, emphasizing prestige and scholarly tradition. The seal serves to represent the university's identity and heritage in academic contexts.


September 12, 2025

## Office Hours

> 885 Evans Hall Mondays, 1:30-3PM Thursdays, 10:30-noon

Office full $\Longrightarrow$ possible move to a nearby classroom

I plan to come to the DCs at least once per week. There will be official Residential Life "lunches with Professor Ribet" at noon at Foothill DC on September 18, September 26, October 3 and October 9. There will also be additional lunch gatherings at DCs and the Faculty Club.

Gatherings are optional and not part of Math 110, but I'll continue to list them on slides for those who are interested. Also, you can send me email to subscribe to email announcements.

- Faculty Club lunch Monday, September 15 at noon
- First official Lunch with Prof. Ribet on Thursday, September 18 after office hour ends.


## Wednesday

We made a list of items concerning lists and talked through all of the items. Here's a quick recap:

- Lists
- Span of a List
- Linear maps (reminder)
- Linear map defined by a list
- Linear independence
- Lists that span
- Bases (certain lists)
- Finite-dimensional spaces

The only thing wrong with this list is that it wasn't numbered. My bad! Lists be like

$$
v_{1}, \ldots, v_{\ell}, \quad \ell \geq 0, \quad \text { all } v_{j} \in V
$$

## Now it's Friday, so let's go!

## Lemma (2.19)

Let $v_{1}, \ldots, v_{\ell}$ be a linearly dependent list of vectors of $V$. Then there is some index $k$ such that $v_{k}$ lies in the span of $v_{1}, \ldots, v_{k-1}$. For this $k$, the span of the list with $v_{k}$ deleted is the same as the span of the list $v_{1}, \ldots, v_{\ell}$.

A nerdy remark is that if $\ell=0$, then the list is linearly independent, so there is nothing to prove. Also, if $v_{1}=0$, then we can take $k=1$ and note that 0 is in the span of the empty list (of length $k-1=0$ ). Thus we can prove the lemma under the assumption that $\ell$ is at least 1 and $v_{1}$ is nonzero. If $\ell=1$ and $v_{1}$ is nonzero, then the list $v_{1}$ is linearly independent and there is nothing to prove. Thus we can and will suppose $\ell \geq 2$ and that $v_{1}$ is nonzero.

## Proof of the lemma

## Lemma (2.19)

Let $v_{1}, \ldots, v_{\ell}$ be a linearly dependent list of vectors of $V$. Then there is some index $k$ such that $v_{k}$ lies in the span of $v_{1}, \ldots, v_{k-1}$. For this $k$, the span of the list with $v_{k}$ deleted is the same as the span of the list $v_{1}, \ldots, v_{\ell}$.

The start of the proof is to note that linear dependence means that there are scalars $\lambda_{j}$, not all equal to 0 , so that

$$
0=\lambda_{1} v_{1}+\cdots+\lambda_{\ell} v_{\ell} .
$$

Let $k$ be the largest index so that $\lambda_{k}$ is nonzero. Then

$$
0=\lambda_{1} v_{1}+\cdots+\lambda_{k} v_{k}, \quad \lambda_{k} \neq 0
$$

The equation

$$
v_{k}=-1 / \lambda_{k}\left(\lambda_{1} v_{1}+\cdots+\lambda_{k-1} v_{k-1}\right)
$$
shows that $v_{k} \in \operatorname{span}\left(v_{1}, \ldots, v_{k-1}\right)$.

## Proof of the lemma

## Lemma (2.19)

Let $v_{1}, \ldots, v_{\ell}$ be a linearly dependent list of vectors of $V$. Then there is some index $k$ such that $v_{k}$ lies in the span of $v_{1}, \ldots, v_{k-1}$. For this $k$, the span of the list with $v_{k}$ deleted is the same as the span of the list $v_{1}, \ldots, v_{\ell}$.

It remains to show that the span of $v_{1}, \ldots, v_{\ell}$ does not shrink if we remove $v_{k}$ from the list. Each element of the span is a linear combination

$$
a_{1} v_{1}+\cdots+a_{k-1} v_{k-1}+a_{k} v_{k}+a_{k+1} v_{k+1}+\cdots+a_{\ell} v_{\ell} .
$$

In this expression, replace $v_{k}$ by $-1 / \lambda_{k}\left(\lambda_{1} v_{1}+\cdots+\lambda_{k-1} v_{k-1}\right)$. After collecting terms, we see that the linear combination $a_{1} v_{1}+\cdots+a_{\ell} v_{\ell}$ has an alternative expression as a linear combination of the vectors in the list other than $v_{k}$.

## Linearly ind. list is no longer than spanning list

## Theorem (2.22, p. 35)

In a finite-dimensional vector space, the length of every linearly independent list of vectors is less than or equal to the length of every spanning list of vectors.

I've copied Axler's 2.22 and will now copy his notation for your convenience. Let $w_{1}, \ldots, w_{n}$ be a list of vectors of $V$ that spans $V$. Let $u_{1}, \ldots, u_{m}$ be a linearly independent list. The theorem asserts

$$
m \stackrel{?}{\leq} n,
$$
and this is what we must prove.

## Lin. independent list is no longer than spanning list

## Theorem (2.22, p. 35)

If $w_{1}, \ldots, w_{n}$ is a spanning list and $u_{1}, \ldots, u_{m}$ is a linearly independent list, then $m \leq n$.

The proof involves replacement. We can do $m$ replacements of a $w$ vector by a $u$ vector. There are only $n w$ vectors to replace; thus if we are able to make more replacements than there are objects to replace, we're in big trouble.

This sounds nuts, but really works.

## Linearly independent list is no longer than span list

I'll explain this with small numbers: $n=2, m=3$. We assume that $V$ is spanned by $w_{1}, w_{2}$ and consider a linearly independent list $u_{1}, \ldots, u_{m}$. The aim is to prove $m \geq 2$. We want to see what is wrong with the contrary case $m>2$. It's enough to show that there cannot be a linearly independent list with three vectors. If that's true, there can't be a linearly independent list with four or more vectors.

## Linearly independent list is short

Again: assume that $V$ is spanned by $w_{1}, w_{2}$ and that $u_{1}, u_{2}, u_{3}$ is linearly independent.

Because $w_{1}, w_{2}$ spans, $u_{3}$ is a linear combination of $w_{1}, w_{2}$ :

$$
u_{3}=a w_{1}+b w_{2} .
$$

By linear independence of the 1 -element list $u_{3}, a$ and $b$ are not both 0 . We can and will assume $b$ is nonzero (explanation on site). Multiplying by $1 / b$ and moving terms around, we find that $w_{2}$ is in the span of $w_{1}$ and $u_{3}$. So is $w_{1}$. Thus $V$ is spanned by $w_{1}, u_{3}$.

Next step: $u_{2}$ is a linear combination of $w_{1}, u_{3}$ :

$$
u_{2}=\lambda w_{1}+\mu u_{3} .
$$

By linear independence of $u_{2}, u_{3}, \lambda$ is nonzero. Repeat previous manipulation to show that $V$ is spanned by $u_{2}, u_{3}$. In particular, $u_{1}$ is a linear combination of $u_{2}, u_{3}$. This is not possible because $u_{1}, u_{2}, u_{3}$ is linearly independent.

## Subspaces of finite-dimensional spaces

Theorem (2.25)
Every subspace of a finite-dimensional vector space is finite-dimensional.

Assume that $U$ is a subspace of $V$ and that $V$ is finite-dimensional. This means that $V$ is spanned by a list of vectors of some length $n$. If $U=\{0\}$, it's finite-dimensional.
Assume $U \neq\{0\}$. Then $U$ has a nonzero vector $u_{1}$. Is $U$ spanned by $u_{1}$ ? If so, done! If not, there is $u_{2} \in U$ not in the span of $u_{1}$. This means that $u_{1}, u_{2}$ is linearly independent. Does this list span? If so, excellent! If not, there is a $u_{3}$ not in the span; then $u_{1}, u_{2}, u_{3}$ is linearly independent. Does it span? If so, fabulous!! If not, there's a linearly independent list in $U$ of length 4.
We can't keep going like this because a linearly independent list can have no more than $n$ elements.

## Bases

Recall that a basis of $V$ is a linearly independent list that spans $V$. To say that $v_{1}, \ldots, v_{n}$ is a basis is to say that the linear map

$$
\mathbf{F}^{n} \rightarrow V, \quad\left(a_{1}, \ldots, a_{n}\right) \mapsto \sum_{j} a_{j} v_{j}
$$
is 1-1 and onto. In words, this means that every vector in $V$ can be written uniquely as a linear combination of $v_{1}, \ldots, v_{n}$.

## Proposition <br> If a vector space has two bases, they are of the same length.

In other words, if $V$ has a basis of length $n$ and also a basis of length $m$, then $m=n$.

A basis is both a spanning list and a linearly independent list. View the basis of length $m$ as a spanning list and the basis of length $n$ as a linearly independent list. We get $n \leq m$.
Reversing roles gives $m \leq n$. Thus $m=n$.

## Spanning lists shrink to bases

## Theorem (2.30)

Every spanning list in a vector space can be reduced to a basis of the vector space.

This means that if $v_{1}, \ldots, v_{m}$ spans, we can get a basis of $V$ by ejecting some of the $v_{j}$ from the list.

## Refining a spanning list to get a basis

## Theorem (2.30)

Every spanning list in a vector space can be reduced to a basis of the vector space.

We can view the proof as an induction on the length of the spanning list $v_{1}, \ldots, v_{m}$. If $m=0, V=\{0\}$ and the spanning list (which is empty) is also a basis. If $m=1$, the list is a single vector, say $v$. Because it spans, $V=\mathbf{F} \cdot v$. If $v$ is nonzero, the list $v$ is linearly independent and represents a basis. If $v=0$, remove it to get the empty list, which is a basis of $V=\{0\}$.

## Refining a spanning list to get a basis

In the induction step, take a spanning list $v_{1}, \ldots, v_{m}$ of length $>1$. If it's linearly independent, it's a basis and we're done.

If it's linearly dependent, some vector in the list is a linear combination of the previous vectors. We can chuck it without disturbing the span of the list-which is all of $V$. The pruned list is still a spanning list, but now it has length $m-1$. Assuming the desired result for lists of that length, we deduce that the pruned list can be pruned further, if necessary, to yield a basis of $V$.

## Finite-dimensional spaces have bases

Corollary (2.31)
If $V$ is finite-dimensional, it has a basis.
To prove it, take a spanning list and prune it if necessary to get a basis.

## Extending linearly independent lists

## Proposition (2.32)

Every linearly independent list of vectors in a finite-dimensional vector space can be extended to a basis of the vector space.

Start with a linearly independent list $v_{1}, \ldots, v_{n}$ in a finite-dimensional vector space $V$. The idea is to extend the list incrementally until it spans (and thus is a basis).
Ask first whether $v_{1}, \ldots, v_{n}$ is already a basis. If not, there is $v_{n+1} \notin \operatorname{span}\left(v_{1}, \ldots, v_{n}\right)$. The key is that $v_{1}, \ldots, v_{n}, v_{n+1}$ is then again linearly independent (to be explained in class). Does it span? If not, it can be grown a second time to a longer linearly independent list $v_{1}, \ldots, v_{n}, v_{n+1}, v_{n+2}$.
list doesn't span? $\Longrightarrow$ it can grow.
However, a linearly independent list can't be longer than $\operatorname{dim} V$. Growth has to stop. When it does, we have a linearly independent spanning list (= a basis).
![The image displays a sign adhered to a trash receptacle. The sign, which is prominently labeled "Cal](https://cdn.mathpix.com/cropped/2025_10_06_98a6140aef1d069f0ab9g-28.jpg?height=594&width=967&top_left_y=141&top_left_x=147)

**Image description:** The image displays a sign adhered to a trash receptacle. The sign, which is prominently labeled "California," reads: "PLEASE HANG UP COATS They do not need to be washed after every use. Thank You & Go Bears!" The purpose of this sign is to encourage individuals to reuse their coats instead of disposing of them, thereby promoting sustainability and reducing waste in the context of an academic environment. The sign emphasizes practicality in garment care during lectures or events.


