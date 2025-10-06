---
course: CS 189
semester: Fall 2025
type: textbook
title: Deep Learning Pages 1-171
source_type: pdf
source_file: DeepLearning_pages_1-171.pdf
processed_date: '2025-10-01'
processor: mathpix
---

# Christopher M.Bishop with Hugh Bishop 

## Deep Learning

## Foundations and Concepts

Deep Learning

# Christopher M. Bishop • Hugh Bishop 

## Deep Learning

Foundations and Concepts

Christopher M. Bishop Microsoft Research Cambridge, UK

Hugh Bishop<br>Wayve Technologies Ltd<br>London, UK

ISBN 978-3-031-45467-7 ISBN 978-3-031-45468-4 (eBook)
https://doi.org/10.1007/978-3-031-45468-4
© The Editor(s) (if applicable) and The Author(s), under exclusive license to Springer Nature Switzerland AG 2024
This work is subject to copyright. All rights are solely and exclusively licensed by the Publisher, whether the whole or part of the material is concerned, specifically the rights of translation, reprinting, reuse of illustrations, recitation, broadcasting, reproduction on microfilms or in any other physical way, and transmission or information storage and retrieval, electronic adaptation, computer software, or by similar or dissimilar methodology now known or hereafter developed.
The use of general descriptive names, registered names, trademarks, service marks, etc. in this publication does not imply, even in the absence of a specific statement, that such names are exempt from the relevant protective laws and regulations and therefore free for general use.
The publisher, the authors, and the editors are safe to assume that the advice and information in this book are believed to be true and accurate at the date of publication. Neither the publisher nor the authors or the editors give a warranty, expressed or implied, with respect to the material contained herein or for any errors or omissions that may have been made. The publisher remains neutral with regard to jurisdictional claims in published maps and institutional affiliations.

Cover illustration: maksimee / Alamy Stock Photo
This Springer imprint is published by the registered company Springer Nature Switzerland AG
The registered company address is: Gewerbestrasse 11, 6330 Cham, Switzerland
Paper in this product is recyclable.

## Preface

Deep learning uses multilayered neural networks trained with large data sets to solve complex information processing tasks and has emerged as the most successful paradigm in the field of machine learning. Over the last decade, deep learning has revolutionized many domains including computer vision, speech recognition, and natural language processing, and it is being used in a growing multitude of applications across healthcare, manufacturing, commerce, finance, scientific discovery, and many other sectors. Recently, massive neural networks, known as large language models and comprising of the order of a trillion learnable parameters, have been found to exhibit the first indications of general artificial intelligence and are now driving one of the biggest disruptions in the history of technology.

## Goals of the book

This expanding impact has been accompanied by an explosion in the number and breadth of research publications in machine learning, and the pace of innovation continues to accelerate. For newcomers to the field, the challenge of getting to grips with the key ideas, let alone catching up to the research frontier, can seem daunting. Against this backdrop, Deep Learning: Foundations and Concepts aims to provide newcomers to machine learning, as well as those already experienced in the field, with a thorough understanding of both the foundational ideas that underpin deep learning as well as the key concepts of modern deep learning architectures and techniques. This material will equip the reader with a strong basis for future specialization. Due to the breadth and pace of change in the field, we have deliberately avoided trying to create a comprehensive survey of the latest research. Instead, much of the value of the book derives from a distillation of key ideas, and although the field itself can be expected to continue its rapid advance, these foundations and concepts are likely to stand the test of time. For example, large language models have been evolving very rapidly at the time of writing, yet the underlying transformer architecture and attention mechanism have remained largely unchanged for the last five years, while many core principles of machine learning have been known for decades.

## Responsible use of technology

Deep learning is a powerful technology with broad applicability that has the potential to create huge value for the world and address some of society's most pressing challenges. However, these same attributes mean that deep learning also has potential both for deliberate misuse and to cause unintended harms. We have chosen not to discuss ethical or societal aspects of the use of deep learning, as these topics are of such importance and complexity that they warrant a more thorough treatment than is possible in a technical textbook such as this. Such considerations should, however, be informed by a solid grounding in the underlying technology and how it works, and so we hope that this book will make a valuable contribution towards these important discussions. The reader is, nevertheless, strongly encouraged to be mindful about the broader implications of their work and to learn about the responsible use of deep learning and artificial intelligence alongside their studies of the technology itself.

## Structure of the book

The book is structured into a relatively large number of smaller bite-sized chapters, each of which explores a specific topic. The book has a linear structure in the sense that each chapter depends only on material covered in earlier chapters. It is well suited to teaching a two-semester undergraduate or postgraduate course on machine learning but is equally relevant to those engaged in active research or in self-study.

A clear understanding of machine learning can be achieved only through the use of some level of mathematics. Specifically, three areas of mathematics lie at the heart of machine learning: probability theory, linear algebra, and multivariate calculus. The book provides a self-contained introduction to the required concepts in probability theory and includes an appendix that summarizes some useful results in linear algebra. It is assumed that the reader already has some familiarity with the basic concepts of multivariate calculus although there are appendices that provide introductions to the calculus of variations and to Lagrange multipliers. The focus of the book, however, is on conveying a clear understanding of ideas, and the emphasis is on techniques that have real-world practical value rather than on abstract theory. Where possible we try to present more complex concepts from multiple complementary perspectives including textual description, diagrams, and mathematical formulae. In addition, many of the key algorithms discussed in the text are summarized in separate boxes. These do not address issues of computational efficiency, but are provided as a complement to the mathematical explanations given in the text. We therefore hope that the material in this book will be accessible to readers from a variety of backgrounds.

Conceptually, this book is perhaps most naturally viewed as a successor to Neural Networks for Pattern Recognition (Bishop, 1995b), which provided the first comprehensive treatment of neural networks from a statistical perspective. It can also be considered as a companion volume to Pattern Recognition and Machine Learning (Bishop, 2006), which covered a broader range of topics in machine learning although it predated the deep learning revolution. However, to ensure that this
new book is self-contained, appropriate material has been carried over from Bishop (2006) and refactored to focus on those foundational ideas that are needed for deep learning. This means that there are many interesting topics in machine learning discussed in Bishop (2006) that remain of interest today but which have been omitted from this new book. For example, Bishop (2006) discusses Bayesian methods in some depth, whereas this book is almost entirely non-Bayesian.

The book is accompanied by a web site that provides supporting material, including a free-to-use digital version of the book as well as solutions to the exercises and downloadable versions of the figures in PDF and JPEG formats:

```
https://www.bishopbook.com
```

The book can be cited using the following BibTex entry:

```
@book{Bishop:DeepLearning24,
    author = {Christopher M. Bishop and Hugh Bishop},
    title = {Deep Learning: Foundations and Concepts},
    year = {2024},
    publisher = {Springer}
}
```

If you have any feedback on the book or would like to report any errors, please send these to feedback@bishopbook.com

## References

In the spirit of focusing on core ideas, we make no attempt to provide a comprehensive literature review, which in any case would be impossible given the scale and pace of change of the field. We do, however, provide references to some of the key research papers as well as review articles and other sources of further reading. In many cases, these also provide important implementation details that we gloss over in the text in order not to distract the reader from the central concepts being discussed.

Many books have been written on the subject of machine learning in general and on deep learning in particular. Those which are closest in level and style to this book include Bishop (2006), Goodfellow, Bengio, and Courville (2016), Murphy (2022), Murphy (2023), and Prince (2023).

Over the last decade, the nature of machine learning scholarship has changed significantly, with many papers being posted online on archival sites ahead of, or even instead of, submission to peer-reviewed conferences and journals. The most popular of these sites is arXiv, pronounced 'archive', and is available at

```
https://arXiv.org
```

The site allows papers to be updated, often leading to multiple versions associated with different calendar years, which can result in some ambiguity as to which version should be cited and for which year. It also provides free access to a PDF of each paper. We have therefore adopted a simple approach of referencing the paper according to the year of first upload, although we recommend reading the most recent version.

Papers on arXiv are indexed using a notation arXiv:YYMM.XXXXX where YY and MM denote the year and month of first upload, respectively. Subsequent versions are denoted by appending a version number $N$ in the form arXiv:YYMM. XXXXXvN.

## Exercises

Each chapter concludes with a set of exercises designed to reinforce the key ideas explained in the text or to develop and generalize them in significant ways. These exercises form an important part of the text and each is graded according to difficulty ranging from $(\star)$, which denotes a simple exercise taking a few moments to complete, through to $(\star \star \star)$, which denotes a significantly more complex exercise. The reader is strongly encouraged to attempt the exercises since active participation with the material greatly increases the effectiveness of learning. Worked solutions to all of the exercises are available as a downloadable PDF file from the book web site.

## Mathematical notation

We follow the same notation as Bishop (2006). For an overview of mathematics in the context of machine learning, see Deisenroth, Faisal, and Ong (2020).

Vectors are denoted by lower case bold roman letters such as $\mathbf{x}$, whereas matrices are denoted by uppercase bold roman letters, such as M . All vectors are assumed to be column vectors unless otherwise stated. A superscript T denotes the transpose of a matrix or vector, so that $\mathbf{x}^{\mathrm{T}}$ will be a row vector. The notation ( $w_{1}, \ldots, w_{M}$ ) denotes a row vector with $M$ elements, and the corresponding column vector is written as $\mathbf{w}=\left(w_{1}, \ldots, w_{M}\right)^{\mathrm{T}}$. The $M \times M$ identity matrix (also known as the unit matrix) is denoted $\mathbf{I}_{M}$, which will be abbreviated to $\mathbf{I}$ if there is no ambiguity about its dimensionality. It has elements $I_{i j}$ that equal 1 if $i=j$ and 0 if $i \neq j$. The elements of a unit matrix are sometimes denoted by $\delta_{i j}$. The notation $\mathbf{1}$ denotes a column vector in which all elements have the value $1 . \mathbf{a} \oplus \mathbf{b}$ denotes the concatenation of vectors $\mathbf{a}$ and $\mathbf{b}$, so that if $\mathbf{a}=\left(a_{1}, \ldots, a_{N}\right)$ and $\mathbf{b}=\left(b_{1}, \ldots, b_{M}\right)$ then $\mathbf{a} \oplus \mathbf{b}= \left(a_{1}, \ldots, a_{N}, b_{1}, \ldots, b_{M}\right) .|x|$ denotes the modulus (the positive part) of a scalar $x$, also known as the absolute value. We use $\operatorname{det} \mathbf{A}$ to denote the determinant of a matrix A.

The notation $x \sim p(x)$ signifies that $x$ is sampled from the distribution $p(x)$. Where there is ambiguity, we will use subscripts as in $p_{x}(\cdot)$ to denote which density is referred to. The expectation of a function $f(x, y)$ with respect to a random variable $x$ is denoted by $\mathbb{E}_{x}[f(x, y)]$. In situations where there is no ambiguity as to which variable is being averaged over, this will be simplified by omitting the suffix, for instance $\mathbb{E}[x]$. If the distribution of $x$ is conditioned on another variable $z$, then the corresponding conditional expectation will be written $\mathbb{E}_{x}[f(x) \mid z]$. Similarly, the variance of $f(x)$ is denoted var $[f(x)]$, and for vector variables, the covariance is written $\operatorname{cov}[\mathbf{x}, \mathbf{y}]$. We will also use $\operatorname{cov}[\mathbf{x}]$ as a shorthand notation for $\operatorname{cov}[\mathbf{x}, \mathbf{x}]$.

The symbol $\forall$ means 'for all', so that $\forall m \in \mathcal{M}$ denotes all values of $m$ within the set $\mathcal{M}$. We use $\mathbb{R}$ to denote the real numbers. On a graph, the set of neighbours of node $i$ is denoted $\mathcal{N}(i)$, which should not be confused with the Gaussian or normal distribution $\mathcal{N}\left(x \mid \mu, \sigma^{2}\right)$. A functional is denoted $f[y]$ where $y(x)$ is some function. The concept of a functional is discussed in Appendix B. Curly braces $\}$ denote a
set. The notation $g(x)=\mathcal{O}(f(x))$ denotes that $|f(x) / g(x)|$ is bounded as $x \rightarrow \infty$. For instance, if $g(x)=3 x^{2}+2$, then $g(x)=\mathcal{O}\left(x^{2}\right)$. The notation $\lfloor x\rfloor$ denotes the 'floor' of $x$, i.e., the largest integer that is less than or equal to $x$.

If we have $N$ independent and identically distributed (i.i.d.) values $\mathbf{x}_{1}, \ldots, \mathbf{x}_{N}$ of a $D$-dimensional vector $\mathbf{x}=\left(x_{1}, \ldots, x_{D}\right)^{\mathrm{T}}$, we can combine the observations into a data matrix $\mathbf{X}$ of dimension $N \times D$ in which the $n$th row of $\mathbf{X}$ corresponds to the row vector $\mathbf{x}_{n}^{\mathrm{T}}$. Thus, the $n, i$ element of $\mathbf{X}$ corresponds to the $i$ th element of the $n$th observation $\mathbf{x}_{n}$ and is written $x_{n i}$. For one-dimensional variables, we denote such a matrix by $\mathbf{x}$, which is a column vector whose $n$th element is $x_{n}$. Note that $\mathbf{x}$ (which has dimensionality $N$ ) uses a different typeface to distinguish it from $\mathbf{x}$ (which has dimensionality $D$ ).

## Acknowledgements

We would like to express our sincere gratitude to the many people who reviewed draft chapters and provided valuable feedback. In particular, we wish to thank Samuel Albanie, Cristian Bodnar, John Bronskill, Wessel Bruinsma, Ignas Budvytis, Chi Chen, Yaoyi Chen, Long Chen, Fergal Cotter, Sam Devlin, Aleksander Durumeric, Sebastian Ehlert, Katarina Elez, Andrew Foong, Hong Ge, Paul Gladkov, Paula Gori Giorgi, John Gossman, Tengda Han, Juyeon Heo, Katja Hofmann, Chin-Wei Huang, Yongchaio Huang, Giulio Isacchini, Matthew Johnson, Pragya Kale, Atharva Kelkar, Leon Klein, Pushmeet Kohli, Bonnie Kruft, Adrian Li, Haiguang Liu, Ziheng Lu, Giulia Luise, Stratis Markou, Sergio Valcarcel Macua, Krzysztof Maziarz, Matěj Mezera, Laurence Midgley, Usman Munir, Félix Musil, Elise van der Pol, Tao Qin, Isaac Reid, David Rosenberger, Lloyd Russell, Maximilian Schebek, Megan Stanley, Karin Strauss, Clark Templeton, Marlon Tobaben, Aldo Sayeg Pasos-Trejo, Richard Turner, Max Welling, Furu Wei, Robert Weston, Chris Williams, Yingce Xia, Shufang Xie, Iryna Zaporozhets, Claudio Zeni, Xieyuan Zhang, and many other colleagues who contributed through valuable discussions. We would also like to thank our editor Paul Drougas and many others at Springer, as well as the copy editor Jonathan Webley, for their support during the production of the book.

We would like to say a special thank-you to Markus Svensén, who provided immense help with the figures and typesetting for Bishop (2006) including the $\mathrm{AT}_{\mathrm{E}} \mathrm{X}$ style files, which have also been used for this new book. We are also grateful to the many scientists who allowed us to reproduce diagrams from their published work. Acknowledgements for specific figures appear in the associated figure captions.

Chris would like to express sincere gratitude to Microsoft for creating a highly stimulating research environment and for providing the opportunity to write this book. The views and opinions expressed in this book, however, are those of the authors and are therefore not necessarily the same as those of Microsoft or its affiliates. It has been a huge privilege and pleasure to collaborate with my son Hugh in preparing this book, which started as a joint project during the first Covid lockdown.

Hugh would like to thank Wayve Technologies Ltd for generously allowing him to work part time so that he could collaborate in writing this book as well as for providing an inspiring and supportive environment for him to work and learn in. The views expressed in this book are not necessarily the same as those of Wayve or its affiliates. He would like to express his gratitude to his fiancée Jemima for her constant support as well as her grammatical and stylistic consultations. He would also like to thank Chris, who has been an excellent colleague and an inspiration to Hugh throughout his life.

Finally, we would both like to say a huge thank-you to our family members Jenna and Mark for so many things far too numerous to list here. It seems a very long time ago that we all gathered on the beach in Antalya to watch a total eclipse of the sun and to take a family photo for the dedication page of Pattern Recognition and Machine Learning!

Chris Bishop and Hugh Bishop
Cambridge, UK
October, 2023

## Contents

Preface ..... v
Contents ..... xi
1 The Deep Learning Revolution ..... 1
1.1 The Impact of Deep Learning ..... 2
1.1.1 Medical diagnosis ..... 2
1.1.2 Protein structure ..... 3
1.1.3 Image synthesis ..... 4
1.1.4 Large language models ..... 5
1.2 A Tutorial Example ..... 6
1.2.1 Synthetic data ..... 6
1.2.2 Linear models ..... 8
1.2.3 Error function ..... 8
1.2.4 Model complexity ..... 9
1.2.5 Regularization ..... 12
1.2.6 Model selection ..... 14
1.3 A Brief History of Machine Learning ..... 16
1.3.1 Single-layer networks ..... 17
1.3.2 Backpropagation ..... 18
1.3.3 Deep networks ..... 20
2 Probabilities ..... 23
2.1 The Rules of Probability ..... 25
2.1.1 A medical screening example ..... 25
2.1.2 The sum and product rules ..... 26
2.1.3 Bayes' theorem ..... 28
2.1.4 Medical screening revisited ..... 30
2.1.5 Prior and posterior probabilities ..... 31
2.1.6 Independent variables ..... 31
2.2 Probability Densities ..... 32
2.2.1 Example distributions ..... 33
2.2.2 Expectations and covariances ..... 34
2.3 The Gaussian Distribution ..... 36
2.3.1 Mean and variance ..... 37
2.3.2 Likelihood function ..... 37
2.3.3 Bias of maximum likelihood ..... 39
2.3.4 Linear regression ..... 40
2.4 Transformation of Densities ..... 42
2.4.1 Multivariate distributions ..... 44
2.5 Information Theory ..... 46
2.5.1 Entropy ..... 46
2.5.2 Physics perspective ..... 47
2.5.3 Differential entropy ..... 49
2.5.4 Maximum entropy ..... 50
2.5.5 Kullback-Leibler divergence ..... 51
2.5.6 Conditional entropy ..... 53
2.5.7 Mutual information ..... 54
2.6 Bayesian Probabilities ..... 54
2.6.1 Model parameters ..... 55
2.6.2 Regularization ..... 56
2.6.3 Bayesian machine learning ..... 57
Exercises ..... 58
3 Standard Distributions ..... 65
3.1 Discrete Variables ..... 66
3.1.1 Bernoulli distribution ..... 66
3.1.2 Binomial distribution ..... 67
3.1.3 Multinomial distribution ..... 68
3.2 The Multivariate Gaussian ..... 70
3.2.1 Geometry of the Gaussian ..... 71
3.2.2 Moments ..... 74
3.2.3 Limitations ..... 75
3.2.4 Conditional distribution ..... 76
3.2.5 Marginal distribution ..... 79
3.2.6 Bayes' theorem ..... 81
3.2.7 Maximum likelihood ..... 84
3.2.8 Sequential estimation ..... 85
3.2.9 Mixtures of Gaussians ..... 86
3.3 Periodic Variables ..... 89
3.3.1 Von Mises distribution ..... 89
3.4 The Exponential Family ..... 94
3.4.1 Sufficient statistics ..... 97
3.5 Nonparametric Methods ..... 98
3.5.1 Histograms ..... 98
3.5.2 Kernel densities ..... 100
3.5.3 Nearest-neighbours ..... 103
Exercises ..... 105
4 Single-layer Networks: Regression ..... 111
4.1 Linear Regression ..... 112
4.1.1 Basis functions ..... 112
4.1.2 Likelihood function ..... 114
4.1.3 Maximum likelihood ..... 115
4.1.4 Geometry of least squares ..... 117
4.1.5 Sequential learning ..... 117
4.1.6 Regularized least squares ..... 118
4.1.7 Multiple outputs ..... 119
4.2 Decision theory ..... 120
4.3 The Bias-Variance Trade-off ..... 123
Exercises ..... 128
5 Single-layer Networks: Classification ..... 131
5.1 Discriminant Functions ..... 132
5.1.1 Two classes ..... 132
5.1.2 Multiple classes ..... 134
5.1.3 1 -of- $K$ coding ..... 135
5.1.4 Least squares for classification ..... 136
5.2 Decision Theory ..... 138
5.2.1 Misclassification rate ..... 139
5.2.2 Expected loss ..... 140
5.2.3 The reject option ..... 142
5.2.4 Inference and decision ..... 143
5.2.5 Classifier accuracy ..... 147
5.2.6 ROC curve ..... 148
5.3 Generative Classifiers ..... 150
5.3.1 Continuous inputs ..... 152
5.3.2 Maximum likelihood solution ..... 153
5.3.3 Discrete features ..... 156
5.3.4 Exponential family ..... 156
5.4 Discriminative Classifiers ..... 157
5.4.1 Activation functions ..... 158
5.4.2 Fixed basis functions ..... 158
5.4.3 Logistic regression ..... 159
5.4.4 Multi-class logistic regression ..... 161
5.4.5 Probit regression ..... 163
5.4.6 Canonical link functions ..... 164
Exercises ..... 166
6 Deep Neural Networks ..... 171
6.1 Limitations of Fixed Basis Functions ..... 172
6.1.1 The curse of dimensionality ..... 172
6.1.2 High-dimensional spaces ..... 175
6.1.3 Data manifolds ..... 176
6.1.4 Data-dependent basis functions ..... 178
6.2 Multilayer Networks ..... 180
6.2.1 Parameter matrices ..... 181
6.2.2 Universal approximation ..... 181
6.2.3 Hidden unit activation functions ..... 182
6.2.4 Weight-space symmetries ..... 185
6.3 Deep Networks ..... 186
6.3.1 Hierarchical representations ..... 187
6.3.2 Distributed representations ..... 187
6.3.3 Representation learning ..... 188
6.3.4 Transfer learning ..... 189
6.3.5 Contrastive learning ..... 191
6.3.6 General network architectures ..... 193
6.3.7 Tensors ..... 194
6.4 Error Functions ..... 194
6.4.1 Regression ..... 194
6.4.2 Binary classification ..... 196
6.4.3 multiclass classification ..... 197
6.5 Mixture Density Networks ..... 198
6.5.1 Robot kinematics example ..... 198
6.5.2 Conditional mixture distributions ..... 199
6.5.3 Gradient optimization ..... 201
6.5.4 Predictive distribution ..... 202
Exercises ..... 204
7 Gradient Descent ..... 209
7.1 Error Surfaces ..... 210
7.1.1 Local quadratic approximation ..... 211
7.2 Gradient Descent Optimization ..... 213
7.2.1 Use of gradient information ..... 214
7.2.2 Batch gradient descent ..... 214
7.2.3 Stochastic gradient descent ..... 214
7.2.4 Mini-batches ..... 216
7.2.5 Parameter initialization ..... 216
7.3 Convergence ..... 218
7.3.1 Momentum ..... 220
7.3.2 Learning rate schedule ..... 222
7.3.3 RMSProp and Adam ..... 223
7.4 Normalization ..... 224
7.4.1 Data normalization ..... 226
7.4.2 Batch normalization ..... 227
7.4.3 Layer normalization ..... 229
Exercises ..... 230
8 Backpropagation ..... 233
8.1 Evaluation of Gradients ..... 234
8.1.1 Single-layer networks ..... 234
8.1.2 General feed-forward networks ..... 235
8.1.3 A simple example ..... 238
8.1.4 Numerical differentiation ..... 239
8.1.5 The Jacobian matrix ..... 240
8.1.6 The Hessian matrix ..... 242
8.2 Automatic Differentiation ..... 244
8.2.1 Forward-mode automatic differentiation ..... 246
8.2.2 Reverse-mode automatic differentiation ..... 249
Exercises ..... 250
9 Regularization ..... 253
9.1 Inductive Bias ..... 254
9.1.1 Inverse problems ..... 254
9.1.2 No free lunch theorem ..... 255
9.1.3 Symmetry and invariance ..... 256
9.1.4 Equivariance ..... 259
9.2 Weight Decay ..... 260
9.2.1 Consistent regularizers ..... 262
9.2.2 Generalized weight decay ..... 264
9.3 Learning Curves ..... 266
9.3.1 Early stopping ..... 266
9.3.2 Double descent ..... 268
9.4 Parameter Sharing ..... 270
9.4.1 Soft weight sharing ..... 271
9.5 Residual Connections ..... 274
9.6 Model Averaging ..... 277
9.6.1 Dropout ..... 279
Exercises ..... 281
10 Convolutional Networks ..... 287
10.1 Computer Vision ..... 288
10.1.1 Image data ..... 289
10.2 Convolutional Filters ..... 290
10.2.1 Feature detectors ..... 290
10.2.2 Translation equivariance ..... 291
10.2.3 Padding ..... 294
10.2.4 Strided convolutions ..... 294
10.2.5 Multi-dimensional convolutions ..... 295
10.2.6 Pooling ..... 296
10.2.7 Multilayer convolutions ..... 298
10.2.8 Example network architectures ..... 299
10.3 Visualizing Trained CNNs ..... 302
10.3.1 Visual cortex ..... 302
10.3.2 Visualizing trained filters ..... 303
10.3.3 Saliency maps ..... 305
10.3.4 Adversarial attacks ..... 306
10.3.5 Synthetic images ..... 308
10.4 Object Detection ..... 308
10.4.1 Bounding boxes ..... 309
10.4.2 Intersection-over-union ..... 310
10.4.3 Sliding windows ..... 311
10.4.4 Detection across scales ..... 313
10.4.5 Non-max suppression ..... 314
10.4.6 Fast region CNNs ..... 314
10.5 Image Segmentation ..... 315
10.5.1 Convolutional segmentation ..... 315
10.5.2 Up-sampling ..... 316
10.5.3 Fully convolutional networks ..... 318
10.5.4 The U-net architecture ..... 319
10.6 Style Transfer ..... 320
Exercises ..... 322
11 Structured Distributions ..... 325
11.1 Graphical Models ..... 326
11.1.1 Directed graphs ..... 326
11.1.2 Factorization ..... 327
11.1.3 Discrete variables ..... 329
11.1.4 Gaussian variables ..... 332
11.1.5 Binary classifier ..... 334
11.1.6 Parameters and observations ..... 334
11.1.7 Bayes' theorem ..... 336
11.2 Conditional Independence ..... 337
11.2.1 Three example graphs ..... 338
11.2.2 Explaining away ..... 341
11.2.3 D-separation ..... 343
11.2.4 Naive Bayes ..... 344
11.2.5 Generative models ..... 346
11.2.6 Markov blanket ..... 347
11.2.7 Graphs as filters ..... 348
11.3 Sequence Models ..... 349
11.3.1 Hidden variables ..... 352
Exercises ..... 353
12 Transformers ..... 357
12.1 Attention ..... 358
12.1.1 Transformer processing ..... 360
12.1.2 Attention coefficients ..... 361
12.1.3 Self-attention ..... 362
12.1.4 Network parameters ..... 363
12.1.5 Scaled self-attention ..... 366
12.1.6 Multi-head attention ..... 366
12.1.7 Transformer layers ..... 368
12.1.8 Computational complexity ..... 370
12.1.9 Positional encoding ..... 371
12.2 Natural Language ..... 374
12.2.1 Word embedding ..... 375
12.2.2 Tokenization ..... 377
12.2.3 Bag of words ..... 378
12.2.4 Autoregressive models ..... 379
12.2.5 Recurrent neural networks ..... 380
12.2.6 Backpropagation through time ..... 381
12.3 Transformer Language Models ..... 382
12.3.1 Decoder transformers ..... 383
12.3.2 Sampling strategies ..... 386
12.3.3 Encoder transformers ..... 388
12.3.4 Sequence-to-sequence transformers ..... 390
12.3.5 Large language models ..... 390
12.4 Multimodal Transformers ..... 394
12.4.1 Vision transformers ..... 395
12.4.2 Generative image transformers ..... 396
12.4.3 Audio data ..... 399
12.4.4 Text-to-speech ..... 400
12.4.5 Vision and language transformers ..... 402
Exercises ..... 403
13 Graph Neural Networks ..... 407
13.1 Machine Learning on Graphs ..... 409
13.1.1 Graph properties ..... 410
13.1.2 Adjacency matrix ..... 410
13.1.3 Permutation equivariance ..... 411
13.2 Neural Message-Passing ..... 412
13.2.1 Convolutional filters ..... 413
13.2.2 Graph convolutional networks ..... 414
13.2.3 Aggregation operators ..... 416
13.2.4 Update operators ..... 418
13.2.5 Node classification ..... 419
13.2.6 Edge classification ..... 420
13.2.7 Graph classification ..... 420
13.3 General Graph Networks ..... 420
13.3.1 Graph attention networks ..... 421
13.3.2 Edge embeddings ..... 421
13.3.3 Graph embeddings ..... 422
13.3.4 Over-smoothing ..... 422
13.3.5 Regularization ..... 423
13.3.6 Geometric deep learning ..... 424
Exercises ..... 425
14 Sampling ..... 429
14.1 Basic Sampling Algorithms ..... 430
14.1.1 Expectations ..... 430
14.1.2 Standard distributions ..... 431
14.1.3 Rejection sampling ..... 433
14.1.4 Adaptive rejection sampling ..... 435
14.1.5 Importance sampling ..... 437
14.1.6 Sampling-importance-resampling ..... 439
14.2 Markov Chain Monte Carlo ..... 440
14.2.1 The Metropolis algorithm ..... 441
14.2.2 Markov chains ..... 442
14.2.3 The Metropolis-Hastings algorithm ..... 445
14.2.4 Gibbs sampling ..... 446
14.2.5 Ancestral sampling ..... 450
14.3 Langevin Sampling ..... 451
14.3.1 Energy-based models ..... 452
14.3.2 Maximizing the likelihood ..... 453
14.3.3 Langevin dynamics ..... 454
Exercises ..... 456
15 Discrete Latent Variables ..... 459
15.1 $K$-means Clustering ..... 460
15.1.1 Image segmentation ..... 464
15.2 Mixtures of Gaussians ..... 466
15.2.1 Likelihood function ..... 468
15.2.2 Maximum likelihood ..... 470
15.3 Expectation-Maximization Algorithm ..... 474
15.3.1 Gaussian mixtures ..... 478
15.3.2 Relation to $K$-means ..... 480
15.3.3 Mixtures of Bernoulli distributions ..... 481
15.4 Evidence Lower Bound ..... 485
15.4.1 EM revisited ..... 486
15.4.2 Independent and identically distributed data ..... 488
15.4.3 Parameter priors ..... 489
15.4.4 Generalized EM ..... 489
15.4.5 Sequential EM ..... 490
Exercises ..... 490
16 Continuous Latent Variables ..... 495
16.1 Principal Component Analysis ..... 497
16.1.1 Maximum variance formulation ..... 497
16.1.2 Minimum-error formulation ..... 499
16.1.3 Data compression ..... 501
16.1.4 Data whitening ..... 502
16.1.5 High-dimensional data ..... 504
16.2 Probabilistic Latent Variables ..... 506
16.2.1 Generative model ..... 506
16.2.2 Likelihood function ..... 507
16.2.3 Maximum likelihood ..... 509
16.2.4 Factor analysis ..... 513
16.2.5 Independent component analysis ..... 514
16.2.6 Kalman filters ..... 515
16.3 Evidence Lower Bound ..... 516
16.3.1 Expectation maximization ..... 518
16.3.2 EM for PCA ..... 519
16.3.3 EM for factor analysis ..... 520
16.4 Nonlinear Latent Variable Models ..... 522
16.4.1 Nonlinear manifolds ..... 522
16.4.2 Likelihood function ..... 524
16.4.3 Discrete data ..... 526
16.4.4 Four approaches to generative modelling ..... 527
Exercises ..... 527
17 Generative Adversarial Networks ..... 533
17.1 Adversarial Training ..... 534
17.1.1 Loss function ..... 535
17.1.2 GAN training in practice ..... 536
17.2 Image GANs ..... 539
17.2.1 CycleGAN ..... 539
Exercises ..... 544
18 Normalizing Flows ..... 547
18.1 Coupling Flows ..... 549
18.2 Autoregressive Flows ..... 552
18.3 Continuous Flows ..... 554
18.3.1 Neural differential equations ..... 554
18.3.2 Neural ODE backpropagation ..... 555
18.3.3 Neural ODE flows ..... 557
Exercises ..... 559
19 Autoencoders ..... 563
19.1 Deterministic Autoencoders ..... 564
19.1.1 Linear autoencoders ..... 564
19.1.2 Deep autoencoders ..... 565
19.1.3 Sparse autoencoders ..... 566
19.1.4 Denoising autoencoders ..... 567
19.1.5 Masked autoencoders ..... 567
19.2 Variational Autoencoders ..... 569
19.2.1 Amortized inference ..... 572
19.2.2 The reparameterization trick ..... 574
Exercises ..... 578
20 Diffusion Models ..... 581
20.1 Forward Encoder ..... 582
20.1.1 Diffusion kernel ..... 583
20.1.2 Conditional distribution ..... 584
20.2 Reverse Decoder ..... 585
20.2.1 Training the decoder ..... 587
20.2.2 Evidence lower bound ..... 588
20.2.3 Rewriting the ELBO ..... 589
20.2.4 Predicting the noise ..... 591
20.2.5 Generating new samples ..... 592
20.3 Score Matching ..... 594
20.3.1 Score loss function ..... 595
20.3.2 Modified score loss ..... 596
20.3.3 Noise variance ..... 597
20.3.4 Stochastic differential equations ..... 598
20.4 Guided Diffusion ..... 599
20.4.1 Classifier guidance ..... 600
20.4.2 Classifier-free guidance ..... 600
Exercises ..... 603
Appendix A Linear Algebra ..... 609
A. 1 Matrix Identities ..... 609
A. 2 Traces and Determinants ..... 610
A. 3 Matrix Derivatives ..... 611
A. 4 Eigenvectors ..... 612
Appendix B Calculus of Variations ..... 617
Appendix C Lagrange Multipliers ..... 621
Bibliography ..... 625
Index ..... 641
![](https://cdn.mathpix.com/cropped/2025_10_01_d90a4d82c88904700833g-021.jpg?height=1244&width=1228&top_left_y=224&top_left_x=424)

**Image Description:** This image features a title slide from an academic lecture, prominently displaying "1 The Deep Learning Revolution" in large, bold font. The background consists of a colorful, abstract pattern resembling a blend of smooth, flowing lines and shapes, evoking a sense of dynamism and complexity. The use of bright colors like red, green, and yellow enhances visual interest, suggesting themes of innovation and exploration in deep learning. No diagram, equation, or strict technical content is present; it serves as an introductory visual for the topic.


Machine learning today is one of the most important, and fastest growing, fields of technology. Applications of machine learning are becoming ubiquitous, and solutions learned from data are increasingly displacing traditional hand-crafted algorithms. This has not only led to improved performance for existing technologies but has opened the door to a vast range of new capabilities that would be inconceivable if new algorithms had to be designed explicitly by hand.

One particular branch of machine learning, known as deep learning, has emerged as an exceptionally powerful and general-purpose framework for learning from data. Deep learning is based on computational models called neural networks which were originally inspired by mechanisms of learning and information processing in the human brain. The field of artificial intelligence, or AI, seeks to recreate the powerful capabilities of the brain in machines, and today the terms machine learning and AI are often used interchangeably. Many of the AI systems in current use represent ap-
plications of machine learning which are designed to solve very specific and focused problems, and while these are extremely useful they fall far short of the tremendous breadth of capabilities of the human brain. This has led to the introduction of the term artificial general intelligence, or AGI, to describe the aspiration of building machines with this much greater flexibility. After many decades of steady progress, machine learning has now entered a phase of very rapid development. Recently, massive deep learning systems called large language models have started to exhibit remarkable capabilities that have been described as the first indications of artificial general intelligence (Bubeck et al., 2023).

### 1.1. The Impact of Deep Learning

We begin our discussion of machine learning by considering four examples drawn from diverse fields to illustrate the huge breadth of applicability of this technology and to introduce some basic concepts and terminology. What is particularly remarkable about these and many other examples is that they have all been addressed using variants of the same fundamental framework of deep learning. This is in sharp contrast to conventional approaches in which different applications are tackled using widely differing and specialist techniques. It should be emphasized that the examples we have chosen represent only a tiny fraction of the breadth of applicability for deep neural networks and that almost every domain where computation has a role is amenable to the transformational impact of deep learning.

### 1.1.1 Medical diagnosis

Consider first the application of machine learning to the problem of diagnosing skin cancer. Melanoma is the most dangerous kind of skin cancer but is curable if detected early. Figure 1.1 shows example images of skin lesions, with malignant melanomas on the top row and benign nevi on the bottom row. Distinguishing between these two classes of image is clearly very challenging, and it would be virtually impossible to write an algorithm by hand that could successfully classify such images with any reasonable level of accuracy.

This problem has been successfully addressed using deep learning (Esteva et al., 2017). The solution was created using a large set of lesion images, known as

Figure 1.1 Examples of skin lesions corresponding to dangerous malignant melanomas on the top row and benign nevi on the bottom row. It is difficult for the untrained eye to distinguish between these two classes.
![](https://cdn.mathpix.com/cropped/2025_10_01_d90a4d82c88904700833g-022.jpg?height=394&width=768&top_left_y=1714&top_left_x=879)

**Image Description:** This image consists of a grid of eight skin lesion photographs, organized into two rows and four columns. Each lesion displays varying shapes, sizes, and colors, ranging from brown to dark red. These images are likely used for educational purposes in dermatology, illustrating different types of skin lesions. The characteristics, like borders, asymmetry, and pigmentation, are visually compared for diagnostic training. The image is intended for recognition and analysis of dermatological conditions.


a training set, each of which is labelled as either malignant or benign, where the labels are obtained from a biopsy test that is considered to provide the true class of the lesion. The training set is used to determine the values of some 25 million adjustable parameters, known as weights, in a deep neural network. This process of setting the parameter values from data is known as learning or training. The goal is for the trained network to predict the correct label for a new lesion just from the image alone without needing the time-consuming step of taking a biopsy. This is an example of a supervised learning problem because, for each training example, the network is told the correct label. It is also an example of a classification problem because each input must be assigned to a discrete set of classes (benign or malignant in this case). Applications in which the output consists of one or more continuous variables are called regression problems. An example of a regression problem would be the prediction of the yield in a chemical manufacturing process in which the inputs consist of the temperature, the pressure, and the concentrations of reactants.

An interesting aspect of this application is that the number of labelled training images available, roughly 129,000 , is considered relatively small, and so the deep neural network was first trained on a much larger data set of 1.28 million images of everyday objects (such as dogs, buildings, and mushrooms) and then fine-tuned on the data set of lesion images. This is an example of transfer learning in which the network learns the general properties of natural images from the large data set of everyday objects and is then specialized to the specific problem of lesion classification. Through the use of deep learning, the classification of skin lesion images has reached a level of accuracy that exceeds that of professional dermatologists (Brinker et al., 2019).

### 1.1.2 Protein structure

Proteins are sometimes called the building blocks of living organisms. They are biological molecules that consist of one or more long chains of units called amino acids, of which there are 22 different types, and the protein is specified by the sequence of amino acids. Once a protein has been synthesized inside a living cell, it folds into a complex three-dimensional structure whose behaviour and interactions are strongly determined by its shape. Calculating this 3D structure, given the amino acid sequence, has been a fundamental open problem in biology for half a century that had seen relatively little progress until the advent of deep learning.

The 3D structure can be measured experimentally using techniques such as Xray crystallography, cryogenic electron microscopy, or nuclear magnetic resonance spectroscopy. However, this can be extremely time-consuming and for some proteins can prove to be challenging, for example due to the difficulty of obtaining a pure sample or because the structure is dependent on the context. In contrast, the amino acid sequence of a protein can be determined experimentally at lower cost and higher throughput. Consequently, there is considerable interest in being able to predict the 3D structures of proteins directly from their amino acid sequences in order to better understand biological processes or for practical applications such as drug discovery. A deep learning model can be trained to take an amino acid sequence as input and generate the 3D structure as output, in which the training data

Figure 1.2 Illustration of the 3D shape of a protein called T1044/6VR4. The green structure shows the ground truth as determined by X-ray crystallography, whereas the superimposed blue structure shows the prediction obtained by a deep learning model called AlphaFold. [From Jumper et al. (2021) with permission.]
![](https://cdn.mathpix.com/cropped/2025_10_01_d90a4d82c88904700833g-024.jpg?height=641&width=644&top_left_y=224&top_left_x=1008)

**Image Description:** The image depicts a 3D representation of a protein structure, showcasing its primary, secondary, and tertiary structures. The protein is illustrated using a ribbon diagram, with helices colored in green and blue. The color differentiation likely indicates two conformations or a comparison between the wild type and a mutant variant. The structure emphasizes the protein's complex folding patterns, highlighting the spatial arrangement of amino acids. No axes are present, as it is a molecular visualization rather than a graph. This image is relevant for studies related to protein structure-function relationships.

consist of a set of proteins for which the amino acid sequence and the 3D structure are both known. Protein structure prediction is therefore another example of supervised learning. Once the system is trained it can take a new amino acid sequence as input and can predict the associated 3D structure (Jumper et al., 2021). Figure 1.2 compares the predicted 3D structure of a protein and the ground truth obtained by X-ray crystallography.

### 1.1.3 Image synthesis

In the two applications discussed so far, a neural network learned to transform an input (a skin image or an amino acid sequence) into an output (a lesion classification or a 3D protein structure, respectively). We turn now to an example where the training data consist simply of a set of sample images and the goal of the trained network is to create new images of the same kind. This is an example of unsupervised learning because the images are unlabelled, in contrast to the lesion classification and protein structure examples. Figure 1.3 shows examples of synthetic images generated by a deep neural network trained on a set of images of human faces taken in a studio against a plain background. Such synthetic images are of exceptionally high quality and it can be difficult tell them apart from photographs of real people.

This is an example of a generative model because it can generate new output examples that differ from those used to train the model but which share the same statistical properties. A variant of this approach allows images to be generated that depend on an input text string known, as a prompt, so that the image content reflects the semantics of the text input. The term generative $A I$ is used to describe deep learning models that generate outputs in the form of images, video, audio, text, candidate drug molecules, or other modalities.

![](https://cdn.mathpix.com/cropped/2025_10_01_d90a4d82c88904700833g-025.jpg?height=766&width=1521&top_left_y=229&top_left_x=148)

**Image Description:** This is a grid of eight portraits of diverse individuals arranged in two rows and four columns. Each portrait presents a close-up view of the person's face, capturing various expressions and features. The backgrounds are neutral, enhancing the focus on facial details. The image showcases diversity in gender, ethnicity, and age among the subjects. The lighting is soft and evenly distributed, highlighting the contours of the faces. This type of image may be used in studies involving facial recognition, social psychology, or diversity representation.

Figure 1.3 Synthetic face images generated by a deep neural network trained using unsupervised learning. [From https://generated.photos.]

### 1.1.4 Large language models

One of most important advances in machine learning in recent years has been the development of powerful models for processing natural language and other forms of sequential data such as source code. A large language model, or LLM, uses deep learning to build rich internal representations that capture the semantic properties of language. An important class of large language models, called autoregressive language models, can generate language as output, and therefore, they are a form of generative AI. Such models take a sequence of words as the input and for the output, generate a single word that represents the next word in the sequence. The augmented sequence, with the new word appended at the end, can then be fed through the model again to generate the subsequent word, and this process can be repeated to generate a long sequence of words. Such models can also output a special 'stop' word that signals the end of text generation, thereby allowing them to output text of finite length and then halt. At that point, a user could append their own series of words to the sequence before feeding the complete sequence back through the model to trigger further word generation. In this way, it is possible for a human to have a conversation with the neural network.

Such models can be trained on large data sets of text by extracting training pairs each consisting of a randomly selected sequence of words as input with the known next word as the target output. This is an example of self-supervised learning in which a function from inputs to outputs is learned but where the labelled outputs are obtained automatically from the input training data without needing separate human-

Figure 1.4 Plot of a training data set of $N=$ 10 points, shown as blue circles, each comprising an observation of the input variable $x$ along with the corresponding target variable $t$. The green curve shows the function $\sin (2 \pi x)$ used to generate the data. Our goal is to predict the value of $t$ for some new value of $x$, without knowledge of the green curve.
![](https://cdn.mathpix.com/cropped/2025_10_01_d90a4d82c88904700833g-026.jpg?height=429&width=697&top_left_y=219&top_left_x=959)

**Image Description:** The image is a graph depicting a sinusoidal function representing a relationship between the variables \( x \) and \( t \). The x-axis ranges from 0 to 1, while the y-axis spans from -1 to 1. A green curve illustrates the continuous sinusoidal function, with blue dots representing discrete data points that appear to align with the curve. The visual emphasizes the correspondence of the observed data to the modeled function, suggesting a fit or trend analysis in a statistical context.


derived labels. Since large volumes of text are available from multiple sources, this approach allows for scaling to very large training sets and associated very large neural networks.

Large language models can exhibit extraordinary capabilities that have been described as the first indications of emerging artificial general intelligence (Bubeck et al., 2023), and we discuss such models at length later in the book. On the next page, we give an illustration of language generation, based on a model called GPT-4 (OpenAI, 2023), in response to an input prompt 'Write a proof of the fact that there are infinitely many primes; do it in the style of a Shakespeare play through a dialogue between two parties arguing over the proof.' .

### 1.2. A Tutorial Example

For the newcomer to the field of machine learning, many of the basic concepts and much of the terminology can be introduced in the context of a simple example involving the fitting of a polynomial to a small synthetic data set (Bishop, 2006). This is a form of supervised learning problem in which we would like to make a prediction for a target variable, given the value of an input variable.

### 1.2.1 Synthetic data

We denote the input variable by $x$ and the target variable by $t$, and we assume that both variables take continuous values on the real axis. Suppose that we are given a training set comprising $N$ observations of $x$, written $x_{1}, \ldots, x_{N}$, together with corresponding observations of the values of $t$, denoted $t_{1}, \ldots, t_{N}$. Our goal is to predict the value of $t$ for some new value of $x$. The ability to make accurate predictions on previously unseen inputs is a key goal in machine learning and is known as generalization.

We can illustrate this using a synthetic data set generated by sampling from a sinusoidal function. Figure 1.4 shows a plot of a training set comprising $N=10$ data points in which the input values were generated by choosing values of $x_{n}$, for $n= 1, \ldots, N$, spaced uniformly in the range [ 0,1 ]. The associated target data values were obtained by first computing the values of the function $\sin (2 \pi x)$ for each value of $x$

## GPT-4: Proof of infinitely many primes

FIRST PARTY:
Thou dost not understand, my friend, There simply is no upper end To primes, those numbers indivisible, No matter if it seems impossible.

SECOND PARTY:
But how dost thou so surely know?
To prove this truth, what canst thou show?

FIRST PARTY:
Consider this, my doubtful peer, A clever proof that will make clear: Assume that there's a final prime, The largest one we'll see in time. Let's call this number "P" for short, And then, my friend, we will purport To build a number "Q" that's new, One larger than "P" and prime too.

SECOND PARTY:
But how, my friend, will this be done?
The proof is not yet fairly won.

FIRST PARTY:
It's simple, friend, just wait and see, We'll multiply the primes, that's key. All primes from two to "P" combined, Will make this new "Q" we will find. But one thing more we must include, To make the proof more watertight, We'll add a one onto the end, And now the proof is in our sight.

SECOND PARTY:
I see it now, it's crystal clear, For "Q" must be a prime, my dear.
It can't divide by any prime, All smaller than itself in time.
And since we chose "P" as the last,
This proves that our assumption's past.
There is no final prime, we see,
The numbers go on endlessly.

## Section 2.3

and then adding a small level of random noise (governed by a Gaussian distribution) to each such point to obtain the corresponding target value $t_{n}$. By generating data in this way, we are capturing an important property of many real-world data sets, namely that they possess an underlying regularity, which we wish to learn, but that individual observations are corrupted by random noise. This noise might arise from intrinsically stochastic (i.e., random) processes such as radioactive decay but more typically is due to there being sources of variability that are themselves unobserved.

In this tutorial example we know the true process that generated the data, namely the sinusoidal function. In a practical application of machine learning, our goal is to discover the underlying trends in the data given the finite training set. Knowing the process that generated the data, however, allows us to illustrate important concepts in machine learning.

### 1.2.2 Linear models

Our goal is to exploit this training set to predict the value $\hat{t}$ of the target variable for some new value $\widehat{x}$ of the input variable. As we will see later, this involves implicitly trying to discover the underlying function $\sin (2 \pi x)$. This is intrinsically a difficult problem as we have to generalize from a finite data set to an entire function. Furthermore, the observed data is corrupted with noise, and so for a given $\widehat{x}$ there is uncertainty as to the appropriate value for $\widehat{t}$. Probability theory provides a framework for expressing such uncertainty in a precise and quantitative manner, whereas decision theory allows us to exploit this probabilistic representation to make predictions that are optimal according to appropriate criteria. Learning probabilities from data lies at the heart of machine learning and will be explored in great detail in this book.

To start with, however, we will proceed rather informally and consider a simple approach based on curve fitting. In particular, we will fit the data using a polynomial function of the form

$$
y(x, \mathbf{w})=w_{0}+w_{1} x+w_{2} x^{2}+\ldots+w_{M} x^{M}=\sum_{j=0}^{M} w_{j} x^{j}
$$
where $M$ is the order of the polynomial, and $x^{j}$ denotes $x$ raised to the power of $j$. The polynomial coefficients $w_{0}, \ldots, w_{M}$ are collectively denoted by the vector $\mathbf{w}$. Note that, although the polynomial function $y(x, \mathbf{w})$ is a nonlinear function of $x$, it is a linear function of the coefficients $\mathbf{w}$. Functions, such as this polynomial, that are linear in the unknown parameters have important properties, as well as significant limitations, and are called linear models.

### 1.2.3 Error function

The values of the coefficients will be determined by fitting the polynomial to the training data. This can be done by minimizing an error function that measures the misfit between the function $y(x, \mathbf{w})$, for any given value of $\mathbf{w}$, and the training set data points. One simple choice of error function, which is widely used, is the sum of

Figure 1.5 The error function (1.2) corresponds to (one half of) the sum of the squares of the displacements (shown by the vertical green arrows) of each data point from the function $y(x, \mathbf{w})$.
![](https://cdn.mathpix.com/cropped/2025_10_01_d90a4d82c88904700833g-029.jpg?height=611&width=768&top_left_y=222&top_left_x=886)

**Image Description:** The image is a diagram depicting a curve in a Cartesian coordinate system. The x-axis represents the variable \(x_n\), while the y-axis, labeled as \(t\), indicates the dependent variable. A red curve illustrates a function \(y(x_n, w)\). Green vertical arrows connect specific points on the curve to corresponding blue points on the \(x\)-axis, representing the relationship between \(x_n\) and \(t\). The points at which these connections are drawn are marked by green circles on the curve and blue circles on the \(x\)-axis, highlighting specific data points.


the squares of the differences between the predictions $y\left(x_{n}, \mathbf{w}\right)$ for each data point $x_{n}$ and the corresponding target value $t_{n}$, given by

$$
E(\mathbf{w})=\frac{1}{2} \sum_{n=1}^{N}\left\{y\left(x_{n}, \mathbf{w}\right)-t_{n}\right\}^{2}
$$

Section 2.3.4

## Exercise 4.1

where the factor of $1 / 2$ is included for later convenience. We will later derive this error function starting from probability theory. Here we simply note that it is a nonnegative quantity that would be zero if, and only if, the function $y(x, \mathbf{w})$ were to pass exactly through each training data point. The geometrical interpretation of the sum-of-squares error function is illustrated in Figure 1.5.

We can solve the curve fitting problem by choosing the value of $\mathbf{w}$ for which $E(\mathbf{w})$ is as small as possible. Because the error function is a quadratic function of the coefficients $\mathbf{w}$, its derivatives with respect to the coefficients will be linear in the elements of $\mathbf{w}$, and so the minimization of the error function has a unique solution, denoted by $\mathbf{w}^{\star}$, which can be found in closed form. The resulting polynomial is given by the function $y\left(x, \mathbf{w}^{\star}\right)$.

### 1.2.4 Model complexity

There remains the problem of choosing the order $M$ of the polynomial, and as we will see this will turn out to be an example of an important concept called model comparison or model selection. In Figure 1.6, we show four examples of the results of fitting polynomials having orders $M=0,1,3$, and 9 to the data set shown in Figure 1.4.

Notice that the constant ( $M=0$ ) and first-order ( $M=1$ ) polynomials give poor fits to the data and consequently poor representations of the function $\sin (2 \pi x)$. The third-order ( $M=3$ ) polynomial seems to give the best fit to the function $\sin (2 \pi x)$ of the examples shown in Figure 1.6. When we go to a much higher order polynomial ( $M=9$ ), we obtain an excellent fit to the training data. In fact, the polynomial

![](https://cdn.mathpix.com/cropped/2025_10_01_d90a4d82c88904700833g-030.jpg?height=973&width=1530&top_left_y=205&top_left_x=136)

**Image Description:** The image consists of four subplots arranged in a 2x2 grid, each depicting a graphical representation of a mathematical function over the interval \( x \) from 0 to 1. The y-axis represents the variable \( t \) ranging from -1 to 1. Each subplot features blue data points and a corresponding green curve, with a red line indicating a fitted model. The subtitles indicate different values for a parameter \( M \) (0, 1, 3, and 9), suggesting varying complexity of the model as \( M \) increases. The diagrams illustrate the impact of parameter changes on the function's behavior.

Figure 1.6 Plots of polynomials having various orders $M$, shown as red curves, fitted to the data set shown in Figure 1.4 by minimizing the error function (1.2).

passes exactly through each data point and $E\left(\mathbf{w}^{\star}\right)=0$. However, the fitted curve oscillates wildly and gives a very poor representation of the function $\sin (2 \pi x)$. This latter behaviour is known as over-fitting.

Our goal is to achieve good generalization by making accurate predictions for new data. We can obtain some quantitative insight into the dependence of the generalization performance on $M$ by considering a separate set of data known as a test set, comprising 100 data points generated using the same procedure as used to generate the training set points. For each value of $M$, we can evaluate the residual value of $E\left(\mathbf{w}^{\star}\right)$ given by (1.2) for the training data, and we can also evaluate $E\left(\mathbf{w}^{\star}\right)$ for the test data set. Instead of evaluating the error function $E(\mathbf{w})$, it is sometimes more convenient to use the root-mean-square (RMS) error defined by

$$
E_{\mathrm{RMS}}=\sqrt{\frac{1}{N} \sum_{n=1}^{N}\left\{y\left(x_{n}, \mathbf{w}\right)-t_{n}\right\}^{2}}
$$
in which the division by $N$ allows us to compare different sizes of data sets on an equal footing, and the square root ensures that $E_{\text {RMS }}$ is measured on the same scale (and in the same units) as the target variable $t$. Graphs of the training-set and test-set

Figure 1.7 Graphs of the root-meansquare error, defined by (1.3), evaluated on the training set, and on an independent test set, for various values of $M$.
![](https://cdn.mathpix.com/cropped/2025_10_01_d90a4d82c88904700833g-031.jpg?height=426&width=861&top_left_y=222&top_left_x=791)

**Image Description:** The slide features a plot with two distinct datasets represented by colored points: red circles indicate "Training" data, while blue circles indicate "Test" data. The x-axis is labeled "M" and ranges from 0 to 9. The y-axis is labeled "ERMS" and spans from 0 to 1. The plot illustrates the relationship between the variable "M" and the performance metrics (ERMS) for both datasets, showing points for various values of M and indicating trends in model performance. A legend identifies the color coding for training and test datasets.


RMS errors are shown, for various values of $M$, in Figure 1.7. The test set error is a measure of how well we are doing in predicting the values of $t$ for new data observations of $x$. Note from Figure 1.7 that small values of $M$ give relatively large values of the test set error, and this can be attributed to the fact that the corresponding polynomials are rather inflexible and are incapable of capturing the oscillations in the function $\sin (2 \pi x)$. Values of $M$ in the range $3 \leqslant M \leqslant 8$ give small values for the test set error, and these also give reasonable representations of the generating function $\sin (2 \pi x)$, as can be seen for $M=3$ in Figure 1.6.

For $M=9$, the training set error goes to zero, as we might expect because this polynomial contains 10 degrees of freedom corresponding to the 10 coefficients $w_{0}, \ldots, w_{9}$, and so can be tuned exactly to the 10 data points in the training set. However, the test set error has become very large and, as we saw in Figure 1.6, the corresponding function $y\left(x, \mathbf{w}^{\star}\right)$ exhibits wild oscillations.

This may seem paradoxical because a polynomial of a given order contains all lower-order polynomials as special cases. The $M=9$ polynomial is therefore capable of generating results at least as good as the $M=3$ polynomial. Furthermore, we might suppose that the best predictor of new data would be the function $\sin (2 \pi x)$ from which the data was generated (and we will see later that this is indeed the case). We know that a power series expansion of the function $\sin (2 \pi x)$ contains terms of all orders, so we might expect that results should improve monotonically as we increase $M$.

We can gain some insight into the problem by examining the values of the coefficients $\mathbf{w}^{\star}$ obtained from polynomials of various orders, as shown in Table 1.1. We see that, as $M$ increases, the magnitude of the coefficients typically gets larger. In particular for the $M=9$ polynomial, the coefficients have become finely tuned to the data. They have large positive and negative values so that the corresponding polynomial function matches each of the data points exactly, but between data points (particularly near the ends of the range) the function exhibits the large oscillations observed in Figure 1.6. Intuitively, what is happening is that the more flexible polynomials with larger values of $M$ are increasingly tuned to the random noise on the target values.

Further insight into this phenomenon can be gained by examining the behaviour of the learned model as the size of the data set is varied, as shown in Figure 1.8. We see that, for a given model complexity, the over-fitting problem become less severe

![](https://cdn.mathpix.com/cropped/2025_10_01_d90a4d82c88904700833g-032.jpg?height=453&width=1525&top_left_y=205&top_left_x=141)

**Image Description:** The image features two plots side by side, representing the results of regression analyses with varying sample sizes (N). The left plot, labeled "N = 15," displays a smooth curve in red, overlaid on a green curve, with blue dots representing sample points that are distributed along the x-axis (0 to 1) and t-axis (-1 to 1). The right plot, labeled "N = 100," shows a similar regression trend with much denser blue points, reflecting a larger sample size. Both graphs illustrate the effect of sample size on the variability and fit of the regression curves.

Figure 1.8 Plots of the solutions obtained by minimizing the sum-of-squares error function (1.2) using the $M=9$ polynomial for $N=15$ data points (left plot) and $N=100$ data points (right plot). We see that increasing the size of the data set reduces the over-fitting problem.

as the size of the data set increases. Another way to say this is that with a larger data set, we can afford to fit a more complex (in other words more flexible) model to the data. One rough heuristic that is sometimes advocated in classical statistics is that the number of data points should be no less than some multiple (say 5 or 10) of the number of learnable parameters in the model. However, when we discuss deep learning later in this book, we will see that excellent results can be obtained using models that have significantly more parameters than the number of training

Section 9.3.2 data points.

### 1.2.5 Regularization

There is something rather unsatisfying about having to limit the number of parameters in a model according to the size of the available training set. It would seem more reasonable to choose the complexity of the model according to the complexity of the problem being solved. One technique that is often used to control the overfitting phenomenon, as an alternative to limiting the number of parameters, is that of regularization, which involves adding a penalty term to the error function (1.2) to discourage the coefficients from having large magnitudes. The simplest such penalty

Table 1.1 Table of the coefficients $\mathbf{w}^{\star}$ for polynomials of various order. Observe how the typical magnitude of the coefficients increases dramatically as the order of the polynomial increases.
|  | $M=0$ | $M=1$ | $M=3$ | $M=9$ |
| :---: | ---: | ---: | ---: | ---: |
| $w_{0}^{\star}$ | 0.11 | 0.90 | 0.12 | 0.26 |
| $w_{1}^{\star}$ |  | -1.58 | 11.20 | -66.13 |
| $w_{2}^{\star}$ |  |  | -33.67 | $1,665.69$ |
| $w_{3}^{\star}$ |  |  | 22.43 | $-15,566.61$ |
| $w_{4}^{\star}$ |  |  |  | $76,321.23$ |
| $w_{5}^{\star}$ |  |  |  | $-217,389.15$ |
| $w_{6}^{\star}$ |  |  |  | $370,626.48$ |
| $w_{7}^{\star}$ |  |  |  | $-372,051.47$ |
| $w_{8}^{\star}$ |  |  |  | $202,540.70$ |
| $w_{9}^{\star}$ |  |  |  | $-46,080.94$ |


![](https://cdn.mathpix.com/cropped/2025_10_01_d90a4d82c88904700833g-033.jpg?height=445&width=1510&top_left_y=210&top_left_x=146)

**Image Description:** The image consists of two plots. The left plot displays a curve representing the function \( \ln \lambda = -18 \) with a green line and data points represented by blue dots. The x-axis ranges from 0 to 1, while the y-axis ranges from -1 to 1. The right plot shows a different function for \( \ln \lambda = 0 \), with a red curve and similar blue data points. Both plots share the same axes settings. The green and red curves illustrate different behaviors of the function under the specified conditions in terms of oscillation and trends.

Figure 1.9 Plots of $M=9$ polynomials fitted to the data set shown in Figure 1.4 using the regularized error function (1.4) for two values of the regularization parameter $\lambda$ corresponding to $\ln \lambda=-18$ and $\ln \lambda=0$. The case of no regularizer, i.e., $\lambda=0$, corresponding to $\ln \lambda=-\infty$, is shown at the bottom right of Figure 1.6.

term takes the form of the sum of the squares of all of the coefficients, leading to a modified error function of the form

$$
\widetilde{E}(\mathbf{w})=\frac{1}{2} \sum_{n=1}^{N}\left\{y\left(x_{n}, \mathbf{w}\right)-t_{n}\right\}^{2}+\frac{\lambda}{2}\|\mathbf{w}\|^{2}
$$
where $\|\mathbf{w}\|^{2} \equiv \mathbf{w}^{\mathrm{T}} \mathbf{w}=w_{0}^{2}+w_{1}^{2}+\ldots+w_{M}^{2}$, and the coefficient $\lambda$ governs the relative importance of the regularization term compared with the sum-of-squares error term. Note that often the coefficient $w_{0}$ is omitted from the regularizer because its inclusion causes the results to depend on the choice of origin for the target variable (Hastie, Tibshirani, and Friedman, 2009), or it may be included but with its own

Section 9.2.1 Exercise 4.2 regularization coefficient. Again, the error function in (1.4) can be minimized exactly in closed form. Techniques such as this are known in the statistics literature as shrinkage methods because they reduce the value of the coefficients. In the context of neural networks, this approach is known as weight decay because the parameters in a neural network are called weights and this regularizer encourages them to decay towards zero.

Figure 1.9 shows the results of fitting the polynomial of order $M=9$ to the same data set as before but now using the regularized error function given by (1.4). We see that, for a value of $\ln \lambda=-18$, the over-fitting has been suppressed and we now obtain a much closer representation of the underlying function $\sin (2 \pi x)$. If, however, we use too large a value for $\lambda$ then we again obtain a poor fit, as shown in Figure 1.9 for $\ln \lambda=0$. The corresponding coefficients from the fitted polynomials are given in Table 1.2, showing that regularization has the desired effect of reducing the magnitude of the coefficients.

The impact of the regularization term on the generalization error can be seen by plotting the value of the RMS error (1.3) for both training and test sets against $\ln \lambda$, as shown in Figure 1.10. We see that $\lambda$ now controls the effective complexity of the model and hence determines the degree of over-fitting.

Figure 1.10 Graph of the root-meansquare error (1.3) versus $\ln \lambda$ for the $M=9$ polynomial.
![](https://cdn.mathpix.com/cropped/2025_10_01_d90a4d82c88904700833g-034.jpg?height=431&width=868&top_left_y=219&top_left_x=786)

**Image Description:** The diagram is a line graph displaying the relationship between two variables: $M$ on the x-axis (ranging from -30 to 0) and $ERRMS$ on the y-axis (ranging from 0 to 0.5). There are two curves: a red line representing "Training" data and a blue line representing "Test" data. The graph shows that as $M$ increases, the Training ERRMS decreases, while the Test ERRMS initially decreases then levels off, indicating a potential gap between training and test performance. A legend identifies the curves’ colors and meanings.


### 1.2.6 Model selection

The quantity $\lambda$ is an example of a hyperparameter whose values are fixed during the minimization of the error function to determine the model parameters $\mathbf{w}$. We cannot simply determine the value of $\lambda$ by minimizing the error function jointly with respect to $\mathbf{w}$ and $\lambda$ since this will lead to $\lambda \rightarrow 0$ and an over-fitted model with small or zero training error. Similarly, the order $M$ of the polynomial is a hyperparameter of the model, and simply optimizing the training set error with respect to $M$ will lead to large values of $M$ and associated over-fitting. We therefore need to find a way to determine suitable values for hyperparameters. The results above suggest a simple way of achieving this, namely by taking the available data and partitioning it into a training set, used to determine the coefficients $\mathbf{w}$, and a separate validation set, also called a hold-out set or a development set. We then select the model having the lowest error on the validation set. If the model design is iterated many times using a data set of limited size, then some over-fitting to the validation data can occur, and so it may be necessary to keep aside a third test set on which the performance of the selected model can finally be evaluated.

For some applications, the supply of data for training and testing will be limited. To build a good model, we should use as much of the available data as possible for training. However, if the validation set is too small, it will give a relatively noisy estimate of predictive performance. One solution to this dilemma is to use cross-

Table 1.2 Table of the coefficients $\mathbf{w}^{\star}$ for $M=9$ polynomials with various values for the regularization parameter $\lambda$. Note that $\ln \lambda=-\infty$ corresponds to a model with no regularization, i.e., to the graph at the bottom right in Figure 1.6. We see that, as the value of $\lambda$ increases, the magnitude of a typical coefficient gets smaller.
|  | $\ln \lambda=-\infty$ | $\ln \lambda=-18$ | $\ln \lambda=0$ |
| :---: | ---: | ---: | ---: |
| $w_{0}^{\star}$ | 0.26 | 0.26 | 0.11 |
| $w_{1}^{\star}$ | -66.13 | 0.64 | -0.07 |
| $w_{2}^{\star}$ | $1,665.69$ | 43.68 | -0.09 |
| $w_{3}^{\star}$ | $-15,566.61$ | -144.00 | -0.07 |
| $w_{4}^{\star}$ | $76,321.23$ | 57.90 | -0.05 |
| $w_{5}^{\star}$ | $-217,389.15$ | 117.36 | -0.04 |
| $w_{6}^{\star}$ | $370,626.48$ | 9.87 | -0.02 |
| $w_{7}^{\star}$ | $-372,051.47$ | -90.02 | -0.01 |
| $w_{8}^{\star}$ | $202,540.70$ | -70.90 | -0.01 |
| $w_{9}^{\star}$ | $-46,080.94$ | 75.26 | 0.00 |


Figure 1.11 The technique of $S$-fold cross-validation, illustrated here for the case of $S=4$, involves taking the available data and partitioning it into $S$ groups of equal size. Then $S-1$ of the groups are used to train a set of models that are then evaluated on the remaining group. This procedure is then repeated for all $S$ possible choices for the held-out group, indicated here by the red blocks, and the performance scores from the $S$ runs are then averaged.
![](https://cdn.mathpix.com/cropped/2025_10_01_d90a4d82c88904700833g-035.jpg?height=304&width=526&top_left_y=219&top_left_x=1128)

**Image Description:** The image presents a series of four horizontal bars, each representing "runs" labeled from 1 to 4. Each bar consists of five equally sized segments, with one or more segments highlighted in red, indicating a specific condition or event occurring within that run. The arrangement allows for quick visual comparison of the highlighted segments across the runs, emphasizing variations in data points or outcomes among the runs. The two-dimensional layout effectively illustrates patterns or trends in the runs being analyzed.


validation, which is illustrated in Figure 1.11. This allows a proportion $(S-1) / S$ of the available data to be used for training while making use of all of the data to assess performance. When data is particularly scarce, it may be appropriate to consider the case $S=N$, where $N$ is the total number of data points, which gives the leave-oneout technique.

The main drawback of cross-validation is that the number of training runs that must be performed is increased by a factor of $S$, and this can prove problematic for models in which the training is itself computationally expensive. A further problem with techniques such as cross-validation that use separate data to assess performance is that we might have multiple complexity hyperparameters for a single model (for instance, there might be several regularization hyperparameters). Exploring combinations of settings for such hyperparameters could, in the worst case, require a number of training runs that is exponential in the number of hyperparameters. The state of the art in modern machine learning involves extremely large models, trained on commensurately large data sets. Consequently, there is limited scope for exploration of hyperparameter settings, and heavy reliance is placed on experience obtained with smaller models and on heuristics.

This simple example of fitting a polynomial to a synthetic data set generated from a sinusoidal function has illustrated many key ideas from machine learning, and we will make further use of this example in future chapters. However, realworld applications of machine learning differ in several important respects. The size of the data sets used for training can be many orders of magnitude larger, and there will generally be many more input variables, perhaps numbering in the millions for image analysis, for example, as well as multiple output variables. The learnable function that relates outputs to inputs is governed by a class of models known as neural networks, and these may have a large number of parameters perhaps numbering in the hundreds of billions, and the error function will be a highly nonlinear function of those parameters. The error function can no longer be minimized through a closed-form solution and instead must be minimized through iterative optimization techniques based on evaluation of the derivatives of the error function with respect to the parameters, all of which may require specialist computational hardware and incur substantial computational cost.

Figure 1.12 Schematic illustration showing two neurons from the human brain. These electrically active cells communicate through junctions called synapses whose strengths change as the network learns.
![](https://cdn.mathpix.com/cropped/2025_10_01_d90a4d82c88904700833g-036.jpg?height=477&width=880&top_left_y=239&top_left_x=747)

**Image Description:** The diagram illustrates two interconnected neurons, highlighting key anatomical features. Each neuron has a cell body (soma) represented as a circular shape. Dendrites, branching structures, extend from the cell body to receive signals. The axon, protruding from the cell body, transmits impulses away from the neuron. A synapse, indicated by a circle, illustrates the connection point between the two neurons where signal transmission occurs. Labels clearly identify the dendrites, axon, and synapse, enhancing clarity for educational purposes.


### 1.3. A Brief History of Machine Learning

Machine learning has a long and rich history, including the pursuit of multiple alternative approaches. Here we focus on the evolution of machine learning methods based on neural networks as these represent the foundation of deep learning and have proven to be the most effective approach to machine learning for real-world applications.

Neural network models were originally inspired by studies of information processing in the brains of humans and other mammals. The basic processing units in the brain are electrically active cells called neurons, as illustrated in Figure 1.12. When a neuron 'fires', it sends an electrical impulse down the axon where it reaches junctions, called synapses, which form connections with other neurons. Chemical signals called neurotransmitters are released at the synapses, and these can stimulate, or inhibit, the firing of subsequent neurons.

A human brain contains around 90 billion neurons in total, each of which has on average several thousand synapses with other neurons, creating a complex network having a total of around 100 trillion $\left(10^{14}\right)$ synapses. If a particular neuron receives sufficient stimulation from the firing of other neurons then it too can be induced to fire. However, some synapses have a negative, or inhibitory, effect whereby the firing of the input neuron makes it less likely that the output neuron will fire. The extent to which one neuron can cause another to fire depends on the strength of the synapse, and it is changes in these strengths that represents a key mechanism whereby the brain can store information and learn from experience.

These properties of neurons have been captured in very simple mathematical models, known as artificial neural networks, which then form the basis for computational approaches to learning (McCulloch and Pitts, 1943). Many of these models describe the properties of a single neuron by forming a linear combination of the outputs of other neurons, which is then transformed using a nonlinear function. This

Figure 1.13 A simple neural network diagram representing the transformations (1.5) and (1.6) describing a single neuron. The polynomial function (1.1) can be seen as a special case of this model.
![](https://cdn.mathpix.com/cropped/2025_10_01_d90a4d82c88904700833g-037.jpg?height=323&width=368&top_left_y=222&top_left_x=1284)

**Image Description:** The image is a diagram illustrating a simple neural network structure. It features nodes labeled \(x_1\), \(x_2\), ..., \(x_M\) representing input neurons, connected by directed arrows to an output neuron labeled \(y\). Each input node is associated with a weight denoted by \(w_1\), \(w_2\), ..., \(w_M\), indicated near the connecting arrows. The output node \(y\) serves as the final output, resulting from the weighted sum of inputs. The diagram is circular and uses purple circles for nodes with connections represented by black arrows.


can be expressed mathematically in the form

$$
\begin{aligned}
a & =\sum_{i=1}^{M} w_{i} x_{i} \\
y & =f(a)
\end{aligned}
$$

where $x_{1}, \ldots, x_{M}$ represent $M$ inputs corresponding to the activities of other neurons that send connections to this neuron, and $w_{1}, \ldots, w_{M}$ are continuous variables, called weights, which represent the strengths of the associated synapses. The quantity $a$ is called the pre-activation, the nonlinear function $f(\cdot)$ is called the activation function, and the output $y$ is called the activation. We can see that the polynomial (1.1) can be viewed as a specific instance of this representation in which the inputs $x_{i}$ are given by powers of a single variable $x$, and the function $f(\cdot)$ is just the identity $f(a)=a$. The simple mathematical formulation given by (1.5) and (1.6) has formed the basis of neural network models from the 1960s up to the present day, and can be represented in diagram form as shown in Figure 1.13.

### 1.3.1 Single-layer networks

The history of artificial neural networks can broadly be divided into three distinct phases according to the level of sophistication of the networks as measured by the number of 'layers' of processing. A simple neural model described by (1.5) and (1.6) can be viewed as having a single layer of processing corresponding to the single layer of connections in Figure 1.13. One of the most important such models in the history of neural computing is the perceptron (Rosenblatt, 1962) in which the activation function $f(\cdot)$ is a step function of the form

$$
f(a)= \begin{cases}0, & \text { if } a \leqslant 0, \\ 1, & \text { if } a>0 .\end{cases}
$$

This can be viewed as a simplified model of neural firing in which a neuron fires if, and only if, the total weighted input exceeds a threshold of 0 . The perceptron was pioneered by Rosenblatt (1962), who developed a specific training algorithm that has the interesting property that if there exists a set of weight values for which the perceptron can achieve perfect classification of its training data then the algorithm is guaranteed to find the solution in a finite number of steps (Bishop, 2006). As well as a learning algorithm, the perceptron also had a dedicated analogue hardware

![](https://cdn.mathpix.com/cropped/2025_10_01_d90a4d82c88904700833g-038.jpg?height=406&width=1554&top_left_y=210&top_left_x=102)

**Image Description:** The image consists of three historical photographs depicting early computer technology and engineering. 

1. The left image shows a man working on an early computing device with connected cables and labeled components, focusing on a spherical shape with the letter "C" displayed prominently.
2. The middle image captures a technician surrounded by a mass of tangled wires, possibly indicating an early wiring setup or troubleshooting work.
3. The right image features a man handling a modular component of a computing system, with rows of circuits and connections visible, highlighting the complexity of early computing hardware.

Figure 1.14 Illustration of the Mark 1 perceptron hardware. The photograph on the left shows how the inputs were obtained using a simple camera system in which an input scene, in this case a printed character, was illuminated by powerful lights, and an image focused onto a $20 \times 20$ array of cadmium sulphide photocells, giving a primitive 400-pixel image. The perceptron also had a patch board, shown in the middle photograph, which allowed different configurations of input features to be tried. Often these were wired up at random to demonstrate the ability of the perceptron to learn without the need for precise wiring, in contrast to a modern digital computer. The photograph on the right shows one of the racks of learnable weights. Each weight was implemented using a rotary variable resistor, also called a potentiometer, driven by an electric motor thereby allowing the value of the weight to be adjusted automatically by the learning algorithm.

implementation, as shown in Figure 1.14. A typical perceptron configuration had multiple layers of processing, but only one of those layers was learnable from data, and so the perceptron is considered to be a 'single-layer' neural network.

At first, the ability of perceptrons to learn from data in a brain-like way was considered remarkable. However, it became apparent that the model also has major limitations. The properties of perceptrons were analysed by Minsky and Papert (1969), who gave formal proofs of the limited capabilities of single-layer networks. Unfortunately, they also speculated that similar limitations would extend to networks having multiple layers of learnable parameters. Although this latter conjecture proved to be wildly incorrect, the effect was to dampen enthusiasm for neural network models, and this contributed to the lack of interest, and funding, for neural networks during the 1970s and early 1980s. Furthermore, researchers were unable to explore the properties of multilayered networks due to the lack of an effective algorithm for training them, since techniques such as the perceptron algorithm were specific to single-layer models. Note that although perceptrons have long disappeared from practical machine learning, the name lives on because a modern neural network is also sometimes called a multilayer perceptron or MLP.

### 1.3.2 Backpropagation

The solution to the problem of training neural networks having more than one layer of learnable parameters came from the use of differential calculus and the application of gradient-based optimization methods. An important change was to replace the step function (1.7) with continuous differentiable activation functions having a non-zero gradient. Another key modification was to introduce differentiable error functions that define how well a given choice of parameter values predicts the target variables in the training set. We saw an example of such an error function when we

Figure 1.15 A neural network having two layers of parameters in which arrows denote the direction of information flow through the network. Each of the hidden units and each of the output units computes a function of the form given by (1.5) and (1.6) in which the activation function $f(\cdot)$ is differentiable.
![](https://cdn.mathpix.com/cropped/2025_10_01_d90a4d82c88904700833g-039.jpg?height=497&width=663&top_left_y=224&top_left_x=981)

**Image Description:** The diagram illustrates a feedforward neural network architecture. It consists of three layers: input units at the bottom, hidden units in the center, and output units at the top. The input layer has several nodes, represented by circles, connecting to the hidden layer with arrows. The hidden layer, also comprised of circular nodes, connects to the output layer, which has fewer nodes. The arrows indicate directed connections denoting the flow of information. The absence of recurrent connections signifies a strictly feedforward structure, emphasizing a unidirectional information flow from inputs to outputs.


Section 1.2.3
used the sum-of-squares error function (1.2) to fit polynomials.
With these changes, we now have an error function whose derivatives with respect to each of the parameters in the network can be evaluated. We can now consider networks having more than one layer of parameters. Figure 1.15 shows a simple network with two processing layers. Nodes in the middle layer called hidden units because their values do not appear in the training set, which only provides values for inputs and outputs. Each of the hidden units and each of the output units in Figure 1.15 computes a function of the form given by (1.5) and (1.6). For a given set of input values, the states of all of the hidden and output units can be evaluated by repeated application of (1.5) and (1.6) in which information is flowing forward through the network in the direction of the arrows. For this reason, such models are sometimes also called feed-forward neural networks.

To train such a network the parameters are first initialized using a random number generator and are then iteratively updated using gradient-based optimization techniques. This involves evaluating the derivatives of the error function, which can be done efficiently in a process known as error backpropagation. In backpropagation, information flows backwards through the network from the outputs towards the inputs (Rumelhart, Hinton, and Williams, 1986). There exist many different optimization algorithms that make use of gradients of the function to be optimized, but the one that is most prevalent in machine learning is also the simplest and is known as stochastic gradient descent.

The ability to train neural networks having multiple layers of weights was a breakthrough that led to a resurgence of interest in the field starting around the mid1980s. This was also a period in which the field moved beyond a focus on neurobiological inspiration and developed a more rigorous and principled foundation (Bishop, 1995b). In particular, it was recognized that probability theory, and ideas from the field of statistics, play a central role in neural networks and machine learning. One key insight is that learning from data involves background assumptions, sometimes called prior knowledge or inductive biases. These might be incorporated explicitly, for example by designing the structure of a neural network such that the classification of a skin lesion does not depend on the location of the lesion within the image, or they might take the form of implicit assumptions that arise from the mathematical
form of the model or the way it is trained.
The development of backpropagation and gradient-based optimization dramatically increased the capability of neural networks to solve practical problems. However, it was also observed that in networks with many layers, it was only weights in the final two layers that would learn useful values. With a few exceptions, notably
Chapter 10 models used for image analysis known as convolutional neural networks (LeCun et al., 1998), there were very few successful applications of networks having more than two layers. Again, this constrained the complexity of the problems that could be addressed effectively with these kinds of network. To achieve reasonable performance on many applications, it was necessary to use hand-crafted pre-processing to transform the input variables into some new space where, it was hoped, the machine learning problem would be easier to solve. This pre-processing stage is sometimes also called feature extraction. Although this approach was sometimes effective, it would clearly be much better if features could be learned from the data rather than being hand-crafted.

By the start of the new millennium, the available neural network methods were once again reaching the limits of their capability. Researchers began to explore a raft of alternatives to neural networks, such as kernel methods, support vector machines, Gaussian processes, and many others. Neural networks fell into disfavour once again, although a core of enthusiastic researchers continued to pursue the goal of a truly effective approach to training networks with many layers.

### 1.3.3 Deep networks

The third, and current, phase in the development of neural networks began during the second decade of the 21st century. A series of developments allowed neural networks with many layers of weights to be trained effectively, thereby removing previous limitations on the capabilities of these techniques. Networks with many layers of weights are called deep neural networks and the sub-field of machine learning that focuses on such networks is called deep learning (LeCun, Bengio, and Hinton, 2015).

One important theme in the origins of deep learning was a significant increase in the scale of neural networks, measured in terms of the number of parameters. Although networks with a few hundred or a few thousand parameters were common in the 1980s, this steadily rose to the millions, and then billions, whereas current state-of-the-art models can have in the region of one trillion ( $10^{12}$ ) parameters. Networks with many parameters require commensurately large data sets so that the training signals can produced good values for those parameters. The combination of massive models and massive data sets in turn requires computation on a massive scale when training the model. Specialist processors called graphics processing units, or GPUs, which had been developed for very fast rendering of graphical data for applications such as video games, proved to be well suited to the training of neural networks because the functions computed by the units in one layer of a network can be evaluated in parallel, and this maps well onto the massive parallelism of GPUs (Krizhevsky, Sutskever, and Hinton, 2012). Today, training for the largest models is performed on large arrays of thousands of GPUs linked by specialist high-speed interconnections.

![](https://cdn.mathpix.com/cropped/2025_10_01_d90a4d82c88904700833g-041.jpg?height=975&width=1442&top_left_y=239&top_left_x=185)

**Image Description:** The image is a logarithmic scale graph that depicts computational performance over time, measured in petaflops per second (x-axis) against years (y-axis). The plot features a curve starting from 1960, highlighting significant milestones in AI advancements, including "Perceptron" and "AlphaGo Zero." There is a blue shaded area indicating the "Modern Era" from 2016 to 2020. Key points are marked with blue dots, each representing notable models such as VGG, ResNets, and Neural Machine Translation, showing the trend of increasing computational capabilities over time.

Figure 1.16 Plot of the number of compute cycles, measured in petaflop/s-days, needed to train a state-of-theart neural network as a function of date, showing two distinct phases of exponential growth. [From OpenAI with permission.]

Figure 1.16 illustrates how the number of compute cycles needed to train a state-of-the-art neural network has grown over the years, showing two distinct phases of growth. The vertical axis has an exponential scale and has units of petaflop/s-days, where a petaflop represents $10^{15}$ (a thousand trillion) floating point operations, and a petaflop/s is one petaflop per second. One petaflop/s-day represents computation at the rate of a petaflop/s for a period of 24 hours, which is roughly $10^{20}$ floating point operations, and therefore, the top line of the graph represents an impressive $10^{24}$ floating point operations. A straight line on the graph represents exponential growth, and we see that from the era of the perceptron up to around 2012, the doubling time was around 2 years, which is consistent with the general growth of computing power as a consequence of Moore's law. From 2012 onward, which marks the era of deep learning, we again see exponential growth but the doubling time is now 3.4 months corresponding to a factor of 10 increase in compute power every year!

It is often found that improvements in performance due to innovations in the architecture or incorporation of more sophisticated forms of inductive bias are soon

Section 12.3.5

Section 10.3

Section 9.5
superseded simply by scaling up the quantity of training data, along with commensurate scaling of the model size and associated compute power used for training (Sutton, 2019). Not only can large models have superior performance on a specific task but they may be capable of solving a broader range of different problems with the same trained neural network. Large language models are a notable example as a single network not only has an extraordinary breadth of capability but is even able to outperform specialist networks designed to solve specific problems.

We have seen that depth plays an important role in allowing neural networks to achieve high performance. One way to view the role of the hidden layers in a deep neural network is that of representation learning (Bengio, Courville, and Vincent, 2012) in which the network learns to transform input data into a new representation that is semantically meaningful thereby creating a much easier problem for the final layer or layers to solve. Such internal representations can be repurposed to allow for the solution of related problems through transfer learning, as we saw for skin lesion classification. It is interesting to note that neural networks used to process images may learn internal representations that are remarkably like those observed in the mammalian visual cortex. Large neural networks that can be adapted or fine-tuned to a range of downstream tasks are called foundation models, and can take advantage of large, heterogeneous data sets to create models having broad applicability (Bommasani et al., 2021).

In addition to scaling, there were other developments that helped in the success of deep learning. For example, in simple neural networks, the training signals become weaker as they are backpropagated through successive layers of a deep network. One technique for addressing this is the introduction of residual connections (He et al., 2015a) that facilitate the training of networks having hundreds of layers. Another key development was the introduction of automatic differentiation methods in which the code that performs backpropagation to evaluate error function gradients is generated automatically from the code used to specify the forward propagation. This allows researchers to experiment rapidly with different architectures for a neural network and to combine different architectural elements in multiple ways very easily since only the relatively simple forward propagation functions need to be coded explicitly. Also, much of the research in machine learning has been conducted through open source, allowing researchers to build on the work of others, thereby further accelerating the rate of progress in the field.
![](https://cdn.mathpix.com/cropped/2025_10_01_d90a4d82c88904700833g-043.jpg?height=1244&width=1230&top_left_y=224&top_left_x=422)

**Image Description:** The image features an abstract background with a colorful, fractal-like design. Superimposed on this background is the title "Probabilities," prominently displayed in black font, and the number "2" in large red font above it. The visual composition suggests a thematic focus on the concept of probabilities, likely related to a lecture on statistical methods or probability theory. The colorful design enhances the aesthetic appeal but does not provide additional informational value.


In almost every application of machine learning we have to deal with uncertainty. For example, a system that classifies images of skin lesions as benign or malignant can never in practice achieve perfect accuracy. We can distinguish between two kinds of uncertainty. The first is epistemic uncertainty (derived from the Greek word episteme meaning knowledge), sometimes called systematic uncertainty. It arises because we only get to see data sets of finite size. As we observe more data, for instance more examples of benign and malignant skin lesion images, we are better able to predict the class of a new example. However, even with an infinitely large data set, we would still not be able to achieve perfect accuracy due to the second kind of uncertainty known as aleatoric uncertainty, also called intrinsic or stochastic uncertainty, or sometimes simply called noise. Generally speaking, the noise arises because we are able to observe only partial information about the world, and therefore, one way to reduce this source of uncertainty is to gather different kinds of data. This is illustrated

![](https://cdn.mathpix.com/cropped/2025_10_01_d90a4d82c88904700833g-044.jpg?height=601&width=1471&top_left_y=205&top_left_x=144)

**Image Description:** The image consists of three parts: 

(a) A 3D surface plot showing a wavy surface defined in a three-dimensional Cartesian coordinate system with axes labeled \(x_1\), \(x_2\), and \(y\). The plot illustrates a function's behavior over the space, characterized by peaks and troughs.

(b) A scatter plot with data points represented as red dots, plotted along the axes \(x_1\) and \(y\). The distribution appears random, indicating variance in the dataset.

(c) Another scatter plot also featuring red dots, but showing a trend in the data points, suggesting a potential underlying relationship between \(x_1\) and \(y\).

Figure 2.1 An extension of the simple sine curve regression problem to two dimensions. (a) A plot of the function $y\left(x_{1}, x_{2}\right)=\sin \left(2 \pi x_{1}\right) \sin \left(2 \pi x_{2}\right)$. Data is generated by selecting values for $x_{1}$ and $x_{2}$, computing the corresponding value of $y\left(x_{1}, x_{2}\right)$, and then adding Gaussian noise. (b) Plot of 100 data points in which $x_{2}$ is unobserved showing high levels of noise. (c) Plot of 100 data points in which $x_{2}$ is fixed to the value $x_{2}=\frac{\pi}{2}$, simulating the effect of being able to measure $x_{2}$ as well as $x_{1}$, showing much lower levels of noise.

Section 1.2

Section 2.1
Section 5.2
using an extension of the sine curve example to two dimensions in Figure 2.1.
As a practical example of this, a biopsy sample of the skin lesion is much more informative than the image alone and might greatly improve the accuracy with which we can determine if a new lesion is malignant. Given both the image and the biopsy data, the intrinsic uncertainty might be very small, and by collecting a large training data set, we may be able to reduce the systematic uncertainty to a low level and thereby make predictions of the class of the lesion with high accuracy.

Both kinds of uncertainty can be handled using the framework of probability theory, which provides a consistent paradigm for the quantification and manipulation of uncertainty and therefore forms one of the central foundations for machine learning. We will see that probabilities are governed by two simple formulae known as the sum rule and the product rule. When coupled with decision theory, these rules allow us, at least in principle, to make optimal predictions given all the information available to us, even though that information may be incomplete or ambiguous.

The concept of probability is often introduced in terms of frequencies of repeatable events. Consider, for example, the bent coin shown in Figure 2.2, and suppose that the shape of the coin is such that if it is flipped a large number of times, it lands concave side up $60 \%$ of the time, and therefore lands convex side up $40 \%$ of the time. We say that the probability of landing concave side up is $60 \%$ or 0.6 . Strictly, the probability is defined in the limit of an infinite number of 'trials' or coin flips in this case. Because the coin must land either concave side up or convex side up, these probabilities add to $100 \%$ or 1.0 . This definition of probability in terms of the frequency of repeatable events is the basis for the frequentist view of statistics.

Now suppose that, although we know that the probability that the coin will land concave side up is 0.6 , we are not allowed to look at the coin itself and we do not

Figure 2.2 Probability can be viewed either as a frequency associated with a repeatable event or as a quantification of uncertainty. A bent coin can be used to illustrate the difference, as discussed in the text.
![](https://cdn.mathpix.com/cropped/2025_10_01_d90a4d82c88904700833g-045.jpg?height=301&width=736&top_left_y=222&top_left_x=913)

**Image Description:** The image features two coins positioned at angles, with one labeled "60%" on the left and the other labeled "40%" on the right. Both coins are deformed, showcasing a contrast in their shapes. The 60% coin has a more pronounced curvature compared to the 40% coin, indicating differences in design or manufacturing. The background is plain, enhancing focus on the coins. This visual likely serves to illustrate concepts in materials science or manufacturing tolerances related to coin production.


know which side is heads and which is tails. If asked to take a bet on whether the coin will land heads or tails when flipped, then symmetry suggests that our bet should be based on the assumption that the probability of seeing heads is 0.5 , and indeed a more careful analysis shows that, in the absence of any additional information, this is indeed the rational choice. Here we are using probabilities in a more general sense than simply the frequency of events. Whether the convex side of the coin is heads or tails is not itself a repeatable event, it is simply unknown. The use of probability as a

Section 2.6

Exercise 2.40 quantification of uncertainty is the Bayesian perspective and is more general in that it includes frequentist probability as a special case. We can learn about which side of the coin is heads if we are given results from a sequence of coin flips by making use of Bayesian reasoning. The more results we observe, the lower our uncertainty as to which side of the coin is which.

Having introduced the concept of probability informally, we turn now to a more detailed exploration of probabilities and discuss how to use them quantitatively. Concepts developed in the remainder of this chapter will form a core foundation for many of the topics discussed throughout the book.

### 2.1. The Rules of Probability

In this section we will derive two simple rules that govern the behaviour of probabilities. However, in spite of their apparent simplicity, these rules will prove to be very powerful and widely applicable. We will motivate the rules of probability by first introducing a simple example.

### 2.1.1 A medical screening example

Consider the problem of screening a population in order to provide early detection of cancer, and let us suppose that $1 \%$ of the population actually have cancer. Ideally our test for cancer would give a positive result for anyone who has cancer and a negative result for anyone who does not. However, tests are not perfect, so we will suppose that when the test is given to people who are free of cancer, $3 \%$ of them will test positive. These are known as false positives. Similarly, when the test is given to people who do have cancer, $10 \%$ of them will test negative. These are called false negatives. The various error rates are illustrated in Figure 2.3.

Given this information, we might ask the following questions: (1) 'If we screen the population, what is the probability that someone will test positive?', (2) 'If some-

Figure 2.3 Illustration of the accuracy of a cancer test. Out of every hundred people taking the test who do not have cancer, shown on the left, on average three will test positive. For those who have cancer, shown on the right, out of every hundred people taking the test, on average 90 will test positive.

![](https://cdn.mathpix.com/cropped/2025_10_01_d90a4d82c88904700833g-046.jpg?height=517&width=345&top_left_y=219&top_left_x=923)

**Image Description:** The image features a grid of blue stick figures arranged in a matrix of 10 rows and 10 columns, totaling 100 figures. Among these, two figures are depicted in red, indicating a notable distinction. This visual representation is likely used to convey statistical data, such as prevalence or proportions within a population, illustrating a certain category or condition in contrast to a larger group. The overall design facilitates easy interpretation of the represented statistics, emphasizing the differences between the groups visually.

No Cancer

![](https://cdn.mathpix.com/cropped/2025_10_01_d90a4d82c88904700833g-046.jpg?height=517&width=345&top_left_y=219&top_left_x=1309)

**Image Description:** The image displays a grid with 100 stylized human figures arranged in rows. There are 90 red figures and 10 blue figures. The red figures are represented in a larger quantity, indicating a majority, while the blue figures signify a minority. The figures are uniformly spaced and aligned, demonstrating a clear contrast between the two colors, emphasizing a ratio or comparison between two groups. This visual representation could be used in discussions about demographics, representation, or population statistics.

Cancer

one receives a positive test result, what is the probability that they actually have cancer?'. We could answer such questions by working through the cancer screening case in detail. Instead, however, we will pause our discussion of this specific example and first derive the general rules of probability, known as the sum rule of probability and the product rule. We will then illustrate the use of these rules by answering our two questions.

### 2.1.2 The sum and product rules

To derive the rules of probability, consider the slightly more general example shown in Figure 2.4 involving two variables $X$ and $Y$. In our cancer example, $X$ could represent the presence or absence of cancer, and $Y$ could be a variable denoting the outcome of the test. Because the values of these variables can vary from one person to another in a way that is generally unknown, they are called random variables or stochastic variables. We will suppose that $X$ can take any of the values $x_{i}$ where $i=1, \ldots, L$ and that $Y$ can take the values $y_{j}$ where $j=1, \ldots, M$. Consider a total of $N$ trials in which we sample both of the variables $X$ and $Y$, and let the number of such trials in which $X=x_{i}$ and $Y=y_{j}$ be $n_{i j}$. Also, let the number of trials in which $X$ takes the value $x_{i}$ (irrespective of the value that $Y$ takes) be denoted by $c_{i}$, and similarly let the number of trials in which $Y$ takes the value $y_{j}$ be denoted by $r_{j}$.

The probability that $X$ will take the value $x_{i}$ and $Y$ will take the value $y_{j}$ is written $p\left(X=x_{i}, Y=y_{j}\right)$ and is called the joint probability of $X=x_{i}$ and $Y=y_{j}$. It is given by the number of points falling in the cell $i, j$ as a fraction of the total number of points, and hence

$$
p\left(X=x_{i}, Y=y_{j}\right)=\frac{n_{i j}}{N}
$$

Here we are implicitly considering the limit $N \rightarrow \infty$. Similarly, the probability that $X$ takes the value $x_{i}$ irrespective of the value of $Y$ is written as $p\left(X=x_{i}\right)$ and is

Figure 2.4 We can derive the sum and product rules of probability by considering a random variable $X$, which takes the values $\left\{x_{i}\right\}$ where $i=1, \ldots, L$, and a second random variable $Y$, which takes the values $\left\{y_{j}\right\}$ where $j= 1, \ldots, M$. In this illustration, we have $L=5$ and $M=3$. If we consider the total number $N$ of instances of these variables, then we denote the number of instances where $X=x_{i}$ and $Y=y_{j}$ by $n_{i j}$, which is the number of in-
![](https://cdn.mathpix.com/cropped/2025_10_01_d90a4d82c88904700833g-047.jpg?height=355&width=531&top_left_y=219&top_left_x=1121)

**Image Description:** The image is a grid diagram, representing a matrix with rows and columns. It consists of a rectangular array with labeled axes. The vertical axis is labeled \( y_j \), representing the row indices, while the horizontal axis is labeled \( x_i \), representing the column indices. Each cell is denoted as \( n_{ij} \), indicating the entry at the intersection of the \( i \)-th column and the \( j \)-th row. The labels \( r_j \) and \( c_i \) suggest the summation of rows and columns, respectively, thereby indicating a joint distribution or matrix representation.

stances in the corresponding cell of the array. The number of instances in column $i$, corresponding to $X=x_{i}$, is denoted by $c_{i}$, and the number of instances in row $j$, corresponding to $Y=y_{j}$, is denoted by $r_{j}$.
given by the fraction of the total number of points that fall in column $i$, so that

$$
p\left(X=x_{i}\right)=\frac{c_{i}}{N} .
$$

Since $\sum_{i} c_{i}=N$, we see that

$$
\sum_{i=1}^{L} p\left(X=x_{i}\right)=1
$$
and, hence, the probabilities sum to one as required. Because the number of instances in column $i$ in Figure 2.4 is just the sum of the number of instances in each cell of that column, we have $c_{i}=\sum_{j} n_{i j}$ and therefore, from (2.1) and (2.2), we have
$$
p\left(X=x_{i}\right)=\sum_{j=1}^{M} p\left(X=x_{i}, Y=y_{j}\right),
$$

which is the sum rule of probability. Note that $p\left(X=x_{i}\right)$ is sometimes called the marginal probability and is obtained by marginalizing, or summing out, the other variables (in this case $Y$ ).

If we consider only those instances for which $X=x_{i}$, then the fraction of such instances for which $Y=y_{j}$ is written $p\left(Y=y_{j} \mid X=x_{i}\right)$ and is called the conditional probability of $Y=y_{j}$ given $X=x_{i}$. It is obtained by finding the fraction of those points in column $i$ that fall in cell $i, j$ and, hence, is given by

$$
p\left(Y=y_{j} \mid X=x_{i}\right)=\frac{n_{i j}}{c_{i}} .
$$

Summing both sides over $j$ and using $\sum_{j} n_{i j}=c_{i}$, we obtain

$$
\sum_{j=1}^{M} p\left(Y=y_{j} \mid X=x_{i}\right)=1
$$
showing that the conditional probabilities are correctly normalized. From (2.1), (2.2), and (2.5), we can then derive the following relationship:
\$\$\begin{aligned}

p\left(X=x_{i}, Y=y_{j}\right) \& =\frac{n_{i j}}{N}=\frac{n_{i j}}{c_{i}} \cdot \frac{c_{i}}{N} <br>
\& =p\left(Y=y_{j} \mid X=x_{i}\right) p\left(X=x_{i}\right)

\end{aligned}\$\$
which is the product rule of probability.
So far, we have been quite careful to make a distinction between a random variable, such as $X$, and the values that the random variable can take, for example $x_{i}$. Thus, the probability that $X$ takes the value $x_{i}$ is denoted $p\left(X=x_{i}\right)$. Although this helps to avoid ambiguity, it leads to a rather cumbersome notation, and in many cases there will be no need for such pedantry. Instead, we may simply write $p(X)$ to denote a distribution over the random variable $X$, or $p\left(x_{i}\right)$ to denote the distribution evaluated for the particular value $x_{i}$, provided that the interpretation is clear from the context.

With this more compact notation, we can write the two fundamental rules of probability theory in the following form:

$$
\begin{array}{lr}
\text { sum rule } & p(X)=\sum_{Y} p(X, Y) \\
\text { product rule } & p(X, Y)=p(Y \mid X) p(X)
\end{array}
$$

Here $p(X, Y)$ is a joint probability and is verbalized as 'the probability of $X$ and $Y^{\prime}$. Similarly, the quantity $p(Y \mid X)$ is a conditional probability and is verbalized as 'the probability of $Y$ given $X$ '. Finally, the quantity $p(X)$ is a marginal probability and is simply 'the probability of $X$ '. These two simple rules form the basis for all of the probabilistic machinery that we will use throughout this book.

### 2.1.3 Bayes' theorem

From the product rule, together with the symmetry property $p(X, Y)=p(Y, X)$, we immediately obtain the following relationship between conditional probabilities:

$$
p(Y \mid X)=\frac{p(X \mid Y) p(Y)}{p(X)}
$$
which is called Bayes' theorem and which plays an important role in machine learning. Note how Bayes' theorem relates the conditional distribution $p(Y \mid X)$ on the left-hand side of the equation, to the 'reversed' conditional distribution $p(X \mid Y)$ on the right-hand side. Using the sum rule, the denominator in Bayes' theorem can be expressed in terms of the quantities appearing in the numerator:
$$
p(X)=\sum_{Y} p(X \mid Y) p(Y)
$$

Thus, we can view the denominator in Bayes' theorem as being the normalization constant required to ensure that the sum over the conditional probability distribution on the left-hand side of (2.10) over all values of $Y$ equals one.

![](https://cdn.mathpix.com/cropped/2025_10_01_d90a4d82c88904700833g-049.jpg?height=1064&width=1503&top_left_y=258&top_left_x=127)

**Image Description:** The image consists of four sub-diagrams, illustrating statistical concepts. Top left: A scatter plot with blue points representing data points in 2D space defined by axes \(X\) and \(Y\), with horizontal lines indicating \(Y=1\) and \(Y=2\). Top right: A bar graph depicting the marginal distribution \(p(Y)\) with categories showing relative frequencies in purple. Bottom left: A histogram representing the marginal distribution \(p(X)\) in bars, with the variable \(X\) on the horizontal axis. Bottom right: Another histogram \(p(X|Y=1)\) showing the distribution of \(X\) given \(Y=1\), with the same variable \(X\) on the horizontal axis.

Figure 2.5 An illustration of a distribution over two variables, $X$, which takes nine possible values, and $Y$, which takes two possible values. The top left figure shows a sample of 60 points drawn from a joint probability distribution over these variables. The remaining figures show histogram estimates of the marginal distributions $p(X)$ and $p(Y)$, as well as the conditional distribution $p(X \mid Y=1)$ corresponding to the bottom row in the top left figure.

Section 3.5.1

In Figure 2.5, we show a simple example involving a joint distribution over two variables to illustrate the concept of marginal and conditional distributions. Here a finite sample of $N=60$ data points has been drawn from the joint distribution and is shown in the top left. In the top right is a histogram of the fractions of data points having each of the two values of $Y$. From the definition of probability, these fractions would equal the corresponding probabilities $p(Y)$ in the limit when the sample size $N \rightarrow \infty$. We can view the histogram as a simple way to model a probability distribution given only a finite number of points drawn from that distribution. The remaining two plots in Figure 2.5 show the corresponding histogram estimates of $p(X)$ and $p(X \mid Y=1)$.

### 2.1.4 Medical screening revisited

Let us now return to our cancer screening example and apply the sum and product rules of probability to answer our two questions. For clarity, when working through this example, we will once again be explicit about distinguishing between the random variables and their instantiations. We will denote the presence or absence of cancer by the variable $C$, which can take two values: $C=0$ corresponds to 'no cancer' and $C=1$ corresponds to 'cancer'. We have assumed that one person in a hundred in the population has cancer, and so we have

$$
\begin{aligned}
p(C=1) & =1 / 100 \\
p(C=0) & =99 / 100
\end{aligned}
$$

respectively. Note that these satisfy $p(C=0)+p(C=1)=1$.
Now let us introduce a second random variable $T$ representing the outcome of a screening test, where $T=1$ denotes a positive result, indicative of cancer, and $T=0$ a negative result, indicative of the absence of cancer. As illustrated in Figure 2.3, we know that for those who have cancer the probability of a positive test result is $90 \%$, while for those who do not have cancer the probability of a positive test result is $3 \%$. We can therefore write out all four conditional probabilities:

$$
\begin{aligned}
p(T=1 \mid C=1) & =90 / 100 \\
p(T=0 \mid C=1) & =10 / 100 \\
p(T=1 \mid C=0) & =3 / 100 \\
p(T=0 \mid C=0) & =97 / 100
\end{aligned}
$$

Again, note that these probabilities are normalized so that

$$
p(T=1 \mid C=1)+p(T=0 \mid C=1)=1
$$
and similarly
$$
p(T=1 \mid C=0)+p(T=0 \mid C=0)=1 .
$$

We can now use the sum and product rules of probability to answer our first question and evaluate the overall probability that someone who is tested at random will have a positive test result:

$$
\begin{aligned}
p(T=1) & =p(T=1 \mid C=0) p(C=0)+p(T=1 \mid C=1) p(C=1) \\
& =\frac{3}{100} \times \frac{99}{100}+\frac{90}{100} \times \frac{1}{100}=\frac{387}{10,000}=0.0387
\end{aligned}
$$

We see that if a person is tested at random there is a roughly $4 \%$ chance that the test will be positive even though there is a $1 \%$ chance that they actually have cancer. From this it follows, using the sum rule, that $p(T=0)=1-387 / 10,000= 9613 / 10,000=0.9613$ and, hence, there is a roughly $96 \%$ chance that the do not have cancer.

Now consider our second question, which is the one that is of particular interest to a person being screened: if a test is positive, what is the probability that the person
has cancer? This requires that we evaluate the probability of cancer conditional on the outcome of the test, whereas the probabilities in (2.14) to (2.17) give the probability distribution over the test outcome conditioned on whether the person has cancer. We can solve the problem of reversing the conditional probability by using Bayes' theorem (2.10) to give

$$
\begin{aligned}
p(C=1 \mid T=1) & =\frac{p(T=1 \mid C=1) p(C=1)}{p(T=1)} \\
& =\frac{90}{100} \times \frac{1}{100} \times \frac{10,000}{387}=\frac{90}{387} \simeq 0.23
\end{aligned}
$$

so that if a person is tested at random and the test is positive, there is a $23 \%$ probability that they actually have cancer. From the sum rule, it then follows that $p(C= 0 \mid T=1)=1-90 / 387=297 / 387 \simeq 0.77$, which is a $77 \%$ chance that they do not have cancer.

### 2.1.5 Prior and posterior probabilities

We can use the cancer screening example to provide an important interpretation of Bayes' theorem as follows. If we had been asked whether someone is likely to have cancer, before they have received a test, then the most complete information we have available is provided by the probability $p(C)$. We call this the prior probability because it is the probability available before we observe the result of the test. Once we are told that this person has received a positive test, we can then use Bayes' theorem to compute the probability $p(C \mid T)$, which we will call the posterior probability because it is the probability obtained after we have observed the test result $T$.

In this example, the prior probability of having cancer is $1 \%$. However, once we have observed that the test result is positive, we find that the posterior probability of cancer is now $23 \%$, which is a substantially higher probability of cancer, as we would intuitively expect. We note, however, that a person with a positive test still has only a $23 \%$ change of actually having cancer, even though the test appears, from Figure 2.3 to be reasonably 'accurate'. This conclusion seems counter-intuitive to many people. The reason has to do with the low prior probability of having cancer. Although the test provides strong evidence of cancer, this has to be combined with the prior probability using Bayes' theorem to arrive at the correct posterior probability.

### 2.1.6 Independent variables

Finally, if the joint distribution of two variables factorizes into the product of the marginals, so that $p(X, Y)=p(X) p(Y)$, then $X$ and $Y$ are said to be independent. An example of independent events would be the successive flips of a coin. From the product rule, we see that $p(Y \mid X)=p(Y)$, and so the conditional distribution of $Y$ given $X$ is indeed independent of the value of $X$. In our cancer screening example, if the probability of a positive test is independent of whether the person has cancer, then $p(T \mid C)=p(T)$, which means that from Bayes' theorem (2.10) we have $p(C \mid T)=p(C)$, and therefore probability of cancer is not changed by observing the test outcome. Of course, such a test would be useless because the outcome of the test tells us nothing about whether the person has cancer.

Figure 2.6 The concept of probability for discrete variables can be extended to that of a probability density $p(x)$ over a continuous variable $x$ and is such that the probability of $x$ lying in the interval ( $x, x+\delta x$ ) is given by $p(x) \delta x$ for $\delta x \rightarrow 0$. The probability density can be expressed as the derivative of a cumulative distribution function $P(x)$.
![](https://cdn.mathpix.com/cropped/2025_10_01_d90a4d82c88904700833g-052.jpg?height=553&width=780&top_left_y=217&top_left_x=884)

**Image Description:** The image is a diagram representing two probability density functions, \( p(x) \) (in red) and \( P(x) \) (in blue), plotted against the x-axis. The x-axis represents a variable \( x \), while the vertical axis indicates probability values. A highlighted section in green on the x-axis between \( \delta x \) marks a specific interval of interest on the graph. The curve \( p(x) \) oscillates while \( P(x) \) appears to represent a cumulative distribution or a related function, showing their interaction in terms of probability over the specified interval.


### 2.2. Probability Densities

As well as considering probabilities defined over discrete sets of values, we also wish to consider probabilities with respect to continuous variables. For instance, we might wish to predict what dose of drug to give to a patient. Since there will be uncertainty in this prediction, we want to quantify this uncertainty and again we can make use of probabilities. However, we cannot simply apply the concepts of probability discussed so far directly, since the probability of observing a specific value for a continuous variable, to infinite precision, will effectively be zero. Instead, we need to introduce the concept of a probability density. Here we will limit ourselves to a relatively informal discussion.

We define the probability density $p(x)$ over a continuous variable $x$ to be such that the probability of $x$ falling in the interval ( $x, x+\delta x$ ) is given by $p(x) \delta x$ for $\delta x \rightarrow 0$. This is illustrated in Figure 2.6. The probability that $x$ will lie in an interval ( $a, b$ ) is then given by

$$
p(x \in(a, b))=\int_{a}^{b} p(x) \mathrm{d} x
$$

Because probabilities are non-negative, and because the value of $x$ must lie somewhere on the real axis, the probability density $p(x)$ must satisfy the two conditions

$$
\begin{aligned}
p(x) & \geqslant 0 \\
\int_{-\infty}^{\infty} p(x) \mathrm{d} x & =1
\end{aligned}
$$

The probability that $x$ lies in the interval $(-\infty, z)$ is given by the cumulative distribution function defined by

$$
P(z)=\int_{-\infty}^{z} p(x) \mathrm{d} x
$$
which satisfies $P^{\prime}(x)=p(x)$, as shown in Figure 2.6.
If we have several continuous variables $x_{1}, \ldots, x_{D}$, denoted collectively by the vector $\mathbf{x}$, then we can define a joint probability density $p(\mathbf{x})=p\left(x_{1}, \ldots, x_{D}\right)$ such that the probability of $\mathbf{x}$ falling in an infinitesimal volume $\delta \mathbf{x}$ containing the point $\mathbf{x}$ is given by $p(\mathbf{x}) \delta \mathbf{x}$. This multivariate probability density must satisfy
\$\$\begin{aligned}

p(\mathbf{x}) \& \geqslant 0 <br>
\int p(\mathbf{x}) \mathrm{d} \mathbf{x} \& =1

\end{aligned}\$\$
in which the integral is taken over the whole of $\mathbf{x}$ space. More generally, we can also consider joint probability distributions over a combination of discrete and continuous variables.

The sum and product rules of probability, as well as Bayes' theorem, also apply to probability densities as well as to combinations of discrete and continuous variables. If $\mathbf{x}$ and $\mathbf{y}$ are two real variables, then the sum and product rules take the form

$$
\begin{array}{ll}
\text { sum rule } & p(\mathbf{x})=\int p(\mathbf{x}, \mathbf{y}) \mathrm{d} \mathbf{y} \\
\text { product rule } & p(\mathbf{x}, \mathbf{y})=p(\mathbf{y} \mid \mathbf{x}) p(\mathbf{x})
\end{array}
$$

Similarly, Bayes' theorem can be written in the form

$$
p(\mathbf{y} \mid \mathbf{x})=\frac{p(\mathbf{x} \mid \mathbf{y}) p(\mathbf{y})}{p(\mathbf{x})}
$$
where the denominator is given by
$$
p(\mathbf{x})=\int p(\mathbf{x} \mid \mathbf{y}) p(\mathbf{y}) \mathrm{d} \mathbf{y}
$$

A formal justification of the sum and product rules for continuous variables requires a branch of mathematics called measure theory (Feller, 1966) and lies outside the scope of this book. Its validity can be seen informally, however, by dividing each real variable into intervals of width $\Delta$ and considering the discrete probability distribution over these intervals. Taking the limit $\Delta \rightarrow 0$ then turns sums into integrals and gives the desired result.

### 2.2.1 Example distributions

There are many forms of probability density that are in widespread use and that are important both in their own right and as building blocks for more complex probabilistic models. The simplest form would be one in which $p(x)$ is a constant, independent of $x$, but this cannot be normalized because the integral in (2.28) will be divergent. Distributions that cannot be normalized are called improper. We can, however, have the uniform distribution that is constant over a finite region, say $(c, d)$, and zero elsewhere, in which case (2.28) implies

$$
p(x)=1 /(d-c), \quad x \in(c, d) .
$$

Figure 2.7 Plots of a uniform distribution over the range $(-1,1)$, shown in red, the exponential distribution with $\lambda=1$, shown in blue, and a Laplace distribution with $\mu=1$ and $\gamma=1$, shown in green.
![](https://cdn.mathpix.com/cropped/2025_10_01_d90a4d82c88904700833g-054.jpg?height=502&width=692&top_left_y=219&top_left_x=962)

**Image Description:** The image is a diagram depicting three probability density functions (PDFs) plotted on a Cartesian coordinate system. The x-axis represents the variable \( x \), ranging approximately from -2 to 4, while the y-axis represents the probability density \( p(x) \), ranging from 0 to 1. The curves shown are in three colors: red, green, and blue. The red curve is a piecewise constant function, while the green and blue curves are smooth and continuous. The highest peak is in the blue curve, indicating the area where the highest probability density occurs, particularly near \( x = 1 \).


Another simple form of density is the exponential distribution given by

$$
p(x \mid \lambda)=\lambda \exp (-\lambda x), \quad x \geqslant 0
$$

A variant of the exponential distribution, known as the Laplace distribution, allows the peak to be moved to a location $\mu$ and is given by

$$
p(x \mid \mu, \gamma)=\frac{1}{2 \gamma} \exp \left(-\frac{|x-\mu|}{\gamma}\right) .
$$

The constant, exponential, and Laplace distributions are illustrated in Figure 2.7.
Another important distribution is the Dirac delta function, which is written

$$
p(x \mid \mu)=\delta(x-\mu) .
$$

This is defined to be zero everywhere except at $x=\mu$ and to have the property of integrating to unity according to (2.28). Informally, we can think of this as an infinitely narrow and infinitely tall spike located at $x=\mu$ with the property of having unit area. Finally, if we have a finite set of observations of $x$ given by $\mathcal{D}=\left\{x_{1}, \ldots, x_{N}\right\}$ then we can use the delta function to construct the empirical distribution given by

$$
p(x \mid \mathcal{D})=\frac{1}{N} \sum_{n=1}^{N} \delta\left(x-x_{n}\right)
$$
which consists of a Dirac delta function centred on each of the data points. The

## Exercise 2.6

probability density defined by (2.37) integrates to one as required.

### 2.2.2 Expectations and covariances

One of the most important operations involving probabilities is that of finding weighted averages of functions. The weighted average of some function $f(x)$ under a probability distribution $p(x)$ is called the expectation of $f(x)$ and will be denoted by $\mathbb{E}[f]$. For a discrete distribution, it is given by summing over all possible values of $x$ in the form

$$
\mathbb{E}[f]=\sum_{x} p(x) f(x)
$$
where the average is weighted by the relative probabilities of the different values of $x$. For continuous variables, expectations are expressed in terms of an integration with respect to the corresponding probability density:
$$
\mathbb{E}[f]=\int p(x) f(x) \mathrm{d} x
$$

In either case, if we are given a finite number $N$ of points drawn from the probability distribution or probability density, then the expectation can be approximated as a
Exercise 2.7 finite sum over these points:

$$
\mathbb{E}[f] \simeq \frac{1}{N} \sum_{n=1}^{N} f\left(x_{n}\right)
$$

The approximation in (2.40) becomes exact in the limit $N \rightarrow \infty$.
Sometimes we will be considering expectations of functions of several variables, in which case we can use a subscript to indicate which variable is being averaged over, so that for instance

$$
\mathbb{E}_{x}[f(x, y)]
$$
denotes the average of the function $f(x, y)$ with respect to the distribution of $x$. Note that $\mathbb{E}_{x}[f(x, y)]$ will be a function of $y$.

We can also consider a conditional expectation with respect to a conditional distribution, so that

$$
\mathbb{E}_{x}[f \mid y]=\sum_{x} p(x \mid y) f(x)
$$
which is also a function of $y$. For continuous variables, the conditional expectation takes the form
$$
\mathbb{E}_{x}[f \mid y]=\int p(x \mid y) f(x) \mathrm{d} x
$$

The variance of $f(x)$ is defined by

$$
\operatorname{var}[f]=\mathbb{E}\left[(f(x)-\mathbb{E}[f(x)])^{2}\right]
$$
and provides a measure of how much $f(x)$ varies around its mean value $\mathbb{E}[f(x)]$. Expanding out the square, we see that the variance can also be written in terms of the expectations of $f(x)$ and $f(x)^{2}$ :
$$
\operatorname{var}[f]=\mathbb{E}\left[f(x)^{2}\right]-\mathbb{E}[f(x)]^{2}
$$

In particular, we can consider the variance of the variable $x$ itself, which is given by

$$
\operatorname{var}[x]=\mathbb{E}\left[x^{2}\right]-\mathbb{E}[x]^{2} .
$$

For two random variables $x$ and $y$, the covariance measures the extent to which the two variables vary together and is defined by

$$
\begin{aligned}
\operatorname{cov}[x, y] & =\mathbb{E}_{x, y}[\{x-\mathbb{E}[x]\}\{y-\mathbb{E}[y]\}] \\
& =\mathbb{E}_{x, y}[x y]-\mathbb{E}[x] \mathbb{E}[y]
\end{aligned}
$$

Figure 2.8 Plot of a Gaussian distribution for a single continuous variable $x$ showing the mean $\mu$ and the standard deviation $\sigma$.
![](https://cdn.mathpix.com/cropped/2025_10_01_d90a4d82c88904700833g-056.jpg?height=560&width=772&top_left_y=217&top_left_x=884)

**Image Description:** The image depicts a probability density function of a normal distribution, denoted as \( N(x | \mu, \sigma^2) \). The horizontal axis represents the variable \( x \), while the vertical axis signifies the probability density. The curve peaks at the mean \( \mu \) and shows the characteristic bell shape of a normal distribution. The distance \( 2\sigma \) is marked, indicating the spread of the distribution, where \( \sigma \) is the standard deviation. The red curve illustrates the continuous nature of the data around the mean.


Exercise 2.9
If $x$ and $y$ are independent, then their covariance equals zero.
For two vectors $\mathbf{x}$ and $\mathbf{y}$, their covariance is a matrix given by

$$
\begin{aligned}
\operatorname{cov}[\mathbf{x}, \mathbf{y}] & =\mathbb{E}_{\mathbf{x}, \mathbf{y}}\left[\{\mathbf{x}-\mathbb{E}[\mathbf{x}]\}\left\{\mathbf{y}^{\mathrm{T}}-\mathbb{E}\left[\mathbf{y}^{\mathrm{T}}\right]\right\}\right] \\
& =\mathbb{E}_{\mathbf{x}, \mathbf{y}}\left[\mathbf{x} \mathbf{y}^{\mathrm{T}}\right]-\mathbb{E}[\mathbf{x}] \mathbb{E}\left[\mathbf{y}^{\mathrm{T}}\right]
\end{aligned}
$$

If we consider the covariance of the components of a vector $\mathbf{x}$ with each other, then we use a slightly simpler notation $\operatorname{cov}[\mathbf{x}] \equiv \operatorname{cov}[\mathbf{x}, \mathbf{x}]$.

### 2.3. The Gaussian Distribution

One of the most important probability distributions for continuous variables is called the normal or Gaussian distribution, and we will make extensive use of this distribution throughout the rest of the book. For a single real-valued variable $x$, the Gaussian distribution is defined by

$$
\mathcal{N}\left(x \mid \mu, \sigma^{2}\right)=\frac{1}{\left(2 \pi \sigma^{2}\right)^{1 / 2}} \exp \left\{-\frac{1}{2 \sigma^{2}}(x-\mu)^{2}\right\},
$$
which represents a probability density over $x$ governed by two parameters: $\mu$, called the mean, and $\sigma^{2}$, called the variance. The square root of the variance, given by $\sigma$, is called the standard deviation, and the reciprocal of the variance, written as $\beta=1 / \sigma^{2}$, is called the precision. We will see the motivation for this terminology shortly. Figure 2.8 shows a plot of the Gaussian distribution. Although the form of the Gaussian distribution might seem arbitrary, we will see later that it arises naturally from the concept of maximum entropy and from the perspective of the central limit theorem.

From (2.49) we see that the Gaussian distribution satisfies

$$
\mathcal{N}\left(x \mid \mu, \sigma^{2}\right)>0 .
$$

Exercise 2.12

Exercise 2.13

Exercise 2.14

Also, it is straightforward to show that the Gaussian is normalized, so that

$$
\int_{-\infty}^{\infty} \mathcal{N}\left(x \mid \mu, \sigma^{2}\right) \mathrm{d} x=1
$$

Thus, (2.49) satisfies the two requirements for a valid probability density.

### 2.3.1 Mean and variance

We can readily find expectations of functions of $x$ under the Gaussian distribution. In particular, the average value of $x$ is given by

$$
\mathbb{E}[x]=\int_{-\infty}^{\infty} \mathcal{N}\left(x \mid \mu, \sigma^{2}\right) x \mathrm{~d} x=\mu
$$

Because the parameter $\mu$ represents the average value of $x$ under the distribution, it is referred to as the mean. The integral in (2.52) is known as the first-order moment of the distribution because it is the expectation of $x$ raised to the power one. We can similarly evaluate the second-order moment given by

$$
\mathbb{E}\left[x^{2}\right]=\int_{-\infty}^{\infty} \mathcal{N}\left(x \mid \mu, \sigma^{2}\right) x^{2} \mathrm{~d} x=\mu^{2}+\sigma^{2}
$$

From (2.52) and (2.53), it follows that the variance of $x$ is given by

$$
\operatorname{var}[x]=\mathbb{E}\left[x^{2}\right]-\mathbb{E}[x]^{2}=\sigma^{2}
$$
and hence $\sigma^{2}$ is referred to as the variance parameter. The maximum of a distribution is known as its mode. For a Gaussian, the mode coincides with the mean.

### 2.3.2 Likelihood function

Suppose that we have a data set of observations represented as a row vector $\mathbf{x}=\left(x_{1}, \ldots, x_{N}\right)$, representing $N$ observations of the scalar variable $x$. Note that we are using the typeface $\mathbf{x}$ to distinguish this from a single observation of a $D$ dimensional vector-valued variable, which we represent by a column vector $\mathbf{x}= \left(x_{1}, \ldots, x_{D}\right)^{\mathrm{T}}$. We will suppose that the observations are drawn independently from a Gaussian distribution whose mean $\mu$ and variance $\sigma^{2}$ are unknown, and we would like to determine these parameters from the data set. The problem of estimating a distribution, given a finite set of observations, is known as density estimation. It should be emphasized that the problem of density estimation is fundamentally illposed, because there are infinitely many probability distributions that could have given rise to the observed finite data set. Indeed, any distribution $p(\mathbf{x})$ that is nonzero at each of the data points $\mathbf{x}_{1}, \ldots, \mathbf{x}_{N}$ is a potential candidate. Here we constrain the space of distributions to be Gaussians, which leads to a well-defined solution.

Data points that are drawn independently from the same distribution are said to be independent and identically distributed, which is often abbreviated to i.i.d. or III. We have seen that the joint probability of two independent events is given by the product of the marginal probabilities for each event separately. Because our data

Figure 2.9 Illustration of the likelihood function for the Gaussian distribution shown by the red curve. Here the grey points denote a data set of values $\left\{x_{n}\right\}$, and the likelihood function (2.55) is given by the product of the corresponding values of $p(x)$ denoted by the blue points. Maximizing the likelihood involves adjusting the mean and variance of the Gaussian so as to maximize this product.
![](https://cdn.mathpix.com/cropped/2025_10_01_d90a4d82c88904700833g-058.jpg?height=497&width=687&top_left_y=219&top_left_x=962)

**Image Description:** The image depicts a probability density function (PDF) plot illustrating a normal distribution. The x-axis labeled \( x \) represents the variable being measured, while the y-axis labeled \( p(x) \) indicates the probability density. The red curve represents the normal distribution, characterized by its mean \( \mu \) and variance \( \sigma^2 \). Blue dots denote specific data points \( x_n \) along the x-axis, with vertical green lines connecting these points to the curve, illustrating their probability density values. Gray dots are also present on the x-axis, likely representing other reference points.

set $\mathbf{X}$ is i.i.d., we can therefore write the probability of the data set, given $\mu$ and $\sigma^{2}$, in the form

$$
p\left(\mathbf{x} \mid \mu, \sigma^{2}\right)=\prod_{n=1}^{N} \mathcal{N}\left(x_{n} \mid \mu, \sigma^{2}\right) .
$$

When viewed as a function of $\mu$ and $\sigma^{2}$, this is called the likelihood function for the Gaussian and is interpreted diagrammatically in Figure 2.9.

One common approach for determining the parameters in a probability distribution using an observed data set, known as maximum likelihood, is to find the parameter values that maximize the likelihood function. This might appear to be a strange criterion because, from our foregoing discussion of probability theory, it would seem more natural to maximize the probability of the parameters given the data, not the
Section 2.6.2 probability of the data given the parameters. In fact, these two criteria are related.

To start with, however, we will determine values for the unknown parameters $\mu$ and $\sigma^{2}$ in the Gaussian by maximizing the likelihood function (2.55). In practice, it is more convenient to maximize the log of the likelihood function. Because the logarithm is a monotonically increasing function of its argument, maximizing the log of a function is equivalent to maximizing the function itself. Taking the log not only simplifies the subsequent mathematical analysis, but it also helps numerically because the product of a large number of small probabilities can easily underflow the numerical precision of the computer, and this is resolved by computing the sum of the log probabilities instead. From (2.49) and (2.55), the log likelihood function can be written in the form

$$
\ln p\left(\mathbf{x} \mid \mu, \sigma^{2}\right)=-\frac{1}{2 \sigma^{2}} \sum_{n=1}^{N}\left(x_{n}-\mu\right)^{2}-\frac{N}{2} \ln \sigma^{2}-\frac{N}{2} \ln (2 \pi) .
$$

## Exercise 2.15

Maximizing (2.56) with respect to $\mu$, we obtain the maximum likelihood solution given by

$$
\mu_{\mathrm{ML}}=\frac{1}{N} \sum_{n=1}^{N} x_{n},
$$

## Exercise 2.16

## Exercise 2.17

which is the sample mean, i.e., the mean of the observed values $\left\{x_{n}\right\}$. Similarly, maximizing (2.56) with respect to $\sigma^{2}$, we obtain the maximum likelihood solution for the variance in the form

$$
\sigma_{\mathrm{ML}}^{2}=\frac{1}{N} \sum_{n=1}^{N}\left(x_{n}-\mu_{\mathrm{ML}}\right)^{2}
$$
which is the sample variance measured with respect to the sample mean $\mu_{\mathrm{ML}}$. Note that we are performing a joint maximization of (2.56) with respect to $\mu$ and $\sigma^{2}$, but for a Gaussian distribution, the solution for $\mu$ decouples from that for $\sigma^{2}$ so that we can first evaluate (2.57) and then subsequently use this result to evaluate (2.58).

### 2.3.3 Bias of maximum likelihood

The technique of maximum likelihood is widely used in deep learning and forms the foundation for most machine learning algorithms. However, it has some limitations, which we can illustrate using a univariate Gaussian.

We first note that the maximum likelihood solutions $\mu_{\mathrm{ML}}$ and $\sigma_{\mathrm{ML}}^{2}$ are functions of the data set values $x_{1}, \ldots, x_{N}$. Suppose that each of these values has been generated independently from a Gaussian distribution whose true parameters are $\mu$ and $\sigma^{2}$. Now consider the expectations of $\mu_{\mathrm{ML}}$ and $\sigma_{\mathrm{ML}}^{2}$ with respect to these data set values. It is straightforward to show that

$$
\begin{aligned}
\mathbb{E}\left[\mu_{\mathrm{ML}}\right] & =\mu \\
\mathbb{E}\left[\sigma_{\mathrm{ML}}^{2}\right] & =\left(\frac{N-1}{N}\right) \sigma^{2} .
\end{aligned}
$$

We see that, when averaged over data sets of a given size, the maximum likelihood solution for the mean will equal the true mean. However, the maximum likelihood estimate of the variance will underestimate the true variance by a factor $(N-1) / N$. This is an example of a phenomenon called bias in which the estimator of a random quantity is systematically different from the true value. The intuition behind this result is given by Figure 2.10.

Note that bias arises because the variance is measured relative to the maximum likelihood estimate of the mean, which itself is tuned to the data. Suppose instead we had access to the true mean $\mu$ and we used this to determine the variance using the estimator

$$
\widehat{\sigma}^{2}=\frac{1}{N} \sum_{n=1}^{N}\left(x_{n}-\mu\right)^{2}
$$

Then we find that

$$
\mathbb{E}\left[\widehat{\sigma}^{2}\right]=\sigma^{2}
$$
which is unbiased. Of course, we do not have access to the true mean but only to the observed data values. From the result (2.60) it follows that for a Gaussian distribution, the following estimate for the variance parameter is unbiased:
$$
\widetilde{\sigma}^{2}=\frac{N}{N-1} \sigma_{\mathrm{ML}}^{2}=\frac{1}{N-1} \sum_{n=1}^{N}\left(x_{n}-\mu_{\mathrm{ML}}\right)^{2} .
$$

![](https://cdn.mathpix.com/cropped/2025_10_01_d90a4d82c88904700833g-060.jpg?height=313&width=1494&top_left_y=217&top_left_x=158)

**Image Description:** The image consists of three plots depicting probability density functions of different distributions, likely illustrating the concept of statistical distributions relative to a parameter \(\mu\). The x-axis represents the value of the random variable, with a dashed vertical line at \(\mu\). The left plot shows a blue curve (left-tailed), the middle plot features a red curve (right-tailed), and the right plot combines both, indicating varying degrees of skewness. Green dots on the x-axis denote expected values or points of interest where the density functions intersect or have specific characteristics.

Figure 2.10 Illustration of how bias arises when using maximum likelihood to determine the mean and variance of a Gaussian. The red curves show the true Gaussian distribution from which data is generated, and the three blue curves show the Gaussian distributions obtained by fitting to three data sets, each consisting of two data points shown in green, using the maximum likelihood results (2.57) and (2.58). Averaged across the three data sets, the mean is correct, but the variance is systematically underestimated because it is measured relative to the sample mean and not relative to the true mean.

Correcting for the bias of maximum likelihood in complex models such as neural networks is not so easy, however.

Note that the bias of the maximum likelihood solution becomes less significant as the number $N$ of data points increases. In the limit $N \rightarrow \infty$ the maximum likelihood solution for the variance equals the true variance of the distribution that generated the data. In the case of the Gaussian, for anything other than small $N$, this bias will not prove to be a serious problem. However, throughout this book we will be interested in complex models with many parameters, for which the bias problems associated with maximum likelihood will be much more severe. In fact, the issue of

Section 2.6.3 bias in maximum likelihood is closely related to the problem of over-fitting.

### 2.3.4 Linear regression

We have seen how the problem of linear regression can be expressed in terms of error minimization. Here we return to this example and view it from a probabilistic perspective, thereby gaining some insights into error functions and regularization.

The goal in the regression problem is to be able to make predictions for the target variable $t$ given some new value of the input variable $x$ by using a set of training data comprising $N$ input values $\mathbf{x}=\left(x_{1}, \ldots, x_{N}\right)$ and their corresponding target values $\mathbf{t}=\left(t_{1}, \ldots, t_{N}\right)$. We can express our uncertainty over the value of the target variable using a probability distribution. For this purpose, we will assume that, given the value of $x$, the corresponding value of $t$ has a Gaussian distribution with a mean equal to the value $y(x, \mathbf{w})$ of the polynomial curve given by (1.1), where $\mathbf{w}$ are the polynomial coefficients, and a variance $\sigma^{2}$. Thus, we have

$$
p\left(t \mid x, \mathbf{w}, \sigma^{2}\right)=\mathcal{N}\left(t \mid y(x, \mathbf{w}), \sigma^{2}\right)
$$

This is illustrated schematically in Figure 2.11.
We now use the training data $\{\mathbf{x}, \mathbf{t}\}$ to determine the values of the unknown parameters $\mathbf{w}$ and $\sigma^{2}$ by maximum likelihood. If the data is assumed to be drawn

Figure 2.11 Schematic illustration of a Gaussian conditional distribution for $t$ given $x$, defined by (2.64), in which the mean is given by the polynomial function $y(x, \mathbf{w})$, and the variance is given by the parameter $\sigma^{2}$.
![](https://cdn.mathpix.com/cropped/2025_10_01_d90a4d82c88904700833g-061.jpg?height=685&width=697&top_left_y=222&top_left_x=957)

**Image Description:** The image depicts a scatter plot with blue points representing data samples on a two-dimensional Cartesian plane. The x-axis is labeled \(x_0\), and the y-axis is labeled \(t\). A red curve illustrates a fitting function \(y(x, w)\) that models the relationship between \(x_0\) and \(t\). A vertical gray line intersects the x-axis, indicating a specific value of \(x_0\). A blue curve labeled \(p(t|x_0, w, \sigma^2)\) is also present, suggesting a probability density function related to \(t\). The diagram visually represents a regression analysis or similar statistical modeling concept.


independently from the distribution (2.64), then the likelihood function is given by

$$
p\left(\mathbf{t} \mid \mathbf{x}, \mathbf{w}, \sigma^{2}\right)=\prod_{n=1}^{N} \mathcal{N}\left(t_{n} \mid y\left(x_{n}, \mathbf{w}\right), \sigma^{2}\right) .
$$

As we did for the simple Gaussian distribution earlier, it is convenient to maximize the logarithm of the likelihood function. Substituting for the Gaussian distribution, given by (2.49), we obtain the log likelihood function in the form

$$
\ln p\left(\mathbf{t} \mid \mathbf{x}, \mathbf{w}, \sigma^{2}\right)=-\frac{1}{2 \sigma^{2}} \sum_{n=1}^{N}\left\{y\left(x_{n}, \mathbf{w}\right)-t_{n}\right\}^{2}-\frac{N}{2} \ln \sigma^{2}-\frac{N}{2} \ln (2 \pi) .
$$

Consider first the evaluation of the maximum likelihood solution for the polynomial coefficients, which will be denoted by $\mathbf{w}_{\text {ML }}$. These are determined by maximizing (2.66) with respect to $\mathbf{w}$. For this purpose, we can omit the last two terms on the right-hand side of (2.66) because they do not depend on $\mathbf{w}$. Also, note that scaling the log likelihood by a positive constant coefficient does not alter the location of the maximum with respect to $\mathbf{w}$, and so we can replace the coefficient $1 / 2 \sigma^{2}$ with $1 / 2$. Finally, instead of maximizing the log likelihood, we can equivalently minimize the negative log likelihood. We therefore see that maximizing the likelihood is equivalent, so far as determining $\mathbf{w}$ is concerned, to minimizing the sum-of-squares error function defined by

$$
E(\mathbf{w})=\frac{1}{2} \sum_{n=1}^{N}\left\{y\left(x_{n}, \mathbf{w}\right)-t_{n}\right\}^{2} .
$$

Thus, the sum-of-squares error function has arisen as a consequence of maximizing the likelihood under the assumption of a Gaussian noise distribution.

We can also use maximum likelihood to determine the variance parameter $\sigma^{2}$. Maximizing (2.66) with respect to $\sigma^{2}$ gives

$$
\sigma_{\mathrm{ML}}^{2}=\frac{1}{N} \sum_{n=1}^{N}\left\{y\left(x_{n}, \mathbf{w}_{\mathrm{ML}}\right)-t_{n}\right\}^{2}
$$

Note that we can first determine the parameter vector $\mathbf{w}_{\text {ML }}$ governing the mean, and subsequently use this to find the variance $\sigma_{\text {ML }}^{2}$ as was the case for the simple Gaussian distribution.

Having determined the parameters $\mathbf{w}$ and $\sigma^{2}$, we can now make predictions for new values of $x$. Because we now have a probabilistic model, these are expressed in terms of the predictive distribution that gives the probability distribution over $t$, rather than simply a point estimate, and is obtained by substituting the maximum likelihood parameters into (2.64) to give

$$
p\left(t \mid x, \mathbf{w}_{\mathrm{ML}}, \sigma_{\mathrm{ML}}^{2}\right)=\mathcal{N}\left(t \mid y\left(x, \mathbf{w}_{\mathrm{ML}}\right), \sigma_{\mathrm{ML}}^{2}\right)
$$

### 2.4. Transformation of Densities

We turn now to a discussion of how a probability density transforms under a nonlinear change of variable. This property will play a crucial role when we discuss a class

## Chapter 18

of generative models called normalizing flows. It also highlights that a probability density has a different behaviour than a simple function under such transformations.

Consider a single variable $x$ and suppose we make a change of variables $x= g(y)$, then a function $f(x)$ becomes a new function $\widetilde{f}(y)$ defined by

$$
\widetilde{f}(y)=f(g(y))
$$

Now consider a probability density $p_{x}(x)$, and again change variables using $x= g(y)$, giving rise to a density $p_{y}(y)$ with respect to the new variable $y$, where the suffixes denote that $p_{x}(x)$ and $p_{y}(y)$ are different densities. Observations falling in the range ( $x, x+\delta x$ ) will, for small values of $\delta x$, be transformed into the range $(y, y+\delta y)$, where $x=g(y)$, and $p_{x}(x) \delta x \simeq p_{y}(y) \delta y$. Hence, if we take the limit $\delta x \rightarrow 0$, we obtain

$$
\begin{aligned}
p_{y}(y) & =p_{x}(x)\left|\frac{\mathrm{d} x}{\mathrm{~d} y}\right| \\
& =p_{x}(g(y))\left|\frac{\mathrm{d} g}{\mathrm{~d} y}\right|
\end{aligned}
$$

Here the modulus $|\cdot|$ arises because the derivative $\mathrm{d} y / \mathrm{d} x$ could be negative, whereas the density is scaled by the ratio of lengths, which is always positive.

This procedure for transforming densities can be very powerful. Any density $p(y)$ can be obtained from a fixed density $q(x)$ that is everywhere non-zero by making a nonlinear change of variable $y=f(x)$ in which $f(x)$ is a monotonic function so that $0 \leqslant f^{\prime}(x)<\infty$.

One consequence of the transformation property (2.71) is that the concept of the maximum of a probability density is dependent on the choice of variable. Suppose $f(x)$ has a mode (i.e., a maximum) at $\widehat{x}$ so that $f^{\prime}(\widehat{x})=0$. The corresponding mode of $\widetilde{f}(y)$ will occur for a value $\widehat{y}$ obtained by differentiating both sides of (2.70) with respect to $y$ :

$$
\widetilde{f}^{\prime}(\widehat{y})=f^{\prime}(g(\widehat{y})) g^{\prime}(\widehat{y})=0
$$

Assuming $g^{\prime}(\widehat{y}) \neq 0$ at the mode, then $f^{\prime}(g(\widehat{y}))=0$. However, we know that $f^{\prime}(\widehat{x})=0$, and so we see that the locations of the mode expressed in terms of each of the variables $x$ and $y$ are related by $\widehat{x}=g(\widehat{y})$, as one would expect. Thus, finding a mode with respect to the variable $x$ is equivalent to first transforming to the variable $y$, then finding a mode with respect to $y$, and then transforming back to $x$.

Now consider the behaviour of a probability density $p_{x}(x)$ under the change of variables $x=g(y)$, where the density with respect to the new variable is $p_{y}(y)$ and is given by (2.71). To deal with the modulus in (2.71) we can write $g^{\prime}(y)=s\left|g^{\prime}(y)\right|$ where $s \in\{-1,+1\}$. Then (2.71) can be written as

$$
p_{y}(y)=p_{x}(g(y)) s g^{\prime}(y)
$$
where we have used $1 / s=s$. Differentiating both sides with respect to $y$ then gives
$$
p_{y}^{\prime}(y)=s p_{x}^{\prime}(g(y))\left\{g^{\prime}(y)\right\}^{2}+s p_{x}(g(y)) g^{\prime \prime}(y)
$$

Due to the presence of the second term on the right-hand side of (2.73), the relationship $\widehat{x}=g(\widehat{y})$ no longer holds. Thus, the value of $x$ obtained by maximizing $p_{x}(x)$ will not be the value obtained by transforming to $p_{y}(y)$ then maximizing with respect to $y$ and then transforming back to $x$. This causes modes of densities to be dependent on the choice of variables. However, for a linear transformation, the second term on the right-hand side of (2.73) vanishes, and so in this case the location of the maximum transforms according to $\widehat{x}=g(\widehat{y})$.

This effect can be illustrated with a simple example, as shown in Figure 2.12. We begin by considering a Gaussian distribution $p_{x}(x)$ over $x$ shown by the red curve in Figure 2.12. Next we draw a sample of $N=50,000$ points from this distribution and plot a histogram of their values, which as expected agrees with the distribution $p_{x}(x)$. Now consider a nonlinear change of variables from $x$ to $y$ given by

$$
x=g(y)=\ln (y)-\ln (1-y)+5
$$

The inverse of this function is given by

$$
y=g^{-1}(x)=\frac{1}{1+\exp (-x+5)}
$$
which is a logistic sigmoid function and is shown in Figure 2.12 by the blue curve.

Figure 2.12 Example of the transformation of the mode of a density under a nonlinear change of variables, illustrating the different behaviour compared to a simple function.
![](https://cdn.mathpix.com/cropped/2025_10_01_d90a4d82c88904700833g-064.jpg?height=502&width=712&top_left_y=224&top_left_x=942)

**Image Description:** The diagram displays a bivariate distribution represented with two overlapping plots. The left side features a histogram (purple) illustrating the marginal distribution \( p_y(y) \) of variable \( y \). The green curve represents a cumulative distribution function (CDF). The right side presents a histogram (red) for the marginal distribution \( p_x(x) \), with an inverse function \( g^{-1}(x) \) shown as a blue curve. The x-axis extends from 0 to 10, while the y-axis ranges from 0 to 0.5, indicating the probability density. The dashed lines connect the plots, demonstrating relationships between the variables.


If we simply transform $p_{x}(x)$ as a function of $x$ we obtain the green curve $p_{x}(g(y))$ shown in Figure 2.12, and we see that the mode of the density $p_{x}(x)$ is transformed via the sigmoid function to the mode of this curve. However, the density over $y$ transforms instead according to (2.71) and is shown by the magenta curve on the left side of the diagram. Note that this has its mode shifted relative to the mode of the green curve.

To confirm this result, we take our sample of 50,000 values of $x$, evaluate the corresponding values of $y$ using (2.75), and then plot a histogram of their values. We see that this histogram matches the magenta curve in Figure 2.12 and not the green curve.

### 2.4.1 Multivariate distributions

We can extend the result (2.71) to densities defined over multiple variables. Consider a density $p(\mathbf{x})$ over a $D$-dimensional variable $\mathbf{x}=\left(x_{1}, \ldots, x_{D}\right)^{\mathrm{T}}$, and suppose we transform to a new variable $\mathbf{y}=\left(y_{1}, \ldots, y_{D}\right)^{\mathrm{T}}$ where $\mathbf{x}=\mathbf{g}(\mathbf{y})$. Here we will limit ourselves to the case where $\mathbf{x}$ and $\mathbf{y}$ have the same dimensionality. The transformed density is then given by the generalization of (2.71) in the form

$$
p_{\mathbf{y}}(\mathbf{y})=p_{\mathbf{x}}(\mathbf{x})|\operatorname{det} \mathbf{J}|
$$
where $\mathbf{J}$ is the Jacobian matrix whose elements are given by the partial derivatives $J_{i j}=\partial g_{i} / \partial y_{j}$, so that
\$\$\mathbf{J}=\left[\begin{array}{ccc}

\frac{\partial g_{1}}{\partial y_{1}} \& \cdots \& \frac{\partial g_{1}}{\partial y_{D}} <br>
\vdots \& \ddots \& \vdots <br>
\frac{\partial g_{D}}{\partial y_{1}} \& \cdots \& \frac{\partial g_{D}}{\partial y_{D}}

\end{array}\right]\$\$

Intuitively, we can view the change of variables as expanding some regions of space and contracting others, with an infinitesimal region $\Delta \mathrm{x}$ around a point x being transformed to a region $\Delta \mathbf{y}$ around the point $\mathbf{y}=\mathbf{g}(\mathbf{x})$. The absolute value of the determinant of the Jacobian represents the ratio of these volumes and is the same factor

![](https://cdn.mathpix.com/cropped/2025_10_01_d90a4d82c88904700833g-065.jpg?height=873&width=1368&top_left_y=236&top_left_x=222)

**Image Description:** The slide contains a diagram consisting of four quadrants. The top left quadrant shows a grid (x1, x2) representation, while the top right illustrates a radial intensity pattern, likely a result of a transformation (e.g., Fourier transform). The bottom left quadrant presents a series of curves (y1, y2), indicating a mapping of the input grid, while the bottom right displays a scatter plot of red points, suggesting a distribution in the transformed space (y1, y2). The x and y axes are labeled correspondingly to denote the input and output variables.

Figure 2.13 Illustration of the effect of a change of variables on a probability distribution in two dimensions. The left column shows the transforming of the variables whereas the middle and right columns show the corresponding effects on a Gaussian distribution and on samples from that distribution, respectively.

that arises when changing variables within an integral. The formula (2.77) follows from the fact that the probability mass in region $\Delta \mathrm{x}$ is the same as the probability mass in $\Delta \mathbf{y}$. Once again, we take the modulus to ensure that the density is nonnegative.

We can illustrate this by applying a change of variables to a Gaussian distribution in two dimensions, as shown in the top row in Figure 2.13. Here the transformation
Exercise 2.20 from $\mathbf{x}$ to $\mathbf{y}$ is given by

$$
\begin{aligned}
& y_{1}=x_{1}+\tanh \left(5 x_{1}\right) \\
& y_{2}=x_{2}+\tanh \left(5 x_{2}\right)+\frac{x_{1}^{3}}{3} .
\end{aligned}
$$

Also shown on the bottom row are samples from a Gaussian distribution in $\mathbf{x}$-space along with the corresponding transformed samples in $\mathbf{y}$-space.

### 2.5. Information Theory

Probability theory forms the basis for another important framework called information theory, which quantifies the information present in a data set and which plays an important role in machine learning. Here we give a brief introduction to some of the key elements of information theory that we will need later in the book, including the important concept of entropy in its various forms. For a more comprehensive introduction to information theory, with connections to machine learning, see MacKay (2003).

### 2.5.1 Entropy

We begin by considering a discrete random variable $x$ and we ask how much information is received when we observe a specific value for this variable. The amount of information can be viewed as the 'degree of surprise' on learning the value of $x$. If we are told that a highly improbable event has just occurred, we will have received more information than if we were told that some very likely event has just occurred, and if we knew that the event was certain to happen, we would receive no information. Our measure of information content will therefore depend on the probability distribution $p(x)$, and so we look for a quantity $h(x)$ that is a monotonic function of the probability $p(x)$ and that expresses the information content. The form of $h(\cdot)$ can be found by noting that if we have two events $x$ and $y$ that are unrelated, then the information gained from observing both of them should be the sum of the information gained from each of them separately, so that $h(x, y)=h(x)+h(y)$. Two unrelated events are statistically independent and so $p(x, y)=p(x) p(y)$. From these two relationships, it is easily shown that $h(x)$ must be given by the logarithm of $p(x)$ and so we have

$$
h(x)=-\log _{2} p(x)
$$
where the negative sign ensures that information is positive or zero. Note that low probability events $x$ correspond to high information content. The choice of base for the logarithm is arbitrary, and for the moment we will adopt the convention prevalent in information theory of using logarithms to the base of 2 . In this case, as we will see shortly, the units of $h(x)$ are bits ('binary digits').

Now suppose that a sender wishes to transmit the value of a random variable to a receiver. The average amount of information that they transmit in the process is obtained by taking the expectation of (2.80) with respect to the distribution $p(x)$ and is given by

$$
\mathrm{H}[x]=-\sum_{x} p(x) \log _{2} p(x)
$$

This important quantity is called the entropy of the random variable $x$. Note that $\lim _{\epsilon \rightarrow 0}(\epsilon \ln \epsilon)=0$ and so we will take $p(x) \ln p(x)=0$ whenever we encounter a value for $x$ such that $p(x)=0$.

So far, we have given a rather heuristic motivation for the definition of information (2.80) and the corresponding entropy (2.81). We now show that these definitions
indeed possess useful properties. Consider a random variable $x$ having eight possible states, each of which is equally likely. To communicate the value of $x$ to a receiver, we would need to transmit a message of length 3 bits. Notice that the entropy of this variable is given by

$$
\mathrm{H}[x]=-8 \times \frac{1}{8} \log _{2} \frac{1}{8}=3 \text { bits. }
$$

Now consider an example (Cover and Thomas, 1991) of a variable having eight possible states $\{a, b, c, d, e, f, g, h\}$ for which the respective probabilities are given by ( $\frac{1}{2}, \frac{1}{4}, \frac{1}{8}, \frac{1}{16}, \frac{1}{64}, \frac{1}{64}, \frac{1}{64}, \frac{1}{64}$ ). The entropy in this case is given by

$$
\mathrm{H}[x]=-\frac{1}{2} \log _{2} \frac{1}{2}-\frac{1}{4} \log _{2} \frac{1}{4}-\frac{1}{8} \log _{2} \frac{1}{8}-\frac{1}{16} \log _{2} \frac{1}{16}-\frac{4}{64} \log _{2} \frac{1}{64}=2 \text { bits. }
$$

We see that the nonuniform distribution has a smaller entropy than the uniform one, and we will gain some insight into this shortly when we discuss the interpretation of entropy in terms of disorder. For the moment, let us consider how we would transmit the identity of the variable's state to a receiver. We could do this, as before, using a 3-bit number. However, we can take advantage of the nonuniform distribution by using shorter codes for the more probable events, at the expense of longer codes for the less probable events, in the hope of getting a shorter average code length. This can be done by representing the states $\{a, b, c, d, e, f, g, h\}$ using, for instance, the following set of code strings: $0,10,110,1110,111100,111101,111110$, and 111111. The average length of the code that has to be transmitted is then
average code length $=\frac{1}{2} \times 1+\frac{1}{4} \times 2+\frac{1}{8} \times 3+\frac{1}{16} \times 4+4 \times \frac{1}{64} \times 6=2$ bits,
which again is the same as the entropy of the random variable. Note that shorter code strings cannot be used because it must be possible to disambiguate a concatenation of such strings into its component parts. For instance, 11001110 decodes uniquely into the state sequence $c, a, d$. This relation between entropy and shortest coding length is a general one. The noiseless coding theorem (Shannon, 1948) states that the entropy is a lower bound on the number of bits needed to transmit the state of a random variable.

From now on, we will switch to the use of natural logarithms in defining entropy, as this will provide a more convenient link with ideas elsewhere in this book. In this case, the entropy is measured in units of nats (from 'natural logarithm') instead of bits, which differ simply by a factor of $\ln 2$.

### 2.5.2 Physics perspective

We have introduced the concept of entropy in terms of the average amount of information needed to specify the state of a random variable. In fact, the concept of entropy has much earlier origins in physics where it was introduced in the context of equilibrium thermodynamics and later given a deeper interpretation as a measure of disorder through developments in statistical mechanics. We can understand this alternative view of entropy by considering a set of $N$ identical objects that are to be divided amongst a set of bins, such that there are $n_{i}$ objects in the $i$ th bin. Consider
the number of different ways of allocating the objects to the bins. There are $N$ ways to choose the first object, ( $N-1$ ) ways to choose the second object, and so on, leading to a total of $N!$ ways to allocate all $N$ objects to the bins, where $N$ ! (pronounced ' $N$ factorial') denotes the product $N \times(N-1) \times \cdots \times 2 \times 1$. However, we do not wish to distinguish between rearrangements of objects within each bin. In the $i$ th bin there are $n_{i}$ ! ways of reordering the objects, and so the total number of ways of allocating the $N$ objects to the bins is given by

$$
W=\frac{N!}{\prod_{i} n_{i}!},
$$
which is called the multiplicity. The entropy is then defined as the logarithm of the multiplicity scaled by a constant factor $1 / N$ so that
$$
\mathrm{H}=\frac{1}{N} \ln W=\frac{1}{N} \ln N!-\frac{1}{N} \sum_{i} \ln n_{i}!.
$$

We now consider the limit $N \rightarrow \infty$, in which the fractions $n_{i} / N$ are held fixed, and apply Stirling's approximation:

$$
\ln N!\simeq N \ln N-N,
$$
which gives
$$
\mathrm{H}=-\lim _{N \rightarrow \infty} \sum_{i}\left(\frac{n_{i}}{N}\right) \ln \left(\frac{n_{i}}{N}\right)=-\sum_{i} p_{i} \ln p_{i}
$$

where we have used $\sum_{i} n_{i}=N$. Here $p_{i}=\lim _{N \rightarrow \infty}\left(n_{i} / N\right)$ is the probability of an object being assigned to the $i$ th bin. In physics terminology, the specific allocation of objects into bins is called a microstate, and the overall distribution of occupation numbers, expressed through the ratios $n_{i} / N$, is called a macrostate. The multiplicity $W$, which expresses the number of microstates in a given macrostate, is also known as the weight of the macrostate.

We can interpret the bins as the states $x_{i}$ of a discrete random variable $X$, where $p\left(X=x_{i}\right)=p_{i}$. The entropy of the random variable $X$ is then

$$
\mathrm{H}[p]=-\sum_{i} p\left(x_{i}\right) \ln p\left(x_{i}\right) .
$$

Distributions $p\left(x_{i}\right)$ that are sharply peaked around a few values will have a relatively low entropy, whereas those that are spread more evenly across many values will have higher entropy, as illustrated in Figure 2.14.

Because $0 \leqslant p_{i} \leqslant 1$, the entropy is non-negative, and it will equal its minimum value of 0 when one of the $p_{i}=1$ and all other $p_{j \neq i}=0$. The maximum entropy configuration can be found by maximizing H using a Lagrange multiplier to enforce the normalization constraint on the probabilities. Thus, we maximize

$$
\widetilde{\mathrm{H}}=-\sum_{i} p\left(x_{i}\right) \ln p\left(x_{i}\right)+\lambda\left(\sum_{i} p\left(x_{i}\right)-1\right)
$$

![](https://cdn.mathpix.com/cropped/2025_10_01_d90a4d82c88904700833g-069.jpg?height=629&width=1491&top_left_y=280&top_left_x=139)

**Image Description:** The image consists of two probability histograms side by side. The left histogram exhibits a peaked distribution with a height of approximately 0.5 and an entropy value \( H = 1.77 \). The right histogram depicts a flatter distribution with a maximum value around 0.25 and an entropy value \( H = 3.09 \). Both histograms are enclosed within a black outline and feature blue shading. The vertical axis represents probabilities, while the horizontal axis represents possible outcomes. The contrast in shapes indicates varying uncertainty in the distributions.

Figure 2.14 Histograms of two probability distributions over 30 bins illustrating the higher value of the entropy H for the broader distribution. The largest entropy would arise from a uniform distribution which would give $\mathrm{H}=-\ln (1 / 30)=3.40$.

Exercise 2.22
Exercise 2.23
from which we find that all of the $p\left(x_{i}\right)$ are equal and are given by $p\left(x_{i}\right)=1 / M$ where $M$ is the total number of states $x_{i}$. The corresponding value of the entropy is then $\mathrm{H}=\ln M$. This result can also be derived from Jensen's inequality (to be discussed shortly). To verify that the stationary point is indeed a maximum, we can evaluate the second derivative of the entropy, which gives

$$
\frac{\partial \widetilde{\mathrm{H}}}{\partial p\left(x_{i}\right) \partial p\left(x_{j}\right)}=-I_{i j} \frac{1}{p_{i}}
$$
where $I_{i j}$ are the elements of the identity matrix. We see that these values are all negative and, hence, the stationary point is indeed a maximum.

### 2.5.3 Differential entropy

We can extend the definition of entropy to include distributions $p(x)$ over continuous variables $x$ as follows. First divide $x$ into bins of width $\Delta$. Then, assuming that $p(x)$ is continuous, the mean value theorem (Weisstein, 1999) tells us that, for each such bin, there must exist a value $x_{i}$ in the range $i \Delta \leqslant x_{i} \leqslant(i+1) \Delta$ such that

$$
\int_{i \Delta}^{(i+1) \Delta} p(x) \mathrm{d} x=p\left(x_{i}\right) \Delta .
$$

We can now quantize the continuous variable $x$ by assigning any value $x$ to the value $x_{i}$ whenever $x$ falls in the $i$ th bin. The probability of observing the value $x_{i}$ is then
$p\left(x_{i}\right) \Delta$. This gives a discrete distribution for which the entropy takes the form

$$
\mathrm{H}_{\Delta}=-\sum_{i} p\left(x_{i}\right) \Delta \ln \left(p\left(x_{i}\right) \Delta\right)=-\sum_{i} p\left(x_{i}\right) \Delta \ln p\left(x_{i}\right)-\ln \Delta
$$
where we have used $\sum_{i} p\left(x_{i}\right) \Delta=1$, which follows from (2.89) and (2.25). We now omit the second term $-\ln \Delta$ on the right-hand side of (2.90), since it is independent of $p(x)$, and then consider the limit $\Delta \rightarrow 0$. The first term on the right-hand side of (2.90) will approach the integral of $p(x) \ln p(x)$ in this limit so that
$$
\lim _{\Delta \rightarrow 0}\left\{-\sum_{i} p\left(x_{i}\right) \Delta \ln p\left(x_{i}\right)\right\}=-\int p(x) \ln p(x) \mathrm{d} x
$$

where the quantity on the right-hand side is called the differential entropy. We see that the discrete and continuous forms of the entropy differ by a quantity $\ln \Delta$, which diverges in the limit $\Delta \rightarrow 0$. This reflects that specifying a continuous variable very precisely requires a large number of bits. For a density defined over multiple continuous variables, denoted collectively by the vector $\mathbf{x}$, the differential entropy is given by

$$
\mathrm{H}[\mathbf{x}]=-\int p(\mathbf{x}) \ln p(\mathbf{x}) \mathrm{d} \mathbf{x}
$$

### 2.5.4 Maximum entropy

We saw for discrete distributions that the maximum entropy configuration corresponds to a uniform distribution of probabilities across the possible states of the variable. Let us now consider the corresponding result for a continuous variable. If this maximum is to be well defined, it will be necessary to constrain the first and second moments of $p(x)$ and to preserve the normalization constraint. We therefore maximize the differential entropy with the three constraints:

$$
\begin{aligned}
\int_{-\infty}^{\infty} p(x) \mathrm{d} x & =1 \\
\int_{-\infty}^{\infty} x p(x) \mathrm{d} x & =\mu \\
\int_{-\infty}^{\infty}(x-\mu)^{2} p(x) \mathrm{d} x & =\sigma^{2}
\end{aligned}
$$

The constrained maximization can be performed using Lagrange multipliers so that we maximize the following functional with respect to $p(x)$ :

$$
\begin{aligned}
& -\int_{-\infty}^{\infty} p(x) \ln p(x) \mathrm{d} x+\lambda_{1}\left(\int_{-\infty}^{\infty} p(x) \mathrm{d} x-1\right) \\
& \quad+\lambda_{2}\left(\int_{-\infty}^{\infty} x p(x) \mathrm{d} x-\mu\right)+\lambda_{3}\left(\int_{-\infty}^{\infty}(x-\mu)^{2} p(x) \mathrm{d} x-\sigma^{2}\right)
\end{aligned}
$$

Appendix B

Exercise 2.24

Exercise 2.25

Using the calculus of variations, we set the derivative of this functional to zero giving

$$
p(x)=\exp \left\{-1+\lambda_{1}+\lambda_{2} x+\lambda_{3}(x-\mu)^{2}\right\}
$$

The Lagrange multipliers can be found by back-substitution of this result into the three constraint equations, leading finally to the result:

$$
p(x)=\frac{1}{\left(2 \pi \sigma^{2}\right)^{1 / 2}} \exp \left\{-\frac{(x-\mu)^{2}}{2 \sigma^{2}}\right\}
$$
and so the distribution that maximizes the differential entropy is the Gaussian. Note that we did not constrain the distribution to be non-negative when we maximized the entropy. However, because the resulting distribution is indeed non-negative, we see with hindsight that such a constraint is not necessary.

If we evaluate the differential entropy of the Gaussian, we obtain

$$
\mathrm{H}[x]=\frac{1}{2}\left\{1+\ln \left(2 \pi \sigma^{2}\right)\right\}
$$

Thus, we see again that the entropy increases as the distribution becomes broader, i.e., as $\sigma^{2}$ increases. This result also shows that the differential entropy, unlike the discrete entropy, can be negative, because $\mathrm{H}(x)<0$ in (2.99) for $\sigma^{2}<1 /(2 \pi e)$.

### 2.5.5 Kullback-Leibler divergence

So far in this section, we have introduced a number of concepts from information theory, including the key notion of entropy. We now start to relate these ideas to machine learning. Consider some unknown distribution $p(\mathbf{x})$, and suppose that we have modelled this using an approximating distribution $q(\mathbf{x})$. If we use $q(\mathbf{x})$ to construct a coding scheme for transmitting values of $\mathbf{x}$ to a receiver, then the average additional amount of information (in nats) required to specify the value of $\mathbf{x}$ (assuming we choose an efficient coding scheme) as a result of using $q(\mathbf{x})$ instead of the true distribution $p(\mathbf{x})$ is given by

$$
\begin{aligned}
\mathrm{KL}(p \| q) & =-\int p(\mathbf{x}) \ln q(\mathbf{x}) \mathrm{d} \mathbf{x}-\left(-\int p(\mathbf{x}) \ln p(\mathbf{x}) \mathrm{d} \mathbf{x}\right) \\
& =-\int p(\mathbf{x}) \ln \left\{\frac{q(\mathbf{x})}{p(\mathbf{x})}\right\} \mathrm{d} \mathbf{x}
\end{aligned}
$$

This is known as the relative entropy or Kullback-Leibler divergence, or KL divergence (Kullback and Leibler, 1951), between the distributions $p(\mathbf{x})$ and $q(\mathbf{x})$. Note that it is not a symmetrical quantity, that is to say $\mathrm{KL}(p \| q) \not \equiv \mathrm{KL}(q \| p)$.

We now show that the Kullback-Leibler divergence satisfies $\operatorname{KL}(p \| q) \geqslant 0$ with equality if, and only if, $p(x)=q(x)$. To do this we first introduce the concept of convex functions. A function $f(x)$ is said to be convex if it has the property that every chord lies on or above the function, as shown in Figure 2.15.

Any value of $x$ in the interval from $x=a$ to $x=b$ can be written in the form $\lambda a+(1-\lambda) b$ where $0 \leqslant \lambda \leqslant 1$. The corresponding point on the chord

Figure 2.15 A convex function $f(x)$ is one for which every chord (shown in blue) lies on or above the function (shown in red).
![](https://cdn.mathpix.com/cropped/2025_10_01_d90a4d82c88904700833g-072.jpg?height=546&width=626&top_left_y=219&top_left_x=1023)

**Image Description:** The image depicts a mathematical function graphed on a Cartesian coordinate system. The x-axis represents the variable \( x \), while the y-axis represents the function \( f(x) \). A red curve illustrates the function, with two vertical green lines denoting points \( a \) and \( b \) on the x-axis. A horizontal blue line illustrates a chord between these two points, intersecting the curve. The point \( x_\lambda \) lies on the chord, labeled with an arrow. The visual emphasizes the relationship between the chord and the curve in discussing secants and calculus concepts.

is given by $\lambda f(a)+(1-\lambda) f(b)$, and the corresponding value of the function is $f(\lambda a+(1-\lambda) b)$. Convexity then implies

$$
f(\lambda a+(1-\lambda) b) \leqslant \lambda f(a)+(1-\lambda) f(b)
$$

Exercise 2.32

Exercise 2.33

This is equivalent to the requirement that the second derivative of the function be everywhere positive. Examples of convex functions are $x \ln x$ (for $x>0$ ) and $x^{2}$. A function is called strictly convex if the equality is satisfied only for $\lambda=0$ and $\lambda=1$. If a function has the opposite property, namely that every chord lies on or below the function, it is called concave, with a corresponding definition for strictly concave. If a function $f(x)$ is convex, then $-f(x)$ will be concave.

Using the technique of proof by induction, we can show from (2.101) that a convex function $f(x)$ satisfies

$$
f\left(\sum_{i=1}^{M} \lambda_{i} x_{i}\right) \leqslant \sum_{i=1}^{M} \lambda_{i} f\left(x_{i}\right)
$$
where $\lambda_{i} \geqslant 0$ and $\sum_{i} \lambda_{i}=1$, for any set of points $\left\{x_{i}\right\}$. The result (2.102) is known as Jensen's inequality. If we interpret the $\lambda_{i}$ as the probability distribution over a discrete variable $x$ taking the values $\left\{x_{i}\right\}$, then (2.102) can be written
$$
f(\mathbb{E}[x]) \leqslant \mathbb{E}[f(x)]
$$

where $\mathbb{E}[\cdot]$ denotes the expectation. For continuous variables, Jensen's inequality takes the form

$$
f\left(\int \mathbf{x} p(\mathbf{x}) \mathrm{d} \mathbf{x}\right) \leqslant \int f(\mathbf{x}) p(\mathbf{x}) \mathrm{d} \mathbf{x}
$$

We can apply Jensen's inequality in the form (2.104) to the Kullback-Leibler divergence (2.100) to give

$$
\mathrm{KL}(p \| q)=-\int p(\mathbf{x}) \ln \left\{\frac{q(\mathbf{x})}{p(\mathbf{x})}\right\} \mathrm{d} \mathbf{x} \geqslant-\ln \int q(\mathbf{x}) \mathrm{d} \mathbf{x}=0
$$

## Exercise 2.34

## Exercise 2.35

where we have used $-\ln x$ is a convex function, together with the normalization condition $\int q(\mathbf{x}) \mathrm{d} \mathbf{x}=1$. In fact, $-\ln x$ is a strictly convex function, so the equality will hold if, and only if, $q(\mathbf{x})=p(\mathbf{x})$ for all $\mathbf{x}$. Thus, we can interpret the KullbackLeibler divergence as a measure of the dissimilarity of the two distributions $p(\mathbf{x})$ and $q(\mathbf{x})$.

We see that there is an intimate relationship between data compression and density estimation (i.e., the problem of modelling an unknown probability distribution) because the most efficient compression is achieved when we know the true distribution. If we use a distribution that is different from the true one, then we must necessarily have a less efficient coding, and on average the additional information that must be transmitted is (at least) equal to the Kullback-Leibler divergence between the two distributions.

Suppose that data is being generated from an unknown distribution $p(\mathbf{x})$ that we wish to model. We can try to approximate this distribution using some parametric distribution $q(\mathbf{x} \mid \boldsymbol{\theta})$, governed by a set of adjustable parameters $\boldsymbol{\theta}$. One way to determine $\boldsymbol{\theta}$ is to minimize the Kullback-Leibler divergence between $p(\mathbf{x})$ and $q(\mathbf{x} \mid \boldsymbol{\theta})$ with respect to $\boldsymbol{\theta}$. We cannot do this directly because we do not know $p(\mathbf{x})$. Suppose, however, that we have observed a finite set of training points $\mathbf{x}_{n}$, for $n=1, \ldots, N$, drawn from $p(\mathbf{x})$. Then the expectation with respect to $p(\mathbf{x})$ can be approximated by a finite sum over these points, using (2.40), so that

$$
\mathrm{KL}(p \| q) \simeq \frac{1}{N} \sum_{n=1}^{N}\left\{-\ln q\left(\mathbf{x}_{n} \mid \boldsymbol{\theta}\right)+\ln p\left(\mathbf{x}_{n}\right)\right\} .
$$

The second term on the right-hand side of (2.106) is independent of $\boldsymbol{\theta}$, and the first term is the negative log likelihood function for $\boldsymbol{\theta}$ under the distribution $q(\mathbf{x} \mid \boldsymbol{\theta})$ evaluated using the training set. Thus, we see that minimizing this Kullback-Leibler divergence is equivalent to maximizing the log likelihood function.

### 2.5.6 Conditional entropy

Now consider the joint distribution between two sets of variables $\mathbf{x}$ and $\mathbf{y}$ given by $p(\mathbf{x}, \mathbf{y})$ from which we draw pairs of values of $\mathbf{x}$ and $\mathbf{y}$. If a value of $\mathbf{x}$ is already known, then the additional information needed to specify the corresponding value of $\mathbf{y}$ is given by $-\ln p(\mathbf{y} \mid \mathbf{x})$. Thus the average additional information needed to specify y can be written as

$$
\mathrm{H}[\mathbf{y} \mid \mathbf{x}]=-\iint p(\mathbf{y}, \mathbf{x}) \ln p(\mathbf{y} \mid \mathbf{x}) \mathrm{d} \mathbf{y} \mathrm{~d} \mathbf{x}
$$
which is called the conditional entropy of $\mathbf{y}$ given $\mathbf{x}$. It is easily seen, using the product rule, that the conditional entropy satisfies the relation:
$$
\mathrm{H}[\mathbf{x}, \mathbf{y}]=\mathrm{H}[\mathbf{y} \mid \mathbf{x}]+\mathrm{H}[\mathbf{x}]
$$

where $\mathrm{H}[\mathbf{x}, \mathbf{y}]$ is the differential entropy of $p(\mathbf{x}, \mathbf{y})$ and $\mathrm{H}[\mathbf{x}]$ is the differential entropy of the marginal distribution $p(\mathbf{x})$. Thus, the information needed to describe $\mathbf{x}$ and $\mathbf{y}$ is given by the sum of the information needed to describe $\mathbf{x}$ alone plus the additional information required to specify $\mathbf{y}$ given $\mathbf{x}$.

### 2.5.7 Mutual information

When two variables $\mathbf{x}$ and $\mathbf{y}$ are independent, their joint distribution will factorize into the product of their marginals $p(\mathbf{x}, \mathbf{y})=p(\mathbf{x}) p(\mathbf{y})$. If the variables are not independent, we can gain some idea of whether they are 'close' to being independent by considering the Kullback-Leibler divergence between the joint distribution and the product of the marginals, given by

$$
\begin{aligned}
\mathrm{I}[\mathbf{x}, \mathbf{y}] & \equiv \mathrm{KL}(p(\mathbf{x}, \mathbf{y}) \| p(\mathbf{x}) p(\mathbf{y})) \\
& =-\iint p(\mathbf{x}, \mathbf{y}) \ln \left(\frac{p(\mathbf{x}) p(\mathbf{y})}{p(\mathbf{x}, \mathbf{y})}\right) \mathrm{d} \mathbf{x} \mathrm{~d} \mathbf{y}
\end{aligned}
$$

Exercise 2.38
which is called the mutual information between the variables $\mathbf{x}$ and $\mathbf{y}$. From the properties of the Kullback-Leibler divergence, we see that $I[\mathbf{x}, \mathbf{y}] \geqslant 0$ with equality if, and only if, $\mathbf{x}$ and $\mathbf{y}$ are independent. Using the sum and product rules of probability, we see that the mutual information is related to the conditional entropy through

$$
\mathrm{I}[\mathbf{x}, \mathbf{y}]=\mathrm{H}[\mathbf{x}]-\mathrm{H}[\mathbf{x} \mid \mathbf{y}]=\mathrm{H}[\mathbf{y}]-\mathrm{H}[\mathbf{y} \mid \mathbf{x}]
$$

Thus, the mutual information represents the reduction in the uncertainty about $\mathbf{x}$ by virtue of being told the value of $\mathbf{y}$ (or vice versa). From a Bayesian perspective, we can view $p(\mathbf{x})$ as the prior distribution for $\mathbf{x}$ and $p(\mathbf{x} \mid \mathbf{y})$ as the posterior distribution after we have observed new data $\mathbf{y}$. The mutual information therefore represents the reduction in uncertainty about $\mathbf{x}$ as a consequence of the new observation $\mathbf{y}$.

### 2.6. Bayesian Probabilities

When we considered the bent coin in Figure 2.2, we introduced the concept of probability in terms of the frequencies of random, repeatable events, such as the probability of the coin landing concave side up. We will refer to this as the classical or frequentist interpretation of probability. We also introduced the more general Bayesian view, in which probabilities provide a quantification of uncertainty. In this case, our uncertainty is whether the concave side of the coin is heads or tails.

The use of probability to represent uncertainty is not an ad hoc choice but is inevitable if we are to respect common sense while making rational and coherent inferences. For example, Cox (1946) showed that if numerical values are used to represent degrees of belief, then a simple set of axioms encoding common sense properties of such beliefs leads uniquely to a set of rules for manipulating degrees of belief that are equivalent to the sum and product rules of probability. It is therefore natural to refer to these quantities as (Bayesian) probabilities.

For the bent coin we assumed, in the absence of further information, that the probability of the concave side of the coin being heads is 0.5 . Now suppose we are told the results of flipping the coin a few times. Intuitively, it seems that this should provide us with some information as to whether the concave side is heads. For instance, suppose we see many more flips that land tails than land heads. Given

Exercise 2.40

Section 3.1.2

Section 1.2
that the coin is more likely to land concave side up, this provides evidence to suggest that the concave side is more likely to be tails. In fact, this intuition is correct, and furthermore, we can quantify this using the rules of probability. Bayes' theorem now acquires a new significance, because it allows us to convert the prior probability for the concave side being heads into a posterior probability by incorporating the data provided by the coin flips. Moreover, this process is iterative, meaning the posterior probability becomes the prior for incorporating data from further coin flips.

One aspect of the Bayesian viewpoint is that the inclusion of prior knowledge arises naturally. Suppose, for instance, that a fair-looking coin is tossed three times and lands heads each time. The maximum likelihood estimate of the probability of landing heads would give 1 , implying that all future tosses will land heads! By contrast, a Bayesian approach with any reasonable prior will lead to a less extreme conclusion.

### 2.6.1 Model parameters

The Bayesian perspective provides valuable insights into several aspects of machine learning, and we can illustrate these using the sine curve regression example. Here we denote the training data set by $\mathcal{D}$. We have already seen in the context of linear regression that the parameters can be chosen using maximum likelihood, in which $\mathbf{w}$ is set to the value that maximizes the likelihood function $p(\mathcal{D} \mid \mathbf{w})$. This corresponds to choosing the value of $\mathbf{w}$ for which the probability of the observed data set is maximized. In the machine learning literature, the negative log of the likelihood function is called an error function. Because the negative logarithm is a monotonically decreasing function, maximizing the likelihood is equivalent to minimizing the error. This leads to a specific choice of parameter values, denoted $\mathbf{w}_{\text {ML }}$, which are then used to make predictions for new data.

We have seen that different choices of training data set, for example containing different numbers of data points, give rise to different solutions for $\mathbf{w}_{\text {ML }}$. From a Bayesian perspective, we can also use the machinery of probability theory to describe this uncertainty in the model parameters. We can capture our assumptions about $\mathbf{w}$, before observing the data, in the form of a prior probability distribution $p(\mathbf{w})$. The effect of the observed data $\mathcal{D}$ is expressed through the likelihood function $p(\mathcal{D} \mid \mathbf{w})$, and Bayes' theorem now takes the form

$$
p(\mathbf{w} \mid \mathcal{D})=\frac{p(\mathcal{D} \mid \mathbf{w}) p(\mathbf{w})}{p(\mathcal{D})},
$$
which allows us to evaluate the uncertainty in $\mathbf{w}$ after we have observed $\mathcal{D}$ in the form of the posterior probability $p(\mathbf{w} \mid \mathcal{D})$.

It is important to emphasize that the quantity $p(\mathcal{D} \mid \mathbf{w})$ is called the likelihood function when it is viewed as a function of the parameter vector $\mathbf{w}$, and it expresses how probable the observed data set is for different values of $\mathbf{w}$. Note that the likeli$\operatorname{hood} p(\mathcal{D} \mid \mathbf{w})$ is not a probability distribution over $\mathbf{w}$, and its integral with respect to w does not (necessarily) equal one.

Given this definition of likelihood, we can state Bayes' theorem in words:

$$
\text { posterior } \propto \text { likelihood } \times \text { prior }
$$
where all of these quantities are viewed as functions of $\mathbf{w}$. The denominator in (2.111) is the normalization constant, which ensures that the posterior distribution on the left-hand side is a valid probability density and integrates to one. Indeed, by integrating both sides of (2.111) with respect to $\mathbf{w}$, we can express the denominator in Bayes' theorem in terms of the prior distribution and the likelihood function:
$$
p(\mathcal{D})=\int p(\mathcal{D} \mid \mathbf{w}) p(\mathbf{w}) \mathrm{d} \mathbf{w}
$$

In both the Bayesian and frequentist paradigms, the likelihood function $p(\mathcal{D} \mid \mathbf{w})$ plays a central role. However, the manner in which it is used is fundamentally different in the two approaches. In a frequentist setting, $\mathbf{w}$ is considered to be a fixed parameter, whose value is determined by some form of 'estimator', and error bars on this estimate are determined (conceptually, at least) by considering the distribution of possible data sets $\mathcal{D}$. By contrast, from the Bayesian viewpoint there is only a single data set $\mathcal{D}$ (namely the one that is actually observed), and the uncertainty in the parameters is expressed through a probability distribution over $\mathbf{w}$.

### 2.6.2 Regularization

We can use this Bayesian perspective to gain insight into the technique of regu- larization that was used in the sine curve regression example to reduce over-fitting. Instead of choosing the model parameters by maximizing the likelihood function with respect to $\mathbf{w}$, we can maximize the posterior probability (2.111). This technique is called the maximum a posteriori estimate, or simply MAP estimate. Equivalently, we can minimize the negative log of the posterior probability. Taking negative logs of both sides of (2.111), we have

$$
-\ln p(\mathbf{w} \mid \mathcal{D})=-\ln p(\mathcal{D} \mid \mathbf{w})-\ln p(\mathbf{w})+\ln p(\mathcal{D})
$$

The first term on the right-hand side of (2.114) is the usual log likelihood. The third term can be omitted since it does not depend on $\mathbf{w}$. The second term takes the form of a function of $\mathbf{w}$, which is added to the log likelihood, and we can recognize this as a form of regularization. To make this more explicit, suppose we choose the prior distribution $p(\mathbf{w})$ to be the product of independent zero-mean Gaussian distributions for each of the elements of $\mathbf{w}$ such that each has the same variance $s^{2}$ so that

$$
p(\mathbf{w} \mid s)=\prod_{i=0}^{M} \mathcal{N}\left(w_{i} \mid 0, s^{2}\right)=\prod_{i=0}^{M}\left(\frac{1}{2 \pi s^{2}}\right)^{1 / 2} \exp \left\{-\frac{w_{i}^{2}}{2 s^{2}}\right\}
$$

Substituting into (2.114), we obtain

$$
-\ln p(\mathbf{w} \mid \mathcal{D})=-\ln p(\mathcal{D} \mid \mathbf{w})+\frac{1}{2 s^{2}} \sum_{i=0}^{M} w_{i}^{2}+\text { const. }
$$

If we consider the particular case of the linear regression model whose log likelihood is given by (2.66), then we find that maximizing the posterior distribution is equivalent to minimizing the function

$$
E(\mathbf{w})=\frac{1}{2 \sigma^{2}} \sum_{n=1}^{N}\left\{y\left(x_{n}, \mathbf{w}\right)-t_{n}\right\}^{2}+\frac{1}{2 s^{2}} \mathbf{w}^{\mathrm{T}} \mathbf{w} .
$$

We see that this takes the form of the regularized sum-of-squares error function encountered earlier in the form (1.4).

### 2.6.3 Bayesian machine learning

The Bayesian perspective has allowed us to motivate the use of regularization and to derive a specific form for the regularization term. However, the use of Bayes' theorem alone does not constitute a truly Bayesian treatment of machine learning since it is still finding a single solution for $\mathbf{w}$ and does not therefore take account of uncertainty in the value of $\mathbf{w}$. Suppose we have a training data set $\mathcal{D}$ and our goal is to predict some target variable $t$ given a new input value $x$. We are therefore interested in the distribution of $t$ given both $x$ and $\mathcal{D}$. From the sum and product rules of probability, we have

$$
p(t \mid x, \mathcal{D})=\int p(t \mid x, \mathbf{w}) p(\mathbf{w} \mid \mathcal{D}) \mathrm{d} \mathbf{w} .
$$

We see that the prediction is obtained by taking a weighted average $p(t \mid x, \mathbf{w})$ over all possible values of $\mathbf{w}$ in which the weighting function is given by the posterior probability distribution $p(\mathbf{w} \mid \mathcal{D})$. The key difference that distinguishes Bayesian methods is this integration over the space of parameters. By contrast, conventional frequentist methods use point estimates for parameters obtained by optimizing a loss function such as a regularized sum-of-squares.

This fully Bayesian treatment of machine learning offers some powerful insights. For example, the problem of over-fitting, encountered earlier in the context

## Section 1.2

## Section 9.6

of polynomial regression, is an example of a pathology arising from the use of maximum likelihood, and does not arise when we marginalize over parameters using the Bayesian approach. Similarly, we may have multiple potential models that we could use to solve a given problem, such as polynomials of different orders in the regression example. A maximum likelihood approach simply picks the model that gives the highest probability of the data, but this favours ever more complex models, leading to over-fitting. A fully Bayesian treatment involves averaging over all possible models, with the contribution of each model weighted by its posterior probability. Moreover, this probability is typically highest for models of intermediate complexity. Very simple models (such as polynomials of low order) have low probability as they are unable to fit the data well, whereas very complex models (such as polynomials of very high order) also have low probability because the Bayesian integration over parameters automatically and elegantly penalizes complexity. For a comprehensive overview of Bayesian methods applied to machine learning, including neural networks, see Bishop (2006).

Unfortunately, there is a major drawback with the Bayesian framework, and this is apparent in (2.118), which involves integrating over the space of parameters. Modern deep learning models can have millions or billions of parameters and even simple approximations to such integrals are typically infeasible. In fact, given a
limited compute budget and an ample source of training data, it will often be better to apply maximum likelihood techniques, generally augmented with one or more forms of regularization, to a large neural network rather than apply a Bayesian treatment to a much smaller model.

## Exercises

$2.1(\star)$ In the cancer screening example, we used a prior probability of cancer of $p(C= 1)=0.01$. In reality, the prevalence of cancer is generally very much lower. Consider a situation in which $p(C=1)=0.001$, and recompute the probability of having cancer given a positive test $p(C=1 \mid T=1)$. Intuitively, the result can appear surprising to many people since the test seems to have high accuracy and yet a positive test still leads to a low probability of having cancer.
$2.2(\star \star)$ Deterministic numbers satisfy the property of transitivity, so that if $x>y$ and $y>z$ then it follows that $x>z$. When we go to random numbers, however, this property need no longer apply. Figure 2.16 shows a set of four cubical dice that have been arranged in a cyclic order. Show that each of the four dice has a $2 / 3$ probability of rolling a higher number than the previous die in the cycle. Such dice are known as non-transitive dice, and the specific examples shown here are called Efron dice.

Figure 2.16 An example of non-transitive cubical dice, in which each die has been 'flattened' to reveal the numbers on each of the faces. The dice have been arranged in a cycle, such that each die has a $2 / 3$ probability of rolling a higher number than the previous die in the cycle.
![](https://cdn.mathpix.com/cropped/2025_10_01_d90a4d82c88904700833g-078.jpg?height=494&width=455&top_left_y=1084&top_left_x=1077)

**Image Description:** The image presents a flow diagram illustrating a transition system involving four states, represented as grids. Each grid contains numbers arranged in a 3x3 matrix format. The states are labeled with different colors: red, green, blue, and orange. Arrows connect the states, indicating transitions between them. The numbers in the matrices signify specific values associated with each state. The diagram effectively visualizes the dynamics of state changes in a system.


2.3 (*) Consider a variable $\mathbf{y}$ given by the sum of two independent random variables $\mathbf{y}=\mathbf{u}+\mathbf{v}$ where $\mathbf{u} \sim p_{\mathbf{u}}(\mathbf{u})$ and $\mathbf{v} \sim p_{\mathbf{v}}(\mathbf{v})$. Show that the distribution $p_{\mathbf{y}}(\mathbf{y})$ is given by

$$
p(\mathbf{y})=\int p_{\mathbf{u}}(\mathbf{u}) p_{\mathbf{v}}(\mathbf{y}-\mathbf{u}) \mathrm{d} \mathbf{u}
$$

This is known as the convolution of $p_{\mathbf{u}}(\mathbf{u})$ and $p_{\mathbf{v}}(\mathbf{v})$.
$2.4(\star \star)$ Verify that the uniform distribution (2.33) is correctly normalized, and find expressions for its mean and variance.
$2.5(\star \star)$ Verify that the exponential distribution (2.34) and the Laplace distribution (2.35) are correctly normalized.
2.6 ( $\star$ ) Using the properties of the Dirac delta function, show that the empirical density (2.37) is correctly normalized.
2.7 ( $\star$ ) By making use of the empirical density (2.37), show that the expectation given by (2.39) can be approximated by a sum over a finite set of samples drawn from the density in the form (2.40).
$2.8(\star)$ Using the definition (2.44), show that $\operatorname{var}[f(x)]$ satisfies (2.45).
$2.9(\star)$ Show that if two variables $x$ and $y$ are independent, then their covariance is zero.
2.10 (*) Suppose that the two variables $x$ and $z$ are statistically independent. Show that the mean and variance of their sum satisfies

$$
\begin{aligned}
\mathbb{E}[x+z] & =\mathbb{E}[x]+\mathbb{E}[z] \\
\operatorname{var}[x+z] & =\operatorname{var}[x]+\operatorname{var}[z]
\end{aligned}
$$

2.11 ( $\star$ ) Consider two variables $x$ and $y$ with joint distribution $p(x, y)$. Prove the following two results:

$$
\begin{aligned}
\mathbb{E}[x] & =\mathbb{E}_{y}\left[\mathbb{E}_{x}[x \mid y]\right] \\
\operatorname{var}[x] & =\mathbb{E}_{y}\left[\operatorname{var}_{x}[x \mid y]\right]+\operatorname{var}_{y}\left[\mathbb{E}_{x}[x \mid y]\right]
\end{aligned}
$$

Here $\mathbb{E}_{x}[x \mid y]$ denotes the expectation of $x$ under the conditional distribution $p(x \mid y)$, with a similar notation for the conditional variance.
$2.12(\star \star \star)$ In this exercise, we prove the normalization condition (2.51) for the univariate Gaussian. To do this consider, the integral

$$
I=\int_{-\infty}^{\infty} \exp \left(-\frac{1}{2 \sigma^{2}} x^{2}\right) \mathrm{d} x
$$
which we can evaluate by first writing its square in the form
$$
I^{2}=\int_{-\infty}^{\infty} \int_{-\infty}^{\infty} \exp \left(-\frac{1}{2 \sigma^{2}} x^{2}-\frac{1}{2 \sigma^{2}} y^{2}\right) \mathrm{d} x \mathrm{~d} y
$$

Now make the transformation from Cartesian coordinates $(x, y)$ to polar coordinates ( $r, \theta$ ) and then substitute $u=r^{2}$. Show that, by performing the integrals over $\theta$ and $u$ and then taking the square root of both sides, we obtain

$$
I=\left(2 \pi \sigma^{2}\right)^{1 / 2}
$$

Finally, use this result to show that the Gaussian distribution $\mathcal{N}\left(x \mid \mu, \sigma^{2}\right)$ is normalized.
$2.13(\star \star)$ By using a change of variables, verify that the univariate Gaussian distribution given by (2.49) satisfies (2.52). Next, by differentiating both sides of the normalization condition

$$
\int_{-\infty}^{\infty} \mathcal{N}\left(x \mid \mu, \sigma^{2}\right) \mathrm{d} x=1
$$
with respect to $\sigma^{2}$, verify that the Gaussian satisfies (2.53). Finally, show that (2.54) holds.
2.14 ( $\star$ ) Show that the mode (i.e., the maximum) of the Gaussian distribution (2.49) is given by $\mu$.
2.15 (*) By setting the derivatives of the log likelihood function (2.56) with respect to $\mu$ and $\sigma^{2}$ equal to zero, verify the results (2.57) and (2.58).
$2.16(\star \star)$ Using the results (2.52) and (2.53), show that
$$
\mathbb{E}\left[x_{n} x_{m}\right]=\mu^{2}+I_{n m} \sigma^{2}
$$

where $x_{n}$ and $x_{m}$ denote data points sampled from a Gaussian distribution with mean $\mu$ and variance $\sigma^{2}$ and $I_{n m}$ satisfies $I_{n m}=1$ if $n=m$ and $I_{n m}=0$ otherwise. Hence prove the results (2.59) and (2.60).
2.17 ( $\star \star$ ) Using the definition (2.61), prove the result (2.62) which shows that the expectation of the variance estimator for a Gaussian distribution based on the true mean is given by the true variance $\sigma^{2}$.
$2.18(\star)$ Show that maximizing (2.66) with respect to $\sigma^{2}$ gives the result (2.68).
$2.19(\star \star)$ Use the transformation property (2.71) of a probability density under a change of variable to show that any density $p(y)$ can be obtained from a fixed density $q(x)$ that is everywhere non-zero by making a nonlinear change of variable $y=f(x)$ in which $f(x)$ is a monotonic function so that $0 \leqslant f^{\prime}(x)<\infty$. Write down the differential equation satisfied by $f(x)$ and draw a diagram illustrating the transformation of the density.
2.20 (*) Evaluate the elements of the Jacobian matrix for the transformation defined by (2.78) and (2.79).
$2.21(\star)$ In Section 2.5, we introduced the idea of entropy $h(x)$ as the information gained on observing the value of a random variable $x$ having distribution $p(x)$. We saw that, for independent variables $x$ and $y$ for which $p(x, y)=p(x) p(y)$, the entropy functions are additive, so that $h(x, y)=h(x)+h(y)$. In this exercise, we derive the relation between $h$ and $p$ in the form of a function $h(p)$. First show that $h\left(p^{2}\right)= 2 h(p)$ and, hence, by induction that $h\left(p^{n}\right)=n h(p)$ where $n$ is a positive integer. Hence, show that $h\left(p^{n / m}\right)=(n / m) h(p)$ where $m$ is also a positive integer. This implies that $h\left(p^{x}\right)=x h(p)$ where $x$ is a positive rational number and, hence, by continuity when it is a positive real number. Finally, show that this implies $h(p)$ must take the form $h(p) \propto \ln p$.
$2.22(\star)$ Use a Lagrange multiplier to show that maximization of the entropy (2.86) for a discrete variable gives a distribution in which all of the probabilities $p\left(x_{i}\right)$ are equal and that the corresponding value of the entropy is then $\ln M$.
2.23 ( $\star$ ) Consider an $M$-state discrete random variable $x$, and use Jensen's inequality in the form (2.102) to show that the entropy of its distribution $p(x)$ satisfies $\mathrm{H}[x] \leqslant \ln M$.
$2.24(\star \star)$ Use the calculus of variations to show that the stationary point of the functional (2.96) is given by (2.97). Then use the constraints (2.93), (2.94), and (2.95) to eliminate the Lagrange multipliers and, hence, show that the maximum entropy solution is given by the Gaussian (2.98).
$2.25(\star)$ Use the results (2.94) and (2.95) to show that the entropy of the univariate Gaussian (2.98) is given by (2.99).
$2.26(\star \star)$ Suppose that $p(\mathbf{x})$ is some fixed distribution and that we wish to approximate it using a Gaussian distribution $q(\mathbf{x})=\mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}, \boldsymbol{\Sigma})$. By writing down the form of the Kullback-Leibler divergence $\mathrm{KL}(p \| q)$ for a Gaussian $q(\mathbf{x})$ and then differentiating, show that minimization of $\mathrm{KL}(p \| q)$ with respect to $\boldsymbol{\mu}$ and $\boldsymbol{\Sigma}$ leads to the result that $\boldsymbol{\mu}$ is given by the expectation of $\mathbf{x}$ under $p(\mathbf{x})$ and that $\boldsymbol{\Sigma}$ is given by the covariance.
2.27 ( $\star \star$ ) Evaluate the Kullback-Leibler divergence (2.100) between the two Gaussians $p(x)=\mathcal{N}\left(x \mid \mu, \sigma^{2}\right)$ and $q(x)=\mathcal{N}\left(x \mid m, s^{2}\right)$.
$2.28(\star \star)$ The alpha family of divergences is defined by

$$
\mathrm{D}_{\alpha}(p \| q)=\frac{4}{1-\alpha^{2}}\left(1-\int p(x)^{(1+\alpha) / 2} q(x)^{(1-\alpha) / 2} \mathrm{~d} x\right)
$$
where $-\infty<\alpha<\infty$ is a continuous parameter. Show that the Kullback-Leibler divergence $\mathrm{KL}(p \| q)$ corresponds to $\alpha \rightarrow 1$. This can be done by writing $p^{\epsilon}= \exp (\epsilon \ln p)=1+\epsilon \ln p+O\left(\epsilon^{2}\right)$ and then taking $\epsilon \rightarrow 0$. Similarly, show that $\mathrm{KL}(q \| p)$ corresponds to $\alpha \rightarrow-1$.
$2.29(\star \star)$ Consider two variables $\mathbf{x}$ and $\mathbf{y}$ having joint distribution $p(\mathbf{x}, \mathbf{y})$. Show that the differential entropy of this pair of variables satisfies
$$
\mathrm{H}[\mathbf{x}, \mathbf{y}] \leqslant \mathrm{H}[\mathbf{x}]+\mathrm{H}[\mathbf{y}]
$$

with equality if, and only if, $\mathbf{x}$ and $\mathbf{y}$ are statistically independent.
$2.30(\star)$ Consider a vector $\mathbf{x}$ of continuous variables with distribution $p(\mathbf{x})$ and corresponding entropy $\mathrm{H}[\mathbf{x}]$. Suppose that we make a non-singular linear transformation of $\mathbf{x}$ to obtain a new variable $\mathbf{y}=\mathbf{A} \mathbf{x}$. Show that the corresponding entropy is given by $\mathrm{H}[\mathbf{y}]=\mathrm{H}[\mathbf{x}]+\ln \operatorname{det} \mathbf{A}$ where $\operatorname{det} \mathbf{A}$ denotes the determinant of $\mathbf{A}$.
2.31 ( $\star \star$ ) Suppose that the conditional entropy $\mathrm{H}[y \mid x]$ between two discrete random variables $x$ and $y$ is zero. Show that, for all values of $x$ such that $p(x)>0$, the variable $y$ must be a function of $x$. In other words, for each $x$ there is only one value of $y$ such that $p(y \mid x) \neq 0$.
$2.32(\star)$ A strictly convex function is defined as one for which every chord lies above the function. Show that this is equivalent to the condition that the second derivative of the function is positive.
2.33 ( $\star \star$ ) Using proof by induction, show that the inequality (2.101) for convex functions implies the result (2.102).
$2.34(\star)$ Show that, up to an additive constant, the Kullback-Leibler divergence (2.100) between the empirical distribution (2.37) and a model distribution $q(\mathbf{x} \mid \boldsymbol{\theta})$ is equal to the negative log likelihood function.
$2.35(\star)$ Using the definition (2.107) together with the product rule of probability, prove the result (2.108).
2.36 ( $* \star \star$ ) Consider two binary variables $x$ and $y$ having the joint distribution given by
![](https://cdn.mathpix.com/cropped/2025_10_01_d90a4d82c88904700833g-082.jpg?height=176&width=296&top_left_y=823&top_left_x=889)

**Image Description:** The image presents a two-dimensional table, functioning as a probability distribution matrix. The axes are labeled with \(x\) and \(y\). The rows represent the variable \(x\) with values 0 and 1, while the columns represent the variable \(y\) with values 0 and 1. The entries in the table indicate the probabilities: \(P(x=0, y=0) = \frac{1}{3}\), \(P(x=0, y=1) = \frac{1}{3}\), \(P(x=1, y=0) = 0\), and \(P(x=1, y=1) = \frac{1}{3}\). This matrix is useful for analyzing joint probabilities in statistics.


Evaluate the following quantities:
(a) $\mathrm{H}[x]$
(b) $\mathrm{H}[y]$
(c) $\mathrm{H}[y \mid x]$
(d) $\mathrm{H}[x \mid y]$
(e) $\mathrm{H}[x, y]$
(f) $\mathrm{I}[x, y]$.

Draw a Venn diagram to show the relationship between these various quantities.
2.37 ( $\star$ ) By applying Jensen's inequality (2.102) with $f(x)=\ln x$, show that the arithmetic mean of a set of real numbers is never less than their geometrical mean.
$2.38(\star)$ Using the sum and product rules of probability, show that the mutual information $I(\mathbf{x}, \mathbf{y})$ satisfies the relation (2.110).
$2.39(\star \star)$ Suppose that two variables $z_{1}$ and $z_{2}$ are independent so that $p\left(z_{1}, z_{2}\right)= p\left(z_{1}\right) p\left(z_{2}\right)$. Show that the covariance matrix between these variables is diagonal. This shows that independence is a sufficient condition for two variables to be uncorrelated. Now consider two variables $y_{1}$ and $y_{2}$ where $y_{1}$ is symmetrically distributed around 0 and $y_{2}=y_{1}^{2}$. Write down the conditional distribution $p\left(y_{2} \mid y_{1}\right)$ and observe that this is dependent on $y_{1}$, thus showing that the two variables are not independent. Now show that the covariance matrix between these two variables is again diagonal. To do this, use the relation $p\left(y_{1}, y_{2}\right)=p\left(y_{1}\right) p\left(y_{2} \mid y_{1}\right)$ to show that the off-diagonal terms are zero. This counterexample shows that zero correlation is not a sufficient condition for independence.
$2.40(\star)$ Consider the bent coin in Figure 2.2. Assume that the prior probability that the convex side is heads is 0.1 . Now suppose the coin is flipped 10 times and we are told that eight of the flips landed heads up and two of the flips landed tails up. Use Bayes' theorem to evaluate the posterior probability that the concave side is heads. Calculate the probability that the next flip will land heads up.
2.41 ( $\star$ ) By substituting (2.115) into (2.114) and making use of the result (2.66) for the $\log$ likelihood of the linear regression model, derive the result (2.117) for the regularized error function.
![](https://cdn.mathpix.com/cropped/2025_10_01_d90a4d82c88904700833g-084.jpg?height=1244&width=1230&top_left_y=224&top_left_x=422)

**Image Description:** This image is a title slide from an academic lecture, featuring the text "Standard Distributions" prominently in black against a colorful, abstract background. The slide is visually engaging, with a gradient of colors creating a fractal-like pattern. The number "3" is displayed in red at the top, indicating it might be part of a sequence in the lecture. There are no diagrams, graphs, or equations present in the image, making it a general thematic or introductory slide.


In this chapter we discuss some specific examples of probability distributions and their properties. As well as being of interest in their own right, these distributions can form building blocks for more complex models and will be used extensively throughout the book.

One role for the distributions discussed in this chapter is to model the probability distribution $p(\mathbf{x})$ of a random variable $\mathbf{x}$, given a finite set $\mathbf{x}_{1}, \ldots, \mathbf{x}_{N}$ of observations. This problem is known as density estimation. It should be emphasized that the problem of density estimation is fundamentally ill-posed, because there are infinitely many probability distributions that could have given rise to the observed finite data set. Indeed, any distribution $p(\mathbf{x})$ that is non-zero at each of the data points $\mathbf{x}_{1}, \ldots, \mathbf{x}_{N}$ is a potential candidate. The issue of choosing an appropriate distribution relates to the problem of model selection, which has already been encountered in the context of polynomial curve fitting and which is a central issue in machine
learning.
We begin by considering distributions for discrete variables before exploring the Gaussian distribution for continuous variables. These are specific examples of parametric distributions, so called because they are governed by a relatively small number of adjustable parameters, such as the mean and variance of a Gaussian. To apply such models to the problem of density estimation, we need a procedure for determining suitable values for the parameters, given an observed data set, and our main focus will be on maximizing the likelihood function. In this chapter, we will assume that the data observations are independent and identically distributed (i.i.d.), whereas in future chapters we will explore more complex scenarios involving structured data where this assumption no longer holds.

One limitation of the parametric approach is that it assumes a specific functional form for the distribution, which may turn out to be inappropriate for a particular application. An alternative approach is given by nonparametric density estimation methods in which the form of the distribution typically depends on the size of the data set. Such models still contain parameters, but these control the model complexity rather than the form of the distribution. We end this chapter by briefly considering three nonparametric methods based respectively on histograms, nearest neighbours, and kernels. A major limitation of nonparametric techniques such as these is that they involve storing all the training data. In other words, the number of parameters grows with the size of the data set, so that the method become very inefficient for large data sets. Deep learning combines the efficiency of parametric models with the generality of nonparametric methods by considering flexible distributions based on neural networks having a large, but fixed, number of parameters.

### 3.1. Discrete Variables

We begin by considering simple distributions for discrete variables, starting with binary variables and then moving on to multi-state variables.

### 3.1.1 Bernoulli distribution

Consider a single binary random variable $x \in\{0,1\}$. For example, $x$ might describe the outcome of flipping a coin, with $x=1$ representing 'heads' and $x=0$ representing 'tails'. If this were a damaged coin, such as the one shown in Figure 2.2, the probability of landing heads is not necessarily the same as that of landing tails. The probability of $x=1$ will be denoted by the parameter $\mu$ so that

$$
p(x=1 \mid \mu)=\mu
$$
where $0 \leqslant \mu \leqslant 1$, from which it follows that $p(x=0 \mid \mu)=1-\mu$. The probability distribution over $x$ can therefore be written in the form
$$
\operatorname{Bern}(x \mid \mu)=\mu^{x}(1-\mu)^{1-x}
$$

which is known as the Bernoulli distribution. It is easily verified that this distribution
is normalized and that it has mean and variance given by

$$
\begin{aligned}
\mathbb{E}[x] & =\mu \\
\operatorname{var}[x] & =\mu(1-\mu) .
\end{aligned}
$$

Now suppose we have a data set $\mathcal{D}=\left\{x_{1}, \ldots, x_{N}\right\}$ of observed values of $x$. We can construct the likelihood function, which is a function of $\mu$, on the assumption that the observations are drawn independently from $p(x \mid \mu)$, so that

$$
p(\mathcal{D} \mid \mu)=\prod_{n=1}^{N} p\left(x_{n} \mid \mu\right)=\prod_{n=1}^{N} \mu^{x_{n}}(1-\mu)^{1-x_{n}}
$$

We can estimate a value for $\mu$ by maximizing the likelihood function or equivalently by maximizing the logarithm of the likelihood, since the log is a monotonic function. The log likelihood function of the Bernoulli distribution is given by

$$
\ln p(\mathcal{D} \mid \mu)=\sum_{n=1}^{N} \ln p\left(x_{n} \mid \mu\right)=\sum_{n=1}^{N}\left\{x_{n} \ln \mu+\left(1-x_{n}\right) \ln (1-\mu)\right\}
$$

At this point, note that the log likelihood function depends on the $N$ observations $x_{n}$ only through their sum $\sum_{n} x_{n}$. This sum provides an example of a sufficient statistic for the data under this distribution. If we set the derivative of $\ln p(\mathcal{D} \mid \mu)$ with respect to $\mu$ equal to zero, we obtain the maximum likelihood estimator:

$$
\mu_{\mathrm{ML}}=\frac{1}{N} \sum_{n=1}^{N} x_{n}
$$
which is also known as the sample mean. Denoting the number of observations of $x=1$ (heads) within this data set by $m$, we can write (3.7) in the form
$$
\mu_{\mathrm{ML}}=\frac{m}{N}
$$

so that the probability of landing heads is given, in this maximum likelihood framework, by the fraction of observations of heads in the data set.

### 3.1.2 Binomial distribution

We can also work out the distribution for the binary variable $x$ of the number $m$ of observations of $x=1$, given that the data set has size $N$. This is called the binomial distribution, and from (3.5) we see that it is proportional to $\mu^{m}(1-\mu)^{N-m}$. To obtain the normalization coefficient, note that out of $N$ coin flips, we have to add up all of the possible ways of obtaining $m$ heads, so that the binomial distribution can be written as

$$
\operatorname{Bin}(m \mid N, \mu)=\binom{N}{m} \mu^{m}(1-\mu)^{N-m}
$$

Figure 3.1 Histogram plot of the binomial distribution (3.9) as a function of $m$ for $N=10$ and $\mu=0.25$.
![](https://cdn.mathpix.com/cropped/2025_10_01_d90a4d82c88904700833g-087.jpg?height=497&width=714&top_left_y=241&top_left_x=920)

**Image Description:** The image is a histogram depicting the distribution of a variable \( m \). The x-axis represents the variable \( m \), ranging from 0 to 10, while the y-axis indicates the relative frequency, ranging from 0 to 0.3. The bars are colored blue, with the heights of the bars representing the frequency of occurrences for each interval of \( m \). The histogram shows a peak around \( m = 2 \), indicating a higher concentration of data points in that range, with tapering frequencies towards the edges.

where

$$
\binom{N}{m} \equiv \frac{N!}{(N-m)!m!}
$$
is the number of ways of choosing $m$ objects out of a total of $N$ identical objects without replacement. Figure 3.1 shows a plot of the binomial distribution for $N=10$ and $\mu=0.25$.

The mean and variance of the binomial distribution can be found by using the results that, for independent events, the mean of the sum is the sum of the means and the variance of the sum is the sum of the variances. Because $m=x_{1}+\ldots+x_{N}$ and because for each observation the mean and variance are given by (3.3) and (3.4), respectively, we have

$$
\begin{aligned}
\mathbb{E}[m] & \equiv \sum_{m=0}^{N} m \operatorname{Bin}(m \mid N, \mu)=N \mu \\
\operatorname{var}[m] & \equiv \sum_{m=0}^{N}(m-\mathbb{E}[m])^{2} \operatorname{Bin}(m \mid N, \mu)=N \mu(1-\mu)
\end{aligned}
$$

Exercise 3.4
These results can also be proved directly by using calculus.

### 3.1.3 Multinomial distribution

Binary variables can be used to describe quantities that can take one of two possible values. Often, however, we encounter discrete variables that can take on one of $K$ possible mutually exclusive states. Although there are various alternative ways to express such variables, we will see shortly that a particularly convenient representation is the 1 -of- $K$ scheme, sometimes called 'one-hot encoding', in which the variable is represented by a $K$-dimensional vector $\mathbf{x}$ in which one of the elements $x_{k}$ equals 1 and all remaining elements equal 0 . So, for instance, if we have a variable that can take $K=6$ states and a particular observation of the variable happens to
correspond to the state where $x_{3}=1$, then $\mathbf{x}$ will be represented by

$$
\mathbf{x}=(0,0,1,0,0,0)^{\mathrm{T}} .
$$

Note that such vectors satisfy $\sum_{k=1}^{K} x_{k}=1$. If we denote the probability of $x_{k}=1$ by the parameter $\mu_{k}$, then the distribution of $\mathbf{x}$ is given by

$$
p(\mathbf{x} \mid \boldsymbol{\mu})=\prod_{k=1}^{K} \mu_{k}^{x_{k}}
$$
where $\boldsymbol{\mu}=\left(\mu_{1}, \ldots, \mu_{K}\right)^{\mathrm{T}}$, and the parameters $\mu_{k}$ are constrained to satisfy $\mu_{k} \geqslant 0$ and $\sum_{k} \mu_{k}=1$, because they represent probabilities. The distribution (3.14) can be regarded as a generalization of the Bernoulli distribution to more than two outcomes. It is easily seen that the distribution is normalized:
$$
\sum_{\mathbf{x}} p(\mathbf{x} \mid \boldsymbol{\mu})=\sum_{k=1}^{K} \mu_{k}=1
$$

and that

$$
\mathbb{E}[\mathbf{x} \mid \boldsymbol{\mu}]=\sum_{\mathbf{x}} p(\mathbf{x} \mid \boldsymbol{\mu}) \mathbf{x}=\boldsymbol{\mu}
$$

Now consider a data set $\mathcal{D}$ of $N$ independent observations $\mathbf{x}_{1}, \ldots, \mathbf{x}_{N}$. The corresponding likelihood function takes the form

$$
p(\mathcal{D} \mid \boldsymbol{\mu})=\prod_{n=1}^{N} \prod_{k=1}^{K} \mu_{k}^{x_{n k}}=\prod_{k=1}^{K} \mu_{k}^{\left(\sum_{n} x_{n k}\right)}=\prod_{k=1}^{K} \mu_{k}^{m_{k}}
$$
where we see that the likelihood function depends on the $N$ data points only through the $K$ quantities:
$$
m_{k}=\sum_{n=1}^{N} x_{n k}
$$

which represent the number of observations of $x_{k}=1$. These are called the sufficient statistics for this distribution. Note that the variables $m_{k}$ are subject to the constraint

$$
\sum_{k=1}^{K} m_{k}=N .
$$

To find the maximum likelihood solution for $\boldsymbol{\mu}$, we need to maximize $\ln p(\mathcal{D} \mid \boldsymbol{\mu})$ with respect to $\mu_{k}$ taking account of the constraint (3.15) that the $\mu_{k}$ must sum to one. This can be achieved using a Lagrange multiplier $\lambda$ and maximizing

$$
\sum_{k=1}^{K} m_{k} \ln \mu_{k}+\lambda\left(\sum_{k=1}^{K} \mu_{k}-1\right) .
$$

Setting the derivative of (3.20) with respect to $\mu_{k}$ to zero, we obtain

$$
\mu_{k}=-m_{k} / \lambda
$$

We can solve for the Lagrange multiplier $\lambda$ by substituting (3.21) into the constraint $\sum_{k} \mu_{k}=1$ to give $\lambda=-N$. Thus, we obtain the maximum likelihood solution for $\mu_{k}$ in the form

$$
\mu_{k}^{\mathrm{ML}}=\frac{m_{k}}{N}
$$
which is the fraction of the $N$ observations for which $x_{k}=1$.
We can also consider the joint distribution of the quantities $m_{1}, \ldots, m_{K}$, conditioned on the parameter vector $\boldsymbol{\mu}$ and on the total number $N$ of observations. From (3.17), this takes the form
$$
\operatorname{Mult}\left(m_{1}, m_{2}, \ldots, m_{K} \mid \boldsymbol{\mu}, N\right)=\binom{N}{m_{1} m_{2} \ldots m_{K}} \prod_{k=1}^{K} \mu_{k}^{m_{k}}
$$

which is known as the multinomial distribution. The normalization coefficient is the number of ways of partitioning $N$ objects into $K$ groups of size $m_{1}, \ldots, m_{K}$ and is given by

$$
\binom{N}{m_{1} m_{2} \ldots m_{K}}=\frac{N!}{m_{1}!m_{2}!\ldots m_{K}!}
$$

Note that two-state quantities can be represented either as binary variables and modelled using the binomial distribution (3.9) or as 1 -of- 2 variables and modelled using the distribution (3.14) with $K=2$.

### 3.2. The Multivariate Gaussian

## Section 2.3

## Section 2.5

The Gaussian, also known as the normal distribution, is a widely used model for the distribution of continuous variables. We have already seen that for of a single variable $x$, the Gaussian distribution can be written in the form

$$
\mathcal{N}\left(x \mid \mu, \sigma^{2}\right)=\frac{1}{\left(2 \pi \sigma^{2}\right)^{1 / 2}} \exp \left\{-\frac{1}{2 \sigma^{2}}(x-\mu)^{2}\right\}
$$
where $\mu$ is the mean and $\sigma^{2}$ is the variance. For a $D$-dimensional vector $\mathbf{x}$, the multivariate Gaussian distribution takes the form
$$
\mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}, \boldsymbol{\Sigma})=\frac{1}{(2 \pi)^{D / 2}} \frac{1}{|\boldsymbol{\Sigma}|^{1 / 2}} \exp \left\{-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^{\mathrm{T}} \boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right\}
$$

where $\boldsymbol{\mu}$ is the $D$-dimensional mean vector, $\boldsymbol{\Sigma}$ is the $D \times D$ covariance matrix, and det $\boldsymbol{\Sigma}$ denotes the determinant of $\boldsymbol{\Sigma}$.

The Gaussian distribution arises in many different contexts and can be motivated from a variety of different perspectives. For example, we have already seen that for

![](https://cdn.mathpix.com/cropped/2025_10_01_d90a4d82c88904700833g-090.jpg?height=314&width=1471&top_left_y=236&top_left_x=144)

**Image Description:** The image consists of three histograms displayed side by side, depicting the distribution of randomly generated numbers within the range of 0 to 1. Each histogram is represented in blue with black borders. The x-axis represents the value range (0 to 1), while the y-axis shows the frequency of occurrences. The histograms correspond to different sample sizes: N=1 on the left, N=2 in the center, and N=10 on the right. As N increases, the distribution appears to smooth out and centralize, illustrating the convergence towards a more normal distribution as sample size increases.

Figure 3.2 Histogram plots of the mean of $N$ uniformly distributed numbers for various values of $N$. We observe that as $N$ increases, the distribution tends towards a Gaussian.

## Exercise 3.8

a single real variable, the distribution that maximizes the entropy is the Gaussian. This property applies also to the multivariate Gaussian.

Another situation in which the Gaussian distribution arises is when we consider the sum of multiple random variables. The central limit theorem tells us that, subject to certain mild conditions, the sum of a set of random variables, which is of course itself a random variable, has a distribution that becomes increasingly Gaussian as the number of terms in the sum increases (Walker, 1969). We can illustrate this by considering $N$ variables $x_{1}, \ldots, x_{N}$ each of which has a uniform distribution over the interval $[0,1]$ and then considering the distribution of the mean $\left(x_{1}+\cdots+x_{N}\right) / N$. For large $N$, this distribution tends to a Gaussian, as illustrated in Figure 3.2. In practice, the convergence to a Gaussian as $N$ increases can be very rapid. One consequence of this result is that the binomial distribution (3.9), which is a distribution over $m$ defined by the sum of $N$ observations of the random binary variable $x$, will tend to a Gaussian as $N \rightarrow \infty$ (see Figure 3.1 for $N=10$ ).

The Gaussian distribution has many important analytical properties, and we will consider several of these in detail. As a result, this section will be rather more technically involved than some of the earlier sections and will require familiarity with various matrix identities.

### 3.2.1 Geometry of the Gaussian

We begin by considering the geometrical from of the Gaussian distribution. The functional dependence of the Gaussian on $\mathbf{x}$ is through the quadratic form

$$
\Delta^{2}=(\mathbf{x}-\boldsymbol{\mu})^{\mathrm{T}} \boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu}),
$$
which appears in the exponent. The quantity $\Delta$ is called the Mahalanobis distance from $\boldsymbol{\mu}$ to $\mathbf{x}$. It reduces to the Euclidean distance when $\boldsymbol{\Sigma}$ is the identity matrix. The Gaussian distribution is constant on surfaces in $\mathbf{x}$-space for which this quadratic form is constant.

First, note that the matrix $\boldsymbol{\Sigma}$ can be taken to be symmetric, without loss of gen-

## Exercise 3.11

erality, because any antisymmetric component would disappear from the exponent. Now consider the eigenvector equation for the covariance matrix

$$
\boldsymbol{\Sigma} \mathbf{u}_{i}=\lambda_{i} \mathbf{u}_{i}
$$

Exercise 3.12
where $i=1, \ldots, D$. Because $\boldsymbol{\Sigma}$ is a real, symmetric matrix, its eigenvalues will be real, and its eigenvectors can be chosen to form an orthonormal set, so that

$$
\mathbf{u}_{i}^{\mathrm{T}} \mathbf{u}_{j}=I_{i j}
$$
where $I_{i j}$ is the $i, j$ element of the identity matrix and satisfies
$$
I_{i j}= \begin{cases}1, & \text { if } i=j \\ 0, & \text { otherwise. }\end{cases}
$$

The covariance matrix $\boldsymbol{\Sigma}$ can be expressed as an expansion in terms of its eigenvectors in the form

$$
\boldsymbol{\Sigma}=\sum_{i=1}^{D} \lambda_{i} \mathbf{u}_{i} \mathbf{u}_{i}^{\mathrm{T}}
$$
and similarly the inverse covariance matrix $\boldsymbol{\Sigma}^{-1}$ can be expressed as
$$
\boldsymbol{\Sigma}^{-1}=\sum_{i=1}^{D} \frac{1}{\lambda_{i}} \mathbf{u}_{i} \mathbf{u}_{i}^{\mathrm{T}}
$$

Substituting (3.32) into (3.27), the quadratic form becomes

$$
\Delta^{2}=\sum_{i=1}^{D} \frac{y_{i}^{2}}{\lambda_{i}}
$$
where we have defined
$$
y_{i}=\mathbf{u}_{i}^{\mathrm{T}}(\mathbf{x}-\boldsymbol{\mu}) .
$$

We can interpret $\left\{y_{i}\right\}$ as a new coordinate system defined by the orthonormal vectors $\mathbf{u}_{i}$ that are shifted and rotated with respect to the original $x_{i}$ coordinates. Forming the vector $\mathbf{y}=\left(y_{1}, \ldots, y_{D}\right)^{\mathrm{T}}$, we have

$$
\mathbf{y}=\mathbf{U}(\mathbf{x}-\boldsymbol{\mu})
$$
where $\mathbf{U}$ is a matrix whose rows are given by $\mathbf{u}_{i}^{\mathrm{T}}$. From (3.29) it follows that $\mathbf{U}$ is an orthogonal matrix, i.e., it satisfies $\mathbf{U U}^{\mathrm{T}}=\mathbf{U}^{\mathrm{T}} \mathbf{U}=\mathbf{I}$, where $\mathbf{I}$ is the identity matrix.

The quadratic form, and hence the Gaussian density, is constant on surfaces for which (3.33) is constant. If all the eigenvalues $\lambda_{i}$ are positive, then these surfaces represent ellipsoids, with their centres at $\boldsymbol{\mu}$ and their axes oriented along $\mathbf{u}_{i}$, and with scaling factors in the directions of the axes given by $\lambda_{i}^{1 / 2}$, as illustrated in Figure 3.3.

For the Gaussian distribution to be well defined, it is necessary for all the eigenvalues $\lambda_{i}$ of the covariance matrix to be strictly positive, otherwise the distribution cannot be properly normalized. A matrix whose eigenvalues are strictly positive is said to be positive definite. When we discuss latent variable models, we will encounter Gaussian distributions for which one or more of the eigenvalues are zero, in

Figure 3.3 The red curve shows the elliptical surface of constant probability density for a Gaussian in a two-dimensional space $\mathbf{x}=$ ( $x_{1}, x_{2}$ ) on which the density is $\exp (-1 / 2)$ of its value at $\mathbf{x}= \mu$. The axes of the ellipse are defined by the eigenvectors $\mathbf{u}_{i}$ of the covariance matrix, with corresponding eigenvalues $\lambda_{i}$.
![](https://cdn.mathpix.com/cropped/2025_10_01_d90a4d82c88904700833g-092.jpg?height=566&width=802&top_left_y=219&top_left_x=845)

**Image Description:** The diagram is a two-dimensional scatter plot representing a multivariate normal distribution. The axes are labeled $x_1$ (horizontal) and $x_2$ (vertical). The blue lines indicate the principal axes of the distribution, representing the eigenvectors $u_1$ and $u_2$. The red ellipse depicts the level curves of constant probability density, centered at the mean $\mu$. The semi-major and semi-minor axes are scaled by the square roots of the eigenvalues $\lambda_1^{1/2}$ and $\lambda_2^{1/2}$, respectively, which denote the variance along each principal axis.


which case the distribution is singular and is confined to a subspace of lower dimensionality. If all the eigenvalues are non-negative, then the covariance matrix is said to be positive semidefinite.

Now consider the form of the Gaussian distribution in the new coordinate system defined by the $y_{i}$. In going from the $\mathbf{x}$ to the $\mathbf{y}$ coordinate system, we have a Jacobian matrix $\mathbf{J}$ with elements given by

$$
J_{i j}=\frac{\partial x_{i}}{\partial y_{j}}=U_{j i}
$$
where $U_{j i}$ are the elements of the matrix $\mathbf{U}^{\mathrm{T}}$. Using the orthonormality property of the matrix $\mathbf{U}$, we see that the square of the determinant of the Jacobian matrix is
$$
|\mathbf{J}|^{2}=\left|\mathbf{U}^{\mathrm{T}}\right|^{2}=\left|\mathbf{U}^{\mathrm{T}}\right||\mathbf{U}|=\left|\mathbf{U}^{\mathrm{T}} \mathbf{U}\right|=|\mathbf{I}|=1
$$

and, hence, $|\mathbf{J}|=1$. Also, the determinant $|\mathbf{\Sigma}|$ of the covariance matrix can be written as the product of its eigenvalues, and hence

$$
|\boldsymbol{\Sigma}|^{1 / 2}=\prod_{j=1}^{D} \lambda_{j}^{1 / 2} .
$$

Thus, in the $y_{j}$ coordinate system, the Gaussian distribution takes the form

$$
p(\mathbf{y})=p(\mathbf{x})|\mathbf{J}|=\prod_{j=1}^{D} \frac{1}{\left(2 \pi \lambda_{j}\right)^{1 / 2}} \exp \left\{-\frac{y_{j}^{2}}{2 \lambda_{j}}\right\},
$$
which is the product of $D$ independent univariate Gaussian distributions. The eigenvectors therefore define a new set of shifted and rotated coordinates with respect to which the joint probability distribution factorizes into a product of independent distributions. The integral of the distribution in the $\mathbf{y}$ coordinate system is then
$$
\int p(\mathbf{y}) \mathrm{d} \mathbf{y}=\prod_{j=1}^{D} \int_{-\infty}^{\infty} \frac{1}{\left(2 \pi \lambda_{j}\right)^{1 / 2}} \exp \left\{-\frac{y_{j}^{2}}{2 \lambda_{j}}\right\} \mathrm{d} y_{j}=1
$$

where we have used the result (2.51) for the normalization of the univariate Gaussian. This confirms that the multivariate Gaussian (3.26) is indeed normalized.

### 3.2.2 Moments

We now look at the moments of the Gaussian distribution and thereby provide an interpretation of the parameters $\boldsymbol{\mu}$ and $\boldsymbol{\Sigma}$. The expectation of x under the Gaussian distribution is given by

$$
\begin{aligned}
\mathbb{E}[\mathbf{x}] & =\frac{1}{(2 \pi)^{D / 2}} \frac{1}{|\boldsymbol{\Sigma}|^{1 / 2}} \int \exp \left\{-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^{\mathrm{T}} \boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right\} \mathbf{x} \mathrm{d} \mathbf{x} \\
& =\frac{1}{(2 \pi)^{D / 2}} \frac{1}{|\boldsymbol{\Sigma}|^{1 / 2}} \int \exp \left\{-\frac{1}{2} \mathbf{z}^{\mathrm{T}} \boldsymbol{\Sigma}^{-1} \mathbf{z}\right\}(\mathbf{z}+\boldsymbol{\mu}) \mathrm{d} \mathbf{z}
\end{aligned}
$$

where we have changed variables using $\mathbf{z}=\mathbf{x}-\boldsymbol{\mu}$. Note that the exponent is an even function of the components of $\mathbf{z}$, and because the integrals over these are taken over the range $(-\infty, \infty)$, the term in $\mathbf{z}$ in the factor ( $\mathbf{z}+\boldsymbol{\mu}$ ) will vanish by symmetry. Thus,

$$
\mathbb{E}[\mathbf{x}]=\boldsymbol{\mu}
$$
and so we refer to $\boldsymbol{\mu}$ as the mean of the Gaussian distribution.
We now consider second-order moments of the Gaussian. In the univariate case, we considered the second-order moment given by $\mathbb{E}\left[x^{2}\right]$. For the multivariate Gaussian, there are $D^{2}$ second-order moments given by $\mathbb{E}\left[x_{i} x_{j}\right]$, which we can group together to form the matrix $\mathbb{E}\left[\mathbf{x x}^{T}\right]$. This matrix can be written as
\$\$\begin{aligned}

\mathbb{E}\left[\mathbf{x x}^{\mathrm{T}}\right] \& =\frac{1}{(2 \pi)^{D / 2}} \frac{1}{|\boldsymbol{\Sigma}|^{1 / 2}} \int \exp \left\{-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^{\mathrm{T}} \boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right\} \mathbf{x x}^{\mathrm{T}} \mathrm{~d} \mathbf{x} <br>
\& =\frac{1}{(2 \pi)^{D / 2}} \frac{1}{|\boldsymbol{\Sigma}|^{1 / 2}} \int \exp \left\{-\frac{1}{2} \mathbf{z}^{\mathrm{T}} \boldsymbol{\Sigma}^{-1} \mathbf{z}\right\}(\mathbf{z}+\boldsymbol{\mu})(\mathbf{z}+\boldsymbol{\mu})^{\mathrm{T}} \mathrm{~d} \mathbf{z}

\end{aligned}$$
where again we have changed variables using $\mathbf{z}=\mathbf{x}-\boldsymbol{\mu}$. Note that the cross-terms involving $\boldsymbol{\mu} \mathbf{z}^{\mathrm{T}}$ and $\boldsymbol{\mu}^{\mathrm{T}} \mathbf{z}$ will again vanish by symmetry. The term $\boldsymbol{\mu} \boldsymbol{\mu}^{\mathrm{T}}$ is constant and can be taken outside the integral, which itself is unity because the Gaussian distribution is normalized. Consider the term involving $\mathbf{z z}^{T}$. Again, we can make use of the eigenvector expansion of the covariance matrix given by (3.28), together with the completeness of the set of eigenvectors, to write
$$\mathbf{z}=\sum_{j=1}^{D} y_{j} \mathbf{u}_{j}\$\$

where $y_{j}=\mathbf{u}_{j}^{\mathrm{T}} \mathbf{z}$, which gives

$$
\begin{aligned}
& \frac{1}{(2 \pi)^{D / 2}} \frac{1}{|\boldsymbol{\Sigma}|^{1 / 2}} \int \exp \left\{-\frac{1}{2} \mathbf{z}^{\mathrm{T}} \boldsymbol{\Sigma}^{-1} \mathbf{z}\right\} \mathbf{z z}^{\mathrm{T}} \mathrm{~d} \mathbf{z} \\
& =\frac{1}{(2 \pi)^{D / 2}} \frac{1}{|\boldsymbol{\Sigma}|^{1 / 2}} \sum_{i=1}^{D} \sum_{j=1}^{D} \mathbf{u}_{i} \mathbf{u}_{j}^{\mathrm{T}} \int \exp \left\{-\sum_{k=1}^{D} \frac{y_{k}^{2}}{2 \lambda_{k}}\right\} y_{i} y_{j} \mathrm{~d} \mathbf{y} \\
& =\sum_{i=1}^{D} \mathbf{u}_{i} \mathbf{u}_{i}^{\mathrm{T}} \lambda_{i}=\boldsymbol{\Sigma}
\end{aligned}
$$

where we have made use of the eigenvector equation (3.28), together with the fact that the integral on the middle line vanishes by symmetry unless $i=j$. In the final line we have made use of the results (2.53) and (3.38), together with (3.31). Thus, we have

$$
\mathbb{E}\left[\mathbf{x} \mathbf{x}^{\mathrm{T}}\right]=\boldsymbol{\mu} \boldsymbol{\mu}^{\mathrm{T}}+\boldsymbol{\Sigma} .
$$

When defining the variance for a single random variable, we subtracted the mean before taking the second moment. Similarly, in the multivariate case it is again convenient to subtract off the mean, giving rise to the covariance of a random vector $\mathbf{x}$ defined by

$$
\operatorname{cov}[\mathbf{x}]=\mathbb{E}\left[(\mathbf{x}-\mathbb{E}[\mathbf{x}])(\mathbf{x}-\mathbb{E}[\mathbf{x}])^{\mathrm{T}}\right]
$$

For the specific case of a Gaussian distribution, we can make use of $\mathbb{E}[\mathbf{x}]=\boldsymbol{\mu}$, together with the result (3.46), to give

$$
\operatorname{cov}[\mathbf{x}]=\boldsymbol{\Sigma}
$$

Because the parameter matrix $\boldsymbol{\Sigma}$ governs the covariance of $\mathbf{x}$ under the Gaussian distribution, it is called the covariance matrix.

### 3.2.3 Limitations

Although the Gaussian distribution (3.26) is often used as a simple density model, it suffers from some significant limitations. Consider the number of free parameters in the distribution. A general symmetric covariance matrix $\boldsymbol{\Sigma}$ will have $D(D+1) / 2$ independent parameters, and there are another $D$ independent parameters in $\boldsymbol{\mu}$, giving $D(D+3) / 2$ parameters in total. For large $D$, the total number of parameters therefore grows quadratically with $D$, and the computational task of manipulating and inverting the large matrices can become prohibitive. One way to address this problem is to use restricted forms of the covariance matrix. If we consider covariance matrices that are diagonal, so that $\boldsymbol{\Sigma}=\operatorname{diag}\left(\sigma_{i}^{2}\right)$, we then have a total of $2 D$ independent parameters in the density model. The corresponding contours of constant density are given by axis-aligned ellipsoids. We could further restrict the covariance matrix to be proportional to the identity matrix, $\boldsymbol{\Sigma}=\sigma^{2} \mathbf{I}$, known as an isotropic covariance, giving $D+1$ independent parameters in the model together with spherical surfaces of constant density. The three possibilities of general, diagonal, and isotropic covariance matrices are illustrated in Figure 3.4. Unfortunately,

Figure 3.4 Contours of constant probability density for a Gaussian distribution in two dimensions in which the covariance matrix is (a) of general form, (b) diagonal, in which case the elliptical contours are aligned with the coordinate axes, and (c) proportional to the identity matrix, in which case the contours are concentric circles.
![](https://cdn.mathpix.com/cropped/2025_10_01_d90a4d82c88904700833g-095.jpg?height=401&width=985&top_left_y=203&top_left_x=671)

**Image Description:** The image consists of three contour plots arranged horizontally, each showing levels of a function defined in a two-dimensional space with axes labeled \(x_1\) and \(x_2\). 

- **(a)**: Concentrated shading at the center, indicating a single peak. 
- **(b)**: Wider concentric contours indicating a smoother gradient around the peak. 
- **(c)**: Similar to (b) but with even broader contours, suggesting a flatter function profile. 

The varying densities of contours visualize changes in the function's value across the defined area.


whereas such approaches limit the number of degrees of freedom in the distribution and make inversion of the covariance matrix a much faster operation, they also greatly restrict the form of the probability density and limit its ability to capture interesting correlations in the data.

A further limitation of the Gaussian distribution is that it is intrinsically unimodal (i.e., has a single maximum) and so is unable to provide a good approximation to multimodal distributions. Thus, the Gaussian distribution can be both too flexible, in the sense of having too many parameters, and too limited in the range of distributions that it can adequately represent. We will see later that the introduction of latent variables, also called hidden variables or unobserved variables, allows both of these problems to be addressed. In particular, a rich family of multimodal distributions is obtained by introducing discrete latent variables leading to mixtures of Gaussians. Similarly, the introduction of continuous latent variables leads to models in which the number of free parameters can be controlled independently of the dimensionality $D$ of the data space while still allowing the model to capture the dominant correlations in the data set.

### 3.2.4 Conditional distribution

An important property of a multivariate Gaussian distribution is that if two sets of variables are jointly Gaussian, then the conditional distribution of one set conditioned on the other is again Gaussian. Similarly, the marginal distribution of either set is also Gaussian.

First, consider the case of conditional distributions. Suppose that $\mathbf{x}$ is a $D$ dimensional vector with Gaussian distribution $\mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}, \boldsymbol{\Sigma})$ and that we partition x into two disjoint subsets $\mathbf{x}_{a}$ and $\mathbf{x}_{b}$. Without loss of generality, we can take $\mathbf{x}_{a}$ to form the first $M$ components of $\mathbf{x}$, with $\mathbf{x}_{b}$ comprising the remaining $D-M$ components, so that

$$
\mathbf{x}=\binom{\mathbf{x}_{a}}{\mathbf{x}_{b}}
$$

We also define corresponding partitions of the mean vector $\boldsymbol{\mu}$ given by

$$
\boldsymbol{\mu}=\binom{\boldsymbol{\mu}_{a}}{\boldsymbol{\mu}_{b}}
$$
and of the covariance matrix $\boldsymbol{\Sigma}$ given by
\$\$\boldsymbol{\Sigma}=\left(\begin{array}{ll}

\boldsymbol{\Sigma}_{a a} \& \boldsymbol{\Sigma}_{a b} <br>
\boldsymbol{\Sigma}_{b a} \& \boldsymbol{\Sigma}_{b b}

\end{array}\right) .\$\$

Note that the symmetry $\boldsymbol{\Sigma}^{\mathrm{T}}=\boldsymbol{\Sigma}$ of the covariance matrix implies that $\boldsymbol{\Sigma}_{a a}$ and $\boldsymbol{\Sigma}_{b b}$ are symmetric and that $\boldsymbol{\Sigma}_{b a}=\boldsymbol{\Sigma}_{a b}^{\mathrm{T}}$.

In many situations, it will be convenient to work with the inverse of the covariance matrix:

$$
\Lambda \equiv \Sigma^{-1}
$$
which is known as the precision matrix. In fact, we will see that some properties of Gaussian distributions are most naturally expressed in terms of the covariance, whereas others take a simpler form when viewed in terms of the precision. We therefore also introduce the partitioned form of the precision matrix:
\$\$\boldsymbol{\Lambda}=\left(\begin{array}{ll}

\boldsymbol{\Lambda}_{a a} \& \boldsymbol{\Lambda}_{a b} <br>
\boldsymbol{\Lambda}_{b a} \& \boldsymbol{\Lambda}_{b b}

\end{array}\right)\$\$

## Exercise 3.16

corresponding to the partitioning (3.49) of the vector $\mathbf{x}$. Because the inverse of a symmetric matrix is also symmetric, we see that $\boldsymbol{\Lambda}_{a a}$ and $\boldsymbol{\Lambda}_{b b}$ are symmetric and that $\boldsymbol{\Lambda}_{b a}=\boldsymbol{\Lambda}_{a b}^{\mathrm{T}}$. It should be stressed at this point that, for instance, $\boldsymbol{\Lambda}_{a a}$ is not simply given by the inverse of $\boldsymbol{\Sigma}_{a a}$. In fact, we will shortly examine the relation between the inverse of a partitioned matrix and the inverses of its partitions.

We begin by finding an expression for the conditional distribution $p\left(\mathbf{x}_{a} \mid \mathbf{x}_{b}\right)$. From the product rule of probability, we see that this conditional distribution can be evaluated from the joint distribution $p(\mathbf{x})=p\left(\mathbf{x}_{a}, \mathbf{x}_{b}\right)$ simply by fixing $\mathbf{x}_{b}$ to the observed value and normalizing the resulting expression to obtain a valid probability distribution over $\mathbf{x}_{a}$. Instead of performing this normalization explicitly, we can obtain the solution more efficiently by considering the quadratic form in the exponent of the Gaussian distribution given by (3.27) and then reinstating the normalization coefficient at the end of the calculation. If we make use of the partitioning (3.49), (3.50), and (3.53), we obtain

$$
\begin{aligned}
& -\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^{\mathrm{T}} \boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})= \\
& \quad-\frac{1}{2}\left(\mathbf{x}_{a}-\boldsymbol{\mu}_{a}\right)^{\mathrm{T}} \boldsymbol{\Lambda}_{a a}\left(\mathbf{x}_{a}-\boldsymbol{\mu}_{a}\right)-\frac{1}{2}\left(\mathbf{x}_{a}-\boldsymbol{\mu}_{a}\right)^{\mathrm{T}} \boldsymbol{\Lambda}_{a b}\left(\mathbf{x}_{b}-\boldsymbol{\mu}_{b}\right) \\
& \quad-\frac{1}{2}\left(\mathbf{x}_{b}-\boldsymbol{\mu}_{b}\right)^{\mathrm{T}} \boldsymbol{\Lambda}_{b a}\left(\mathbf{x}_{a}-\boldsymbol{\mu}_{a}\right)-\frac{1}{2}\left(\mathbf{x}_{b}-\boldsymbol{\mu}_{b}\right)^{\mathrm{T}} \boldsymbol{\Lambda}_{b b}\left(\mathbf{x}_{b}-\boldsymbol{\mu}_{b}\right)
\end{aligned}
$$

We see that as a function of $\mathbf{x}_{a}$, this is again a quadratic form, and hence, the corresponding conditional distribution $p\left(\mathbf{x}_{a} \mid \mathbf{x}_{b}\right)$ will be Gaussian. Because this distribution is completely characterized by its mean and its covariance, our goal will be to identify expressions for the mean and covariance of $p\left(\mathbf{x}_{a} \mid \mathbf{x}_{b}\right)$ by inspection of (3.54).

This is an example of a rather common operation associated with Gaussian distributions, sometimes called 'completing the square', in which we are given a
quadratic form defining the exponent terms in a Gaussian distribution and we need to determine the corresponding mean and covariance. Such problems can be solved straightforwardly by noting that the exponent in a general Gaussian distribution $\mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}, \boldsymbol{\Sigma})$ can be written as

$$
-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^{\mathrm{T}} \boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})=-\frac{1}{2} \mathbf{x}^{\mathrm{T}} \boldsymbol{\Sigma}^{-1} \mathbf{x}+\mathbf{x}^{\mathrm{T}} \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}+\text { const }
$$
where 'const' denotes terms that are independent of $\mathbf{x}$, We have also made use of the symmetry of $\boldsymbol{\Sigma}$. Thus, if we take our general quadratic form and express it in the form given by the right-hand side of (3.55), then we can immediately equate the matrix of coefficients entering the second-order term in $\mathbf{x}$ to the inverse covariance matrix $\boldsymbol{\Sigma}^{-1}$ and the coefficient of the linear term in $\mathbf{x}$ to $\boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}$, from which we can obtain $\mu$.

Now let us apply this procedure to the conditional Gaussian distribution $p\left(\mathbf{x}_{a} \mid \mathbf{x}_{b}\right)$ for which the quadratic form in the exponent is given by (3.54). We will denote the mean and covariance of this distribution by $\boldsymbol{\mu}_{a \mid b}$ and $\boldsymbol{\Sigma}_{a \mid b}$, respectively. Consider the functional dependence of (3.54) on $\mathbf{x}_{a}$ in which $\mathbf{x}_{b}$ is regarded as a constant. If we pick out all terms that are second order in $\mathbf{x}_{a}$, we have

$$
-\frac{1}{2} \mathbf{x}_{a}^{\mathrm{T}} \boldsymbol{\Lambda}_{a a} \mathbf{x}_{a}
$$
from which we can immediately conclude that the covariance (inverse precision) of $p\left(\mathbf{x}_{a} \mid \mathbf{x}_{b}\right)$ is given by
$$
\boldsymbol{\Sigma}_{a \mid b}=\boldsymbol{\Lambda}_{a a}^{-1}
$$

Now consider all the terms in (3.54) that are linear in $\mathbf{x}_{a}$ :

$$
\mathbf{x}_{a}^{\mathrm{T}}\left\{\boldsymbol{\Lambda}_{a a} \boldsymbol{\mu}_{a}-\boldsymbol{\Lambda}_{a b}\left(\mathbf{x}_{b}-\boldsymbol{\mu}_{b}\right)\right\}
$$
where we have used $\boldsymbol{\Lambda}_{b a}^{\mathrm{T}}=\boldsymbol{\Lambda}_{a b}$. From our discussion of the general from (3.55), the coefficient of $\mathrm{x}_{a}$ in this expression must equal $\boldsymbol{\Sigma}_{a \mid b}^{-1} \boldsymbol{\mu}_{a \mid b}$ and, hence,
\$\$\begin{aligned}

\boldsymbol{\mu}_{a \mid b} \& =\boldsymbol{\Sigma}_{a \mid b}\left\{\boldsymbol{\Lambda}_{a a} \boldsymbol{\mu}_{a}-\boldsymbol{\Lambda}_{a b}\left(\mathbf{x}_{b}-\boldsymbol{\mu}_{b}\right)\right\} <br>
\& =\boldsymbol{\mu}_{a}-\boldsymbol{\Lambda}_{a a}^{-1} \boldsymbol{\Lambda}_{a b}\left(\mathbf{x}_{b}-\boldsymbol{\mu}_{b}\right)

\end{aligned}$$
where we have made use of (3.57).
The results (3.57) and (3.59) are expressed in terms of the partitioned precision matrix of the original joint distribution $p\left(\mathbf{x}_{a}, \mathbf{x}_{b}\right)$. We can also express these results in terms of the corresponding partitioned covariance matrix. To do this, we make use of the following identity for the inverse of a partitioned matrix:
$$\left(\begin{array}{ll}

\mathbf{A} \& \mathbf{B} <br>
\mathbf{C} \& \mathbf{D}
\end{array}\right)^{-1}=\left(\begin{array}{cc}
\mathbf{M} \& -\mathbf{M B D}^{-1} <br>
-\mathbf{D}^{-1} \mathbf{C M} \& \mathbf{D}^{-1}+\mathbf{D}^{-1} \mathbf{C M B D}^{-1}

\end{array}\right)$$
where we have defined
$$\mathbf{M}=\left(\mathbf{A}-\mathbf{B D}^{-1} \mathbf{C}\right)^{-1}\$\$

The quantity $\mathbf{M}^{-1}$ is known as the Schur complement of the matrix on the left-hand side of (3.60) with respect to the submatrix $\mathbf{D}$. Using the definition

$$
\left(\begin{array}{ll}
\boldsymbol{\Sigma}_{a a} & \boldsymbol{\Sigma}_{a b} \\
\boldsymbol{\Sigma}_{b a} & \boldsymbol{\Sigma}_{b b}
\end{array}\right)^{-1}=\left(\begin{array}{ll}
\boldsymbol{\Lambda}_{a a} & \boldsymbol{\Lambda}_{a b} \\
\boldsymbol{\Lambda}_{b a} & \boldsymbol{\Lambda}_{b b}
\end{array}\right)
$$

and making use of (3.60), we have

$$
\begin{aligned}
& \boldsymbol{\Lambda}_{a a}=\left(\boldsymbol{\Sigma}_{a a}-\boldsymbol{\Sigma}_{a b} \boldsymbol{\Sigma}_{b b}^{-1} \boldsymbol{\Sigma}_{b a}\right)^{-1} \\
& \boldsymbol{\Lambda}_{a b}=-\left(\boldsymbol{\Sigma}_{a a}-\boldsymbol{\Sigma}_{a b} \boldsymbol{\Sigma}_{b b}^{-1} \boldsymbol{\Sigma}_{b a}\right)^{-1} \boldsymbol{\Sigma}_{a b} \boldsymbol{\Sigma}_{b b}^{-1} .
\end{aligned}
$$

From these we obtain the following expressions for the mean and covariance of the conditional distribution $p\left(\mathbf{x}_{a} \mid \mathbf{x}_{b}\right)$ :

$$
\begin{aligned}
\boldsymbol{\mu}_{a \mid b} & =\boldsymbol{\mu}_{a}+\boldsymbol{\Sigma}_{a b} \boldsymbol{\Sigma}_{b b}^{-1}\left(\mathbf{x}_{b}-\boldsymbol{\mu}_{b}\right) \\
\boldsymbol{\Sigma}_{a \mid b} & =\boldsymbol{\Sigma}_{a a}-\boldsymbol{\Sigma}_{a b} \boldsymbol{\Sigma}_{b b}^{-1} \boldsymbol{\Sigma}_{b a} .
\end{aligned}
$$

Comparing (3.57) and (3.66), we see that the conditional distribution $p\left(\mathbf{x}_{a} \mid \mathbf{x}_{b}\right)$ takes a simpler form when expressed in terms of the partitioned precision matrix than when it is expressed in terms of the partitioned covariance matrix. Note that the mean of the conditional distribution $p\left(\mathbf{x}_{a} \mid \mathbf{x}_{b}\right)$, given by (3.65), is a linear function of $\mathbf{x}_{b}$ and that the covariance, given by (3.66), is independent of $\mathbf{x}_{b}$. This represents an example of a linear-Gaussian model.

### 3.2.5 Marginal distribution

We have seen that if a joint distribution $p\left(\mathbf{x}_{a}, \mathbf{x}_{b}\right)$ is Gaussian, then the conditional distribution $p\left(\mathbf{x}_{a} \mid \mathbf{x}_{b}\right)$ will again be Gaussian. Now we turn to a discussion of the marginal distribution given by

$$
p\left(\mathbf{x}_{a}\right)=\int p\left(\mathbf{x}_{a}, \mathbf{x}_{b}\right) \mathrm{d} \mathbf{x}_{b},
$$
which, as we will see, is also Gaussian. Once again, our strategy for calculating this distribution will be to focus on the quadratic form in the exponent of the joint distribution and thereby to identify the mean and covariance of the marginal distribution $p\left(\mathbf{x}_{a}\right)$.

The quadratic form for the joint distribution can be expressed, using the partitioned precision matrix, in the form (3.54). Our goal is to integrate out $\mathrm{x}_{b}$, which is most easily achieved by first considering the terms involving $\mathbf{x}_{b}$ and then completing the square to facilitate the integration. Picking out just those terms that involve $\mathbf{x}_{b}$, we have

$$
-\frac{1}{2} \mathbf{x}_{b}^{\mathrm{T}} \boldsymbol{\Lambda}_{b b} \mathbf{x}_{b}+\mathbf{x}_{b}^{T} \mathbf{m}=-\frac{1}{2}\left(\mathbf{x}_{b}-\boldsymbol{\Lambda}_{b b}^{-1} \mathbf{m}\right)^{\mathrm{T}} \boldsymbol{\Lambda}_{b b}\left(\mathbf{x}_{b}-\boldsymbol{\Lambda}_{b b}^{-1} \mathbf{m}\right)+\frac{1}{2} \mathbf{m}^{\mathrm{T}} \boldsymbol{\Lambda}_{b b}^{-1} \mathbf{m}
$$
where we have defined
$$
\mathbf{m}=\boldsymbol{\Lambda}_{b b} \boldsymbol{\mu}_{b}-\boldsymbol{\Lambda}_{b a}\left(\mathbf{x}_{a}-\boldsymbol{\mu}_{a}\right) .
$$

We see that the dependence on $\mathbf{x}_{b}$ has been cast into the standard quadratic form of a Gaussian distribution corresponding to the first term on the right-hand side of (3.68) plus a term that does not depend on $\mathbf{x}_{b}$ (but that does depend on $\mathbf{x}_{a}$ ). Thus, when we take the exponential of this quadratic form, we see that the integration over $\mathbf{x}_{b}$ required by (3.67) will take the form

$$
\int \exp \left\{-\frac{1}{2}\left(\mathbf{x}_{b}-\boldsymbol{\Lambda}_{b b}^{-1} \mathbf{m}\right)^{\mathrm{T}} \boldsymbol{\Lambda}_{b b}\left(\mathbf{x}_{b}-\boldsymbol{\Lambda}_{b b}^{-1} \mathbf{m}\right)\right\} \mathrm{d} \mathbf{x}_{b}
$$

This integration is easily performed by noting that it is the integral over an unnormalized Gaussian, and so the result will be the reciprocal of the normalization coefficient. We know from the form of the normalized Gaussian given by (3.26) that this coefficient is independent of the mean and depends only on the determinant of the covariance matrix. Thus, by completing the square with respect to $\mathbf{x}_{b}$, we can integrate out $\mathrm{x}_{b}$ so that the only term remaining from the contributions on the left-hand side of (3.68) that depends on $\mathbf{x}_{a}$ is the last term on the right-hand side of (3.68) in which $\mathbf{m}$ is given by (3.69). Combining this term with the remaining terms from (3.54) that depend on $\mathrm{x}_{a}$, we obtain

$$
\begin{aligned}
& \frac{1}{2}\left[\boldsymbol{\Lambda}_{b b} \boldsymbol{\mu}_{b}-\boldsymbol{\Lambda}_{b a}\left(\mathbf{x}_{a}-\boldsymbol{\mu}_{a}\right)\right]^{\mathrm{T}} \boldsymbol{\Lambda}_{b b}^{-1}\left[\boldsymbol{\Lambda}_{b b} \boldsymbol{\mu}_{b}-\boldsymbol{\Lambda}_{b a}\left(\mathbf{x}_{a}-\boldsymbol{\mu}_{a}\right)\right] \\
&-\frac{1}{2} \mathbf{x}_{a}^{\mathrm{T}} \boldsymbol{\Lambda}_{a a} \mathbf{x}_{a}+\mathbf{x}_{a}^{\mathrm{T}}\left(\boldsymbol{\Lambda}_{a a} \boldsymbol{\mu}_{a}+\boldsymbol{\Lambda}_{a b} \boldsymbol{\mu}_{b}\right)+\mathrm{const} \\
&=-\frac{1}{2} \mathbf{x}_{a}^{\mathrm{T}}\left(\boldsymbol{\Lambda}_{a a}-\boldsymbol{\Lambda}_{a b} \boldsymbol{\Lambda}_{b b}^{-1} \boldsymbol{\Lambda}_{b a}\right) \mathbf{x}_{a} \\
&+\mathbf{x}_{a}^{\mathrm{T}}\left(\boldsymbol{\Lambda}_{a a}-\boldsymbol{\Lambda}_{a b} \boldsymbol{\Lambda}_{b b}^{-1} \boldsymbol{\Lambda}_{b a}\right) \boldsymbol{\mu}_{a}+\mathrm{const}
\end{aligned}
$$

where 'const' denotes quantities independent of $\mathbf{x}_{a}$. Again, by comparison with (3.55), we see that the covariance of the marginal distribution $p\left(\mathbf{x}_{a}\right)$ is given by

$$
\boldsymbol{\Sigma}_{a}=\left(\boldsymbol{\Lambda}_{a a}-\boldsymbol{\Lambda}_{a b} \boldsymbol{\Lambda}_{b b}^{-1} \boldsymbol{\Lambda}_{b a}\right)^{-1}
$$

Similarly, the mean is given by

$$
\boldsymbol{\Sigma}_{a}\left(\boldsymbol{\Lambda}_{a a}-\boldsymbol{\Lambda}_{a b} \boldsymbol{\Lambda}_{b b}^{-1} \boldsymbol{\Lambda}_{b a}\right) \boldsymbol{\mu}_{a}=\boldsymbol{\mu}_{a}
$$
where we have used (3.72). The covariance (3.72) is expressed in terms of the partitioned precision matrix given by (3.53). We can rewrite this in terms of the corresponding partitioning of the covariance matrix given by (3.51), as we did for the conditional distribution. These partitioned matrices are related by
\$\$\left(\begin{array}{ll}

\boldsymbol{\Lambda}_{a a} \& \boldsymbol{\Lambda}_{a b} <br>
\boldsymbol{\Lambda}_{b a} \& \boldsymbol{\Lambda}_{b b}
\end{array}\right)^{-1}=\left(\begin{array}{ll}
\boldsymbol{\Sigma}_{a a} \& \boldsymbol{\Sigma}_{a b} <br>
\boldsymbol{\Sigma}_{b a} \& \boldsymbol{\Sigma}_{b b}

\end{array}\right)\$\$

Making use of (3.60), we then have

$$
\left(\boldsymbol{\Lambda}_{a a}-\boldsymbol{\Lambda}_{a b} \boldsymbol{\Lambda}_{b b}^{-1} \boldsymbol{\Lambda}_{b a}\right)^{-1}=\boldsymbol{\Sigma}_{a a}
$$

Thus, we obtain the intuitively satisfying result that the marginal distribution $p\left(\mathbf{x}_{a}\right)$ has mean and covariance given by

$$
\begin{aligned}
\mathbb{E}\left[\mathbf{x}_{a}\right] & =\boldsymbol{\mu}_{a} \\
\operatorname{cov}\left[\mathbf{x}_{a}\right] & =\boldsymbol{\Sigma}_{a a} .
\end{aligned}
$$

We see that for a marginal distribution, the mean and covariance are most simply expressed in terms of the partitioned covariance matrix, in contrast to the conditional distribution for which the partitioned precision matrix gives rise to simpler expressions.

Our results for the marginal and conditional distributions of a partitioned Gaussian can be summarized as follows. Given a joint Gaussian distribution $\mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}, \boldsymbol{\Sigma})$ with $\boldsymbol{\Lambda} \equiv \boldsymbol{\Sigma}^{-1}$ and the following partitions

$$
\begin{aligned}
\mathbf{x}=\binom{\mathbf{x}_{a}}{\mathbf{x}_{b}}, & \boldsymbol{\mu}=\binom{\boldsymbol{\mu}_{a}}{\boldsymbol{\mu}_{b}} \\
\boldsymbol{\Sigma}=\left(\begin{array}{cc}
\boldsymbol{\Sigma}_{a a} & \boldsymbol{\Sigma}_{a b} \\
\boldsymbol{\Sigma}_{b a} & \boldsymbol{\Sigma}_{b b}
\end{array}\right), & \boldsymbol{\Lambda}=\left(\begin{array}{cc}
\boldsymbol{\Lambda}_{a a} & \boldsymbol{\Lambda}_{a b} \\
\boldsymbol{\Lambda}_{b a} & \boldsymbol{\Lambda}_{b b}
\end{array}\right)
\end{aligned}
$$

then the conditional distribution is given by

$$
\begin{aligned}
p\left(\mathbf{x}_{a} \mid \mathbf{x}_{b}\right) & =\mathcal{N}\left(\mathbf{x} \mid \boldsymbol{\mu}_{a \mid b}, \boldsymbol{\Lambda}_{a a}^{-1}\right) \\
\boldsymbol{\mu}_{a \mid b} & =\boldsymbol{\mu}_{a}-\boldsymbol{\Lambda}_{a a}^{-1} \boldsymbol{\Lambda}_{a b}\left(\mathbf{x}_{b}-\boldsymbol{\mu}_{b}\right)
\end{aligned}
$$

and the marginal distribution is given by

$$
p\left(\mathbf{x}_{a}\right)=\mathcal{N}\left(\mathbf{x}_{a} \mid \boldsymbol{\mu}_{a}, \boldsymbol{\Sigma}_{a a}\right)
$$

We illustrate the idea of conditional and marginal distributions associated with a multivariate Gaussian using an example involving two variables in Figure 3.5.

### 3.2.6 Bayes' theorem

In Sections 3.2.4 and 3.2.5 we considered a Gaussian $p(\mathbf{x})$ in which we partitioned the vector $\mathbf{x}$ into two subvectors $\mathbf{x}=\left(\mathbf{x}_{a}, \mathbf{x}_{b}\right)$ and then found expressions for the conditional distribution $p\left(\mathbf{x}_{a} \mid \mathbf{x}_{b}\right)$ and the marginal distribution $p\left(\mathbf{x}_{a}\right)$. We noted that the mean of the conditional distribution $p\left(\mathbf{x}_{a} \mid \mathbf{x}_{b}\right)$ was a linear function of $\mathbf{x}_{b}$. Here we will suppose that we are given a Gaussian marginal distribution $p(\mathbf{x})$ and a Gaussian conditional distribution $p(\mathbf{y} \mid \mathbf{x})$ in which $p(\mathbf{y} \mid \mathbf{x})$ has a mean that is a linear function of $\mathbf{x}$ and a covariance that is independent of $\mathbf{x}$. This is an example of a linear-Gaussian model (Roweis and Ghahramani, 1999). We wish to find the marginal distribution $p(\mathbf{y})$ and the conditional distribution $p(\mathbf{x} \mid \mathbf{y})$. This is a struc- ture that arises in several types of generative model and it will prove convenient to derive the general results here.

We will take the marginal and conditional distributions to be

$$
\begin{aligned}
p(\mathbf{x}) & =\mathcal{N}\left(\mathbf{x} \mid \boldsymbol{\mu}, \mathbf{\Lambda}^{-1}\right) \\
p(\mathbf{y} \mid \mathbf{x}) & =\mathcal{N}\left(\mathbf{y} \mid \mathbf{A} \mathbf{x}+\mathbf{b}, \mathbf{L}^{-1}\right)
\end{aligned}
$$

![](https://cdn.mathpix.com/cropped/2025_10_01_d90a4d82c88904700833g-101.jpg?height=719&width=1354&top_left_y=205&top_left_x=236)

**Image Description:** The image contains two panels labeled (a) and (b). 

In panel (a), a contour plot depicts a bivariate probability distribution \( p(x_a, x_b) \). The x-axis is labeled \( x_a \) (ranging from 0 to 1), and the y-axis is labeled \( x_b \) (also ranging from 0 to 1). A red horizontal line at \( x_b = 0.7 \) indicates a conditional probability.

Panel (b) presents two probability density functions, \( p(x_a | x_b = 0.7) \) in red and \( p(x_a) \) in blue, graphed against \( x_a \). The x-axis corresponds to \( x_a \), while the y-axis represents probability density (0 to 10).

Figure 3.5 (a) Contours of a Gaussian distribution $p\left(x_{a}, x_{b}\right)$ over two variables. (b) The marginal distribution $p\left(x_{a}\right)$ (blue curve) and the conditional distribution $p\left(x_{a} \mid x_{b}\right)$ for $x_{b}=0.7$ (red curve).

where $\boldsymbol{\mu}, \mathbf{A}$, and $\mathbf{b}$ are parameters governing the means, and $\boldsymbol{\Lambda}$ and $\mathbf{L}$ are precision matrices. If $\mathbf{x}$ has dimensionality $M$ and $\mathbf{y}$ has dimensionality $D$, then the matrix $\mathbf{A}$ has size $D \times M$.

First we find an expression for the joint distribution over $\mathbf{x}$ and $\mathbf{y}$. To do this, we define

$$
\mathbf{z}=\binom{\mathbf{x}}{\mathbf{y}}
$$
and then consider the log of the joint distribution:
\$\$\begin{aligned}

\ln p(\mathbf{z})= \& \ln p(\mathbf{x})+\ln p(\mathbf{y} \mid \mathbf{x}) <br>
= \& -\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^{\mathrm{T}} \boldsymbol{\Lambda}(\mathbf{x}-\boldsymbol{\mu}) <br>
\& -\frac{1}{2}(\mathbf{y}-\mathbf{A x}-\mathbf{b})^{\mathrm{T}} \mathbf{L}(\mathbf{y}-\mathbf{A x}-\mathbf{b})+const

\end{aligned}$$
where 'const' denotes terms independent of $\mathbf{x}$ and $\mathbf{y}$. As before, we see that this is a quadratic function of the components of $\mathbf{z}$, and hence, $p(\mathbf{z})$ is Gaussian distribution. To find the precision of this Gaussian, we consider the second-order terms in (3.86), which can be written as
$$\begin{aligned}

\& -\frac{1}{2} \mathbf{x}^{\mathrm{T}}\left(\boldsymbol{\Lambda}+\mathbf{A}^{\mathrm{T}} \mathbf{L} \mathbf{A}\right) \mathbf{x}-\frac{1}{2} \mathbf{y}^{\mathrm{T}} \mathbf{L} \mathbf{y}+\frac{1}{2} \mathbf{y}^{\mathrm{T}} \mathbf{L} \mathbf{A} \mathbf{x}+\frac{1}{2} \mathbf{x}^{\mathrm{T}} \mathbf{A}^{\mathrm{T}} \mathbf{L} \mathbf{y} <br>
\& =-\frac{1}{2}\binom{\mathbf{x}}{\mathbf{y}}^{\mathrm{T}}\left($$
\begin{array}{cc}
\boldsymbol{\Lambda}+\mathbf{A}^{\mathrm{T}} \mathbf{L} \mathbf{A} & -\mathbf{A}^{\mathrm{T}} \mathbf{L} \\
-\mathbf{L} \mathbf{A} & \mathbf{L}
\end{array}
$$\right)\binom{\mathbf{x}}{\mathbf{y}}=-\frac{1}{2} \mathbf{z}^{\mathrm{T}} \mathbf{R} \mathbf{z}

\end{aligned}$$
and so the Gaussian distribution over $\mathbf{z}$ has precision (inverse covariance) matrix
given by
$$\mathbf{R}=\left(\begin{array}{cc}

\boldsymbol{\Lambda}+\mathbf{A}^{\mathrm{T}} \mathbf{L} \mathbf{A} \& -\mathbf{A}^{\mathrm{T}} \mathbf{L} <br>
-\mathbf{L} \mathbf{A} \& \mathbf{L}

\end{array}\right) .\$\$

The covariance matrix is found by taking the inverse of the precision, which can be Exercise 3.23 done using the matrix inversion formula (3.60) to give

$$
\operatorname{cov}[\mathbf{z}]=\mathbf{R}^{-1}=\left(\begin{array}{cc}
\mathbf{\Lambda}^{-1} & \mathbf{\Lambda}^{-1} \mathbf{A}^{\mathrm{T}} \\
\mathbf{A} \mathbf{\Lambda}^{-1} & \mathbf{L}^{-1}+\mathbf{A} \mathbf{\Lambda}^{-1} \mathbf{A}^{\mathrm{T}}
\end{array}\right) .
$$

Similarly, we can find the mean of the Gaussian distribution over $\mathbf{z}$ by identifying the linear terms in (3.86), which are given by

$$
\mathbf{x}^{\mathrm{T}} \boldsymbol{\Lambda} \boldsymbol{\mu}-\mathbf{x}^{\mathrm{T}} \mathbf{A}^{\mathrm{T}} \mathbf{L} \mathbf{b}+\mathbf{y}^{\mathrm{T}} \mathbf{L} \mathbf{b}=\binom{\mathbf{x}}{\mathbf{y}}^{\mathrm{T}}\binom{\boldsymbol{\Lambda} \boldsymbol{\mu}-\mathbf{A}^{\mathrm{T}} \mathbf{L} \mathbf{b}}{\mathbf{L} \mathbf{b}} .
$$

Using our earlier result (3.55) obtained by completing the square over the quadratic form of a multivariate Gaussian, we find that the mean of $\mathbf{z}$ is given by

$$
\mathbb{E}[\mathbf{z}]=\mathbf{R}^{-1}\binom{\mathbf{\Lambda} \boldsymbol{\mu}-\mathbf{A}^{\mathrm{T}} \mathbf{L} \mathbf{b}}{\mathbf{L b}}
$$

Exercise 3.24
Making use of (3.89), we then obtain

$$
\mathbb{E}[\mathbf{z}]=\binom{\boldsymbol{\mu}}{\mathbf{A} \boldsymbol{\mu}+\mathbf{b}} .
$$

## Section 3.2

## Section 3.2

$$
\begin{aligned}
\mathbb{E}[\mathbf{y}] & =\mathbf{A} \boldsymbol{\mu}+\mathbf{b} \\
\operatorname{cov}[\mathbf{y}] & =\mathbf{L}^{-1}+\mathbf{A} \mathbf{\Lambda}^{-1} \mathbf{A}^{\mathrm{T}} .
\end{aligned}
$$

A special case of this result is when $\mathbf{A}=\mathbf{I}$, in which case the marginal distribution reduces to the convolution of two Gaussians, for which we see that the mean of the convolution is the sum of the means of the two Gaussians and the covariance of the convolution is the sum of their covariances.

Finally, we seek an expression for the conditional $p(\mathbf{x} \mid \mathbf{y})$. Recall that the results for the conditional distribution are most easily expressed in terms of the partitioned precision matrix, using (3.57) and (3.59). Applying these results to (3.89) and (3.92), we see that the conditional distribution $p(\mathbf{x} \mid \mathbf{y})$ has mean and covariance given by

$$
\begin{aligned}
\mathbb{E}[\mathbf{x} \mid \mathbf{y}] & =\left(\boldsymbol{\Lambda}+\mathbf{A}^{\mathrm{T}} \mathbf{L} \mathbf{A}\right)^{-1}\left\{\mathbf{A}^{\mathrm{T}} \mathbf{L}(\mathbf{y}-\mathbf{b})+\boldsymbol{\Lambda} \boldsymbol{\mu}\right\} \\
\operatorname{cov}[\mathbf{x} \mid \mathbf{y}] & =\left(\boldsymbol{\Lambda}+\mathbf{A}^{\mathrm{T}} \mathbf{L} \mathbf{A}\right)^{-1} .
\end{aligned}
$$

The evaluation of this conditional distribution can be seen as an example of Bayes' theorem, in which we interpret $p(\mathbf{x})$ as a prior distribution over $\mathbf{x}$. If the variable $\mathbf{y}$ is observed, then the conditional distribution $p(\mathbf{x} \mid \mathbf{y})$ represents the corresponding posterior distribution over $\mathbf{x}$. Having found the marginal and conditional distributions, we have effectively expressed the joint distribution $p(\mathbf{z})=p(\mathbf{x}) p(\mathbf{y} \mid \mathbf{x})$ in the form $p(\mathbf{x} \mid \mathbf{y}) p(\mathbf{y})$.

These results can be summarized as follows. Given a marginal Gaussian distribution for $\mathbf{x}$ and a conditional Gaussian distribution for $\mathbf{y}$ given $\mathbf{x}$ in the form

$$
\begin{aligned}
p(\mathbf{x}) & =\mathcal{N}\left(\mathbf{x} \mid \boldsymbol{\mu}, \mathbf{\Lambda}^{-1}\right) \\
p(\mathbf{y} \mid \mathbf{x}) & =\mathcal{N}\left(\mathbf{y} \mid \mathbf{A} \mathbf{x}+\mathbf{b}, \mathbf{L}^{-1}\right),
\end{aligned}
$$

then the marginal distribution of $\mathbf{y}$ and the conditional distribution of $\mathbf{x}$ given $\mathbf{y}$ are given by

$$
\begin{aligned}
p(\mathbf{y}) & =\mathcal{N}\left(\mathbf{y} \mid \mathbf{A} \boldsymbol{\mu}+\mathbf{b}, \mathbf{L}^{-1}+\mathbf{A} \mathbf{\Lambda}^{-1} \mathbf{A}^{\mathrm{T}}\right) \\
p(\mathbf{x} \mid \mathbf{y}) & =\mathcal{N}\left(\mathbf{x} \mid \mathbf{\Sigma}\left\{\mathbf{A}^{\mathrm{T}} \mathbf{L}(\mathbf{y}-\mathbf{b})+\boldsymbol{\Lambda} \boldsymbol{\mu}\right\}, \mathbf{\Sigma}\right)
\end{aligned}
$$

where

$$
\boldsymbol{\Sigma}=\left(\boldsymbol{\Lambda}+\mathbf{A}^{\mathrm{T}} \mathbf{L} \mathbf{A}\right)^{-1}
$$

### 3.2.7 Maximum likelihood

Given a data set $\mathbf{X}=\left(\mathbf{x}_{1}, \ldots, \mathbf{x}_{N}\right)^{\mathrm{T}}$ in which the observations $\left\{\mathbf{x}_{n}\right\}$ are assumed to be drawn independently from a multivariate Gaussian distribution, we can estimate the parameters of the distribution by maximum likelihood. The log likelihood function is given by

$$
\ln p(\mathbf{X} \mid \boldsymbol{\mu}, \boldsymbol{\Sigma})=-\frac{N D}{2} \ln (2 \pi)-\frac{N}{2} \ln |\boldsymbol{\Sigma}|-\frac{1}{2} \sum_{n=1}^{N}\left(\mathbf{x}_{n}-\boldsymbol{\mu}\right)^{\mathrm{T}} \boldsymbol{\Sigma}^{-1}\left(\mathbf{x}_{n}-\boldsymbol{\mu}\right) .
$$

By simple rearrangement, we see that the likelihood function depends on the data set only through the two quantities

$$
\sum_{n=1}^{N} \mathbf{x}_{n}, \quad \sum_{n=1}^{N} \mathbf{x}_{n} \mathbf{x}_{n}^{\mathrm{T}}
$$

These are known as the sufficient statistics for the Gaussian distribution. Using (A.19), the derivative of the log likelihood with respect to $\boldsymbol{\mu}$ is given by

$$
\frac{\partial}{\partial \boldsymbol{\mu}} \ln p(\mathbf{X} \mid \boldsymbol{\mu}, \boldsymbol{\Sigma})=\sum_{n=1}^{N} \boldsymbol{\Sigma}^{-1}\left(\mathbf{x}_{n}-\boldsymbol{\mu}\right)
$$
and setting this derivative to zero, we obtain the solution for the maximum likelihood estimate of the mean:
$$
\boldsymbol{\mu}_{\mathrm{ML}}=\frac{1}{N} \sum_{n=1}^{N} \mathbf{x}_{n}
$$

Exercise 3.28

Exercise 3.29
which is the mean of the observed set of data points. The maximization of (3.102) with respect to $\boldsymbol{\Sigma}$ is rather more involved. The simplest approach is to ignore the symmetry constraint and show that the resulting solution is symmetric as required. Alternative derivations of this result, which impose the symmetry and positive definiteness constraints explicitly, can be found in Magnus and Neudecker (1999). The result is as expected and takes the form

$$
\boldsymbol{\Sigma}_{\mathrm{ML}}=\frac{1}{N} \sum_{n=1}^{N}\left(\mathbf{x}_{n}-\boldsymbol{\mu}_{\mathrm{ML}}\right)\left(\mathbf{x}_{n}-\boldsymbol{\mu}_{\mathrm{ML}}\right)^{\mathrm{T}},
$$
which involves $\boldsymbol{\mu}_{\mathrm{ML}}$ because this is the result of a joint maximization with respect to $\boldsymbol{\mu}$ and $\boldsymbol{\Sigma}$. Note that the solution (3.105) for $\boldsymbol{\mu}_{\mathrm{ML}}$ does not depend on $\boldsymbol{\Sigma}_{\mathrm{ML}}$, and so we can first evaluate $\boldsymbol{\mu}_{\mathrm{ML}}$ and then use this to evaluate $\boldsymbol{\Sigma}_{\mathrm{ML}}$.

If we evaluate the expectations of the maximum likelihood solutions under the true distribution, we obtain the following results

$$
\begin{aligned}
\mathbb{E}\left[\boldsymbol{\mu}_{\mathrm{ML}}\right] & =\boldsymbol{\mu} \\
\mathbb{E}\left[\boldsymbol{\Sigma}_{\mathrm{ML}}\right] & =\frac{N-1}{N} \boldsymbol{\Sigma} .
\end{aligned}
$$

We see that the expectation of the maximum likelihood estimate for the mean is equal to the true mean. However, the maximum likelihood estimate for the covariance has an expectation that is less than the true value, and hence, it is biased. We can correct this bias by defining a different estimator $\widetilde{\boldsymbol{\Sigma}}$ given by

$$
\widetilde{\boldsymbol{\Sigma}}=\frac{1}{N-1} \sum_{n=1}^{N}\left(\mathbf{x}_{n}-\boldsymbol{\mu}_{\mathrm{ML}}\right)\left(\mathbf{x}_{n}-\boldsymbol{\mu}_{\mathrm{ML}}\right)^{\mathrm{T}} .
$$

Clearly from (3.106) and (3.108), the expectation of $\widetilde{\boldsymbol{\Sigma}}$ is equal to $\boldsymbol{\Sigma}$.

### 3.2.8 Sequential estimation

Our discussion of the maximum likelihood solution represents a batch method in which the entire training data set is considered at once. An alternative is to use sequential methods, which allow data points to be processed one at a time and then discarded. These are important for online applications and for large data when the batch processing of all data points at once is infeasible.

Consider the result (3.105) for the maximum likelihood estimator of the mean $\boldsymbol{\mu}_{\mathrm{ML}}$, which we will denote by $\boldsymbol{\mu}_{\mathrm{ML}}^{(N)}$ when it is based on $N$ observations. If we

Figure 3.6 Plots of the Old Faithful data in which the red curves are contours of constant probability density. (a) A single Gaussian distribution which has been fitted to the data using maximum likelihood. Note that this distribution fails to capture the two clumps in the data and indeed places much of its probability mass in the central region between the clumps where the data are relatively sparse. (b) The distribution given by a linear combination of two Gaussians, also fitted by maximum likelihood, which gives a better representation of the data.
![](https://cdn.mathpix.com/cropped/2025_10_01_d90a4d82c88904700833g-105.jpg?height=570&width=1022&top_left_y=217&top_left_x=630)

**Image Description:** The image consists of two scatter plot diagrams (labeled a and b). 

1. **Diagram Details**: The left diagram (a) shows blue dots representing data points overlaid with contour lines (in orange) indicating density. The axes range from approximately 1 to 6 on the x-axis and 40 to 100 on the y-axis. The contours depict the distribution of the data points in variable space. The right diagram (b) also features blue dots and orange contour lines but displays a distinct, more clustered distribution with slightly altered axis ranges, emphasizing differences in data clustering or density.

dissect out the contribution from the final data point $\mathbf{x}_{N}$, we obtain

$$
\begin{aligned}
\boldsymbol{\mu}_{\mathrm{ML}}^{(N)} & =\frac{1}{N} \sum_{n=1}^{N} \mathbf{x}_{n} \\
& =\frac{1}{N} \mathbf{x}_{N}+\frac{1}{N} \sum_{n=1}^{N-1} \mathbf{x}_{n} \\
& =\frac{1}{N} \mathbf{x}_{N}+\frac{N-1}{N} \boldsymbol{\mu}_{\mathrm{ML}}^{(N-1)} \\
& =\boldsymbol{\mu}_{\mathrm{ML}}^{(N-1)}+\frac{1}{N}\left(\mathbf{x}_{N}-\boldsymbol{\mu}_{\mathrm{ML}}^{(N-1)}\right)
\end{aligned}
$$

This result has a nice interpretation, as follows. After observing $N-1$ data points, we estimate $\boldsymbol{\mu}$ by $\boldsymbol{\mu}_{\mathrm{ML}}^{(N-1)}$. We now observe data point $\mathbf{x}_{N}$, and we obtain our revised estimate $\boldsymbol{\mu}_{\mathrm{ML}}^{(N)}$ by moving the old estimate a small amount, proportional to $1 / N$, in the direction of the 'error signal' ( $\mathbf{x}_{N}-\boldsymbol{\mu}_{\mathrm{ML}}^{(N-1)}$ ). Note that, as $N$ increases, so the contributions from successive data points get smaller.

### 3.2.9 Mixtures of Gaussians

Although the Gaussian distribution has some important analytical properties, it suffers from significant limitations when used to model modelling real data sets. Consider the example shown in Figure 3.6(a). This is known as the 'Old Faithful' data set, and comprises 272 measurements of the eruption of the Old Faithful geyser in Yellowstone National Park in the USA. Each measurement gives the duration of the eruption in minutes (horizontal axis) and the time in minutes to the next eruption (vertical axis). We see that the data set forms two dominant clumps, and that a simple Gaussian distribution is unable to capture this structure.

We might expect that a superposition of two Gaussian distributions would be able to do a much better job of representing the structure in this data set, and indeed

Figure 3.7 Example of a Gaussian mixture distribution in one dimension showing three Gaussians (each scaled by a coefficient) in blue and their sum in red.
![](https://cdn.mathpix.com/cropped/2025_10_01_d90a4d82c88904700833g-106.jpg?height=406&width=593&top_left_y=222&top_left_x=1059)

**Image Description:** The image is a graphical representation of a probability density function, labeled \( p(t|x) \), plotted against time \( t \) on the x-axis. Two curves are displayed: a red curve and a blue curve, indicating different probabilistic distributions for the variable \( t \) conditioned on \( x \). The curves exhibit multiple peaks, suggesting varying probabilities at specific intervals of \( t \). The absence of grid lines and clear labels enhances the focus on the comparative nature of the distributions depicted.


this proves to be the case, as can be seen from Figure 3.6(b). Such superpositions, formed by taking linear combinations of more basic distributions such as Gaussians, can be formulated as probabilistic models known as mixture distributions. In this section we will consider Gaussians to illustrate the framework of mixture models. More generally, mixture models can comprise linear combinations of other distributions, for example mixtures of Bernoulli distributions for binary variables. In Figure 3.7 we see that a linear combination of Gaussians can give rise to very complex densities. By using a sufficient number of Gaussians and by adjusting their means and covariances as well as the coefficients in the linear combination, almost any continuous distribution can be approximated to arbitrary accuracy.

We therefore consider a superposition of $K$ Gaussian densities of the form

$$
p(\mathbf{x})=\sum_{k=1}^{K} \pi_{k} \mathcal{N}\left(\mathbf{x} \mid \boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{k}\right),
$$
which is called a mixture of Gaussians. Each Gaussian density $\mathcal{N}\left(\mathbf{x} \mid \boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{k}\right)$ is called a component of the mixture and has its own mean $\boldsymbol{\mu}_{k}$ and covariance $\boldsymbol{\Sigma}_{k}$. Contour and surface plots for a Gaussian mixture in two dimensions having three components are shown in Figure 3.8.

The parameters $\pi_{k}$ in (3.111) are called mixing coefficients. If we integrate both sides of (3.111) with respect to $\mathbf{x}$, and note that both $p(\mathbf{x})$ and the individual Gaussian components are normalized, we obtain

$$
\sum_{k=1}^{K} \pi_{k}=1 .
$$

Also, given that $\mathcal{N}\left(\mathbf{x} \mid \boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{k}\right) \geqslant 0$, a sufficient condition for the requirement $p(\mathbf{x}) \geqslant$ 0 is that $\pi_{k} \geqslant 0$ for all $k$. Combining this with the condition (3.112), we obtain

$$
0 \leqslant \pi_{k} \leqslant 1 .
$$

We can therefore see that the mixing coefficients satisfy the requirements to be probabilities, and we will show that this probabilistic interpretation of mixture distribu- tions is very powerful.

![](https://cdn.mathpix.com/cropped/2025_10_01_d90a4d82c88904700833g-107.jpg?height=578&width=1493&top_left_y=207&top_left_x=100)

**Image Description:** The image contains three sub-diagrams. 

(a) A contour plot in the x1-x2 plane shows the probability density functions of three distributions (π1, π2, π3) with values 0.5, 0.3, and 0.2, respectively. The contours are color-coded: blue for π1, green for π2, and red for π3.

(b) Another contour plot visualizes a probability distribution \( p(x) \) in the x1-x2 plane with lighter hues indicating lower probabilities.

(c) A 3D surface plot represents the same probability distribution \( p(x) \) as a mesh, where the z-axis illustrates the probability density, enhancing the visual interpretation of the data.

Figure 3.8 Illustration of a mixture of three Gaussians in a two-dimensional space. (a) Contours of constant density for each of the mixture components, in which the three components are denoted red, blue, and green, and the values of the mixing coefficients are shown below each component. (b) Contours of the marginal probability density $p(\mathbf{x})$ of the mixture distribution. (c) A surface plot of the distribution $p(\mathbf{x})$.

From the sum and product rules of probability, the marginal density can be written as

$$
p(\mathbf{x})=\sum_{k=1}^{K} p(k) p(\mathbf{x} \mid k),
$$
which is equivalent to (3.111) in which we can view $\pi_{k}=p(k)$ as the prior probability of picking the $k$ th component, and the density $\mathcal{N}\left(\mathbf{x} \mid \boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{k}\right)=p(\mathbf{x} \mid k)$ as the probability of $\mathbf{x}$ conditioned on $k$. As we will see in later chapters, an important role is played by the corresponding posterior probabilities $p(k \mid \mathbf{x})$, which are also known as responsibilities. From Bayes' theorem, these are given by
\$\$\begin{aligned}

\gamma_{k}(\mathbf{x}) \& \equiv p(k \mid \mathbf{x}) <br>
\& =\frac{p(k) p(\mathbf{x} \mid k)}{\sum_{l} p(l) p(\mathbf{x} \mid l)} <br>
\& =\frac{\pi_{k} \mathcal{N}\left(\mathbf{x} \mid \boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{k}\right)}{\sum_{l} \pi_{l} \mathcal{N}\left(\mathbf{x} \mid \boldsymbol{\mu}_{l}, \boldsymbol{\Sigma}_{l}\right)}

\end{aligned}\$\$

The form of the Gaussian mixture distribution is governed by the parameters $\boldsymbol{\pi}$, $\boldsymbol{\mu}$, and $\boldsymbol{\Sigma}$, where we have used the notation $\boldsymbol{\pi} \equiv\left\{\pi_{1}, \ldots, \pi_{K}\right\}, \boldsymbol{\mu} \equiv\left\{\boldsymbol{\mu}_{1}, \ldots, \boldsymbol{\mu}_{K}\right\}$, and $\boldsymbol{\Sigma} \equiv\left\{\boldsymbol{\Sigma}_{1}, \ldots \boldsymbol{\Sigma}_{K}\right\}$. One way to set the values of these parameters is to use maximum likelihood. From (3.111), the log of the likelihood function is given by

$$
\ln p(\mathbf{X} \mid \boldsymbol{\pi}, \boldsymbol{\mu}, \boldsymbol{\Sigma})=\sum_{n=1}^{N} \ln \left\{\sum_{k=1}^{K} \pi_{k} \mathcal{N}\left(\mathbf{x}_{n} \mid \boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{k}\right)\right\}
$$
where $\mathbf{X}=\left\{\mathbf{x}_{1}, \ldots, \mathbf{x}_{N}\right\}$. We immediately see that the situation is now much more complex than with a single Gaussian, due to the summation over $k$ inside the log-
arithm. As a result, the maximum likelihood solution for the parameters no longer has a closed-form analytical solution. One approach for maximizing the likelihood function is to use iterative numerical optimization techniques. Alternatively, we can employ a powerful framework called expectation maximization, which has wide applicability to a variety of different deep generative models.

### 3.3. Periodic Variables

Although Gaussian distributions are of great practical significance, both in their own right and as building blocks for more complex probabilistic models, there are situations in which they are inappropriate as density models for continuous variables. One important case, which arises in practical applications, is that of periodic variables.

An example of a periodic variable is the wind direction at a particular geographical location. We might, for instance, measure the wind direction at multiple locations and wish to summarize this data using a parametric distribution. Another example is calendar time, where we may be interested in modelling quantities that are believed to be periodic over 24 hours or over an annual cycle. Such quantities can conveniently be represented using an angular (polar) coordinate $0 \leqslant \theta<2 \pi$.

We might be tempted to treat periodic variables by choosing some direction as the origin and then applying a conventional distribution such as the Gaussian. Such an approach, however, would give results that were strongly dependent on the arbitrary choice of origin. Suppose, for instance, that we have two observations at $\theta_{1}=1^{\circ}$ and $\theta_{2}=359^{\circ}$, and we model them using a standard univariate Gaussian distribution. If we place the origin at $0^{\circ}$, then the sample mean of this data set will be $180^{\circ}$ with standard deviation $179^{\circ}$, whereas if we place the origin at $180^{\circ}$, then the mean will be $0^{\circ}$ and the standard deviation will be $1^{\circ}$. We clearly need to develop a special approach for periodic variables.

### 3.3.1 Von Mises distribution

Let us consider the problem of evaluating the mean of a set of observations $\mathcal{D}=\left\{\theta_{1}, \ldots, \theta_{N}\right\}$ of a periodic variable $\theta$ where $\theta$ is measured in radians. We have already seen that the simple average $\left(\theta_{1}+\cdots+\theta_{N}\right) / N$ will be strongly coordinate dependent. To find an invariant measure of the mean, note that the observations can be viewed as points on the unit circle and can therefore be described instead by two-dimensional unit vectors $\mathrm{x}_{1}, \ldots, \mathrm{x}_{N}$ where $\left\|\mathrm{x}_{n}\right\|=1$ for $n=1, \ldots, N$, as illustrated in Figure 3.9. We can average the vectors $\left\{\mathbf{x}_{n}\right\}$ instead to give

$$
\overline{\mathbf{x}}=\frac{1}{N} \sum_{n=1}^{N} \mathbf{x}_{n}
$$
and then find the corresponding angle $\bar{\theta}$ of this average. Clearly, this definition will ensure that the location of the mean is independent of the origin of the angular coordinate. Note that $\overline{\mathbf{x}}$ will typically lie inside the unit circle. The Cartesian coordinates

Figure 3.9 Illustration of the representation of values $\theta_{n}$ of a periodic variable as twodimensional vectors $\mathbf{x}_{n}$ living on the unit circle. Also shown is the average $\overline{\mathrm{x}}$ of those vectors.
![](https://cdn.mathpix.com/cropped/2025_10_01_d90a4d82c88904700833g-109.jpg?height=617&width=649&top_left_y=219&top_left_x=998)

**Image Description:** The image depicts a circular diagram in a Cartesian coordinate system. The circle is centered at the origin and is drawn with a red outline. The axes are labeled \(x_1\) (horizontal) and \(x_2\) (vertical). Points \(x_2, x_3, x_4\) are marked on the circumference with blue dots. A vector from the origin to a point on the circle is represented, labeled \(\bar{x}\), and a green sector indicates an angle \(\theta\) formed with the radius \(\bar{r}\) extending outward. The diagram illustrates geometric relationships in a multidimensional vector space.

of the observations are given by $\mathbf{x}_{n}=\left(\cos \theta_{n}, \sin \theta_{n}\right)$, and we can write the Cartesian coordinates of the sample mean in the form $\overline{\mathbf{x}}=(\bar{r} \cos \bar{\theta}, \bar{r} \sin \bar{\theta})$. Substituting into (3.117) and equating the $x_{1}$ and $x_{2}$ components then gives

$$
\bar{x}_{1}=\bar{r} \cos \bar{\theta}=\frac{1}{N} \sum_{n=1}^{N} \cos \theta_{n}, \quad \bar{x}_{2}=\bar{r} \sin \bar{\theta}=\frac{1}{N} \sum_{n=1}^{N} \sin \theta_{n} .
$$

Taking the ratio, and using the identity $\tan \theta=\sin \theta / \cos \theta$, we can solve for $\bar{\theta}$ to give

$$
\bar{\theta}=\tan ^{-1}\left\{\frac{\sum_{n} \sin \theta_{n}}{\sum_{n} \cos \theta_{n}}\right\}
$$

Shortly, we will see how this result arises naturally as a maximum likelihood estimator.

First, we need to define a periodic generalization of the Gaussian called the von Mises distribution. Here we will limit our attention to univariate distributions, although analogous periodic distributions can also be found over hyperspheres of arbitrary dimension (Mardia and Jupp, 2000).

By convention, we will consider distributions $p(\theta)$ that have period $2 \pi$. Any probability density $p(\theta)$ defined over $\theta$ must not only be non-negative and integrate to one, but it must also be periodic. Thus, $p(\theta)$ must satisfy the three conditions:

$$
\begin{aligned}
p(\theta) & \geqslant 0 \\
\int_{0}^{2 \pi} p(\theta) \mathrm{d} \theta & =1 \\
p(\theta+2 \pi) & =p(\theta) .
\end{aligned}
$$

From (3.122), it follows that $p(\theta+M 2 \pi)=p(\theta)$ for any integer $M$.
We can easily obtain a Gaussian-like distribution that satisfies these three properties as follows. Consider a Gaussian distribution over two variables $\mathbf{x}=\left(x_{1}, x_{2}\right)$

Figure 3.10 The von Mises distribution can be derived by considering a two-dimensional Gaussian of the form (3.123), whose density contours are shown in blue, and conditioning on the unit circle shown in red.
![](https://cdn.mathpix.com/cropped/2025_10_01_d90a4d82c88904700833g-110.jpg?height=382&width=421&top_left_y=222&top_left_x=1226)

**Image Description:** The image depicts a two-dimensional Cartesian coordinate system with axes labeled \( x_1 \) (horizontal) and \( x_2 \) (vertical). It features a red circle representing the equation \( r = 1 \) and a series of blue spiral curves labeled \( p(x) \). The circle is centered at the origin with a radius of 1. The blue spirals originate from the center, expanding outward. This diagram illustrates a spatial relationship between the circle and the spirals, commonly used in discussions of functions in polar coordinates or systems in physics.


having mean $\boldsymbol{\mu}=\left(\mu_{1}, \mu_{2}\right)$ and a covariance matrix $\boldsymbol{\Sigma}=\sigma^{2} \mathbf{I}$ where $\mathbf{I}$ is the $2 \times 2$ identity matrix, so that

$$
p\left(x_{1}, x_{2}\right)=\frac{1}{2 \pi \sigma^{2}} \exp \left\{-\frac{\left(x_{1}-\mu_{1}\right)^{2}+\left(x_{2}-\mu_{2}\right)^{2}}{2 \sigma^{2}}\right\} .
$$

The contours of constant $p(\mathbf{x})$ are circles, as illustrated in Figure 3.10.
Now suppose we consider the value of this distribution along a circle of fixed radius. Then by construction, this distribution will be periodic, although it will not be normalized. We can determine the form of this distribution by transforming from Cartesian coordinates ( $x_{1}, x_{2}$ ) to polar coordinates ( $r, \theta$ ) so that

$$
x_{1}=r \cos \theta, \quad x_{2}=r \sin \theta .
$$

We also map the mean $\boldsymbol{\mu}$ into polar coordinates by writing

$$
\mu_{1}=r_{0} \cos \theta_{0}, \quad \mu_{2}=r_{0} \sin \theta_{0}
$$

Next we substitute these transformations into the two-dimensional Gaussian distribution (3.123), and then condition on the unit circle $r=1$, noting that we are interested only in the dependence on $\theta$. Focusing on the exponent in the Gaussian distribution we have

$$
\begin{aligned}
& -\frac{1}{2 \sigma^{2}}\left\{\left(r \cos \theta-r_{0} \cos \theta_{0}\right)^{2}+\left(r \sin \theta-r_{0} \sin \theta_{0}\right)^{2}\right\} \\
& \quad=-\frac{1}{2 \sigma^{2}}\left\{1+r_{0}^{2}-2 r_{0} \cos \theta \cos \theta_{0}-2 r_{0} \sin \theta \sin \theta_{0}\right\} \\
& \quad=\frac{r_{0}}{\sigma^{2}} \cos \left(\theta-\theta_{0}\right)+\mathrm{const}
\end{aligned}
$$

where 'const' denotes terms independent of $\theta$. We have made use of the following trigonometrical identities:

$$
\begin{aligned}
\cos ^{2} A+\sin ^{2} A & =1 \\
\cos A \cos B+\sin A \sin B & =\cos (A-B) .
\end{aligned}
$$

If we now define $m=r_{0} / \sigma^{2}$, we obtain our final expression for the distribution of $p(\theta)$ along the unit circle $r=1$ in the form

$$
p\left(\theta \mid \theta_{0}, m\right)=\frac{1}{2 \pi I_{0}(m)} \exp \left\{m \cos \left(\theta-\theta_{0}\right)\right\},
$$

![](https://cdn.mathpix.com/cropped/2025_10_01_d90a4d82c88904700833g-111.jpg?height=590&width=589&top_left_y=234&top_left_x=200)

**Image Description:** The image presents a graph depicting two curves. The x-axis is labeled for a continuous variable, while the y-axis represents the output of a function. The red curve corresponds to parameters \( m = 5 \) and \( \theta_0 = \frac{\pi}{4} \), showing a peak feature, while the blue curve represents \( m = 1 \) and \( \theta_0 = \frac{3\pi}{4} \), demonstrating a more moderate oscillation. Both curves are clearly distinguishable due to their colors and varying shapes, indicating different functional behaviors or characteristics based on the given parameters.

Figure 3.11 The von Mises distribution plotted for two different parameter values, shown as a Cartesian plot on the left and as the corresponding polar plot on the right.

![](https://cdn.mathpix.com/cropped/2025_10_01_d90a4d82c88904700833g-111.jpg?height=582&width=584&top_left_y=234&top_left_x=967)

**Image Description:** The image is a diagram illustrating two mathematical functions represented by curves in a polar coordinate system. The x-axis ranges from 0 to \(2\pi\), while the y-axis indicates angles from \(0\) to \(3\pi/4\). The red curve corresponds to the parameters \(m = 5\) and \(\theta_0 = \pi/4\), forming an elliptical shape, while the blue curve represents \(m = 1\) and \(\theta_0 = 3\pi/4\), depicted as a circle. The axes are labeled appropriately, and both curves are displayed in relation to the origin within the polar plot.

which is called the von Mises distribution or the circular normal. Here the parameter $\theta_{0}$ corresponds to the mean of the distribution, whereas $m$, which is known as the concentration parameter, is analogous to the inverse variance (i.e. the precision) for the Gaussian. The normalization coefficient in (3.129) is expressed in terms of $I_{0}(m)$, which is the zeroth-order modified Bessel function of the first kind (Abramowitz and Stegun, 1965) and is defined by

$$
I_{0}(m)=\frac{1}{2 \pi} \int_{0}^{2 \pi} \exp \{m \cos \theta\} \mathrm{d} \theta
$$

## Exercise 3.31

For large $m$, the distribution becomes approximately Gaussian. The von Mises distribution is plotted in Figure 3.11, and the function $I_{0}(m)$ is plotted in Figure 3.12.

Now consider the maximum likelihood estimators for the parameters $\theta_{0}$ and $m$ for the von Mises distribution. The log likelihood function is given by

$$
\ln p\left(\mathcal{D} \mid \theta_{0}, m\right)=-N \ln (2 \pi)-N \ln I_{0}(m)+m \sum_{n=1}^{N} \cos \left(\theta_{n}-\theta_{0}\right) .
$$

Setting the derivative with respect to $\theta_{0}$ equal to zero gives

$$
\sum_{n=1}^{N} \sin \left(\theta_{n}-\theta_{0}\right)=0 .
$$

To solve for $\theta_{0}$, we make use of the trigonometric identity

$$
\sin (A-B)=\cos B \sin A-\cos A \sin B
$$

Exercise 3.32 from which we obtain

![](https://cdn.mathpix.com/cropped/2025_10_01_d90a4d82c88904700833g-112.jpg?height=509&width=1488&top_left_y=236&top_left_x=129)

**Image Description:** The image consists of two graphs side by side. 

1. The left graph depicts the function \( I_0(m) \) with the x-axis labeled as \( m \) and the y-axis showing values up to approximately 3000, featuring an exponential curve that rises sharply and levels off. 
2. The right graph represents the function \( A(m) \) with the same x-axis \( m \) and a y-axis ranging from 0 to 1. This graph shows a sigmoid-like curve that increases gradually, approaching its asymptote of 1.

Both graphs are rendered in red on a white background.

Figure 3.12 Plot of the Bessel function $I_{0}(m)$ defined by (3.130), together with the function $A(m)$ defined by (3.136).

$$
\theta_{0}^{\mathrm{ML}}=\tan ^{-1}\left\{\frac{\sum_{n} \sin \theta_{n}}{\sum_{n} \cos \theta_{n}}\right\},
$$
which we recognize as the result (3.119) obtained earlier for the mean of the observations viewed in a two-dimensional Cartesian space.

Similarly, maximizing (3.131) with respect to $m$ and making use of $I_{0}^{\prime}(m)= I_{1}(m)$ (Abramowitz and Stegun, 1965), we have

$$
A\left(m_{\mathrm{ML}}\right)=\frac{1}{N} \sum_{n=1}^{N} \cos \left(\theta_{n}-\theta_{0}^{\mathrm{ML}}\right)
$$
where we have substituted for the maximum likelihood solution for $\theta_{0}^{\mathrm{ML}}$ (recalling that we are performing a joint optimization over $\theta$ and $m$ ), and we have defined
$$
A(m)=\frac{I_{1}(m)}{I_{0}(m)} .
$$

The function $A(m)$ is plotted in Figure 3.12. Making use of the trigonometric identity (3.128), we can write (3.135) in the form

$$
A\left(m_{\mathrm{ML}}\right)=\left(\frac{1}{N} \sum_{n=1}^{N} \cos \theta_{n}\right) \cos \theta_{0}^{\mathrm{ML}}+\left(\frac{1}{N} \sum_{n=1}^{N} \sin \theta_{n}\right) \sin \theta_{0}^{\mathrm{ML}} .
$$

The right-hand side of (3.137) is easily evaluated, and the function $A(m)$ can be inverted numerically. One limitation of the von Mises distribution is that it is unimodal. By forming mixtures of von Mises distributions, we obtain a flexible framework for modelling periodic variables that can handle multimodality.

For completeness, we mention briefly some alternative techniques for constructing periodic distributions. The simplest approach is to use a histogram of observations in which the angular coordinate is divided into fixed bins. This has the virtue of
simplicity and flexibility but also suffers from significant limitations, as we will see when we discuss histogram methods in more detail later. Another approach starts, like the von Mises distribution, from a Gaussian distribution over a Euclidean space but now marginalizes onto the unit circle rather than conditioning (Mardia and Jupp, 2000). However, this leads to more complex forms of distribution and will not be discussed further. Finally, any valid distribution over the real axis (such as a Gaussian) can be turned into a periodic distribution by mapping successive intervals of width $2 \pi$ onto the periodic variable ( $0,2 \pi$ ), which corresponds to 'wrapping' the real axis around the unit circle. Again, the resulting distribution is more complex to handle than the von Mises distribution.

### 3.4. The Exponential Family

The probability distributions that we have studied so far in this chapter (with the exception of mixture models) are specific examples of a broad class of distributions called the exponential family (Duda and Hart, 1973; Bernardo and Smith, 1994). Members of the exponential family have many important properties in common, and it is illuminating to discuss these properties in some generality.

The exponential family of distributions over $\mathbf{x}$, given parameters $\boldsymbol{\eta}$, is defined to be the set of distributions of the form

$$
p(\mathbf{x} \mid \boldsymbol{\eta})=h(\mathbf{x}) g(\boldsymbol{\eta}) \exp \left\{\boldsymbol{\eta}^{\mathrm{T}} \mathbf{u}(\mathbf{x})\right\}
$$
where $\mathbf{x}$ may be scalar or vector and may be discrete or continuous. Here $\boldsymbol{\eta}$ are called the natural parameters of the distribution, and $\mathbf{u}(\mathbf{x})$ is some function of $\mathbf{x}$. The function $g(\boldsymbol{\eta})$ can be interpreted as the coefficient that ensures that the distribution is normalized, and therefore, it satisfies
$$
g(\boldsymbol{\eta}) \int h(\mathbf{x}) \exp \left\{\boldsymbol{\eta}^{\mathrm{T}} \mathbf{u}(\mathbf{x})\right\} \mathrm{d} \mathbf{x}=1
$$

where the integration is replaced by summation if $\mathbf{x}$ is a discrete variable.
We begin by taking some examples of the distributions introduced earlier in the chapter and showing that they are indeed members of the exponential family. Consider first the Bernoulli distribution:

$$
p(x \mid \mu)=\operatorname{Bern}(x \mid \mu)=\mu^{x}(1-\mu)^{1-x}
$$

Expressing the right-hand side as the exponential of the logarithm, we have

$$
\begin{aligned}
p(x \mid \mu) & =\exp \{x \ln \mu+(1-x) \ln (1-\mu)\} \\
& =(1-\mu) \exp \left\{\ln \left(\frac{\mu}{1-\mu}\right) x\right\}
\end{aligned}
$$

Comparison with (3.138) allows us to identify

$$
\eta=\ln \left(\frac{\mu}{1-\mu}\right)
$$
which we can solve for $\mu$ to give $\mu=\sigma(\eta)$, where
$$
\sigma(\eta)=\frac{1}{1+\exp (-\eta)}
$$

is called the logistic sigmoid function. Thus, we can write the Bernoulli distribution using the standard representation (3.138) in the form

$$
p(x \mid \eta)=\sigma(-\eta) \exp (\eta x)
$$
where we have used $1-\sigma(\eta)=\sigma(-\eta)$, which is easily proved from (3.143). Comparison with (3.138) shows that
\$\$\begin{aligned}

u(x) \& =x <br>
h(x) \& =1 <br>
g(\eta) \& =\sigma(-\eta) .

\end{aligned}\$\$

Next consider the multinomial distribution which, for a single observation $\mathbf{x}$, takes the form

$$
p(\mathbf{x} \mid \boldsymbol{\mu})=\prod_{k=1}^{M} \mu_{k}^{x_{k}}=\exp \left\{\sum_{k=1}^{M} x_{k} \ln \mu_{k}\right\}
$$
where $\mathbf{x}=\left(x_{1}, \ldots, x_{M}\right)^{\mathrm{T}}$. Again, we can write this in the standard representation (3.138) so that
$$
p(\mathbf{x} \mid \boldsymbol{\eta})=\exp \left(\boldsymbol{\eta}^{\mathrm{T}} \mathbf{x}\right)
$$

where $\eta_{k}=\ln \mu_{k}$, and we have defined $\boldsymbol{\eta}=\left(\eta_{1}, \ldots, \eta_{M}\right)^{\mathrm{T}}$. Again, comparing with (3.138) we have

$$
\begin{aligned}
\mathbf{u}(\mathbf{x}) & =\mathbf{x} \\
h(\mathbf{x}) & =1 \\
g(\boldsymbol{\eta}) & =1 .
\end{aligned}
$$

Note that the parameters $\eta_{k}$ are not independent because the parameters $\mu_{k}$ are subject to the constraint

$$
\sum_{k=1}^{M} \mu_{k}=1
$$
so that, given any $M-1$ of the parameters $\mu_{k}$, the value of the remaining parameter is fixed. In some circumstances, it will be convenient to remove this constraint by expressing the distribution in terms of only $M-1$ parameters. This can be achieved by using the relationship (3.153) to eliminate $\mu_{M}$ by expressing it in terms of the remaining $\left\{\mu_{k}\right\}$ where $k=1, \ldots, M-1$, thereby leaving $M-1$ parameters. Note that these remaining parameters are still subject to the constraints
$$
0 \leqslant \mu_{k} \leqslant 1, \quad \sum_{k=1}^{M-1} \mu_{k} \leqslant 1 .
$$

Making use of the constraint (3.153), the multinomial distribution in this representation then becomes

$$
\begin{aligned}
& \exp \left\{\sum_{k=1}^{M} x_{k} \ln \mu_{k}\right\} \\
& \quad=\exp \left\{\sum_{k=1}^{M-1} x_{k} \ln \mu_{k}+\left(1-\sum_{k=1}^{M-1} x_{k}\right) \ln \left(1-\sum_{k=1}^{M-1} \mu_{k}\right)\right\} \\
& \quad=\exp \left\{\sum_{k=1}^{M-1} x_{k} \ln \left(\frac{\mu_{k}}{1-\sum_{j=1}^{M-1} \mu_{j}}\right)+\ln \left(1-\sum_{k=1}^{M-1} \mu_{k}\right)\right\}
\end{aligned}
$$

We now identify

$$
\ln \left(\frac{\mu_{k}}{1-\sum_{j} \mu_{j}}\right)=\eta_{k}
$$
which we can solve for $\mu_{k}$ by first summing both sides over $k$ and then rearranging and back-substituting to give
$$
\mu_{k}=\frac{\exp \left(\eta_{k}\right)}{1+\sum_{j} \exp \left(\eta_{j}\right)}
$$

This is called the softmax function or the normalized exponential. In this representation, the multinomial distribution therefore takes the form

$$
p(\mathbf{x} \mid \boldsymbol{\eta})=\left(1+\sum_{k=1}^{M-1} \exp \left(\eta_{k}\right)\right)^{-1} \exp \left(\boldsymbol{\eta}^{\mathrm{T}} \mathbf{x}\right)
$$

This is the standard form of the exponential family, with parameter vector $\boldsymbol{\eta}= \left(\eta_{1}, \ldots, \eta_{M-1}\right)^{\mathrm{T}}$ in which

$$
\begin{aligned}
& \mathbf{u}(\mathbf{x})=\mathbf{x} \\
& h(\mathbf{x})=1 \\
& g(\boldsymbol{\eta})=\left(1+\sum_{k=1}^{M-1} \exp \left(\eta_{k}\right)\right)^{-1} .
\end{aligned}
$$

Finally, let us consider the Gaussian distribution. For the univariate Gaussian, we have

$$
\begin{aligned}
p\left(x \mid \mu, \sigma^{2}\right) & =\frac{1}{\left(2 \pi \sigma^{2}\right)^{1 / 2}} \exp \left\{-\frac{1}{2 \sigma^{2}}(x-\mu)^{2}\right\} \\
& =\frac{1}{\left(2 \pi \sigma^{2}\right)^{1 / 2}} \exp \left\{-\frac{1}{2 \sigma^{2}} x^{2}+\frac{\mu}{\sigma^{2}} x-\frac{1}{2 \sigma^{2}} \mu^{2}\right\}
\end{aligned}
$$

which, after some simple rearranging, can be cast in the standard exponential family form (3.138) with

$$
\begin{aligned}
\boldsymbol{\eta} & =\binom{\mu / \sigma^{2}}{-1 / 2 \sigma^{2}} \\
\mathbf{u}(x) & =\binom{x}{x^{2}} \\
h(\mathbf{x}) & =(2 \pi)^{-1 / 2} \\
g(\boldsymbol{\eta}) & =\left(-2 \eta_{2}\right)^{1 / 2} \exp \left(\frac{\eta_{1}^{2}}{4 \eta_{2}}\right) .
\end{aligned}
$$

Finally, we shall sometimes make use of a restricted form of (3.138) in which we choose $\mathbf{u}(\mathbf{x})=\mathbf{x}$. However, this can be somewhat generalized by noting that if $f(\mathbf{x})$ is a normalized density then

$$
\frac{1}{s} f\left(\frac{1}{s} \mathbf{x}\right)
$$
is also a normalized density, where $s>0$ is a scale parameter. Combining these, we arrive at a restricted set of exponential family class-conditional densities of the form
$$
p\left(\mathbf{x} \mid \boldsymbol{\lambda}_{k}, s\right)=\frac{1}{s} h\left(\frac{1}{s} \mathbf{x}\right) g\left(\boldsymbol{\lambda}_{k}\right) \exp \left\{\frac{1}{s} \boldsymbol{\lambda}_{k}^{\mathrm{T}} \mathbf{x}\right\} .
$$

Note that we are allowing each class to have its own parameter vector $\boldsymbol{\lambda}_{k}$ but we are assuming that the classes share the same scale parameter $s$.

### 3.4.1 Sufficient statistics

Let us now consider the problem of estimating the parameter vector $\boldsymbol{\eta}$ in the general exponential family distribution (3.138) using the technique of maximum likelihood. Taking the gradient of both sides of (3.139) with respect to $\boldsymbol{\eta}$, we have

$$
\begin{aligned}
& \nabla g(\boldsymbol{\eta}) \int h(\mathbf{x}) \exp \left\{\boldsymbol{\eta}^{\mathrm{T}} \mathbf{u}(\mathbf{x})\right\} \mathrm{d} \mathbf{x} \\
& \quad+g(\boldsymbol{\eta}) \int h(\mathbf{x}) \exp \left\{\boldsymbol{\eta}^{\mathrm{T}} \mathbf{u}(\mathbf{x})\right\} \mathbf{u}(\mathbf{x}) \mathrm{d} \mathbf{x}=0
\end{aligned}
$$

Rearranging and making use again of (3.139) then gives

$$
-\frac{1}{g(\boldsymbol{\eta})} \nabla g(\boldsymbol{\eta})=g(\boldsymbol{\eta}) \int h(\mathbf{x}) \exp \left\{\boldsymbol{\eta}^{\mathrm{T}} \mathbf{u}(\mathbf{x})\right\} \mathbf{u}(\mathbf{x}) \mathrm{d} \mathbf{x}=\mathbb{E}[\mathbf{u}(\mathbf{x})]
$$

We therefore obtain the result

$$
-\nabla \ln g(\boldsymbol{\eta})=\mathbb{E}[\mathbf{u}(\mathbf{x})]
$$

Note that the covariance of $\mathbf{u}(\mathbf{x})$ can be expressed in terms of the second derivatives of $g(\boldsymbol{\eta})$, and similarly for higher-order moments. Thus, provided we can normalize a distribution from the exponential family, we can always find its moments by simple differentiation.

Now consider a set of independent identically distributed data denoted by $\mathbf{X}= \left\{\mathbf{x}_{1}, \ldots, \mathbf{x}_{n}\right\}$, for which the likelihood function is given by

$$
p(\mathbf{X} \mid \boldsymbol{\eta})=\left(\prod_{n=1}^{N} h\left(\mathbf{x}_{n}\right)\right) g(\boldsymbol{\eta})^{N} \exp \left\{\boldsymbol{\eta}^{\mathrm{T}} \sum_{n=1}^{N} \mathbf{u}\left(\mathbf{x}_{n}\right)\right\} .
$$

Setting the gradient of $\ln p(\mathbf{X} \mid \boldsymbol{\eta})$ with respect to $\boldsymbol{\eta}$ to zero, we get the following condition to be satisfied by the maximum likelihood estimator $\boldsymbol{\eta}_{\mathrm{ML}}$ :

$$
-\nabla \ln g\left(\boldsymbol{\eta}_{\mathrm{ML}}\right)=\frac{1}{N} \sum_{n=1}^{N} \mathbf{u}\left(\mathbf{x}_{n}\right),
$$
which can in principle be solved to obtain $\boldsymbol{\eta}_{\mathrm{ML}}$. We see that the solution for the maximum likelihood estimator depends on the data only through $\sum_{n} \mathbf{u}\left(\mathbf{x}_{n}\right)$, which is therefore called the sufficient statistic of the distribution (3.138). We do not need to store the entire data set itself but only the value of the sufficient statistic. For the Bernoulli distribution, for example, the function $\mathbf{u}(x)$ is given just by $x$ and so we need only keep the sum of the data points $\left\{x_{n}\right\}$, whereas for the Gaussian $\mathbf{u}(x)=\left(x, x^{2}\right)^{\mathrm{T}}$, and so we should keep both the sum of $\left\{x_{n}\right\}$ and the sum of $\left\{x_{n}^{2}\right\}$.

If we consider the limit $N \rightarrow \infty$, then the right-hand side of (3.174) becomes $\mathbb{E}[\mathbf{u}(\mathbf{x})]$, and so by comparing with (3.172) we see that in this limit, $\boldsymbol{\eta}_{\mathrm{ML}}$ will equal the true value $\boldsymbol{\eta}$.

### 3.5. Nonparametric Methods

Throughout this chapter, we have focused on the use of probability distributions having specific functional forms governed by a small number of parameters whose values are to be determined from a data set. This is called the parametric approach to density modelling. An important limitation of this approach is that the chosen density might be a poor model of the distribution that generates the data, which can result in poor predictive performance. For instance, if the process that generates the data is multimodal, then this aspect of the distribution can never be captured by a Gaussian, which is necessarily unimodal. In this final section, we consider some nonparametric approaches to density estimation that make few assumptions about the form of the distribution.

### 3.5.1 Histograms

Let us start with a discussion of histogram methods for density estimation, which we have already encountered in the context of marginal and conditional distributions in Figure 2.5 and in the context of the central limit theorem in Figure 3.2. Here we explore the properties of histogram density models in more detail, focusing on cases with a single continuous variable $x$. Standard histograms simply partition $x$ into distinct bins of width $\Delta_{i}$ and then count the number $n_{i}$ of observations of $x$ falling

Figure 3.13 An illustration of the histogram approach to density estimation, in which a data set of 50 data points is generated from the distribution shown by the green curve. Histogram density estimates, based on (3.175) with a common bin width $\Delta$, are shown for various values of $\Delta$.
![](https://cdn.mathpix.com/cropped/2025_10_01_d90a4d82c88904700833g-118.jpg?height=509&width=627&top_left_y=249&top_left_x=959)

**Image Description:** The image consists of three individual panels, each presenting a histogram and a smooth curve. The horizontal axis ranges from 0 to 1, representing a variable of interest, while the vertical axis indicates frequency or density, extending up to 5. Each panel is labeled with a different delta value (Δ = 0.04, Δ = 0.08, Δ = 0.25). The histograms are depicted in blue and vary in shape and height across panels, while the overlaid smooth green curve represents a probability density function, indicating the relationship between the variable and its distribution for the corresponding delta.


in bin $i$. To turn this count into a normalized probability density, we simply divide by the total number $N$ of observations and by the width $\Delta_{i}$ of the bins to obtain probability values for each bin:

$$
p_{i}=\frac{n_{i}}{N \Delta_{i}}
$$
for which it is easily seen that $\int p(x) \mathrm{d} x=1$. This gives a model for the density $p(x)$ that is constant over the width of each bin. Often the bins are chosen to have the same width $\Delta_{i}=\Delta$.

In Figure 3.13, we show an example of histogram density estimation. Here the data is drawn from the distribution corresponding to the green curve, which is formed from a mixture of two Gaussians. Also shown are three examples of histogram density estimates corresponding to three different choices for the bin width $\Delta$. We see that when $\Delta$ is very small (top figure), the resulting density model is very spiky, with a lot of structure that is not present in the underlying distribution that generated the data set. Conversely, if $\Delta$ is too large (bottom figure) then the result is a model that is too smooth and consequently fails to capture the bimodal property of the green curve. The best results are obtained for some intermediate value of $\Delta$ (middle figure). In principle, a histogram density model is also dependent on the choice of edge location for the bins, though this is typically much less significant than the bin width $\Delta$.

Note that the histogram method has the property (unlike the methods to be discussed shortly) that, once the histogram has been computed, the data set itself can be discarded, which can be advantageous if the data set is large. Also, the histogram approach is easily applied if the data points arrive sequentially.

In practice, the histogram technique can be useful for obtaining a quick visualization of data in one or two dimensions but is unsuited to most density estimation applications. One obvious problem is that the estimated density has discontinuities that are due to the bin edges rather than any property of the underlying distribution that generated the data. A major limitation of the histogram approach is its scaling with dimensionality. If we divide each variable in a $D$-dimensional space into

## Chapter 1

$M$ bins, then the total number of bins will be $M^{D}$. This exponential scaling with $D$ is an example of the curse of dimensionality. In a space of high dimensionality, the quantity of data needed to provide meaningful estimates of the local probability density would be prohibitive.

The histogram approach to density estimation does, however, teach us two important lessons. First, to estimate the probability density at a particular location, we should consider the data points that lie within some local neighbourhood of that point. Note that the concept of locality requires that we assume some form of distance measure, and here we have been assuming Euclidean distance. For histograms, this neighbourhood property was defined by the bins, and there is a natural 'smoothing' parameter describing the spatial extent of the local region, in this case the bin width. Second, to obtain good results, the value of the smoothing parameter should be neither too large nor too small. This is reminiscent of the choice of model complexity in polynomial regression where the degree $M$ of the polynomial, or alternatively the value $\lambda$ of the regularization parameter, was optimal for some intermediate value, neither too large nor too small. Armed with these insights, we turn now to a discussion of two widely used nonparametric techniques for density estimation, kernel estimators and nearest neighbours, which have better scaling with dimensionality than the simple histogram model.

### 3.5.2 Kernel densities

Let us suppose that observations are being drawn from some unknown probability density $p(\mathbf{x})$ in some $D$-dimensional space, which we will take to be Euclidean, and we wish to estimate the value of $p(\mathbf{x})$. From our earlier discussion of locality, let us consider some small region $\mathcal{R}$ containing $\mathbf{x}$. The probability mass associated with this region is given by

$$
P=\int_{\mathcal{R}} p(\mathbf{x}) \mathrm{d} \mathbf{x}
$$

Now suppose that we have collected a data set comprising $N$ observations drawn from $p(\mathbf{x})$. Because each data point has a probability $P$ of falling within $\mathcal{R}$, the total number $K$ of points that lie inside $\mathcal{R}$ will be distributed according to the binomial distribution:

$$
\operatorname{Bin}(K \mid N, P)=\frac{N!}{K!(N-K)!} P^{K}(1-P)^{N-K}
$$

Using (3.11), we see that the mean fraction of points falling inside the region is $\mathbb{E}[K / N]=P$, and similarly using (3.12), we see that the variance around this mean is $\operatorname{var}[K / N]=P(1-P) / N$. For large $N$, this distribution will be sharply peaked around the mean and so

$$
K \simeq N P .
$$

If, however, we also assume that the region $\mathcal{R}$ is sufficiently small so that the probability density $p(\mathbf{x})$ is roughly constant over the region, then we have

$$
P \simeq p(\mathbf{x}) V
$$
where $V$ is the volume of $\mathcal{R}$. Combining (3.178) and (3.179), we obtain our density estimate in the form
$$
p(\mathbf{x})=\frac{K}{N V}
$$

Note that the validity of (3.180) depends on two contradictory assumptions, namely that the region $\mathcal{R}$ is sufficiently small that the density is approximately constant over the region and yet sufficiently large (in relation to the value of that density) that the number $K$ of points falling inside the region is sufficient for the binomial distribution to be sharply peaked.

We can exploit the result (3.180) in two different ways. Either we can fix $K$ and determine the value of $V$ from the data, which gives rise to the $K$-nearest-neighbour technique discussed shortly, or we can fix $V$ and determine $K$ from the data, giving rise to the kernel approach. It can be shown that both the $K$-nearest-neighbour density estimator and the kernel density estimator converge to the true probability density in the limit $N \rightarrow \infty$ provided that $V$ shrinks with $N$ and that $K$ grows with $N$, at an appropriate rate (Duda and Hart, 1973).

We begin by discussing the kernel method in detail. To start with we take the region $\mathcal{R}$ to be a small hypercube centred on the point $\mathbf{x}$ at which we wish to determine the probability density. To count the number $K$ of points falling within this region, it is convenient to define the following function:

$$
k(\mathbf{u})= \begin{cases}1, & \left|u_{i}\right| \leqslant 1 / 2, \quad i=1, \ldots, D \\ 0, & \text { otherwise, }\end{cases}
$$
which represents a unit cube centred on the origin. The function $k(\mathbf{u})$ is an example of a kernel function, and in this context, it is also called a Parzen window. From (3.181), the quantity $k\left(\left(\mathbf{x}-\mathbf{x}_{n}\right) / h\right)$ will be 1 if the data point $\mathbf{x}_{n}$ lies inside a cube of side $h$ centred on $\mathbf{x}$, and zero otherwise. The total number of data points lying inside this cube will therefore be
$$
K=\sum_{n=1}^{N} k\left(\frac{\mathbf{x}-\mathbf{x}_{n}}{h}\right) .
$$

Substituting this expression into (3.180) then gives the following result for the estimated density at $\mathbf{x}$ :

$$
p(\mathbf{x})=\frac{1}{N} \sum_{n=1}^{N} \frac{1}{h^{D}} k\left(\frac{\mathbf{x}-\mathbf{x}_{n}}{h}\right)
$$
where we have used $V=h^{D}$ for the volume of a hypercube of side $h$ in $D$ dimensions. Using the symmetry of the function $k(\mathbf{u})$, we can now reinterpret this equation, not as a single cube centred on $\mathbf{x}$ but as the sum over $N$ cubes centred on the $N$ data points $\mathbf{x}_{n}$.

As it stands, the kernel density estimator (3.183) will suffer from one of the same problems that the histogram method suffered from, namely the presence of artificial discontinuities, in this case at the boundaries of the cubes. We can obtain a smoother

Figure 3.14 Illustration of the kernel density model (3.184) applied to the same data set used to demonstrate the histogram approach in Figure 3.13. We see that $h$ acts as a smoothing parameter and that if it is set too small (top panel), the result is a very noisy density model, whereas if it is set too large (bottom panel), then the bimodal nature of the underlying distribution from which the data is generated (shown by the green curve) is washed out. The best density model is obtained for some intermediate value of $h$
![](https://cdn.mathpix.com/cropped/2025_10_01_d90a4d82c88904700833g-121.jpg?height=509&width=624&top_left_y=249&top_left_x=962)

**Image Description:** The image consists of three distinct plots arranged vertically, each illustrating functions over the interval [0, 1]. The x-axis represents the independent variable, while the y-axis indicates the function values, ranging approximately from 0 to 5. Each plot is labeled with distinct values of \( h \): 0.005, 0.07, and 0.2. The top plot showcases a high-frequency pattern, the middle plot displays smoother transitions, and the bottom plot reveals a more uniform trend. The curves are color-coded, with one appearing in blue and the other in green, representing different functions or scenarios.

(middle panel).
density model if we choose a smoother kernel function, and a common choice is the Gaussian, which gives rise to the following kernel density model:

$$
p(\mathbf{x})=\frac{1}{N} \sum_{n=1}^{N} \frac{1}{\left(2 \pi h^{2}\right)^{D / 2}} \exp \left\{-\frac{\left\|\mathbf{x}-\mathbf{x}_{n}\right\|^{2}}{2 h^{2}}\right\}
$$
where $h$ represents the standard deviation of the Gaussian components. Thus, our density model is obtained by placing a Gaussian over each data point, adding up the contributions over the whole data set, and then dividing by $N$ so that the density is correctly normalized. In Figure 3.14, we apply the model (3.184) to the data set used earlier to demonstrate the histogram technique. We see that, as expected, the parameter $h$ plays the role of a smoothing parameter, and there is a trade-off between sensitivity to noise at small $h$ and over-smoothing at large $h$. Again, the optimization of $h$ is a problem in model complexity, analogous to the choice of bin width in histogram density estimation or the degree of the polynomial used in curve fitting.

We can choose any other kernel function $k(\mathbf{u})$ in (3.183) subject to the conditions

$$
\begin{aligned}
k(\mathbf{u}) & \geqslant 0 \\
\int k(\mathbf{u}) \mathrm{d} \mathbf{u} & =1
\end{aligned}
$$

which ensure that the resulting probability distribution is non-negative everywhere and integrates to one. The class of density model given by (3.183) is called a kernel density estimator or Parzen estimator. It has a great merit that there is no computation involved in the 'training' phase because this simply requires the training set to be stored. However, this is also one of its great weaknesses because the computational cost of evaluating the density grows linearly with the size of the data set.

Figure 3.15 Illustration of $K$-nearestneighbour density estimation using the same data set as in Figures 3.14 and 3.13. We see that the parameter $K$ governs the degree of smoothing, so that a small value of $K$ leads to a very noisy density model (top panel), whereas a large value (bottom panel) smooths out the bimodal nature of the true distribution (shown by the green curve) from which the data set was generated.
![](https://cdn.mathpix.com/cropped/2025_10_01_d90a4d82c88904700833g-122.jpg?height=509&width=627&top_left_y=249&top_left_x=959)

**Image Description:** The image depicts three plots in a vertical arrangement, illustrating the behavior of a system parameterized by \( K \) values of 1, 5, and 30. Each plot has two curves: a blue curve and a green curve. The x-axis ranges from 0 to 1, representing an independent variable, while the y-axis scales vary across the plots, indicating the respective dependent variable's magnitude. The blue curve exhibits oscillatory behavior that becomes less pronounced as \( K \) increases, while the green curve shows a smoother trend across all three plots.


### 3.5.3 Nearest-neighbours

One of the difficulties with the kernel approach to density estimation is that the parameter $h$ governing the kernel width is fixed for all kernels. In regions of high data density, a large value of $h$ may lead to over-smoothing and a washing out of structure that might otherwise be extracted from the data. However, reducing $h$ may lead to noisy estimates elsewhere in the data space where the density is smaller. Thus, the optimal choice for $h$ may be dependent on the location within the data space. This issue is addressed by nearest-neighbour methods for density estimation.

We therefore return to our general result (3.180) for local density estimation, and instead of fixing $V$ and determining the value of $K$ from the data, we consider a fixed value of $K$ and use the data to find an appropriate value for $V$. To do this, we consider a small sphere centred on the point $\mathbf{x}$ at which we wish to estimate the density $p(\mathbf{x})$, and we allow the radius of the sphere to grow until it contains precisely $K$ data points. The estimate of the density $p(\mathbf{x})$ is then given by (3.180) with $V$ set to the volume of the resulting sphere. This technique is known as $K$ nearest neighbours and is illustrated in Figure 3.15 for various choices of the parameter $K$ using the same data set as used in Figures 3.13 and 3.14. We see that the value of $K$ now governs the degree of smoothing and that again there is an optimum choice for $K$ that is neither too large nor too small. Note that the model produced by $K$ nearest neighbours is not a true density model because the integral over all space diverges.

We close this chapter by showing how the $K$-nearest-neighbour technique for density estimation can be extended to the problem of classification. To do this, we apply the $K$-nearest-neighbour density estimation technique to each class separately and then make use of Bayes' theorem. Let us suppose that we have a data set comprising $N_{k}$ points in class $\mathcal{C}_{k}$ with $N$ points in total, so that $\sum_{k} N_{k}=N$. If we wish to classify a new point $\mathbf{x}$, we draw a sphere centred on $\mathbf{x}$ containing precisely $K$ points irrespective of their class. Suppose this sphere has volume $V$ and contains $K_{k}$ points from class $\mathcal{C}_{k}$. Then (3.180) provides an estimate of the density associated

Figure 3.16 (a) In the $K$-nearestneighbour classifier, a new point, shown by the black diamond, is classified according to the majority class membership of the $K$ closest training data points, in this case $K=$ 3. (b) In the nearest-neighbour ( $K=1$ ) approach to classification, the resulting decision boundary is composed of hyperplanes that form perpendicular bisectors of pairs of points from different classes.

![](https://cdn.mathpix.com/cropped/2025_10_01_d90a4d82c88904700833g-123.jpg?height=495&width=511&top_left_y=219&top_left_x=630)

**Image Description:** The image is a scatter plot representing two classes of data points, colored red and blue. The axes are labeled \(x_1\) (horizontal) and \(x_2\) (vertical). Each point corresponds to a specific observation. A highlighted point is marked with a green square, and a green line connects this point to another, illustrating distances or relationships between points in the feature space. The diagram can be used for visualizing clustering or classification in machine learning.

(a)

![](https://cdn.mathpix.com/cropped/2025_10_01_d90a4d82c88904700833g-123.jpg?height=497&width=512&top_left_y=219&top_left_x=1135)

**Image Description:** The image depicts a scatter plot illustrating a decision boundary in a binary classification problem. The axes are labeled as \( x_1 \) (horizontal) and \( x_2 \) (vertical). There are two clusters of points: red dots representing one class and blue dots representing another. The green line represents the decision boundary that separates the two classes, indicating the regions where the model predicts different outcomes. The diagram visually demonstrates the concept of classification in machine learning, highlighting how decision boundaries can differentiate between data points of different categories.

(b)

with each class:

$$
p\left(\mathbf{x} \mid \mathcal{C}_{k}\right)=\frac{K_{k}}{N_{k} V}
$$

Similarly, the unconditional density is given by

$$
p(\mathbf{x})=\frac{K}{N V}
$$
and the class priors are given by
$$
p\left(\mathcal{C}_{k}\right)=\frac{N_{k}}{N}
$$

We can now combine (3.187), (3.188), and (3.189) using Bayes' theorem to obtain the posterior probability of class membership:

$$
p\left(\mathcal{C}_{k} \mid \mathbf{x}\right)=\frac{p\left(\mathbf{x} \mid \mathcal{C}_{k}\right) p\left(\mathcal{C}_{k}\right)}{p(\mathbf{x})}=\frac{K_{k}}{K} .
$$

We can minimize the probability of misclassification by assigning the test point $\mathbf{x}$ to the class having the largest posterior probability, corresponding to the largest value of $K_{k} / K$. Thus, to classify a new point, we identify the $K$ nearest points from the training data set and then assign the new point to the class having the largest number of representatives amongst this set. Ties can be broken at random. The particular case of $K=1$ is called the nearest-neighbour rule, because a test point is simply assigned to the same class as the nearest point from the training set. These concepts are illustrated in Figure 3.16.

An interesting property of the nearest-neighbour ( $K=1$ ) classifier is that, in the limit $N \rightarrow \infty$, the error rate is never more than twice the minimum achievable error rate of an optimal classifier, i.e., one that uses the true class distributions (Cover and Hart, 1967) .

As discussed so far, both the $K$-nearest-neighbour method and the kernel density estimator require the entire training data set to be stored, leading to expensive
computation if the data set is large. This effect can be offset, at the expense of some additional one-off computation, by constructing tree-based search structures to allow (approximate) near neighbours to be found efficiently without doing an exhaustive search of the data set. Nevertheless, these nonparametric methods are still severely limited. On the other hand, we have seen that simple parametric models are very restricted in terms of the forms of distribution that they can represent. We therefore need to find density models that are very flexible and yet for which the complexity of the models can be controlled independently of the size of the training set, and this can be achieved using deep neural networks.

## Exercises

3.1 ( $\star$ ) Verify that the Bernoulli distribution (3.2) satisfies the following properties:

$$
\begin{aligned}
\sum_{x=0}^{1} p(x \mid \mu) & =1 \\
\mathbb{E}[x] & =\mu \\
\operatorname{var}[x] & =\mu(1-\mu) .
\end{aligned}
$$

Show that the entropy $\mathrm{H}[x]$ of a Bernoulli-distributed random binary variable $x$ is given by

$$
\mathrm{H}[x]=-\mu \ln \mu-(1-\mu) \ln (1-\mu)
$$
$3.2(\star \star)$ The form of the Bernoulli distribution given by (3.2) is not symmetric between the two values of $x$. In some situations, it will be more convenient to use an equivalent formulation for which $x \in\{-1,1\}$, in which case the distribution can be written
$$
p(x \mid \mu)=\left(\frac{1-\mu}{2}\right)^{(1-x) / 2}\left(\frac{1+\mu}{2}\right)^{(1+x) / 2}
$$

where $\mu \in[-1,1]$. Show that the distribution (3.195) is normalized, and evaluate its mean, variance, and entropy.
$3.3(\star \star)$ In this exercise, we prove that the binomial distribution (3.9) is normalized. First, use the definition (3.10) of the number of combinations of $m$ identical objects chosen from a total of $N$ to show that

$$
\binom{N}{m}+\binom{N}{m-1}=\binom{N+1}{m} .
$$

Use this result to prove by induction the following result:

$$
(1+x)^{N}=\sum_{m=0}^{N}\binom{N}{m} x^{m}
$$
which is known as the binomial theorem and which is valid for all real values of $x$. Finally, show that the binomial distribution is normalized, so that
$$
\sum_{m=0}^{N}\binom{N}{m} \mu^{m}(1-\mu)^{N-m}=1
$$

which can be done by first pulling a factor $(1-\mu)^{N}$ out of the summation and then making use of the binomial theorem.
$3.4(\star \star)$ Show that the mean of the binomial distribution is given by (3.11). To do this, differentiate both sides of the normalization condition (3.198) with respect to $\mu$ and then rearrange to obtain an expression for the mean of $n$. Similarly, by differentiating (3.198) twice with respect to $\mu$ and making use of the result (3.11) for the mean of the binomial distribution, prove the result (3.12) for the variance of the binomial.
$3.5(\star)$ Show that the mode of the multivariate Gaussian (3.26) is given by $\boldsymbol{\mu}$.
$3.6(\star \star)$ Suppose that $\mathbf{x}$ has a Gaussian distribution with mean $\boldsymbol{\mu}$ and covariance $\boldsymbol{\Sigma}$. Show that the linearly transformed variable $\mathbf{A x}+\mathbf{b}$ is also Gaussian, and find its mean and covariance.
3.7 ( $\star \star \star$ ) Show that the Kullback-Leibler divergence between two Gaussian distributions $q(\mathbf{x})=\mathcal{N}\left(\mathbf{x} \mid \boldsymbol{\mu}_{q}, \boldsymbol{\Sigma}_{q}\right)$ and $p(\mathbf{x})=\mathcal{N}\left(\mathbf{x} \mid \boldsymbol{\mu}_{p}, \boldsymbol{\Sigma}_{p}\right)$ is given by

$$
\begin{aligned}
& \operatorname{KL}(q(\mathbf{x}) \| p(\mathbf{x})) \\
& =\frac{1}{2}\left\{\ln \frac{\left|\boldsymbol{\Sigma}_{p}\right|}{\left|\boldsymbol{\Sigma}_{q}\right|}-D+\operatorname{Tr}\left(\boldsymbol{\Sigma}_{p}^{-1} \boldsymbol{\Sigma}_{q}\right)+\left(\boldsymbol{\mu}_{p}-\boldsymbol{\mu}_{q}\right)^{\mathrm{T}} \boldsymbol{\Sigma}_{p}^{-1}\left(\boldsymbol{\mu}_{p}-\boldsymbol{\mu}_{q}\right)\right\}
\end{aligned}
$$

where $\operatorname{Tr}(\cdot)$ denotes the trace of a matrix, and $D$ is the dimensionality of $\mathbf{x}$.
$3.8(\star \star)$ This exercise demonstrates that the multivariate distribution with maximum entropy, for a given covariance, is a Gaussian. The entropy of a distribution $p(\mathbf{x})$ is given by

$$
\mathrm{H}[\mathbf{x}]=-\int p(\mathbf{x}) \ln p(\mathbf{x}) \mathrm{d} \mathbf{x}
$$

We wish to maximize $\mathrm{H}[\mathbf{x}]$ over all distributions $p(\mathbf{x})$ subject to the constraints that $p(\mathbf{x})$ is normalized and that it has a specific mean and covariance, so that

$$
\begin{aligned}
& \int p(\mathbf{x}) \mathrm{d} \mathbf{x}=1 \\
& \int p(\mathbf{x}) \mathbf{x} \mathrm{d} \mathbf{x}=\boldsymbol{\mu} \\
& \int p(\mathbf{x})(\mathbf{x}-\boldsymbol{\mu})(\mathbf{x}-\boldsymbol{\mu})^{\mathrm{T}} \mathrm{~d} \mathbf{x}=\mathbf{\Sigma}
\end{aligned}
$$

By performing a variational maximization of (3.200) and using Lagrange multipliers to enforce the constraints (3.201), (3.202), and (3.203), show that the maximum likelihood distribution is given by the Gaussian (3.26).
$3.9(\star \star \star)$ Show that the entropy of the multivariate Gaussian $\mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}, \boldsymbol{\Sigma})$ is given by

$$
\mathrm{H}[\mathbf{x}]=\frac{1}{2} \ln |\boldsymbol{\Sigma}|+\frac{D}{2}(1+\ln (2 \pi))
$$
where $D$ is the dimensionality of $\mathbf{x}$.
$3.10(\star \star \star)$ Consider two random variables $x_{1}$ and $x_{2}$ having Gaussian distributions with means $\mu_{1}$ and $\mu_{2}$ and precisions $\tau_{1}$ and $\tau_{2}$, respectively. Derive an expression for the differential entropy of the variable $x=x_{1}+x_{2}$. To do this, first find the distribution of $x$ by using the relation
$$
p(x)=\int_{-\infty}^{\infty} p\left(x \mid x_{2}\right) p\left(x_{2}\right) \mathrm{d} x_{2}
$$

and completing the square in the exponent. Then observe that this represents the convolution of two Gaussian distributions, which itself will be Gaussian, and finally make use of the result (2.99) for the entropy of the univariate Gaussian.
3.11 ( $\star$ ) Consider the multivariate Gaussian distribution given by (3.26). By writing the precision matrix (inverse covariance matrix) as the sum of a symmetric and an antisymmetric matrix, show that the antisymmetric term does not appear in the exponent of the Gaussian, and hence, that the precision matrix may be taken to be symmetric without loss of generality. Because the inverse of a symmetric matrix is also symmetric (see Exercise 3.16), it follows that the covariance matrix may also be chosen to be symmetric without loss of generality.
$3.12(\star \star \star)$ Consider a real, symmetric matrix $\boldsymbol{\Sigma}$ whose eigenvalue equation is given by (3.28). By taking the complex conjugate of this equation, subtracting the original equation, and then forming the inner product with eigenvector $\mathbf{u}_{i}$, show that the eigenvalues $\lambda_{i}$ are real. Similarly, use the symmetry property of $\boldsymbol{\Sigma}$ to show that two eigenvectors $\mathbf{u}_{i}$ and $\mathbf{u}_{j}$ will be orthogonal provided $\lambda_{j} \neq \lambda_{i}$. Finally, show that, without loss of generality, the set of eigenvectors can be chosen to be orthonormal, so that they satisfy (3.29), even if some of the eigenvalues are zero.
$3.13(\star \star)$ Show that a real, symmetric matrix $\boldsymbol{\Sigma}$ having the eigenvector equation (3.28) can be expressed as an expansion in the eigenvectors, with coefficients given by the eigenvalues, of the form (3.31). Similarly, show that the inverse matrix $\boldsymbol{\Sigma}^{-1}$ has a representation of the from (3.32).
$3.14(\star \star)$ A positive definite matrix $\boldsymbol{\Sigma}$ can be defined as one for which the quadratic form

$$
\mathbf{a}^{\mathrm{T}} \boldsymbol{\Sigma} \mathbf{a}
$$
is positive for any real value of the vector $\mathbf{a}$. Show that a necessary and sufficient condition for $\boldsymbol{\Sigma}$ to be positive definite is that all the eigenvalues $\lambda_{i}$ of $\boldsymbol{\Sigma}$, defined by (3.28), are positive.
$3.15(\star)$ Show that a real, symmetric matrix of size $D \times D$ has $D(D+1) / 2$ independent parameters.
$3.16(\star)$ Show that the inverse of a symmetric matrix is itself symmetric.
$3.17(\star \star)$ By diagonalizing the coordinate system using the eigenvector expansion (3.31), show that the volume contained within the hyperellipsoid corresponding to a constant Mahalanobis distance $\Delta$ is given by
$$
V_{D}|\boldsymbol{\Sigma}|^{1 / 2} \Delta^{D}
$$

where $V_{D}$ is the volume of the unit sphere in $D$ dimensions, and the Mahalanobis distance is defined by (3.27).
$3.18(\star \star)$ Prove the identity (3.60) by multiplying both sides by the matrix

$$
\left(\begin{array}{ll}
\mathrm{A} & \mathrm{~B} \\
\mathrm{C} & \mathrm{D}
\end{array}\right)
$$

and making use of the definition (3.61).
$3.19(\star \star \star)$ In Sections 3.2.4 and 3.2.5, we considered the conditional and marginal distributions for a multivariate Gaussian. More generally, we can consider a partitioning of the components of $\mathbf{x}$ into three groups $\mathbf{x}_{a}, \mathbf{x}_{b}$, and $\mathbf{x}_{c}$, with a corresponding partitioning of the mean vector $\boldsymbol{\mu}$ and of the covariance matrix $\boldsymbol{\Sigma}$ in the form

$$
\boldsymbol{\mu}=\left(\begin{array}{c}
\boldsymbol{\mu}_{a} \\
\boldsymbol{\mu}_{b} \\
\boldsymbol{\mu}_{c}
\end{array}\right), \quad \boldsymbol{\Sigma}=\left(\begin{array}{ccc}
\boldsymbol{\Sigma}_{a a} & \boldsymbol{\Sigma}_{a b} & \boldsymbol{\Sigma}_{a c} \\
\boldsymbol{\Sigma}_{b a} & \boldsymbol{\Sigma}_{b b} & \boldsymbol{\Sigma}_{b c} \\
\boldsymbol{\Sigma}_{c a} & \boldsymbol{\Sigma}_{c b} & \boldsymbol{\Sigma}_{c c}
\end{array}\right) .
$$

By making use of the results of Section 3.2, find an expression for the conditional distribution $p\left(\mathbf{x}_{a} \mid \mathbf{x}_{b}\right)$ in which $\mathbf{x}_{c}$ has been marginalized out.
$3.20(\star \star)$ A very useful result from linear algebra is the Woodbury matrix inversion formula given by

$$
(\mathbf{A}+\mathbf{B C D})^{-1}=\mathbf{A}^{-1}-\mathbf{A}^{-1} \mathbf{B}\left(\mathbf{C}^{-1}+\mathbf{D A}^{-1} \mathbf{B}\right)^{-1} \mathbf{D A}^{-1}
$$

By multiplying both sides by ( $\mathbf{A}+\mathbf{B C D}$ ), prove the correctness of this result.
$3.21(\star)$ Let $\mathbf{x}$ and $\mathbf{z}$ be two independent random vectors, so that $p(\mathbf{x}, \mathbf{z})=p(\mathbf{x}) p(\mathbf{z})$. Show that the mean of their sum $\mathbf{y}=\mathbf{x}+\mathbf{z}$ is given by the sum of the means of each of the variables separately. Similarly, show that the covariance matrix of $\mathbf{y}$ is given by the sum of the covariance matrices of $\mathbf{x}$ and $\mathbf{z}$.
3.22 ( $\star \star \star$ ) Consider a joint distribution over the variable

$$
\mathbf{z}=\binom{\mathbf{x}}{\mathbf{y}}
$$
whose mean and covariance are given by (3.92) and (3.89), respectively. By making use of the results (3.76) and (3.77), show that the marginal distribution $p(\mathbf{x})$ is given by (3.83). Similarly, by making use of the results (3.65) and (3.66), show that the conditional distribution $p(\mathbf{y} \mid \mathbf{x})$ is given by (3.84).
$3.23(\star \star)$ Using the partitioned matrix inversion formula (3.60), show that the inverse of the precision matrix (3.88) is given by the covariance matrix (3.89).
$3.24(\star \star)$ By starting from (3.91) and making use of the result (3.89), verify the result (3.92).
$3.25(\star \star)$ Consider two multi-dimensional random vectors $\mathbf{x}$ and $\mathbf{z}$ having Gaussian distributions $p(\mathbf{x})=\mathcal{N}\left(\mathbf{x} \mid \boldsymbol{\mu}_{\mathbf{x}}, \boldsymbol{\Sigma}_{\mathbf{x}}\right)$ and $p(\mathbf{z})=\mathcal{N}\left(\mathbf{z} \mid \boldsymbol{\mu}_{\mathbf{z}}, \boldsymbol{\Sigma}_{\mathbf{z}}\right)$, respectively, together with their sum $\mathbf{y}=\mathbf{x}+\mathbf{z}$. By considering the linear-Gaussian model comprising the product of the marginal distribution $p(\mathbf{x})$ and the conditional distribution $p(\mathbf{y} \mid \mathbf{x})$ and making use of the results (3.93) and (3.94), show that the marginal distribution of $p(\mathbf{y})$ is given by
$$
p(\mathbf{y})=\mathcal{N}\left(\mathbf{y} \mid \boldsymbol{\mu}_{\mathbf{x}}+\boldsymbol{\mu}_{\mathbf{z}}, \boldsymbol{\Sigma}_{\mathbf{x}}+\boldsymbol{\Sigma}_{\mathbf{z}}\right) .
$$

$3.26(\star \star \star)$ This exercise and the next provide practice at manipulating the quadratic forms that arise in linear-Gaussian models, and they also serve as an independent check of results derived in the main text. Consider a joint distribution $p(\mathbf{x}, \mathbf{y})$ defined by the marginal and conditional distributions given by (3.83) and (3.84). By examining the quadratic form in the exponent of the joint distribution and using the technique of 'completing the square' discussed in Section 3.2, find expressions for the mean and covariance of the marginal distribution $p(\mathbf{y})$ in which the variable $\mathbf{x}$ has been integrated out. To do this, make use of the Woodbury matrix inversion formula (3.210). Verify that these results agree with (3.93) and (3.94).
3.27 ( $\star \star \star$ ) Consider the same joint distribution as in Exercise 3.26, but now use the technique of completing the square to find expressions for the mean and covariance of the conditional distribution $p(\mathbf{x} \mid \mathbf{y})$. Again, verify that these agree with the corresponding expressions (3.95) and (3.96).
$3.28(\star \star)$ To find the maximum likelihood solution for the covariance matrix of a multivariate Gaussian, we need to maximize the log likelihood function (3.102) with respect to $\boldsymbol{\Sigma}$, noting that the covariance matrix must be symmetric and positive definite. Here we proceed by ignoring these constraints and doing a straightforward maximization. Using the results (A.21), (A.26), and (A.28) from Appendix A, show that the covariance matrix $\boldsymbol{\Sigma}$ that maximizes the log likelihood function (3.102) is given by the sample covariance (3.106). We note that the final result is necessarily symmetric and positive definite (provided the sample covariance is non-singular).
$3.29(\star \star)$ Use the result (3.42) to prove (3.46). Now, using the results (3.42) and (3.46), show that

$$
\mathbb{E}\left[\mathbf{x}_{n} \mathbf{x}_{m}^{\mathrm{T}}\right]=\boldsymbol{\mu} \boldsymbol{\mu}^{\mathrm{T}}+I_{n m} \boldsymbol{\Sigma}
$$
where $\mathbf{x}_{n}$ denotes a data point sampled from a Gaussian distribution with mean $\boldsymbol{\mu}$ and covariance $\boldsymbol{\Sigma}$, and $I_{n m}$ denotes the $(n, m)$ element of the identity matrix. Hence, prove the result (3.108).
$3.30(\star)$ The various trigonometric identities used in the discussion of periodic variables in this chapter can be proven easily from the relation
$$
\exp (i A)=\cos A+i \sin A
$$

in which $i$ is the square root of minus one. By considering the identity

$$
\exp (i A) \exp (-i A)=1
$$
prove the result (3.127). Similarly, using the identity
$$
\cos (A-B)=\Re \exp \{i(A-B)\}
$$

where $\Re$ denotes the real part, prove (3.128). Finally, by using $\sin (A-B)= \Im \exp \{i(A-B)\}$, where $\Im$ denotes the imaginary part, prove the result (3.133).
3.31 ( $\star \star$ ) For large $m$, the von Mises distribution (3.129) becomes sharply peaked around the mode $\theta_{0}$. By defining $\xi=m^{1 / 2}\left(\theta-\theta_{0}\right)$ and taking the Taylor expansion of the cosine function given by

$$
\cos \alpha=1-\frac{\alpha^{2}}{2}+O\left(\alpha^{4}\right)
$$
show that as $m \rightarrow \infty$, the von Mises distribution tends to a Gaussian.
$3.32(\star)$ Using the trigonometric identity (3.133), show that solution of (3.132) for $\theta_{0}$ is given by (3.134).
3.33 ( $\star$ ) By computing the first and second derivatives of the von Mises distribution (3.129), and using $I_{0}(m)>0$ for $m>0$, show that the maximum of the distribution occurs when $\theta=\theta_{0}$ and that the minimum occurs when $\theta=\theta_{0}+\pi(\bmod 2 \pi)$.
$3.34(\star)$ By making use of the result (3.118) together with (3.134) and the trigonometric identity (3.128), show that the maximum likelihood solution $m_{\text {ML }}$ for the concentration of the von Mises distribution satisfies $A\left(m_{\mathrm{ML}}\right)=\bar{r}$ where $\bar{r}$ is the radius of the mean of the observations viewed as unit vectors in the two-dimensional Euclidean plane, as illustrated in Figure 3.9.
$3.35(\star)$ Verify that the multivariate Gaussian distribution can be cast in exponential family form (3.138), and derive expressions for $\boldsymbol{\eta}, \mathbf{u}(\mathbf{x}), h(\mathbf{x})$, and $g(\boldsymbol{\eta})$ analogous to (3.164) to (3.167).
$3.36(\star)$ The result (3.172) showed that the negative gradient of $\ln g(\boldsymbol{\eta})$ for the exponential family is given by the expectation of $\mathbf{u}(\mathbf{x})$. By taking the second derivatives of (3.139), show that
$$
-\nabla \nabla \ln g(\boldsymbol{\eta})=\mathbb{E}\left[\mathbf{u}(\mathbf{x}) \mathbf{u}(\mathbf{x})^{\mathrm{T}}\right]-\mathbb{E}[\mathbf{u}(\mathbf{x})] \mathbb{E}\left[\mathbf{u}(\mathbf{x})^{\mathrm{T}}\right]=\operatorname{cov}[\mathbf{u}(\mathbf{x})] .
$$

$3.37(\star \star)$ Consider a histogram-like density model in which the space $\mathbf{x}$ is divided into fixed regions for which the density $p(\mathbf{x})$ takes the constant value $h_{i}$ over the $i$ th region. The volume of region $i$ is denoted $\Delta_{i}$. Suppose we have a set of $N$ observations of $\mathbf{x}$ such that $n_{i}$ of these observations fall in region $i$. Using a Lagrange multiplier to enforce the normalization constraint on the density, derive an expression for the maximum likelihood estimator for the $\left\{h_{i}\right\}$.
3.38 ( $\star$ ) Show that the $K$-nearest-neighbour density model defines an improper distribution whose integral over all space is divergent.

![](https://cdn.mathpix.com/cropped/2025_10_01_d90a4d82c88904700833g-130.jpg?height=1246&width=1230&top_left_y=222&top_left_x=422)

**Image Description:** The image is a title slide from an academic lecture on "Single-layer Networks: Regression." It features a colorful abstract background with swirls of colors blending into each other, predominantly in shades of red, green, and yellow. The text "4" is displayed prominently at the top in a bold, large font, followed by the title in a slightly smaller font, which describes the focus area of the lecture. The overall design is visually striking, aimed at drawing attention to the topic being discussed.

Section 1.2

In this chapter we discuss some of the basic ideas behind neural networks using the framework of linear regression, which we encountered briefly in the context of polynomial curve fitting. We will see that a linear regression model corresponds to a simple form of neural network having a single layer of learnable parameters. Although single-layer networks have very limited practical applicability, they have simple analytical properties and provide an excellent framework for introducing many of the core concepts that will lay a foundation for our discussion of deep neural networks in later chapters.

### 4.1. Linear Regression

The goal of regression is to predict the value of one or more continuous target variables $t$ given the value of a $D$-dimensional vector $\mathbf{x}$ of input variables. Typically we are given a training data set comprising $N$ observations $\left\{\mathbf{x}_{n}\right\}$, where $n=1, \ldots, N$, together with corresponding target values $\left\{t_{n}\right\}$, and the goal is to predict the value of $t$ for a new value of $\mathbf{x}$. To do this, we formulate a function $y(\mathbf{x}, \mathbf{w})$ whose values for new inputs $\mathbf{x}$ constitute the predictions for the corresponding values of $t$, and where $\mathbf{w}$ represents a vector of parameters that can be learned from the training data.

The simplest model for regression is one that involves a linear combination of the input variables:

$$
y(\mathbf{x}, \mathbf{w})=w_{0}+w_{1} x_{1}+\ldots+w_{D} x_{D}
$$
where $\mathbf{x}=\left(x_{1}, \ldots, x_{D}\right)^{\mathrm{T}}$. The term linear regression sometimes refers specifically to this form of model. The key property of this model is that it is a linear function of the parameters $w_{0}, \ldots, w_{D}$. It is also, however, a linear function of the input variables $x_{i}$, and this imposes significant limitations on the model.

### 4.1.1 Basis functions

We can extend the class of models defined by (4.1) by considering linear combinations of fixed nonlinear functions of the input variables, of the form

$$
y(\mathbf{x}, \mathbf{w})=w_{0}+\sum_{j=1}^{M-1} w_{j} \phi_{j}(\mathbf{x})
$$

Section 4.3

Section 6.1
where $\phi_{j}(\mathbf{x})$ are known as basis functions. By denoting the maximum value of the index $j$ by $M-1$, the total number of parameters in this model will be $M$.

The parameter $w_{0}$ allows for any fixed offset in the data and is sometimes called a bias parameter (not to be confused with bias in a statistical sense). It is often convenient to define an additional dummy basis function $\phi_{0}(x)$ whose value is fixed at $\phi_{0}(x)=1$ so that (4.2) becomes

$$
y(\mathbf{x}, \mathbf{w})=\sum_{j=0}^{M-1} w_{j} \phi_{j}(\mathbf{x})=\mathbf{w}^{\mathrm{T}} \boldsymbol{\phi}(\mathbf{x})
$$
where $\mathbf{w}=\left(w_{0}, \ldots, w_{M-1}\right)^{\mathrm{T}}$ and $\boldsymbol{\phi}=\left(\phi_{0}, \ldots, \phi_{M-1}\right)^{\mathrm{T}}$. We can represent the model (4.3) using a neural network diagram, as shown in Figure 4.1.

By using nonlinear basis functions, we allow the function $y(\mathbf{x}, \mathbf{w})$ to be a nonlinear function of the input vector $\mathbf{x}$. Functions of the form (4.2) are called linear models, however, because they are linear in $\mathbf{w}$. It is this linearity in the parameters that will greatly simplify the analysis of this class of models. However, it also leads to some significant limitations.

Figure 4.1 The linear regression model (4.3) can be expressed as a simple neural network diagram involving a single layer of parameters. Here each basis function $\phi_{j}(\mathbf{x})$ is represented by an input node, with the solid node representing the 'bias' basis function $\phi_{0}$, and the function $y(\mathbf{x}, \mathbf{w})$ is represented by an output node. Each of the parameters $w_{j}$ is shown by a line connecting the corresponding basis function to the output.
![](https://cdn.mathpix.com/cropped/2025_10_01_d90a4d82c88904700833g-132.jpg?height=323&width=514&top_left_y=222&top_left_x=1123)

**Image Description:** The diagram illustrates a neural network architecture, depicting nodes and their connections. It features three layers labeled \(\phi_0(x)\), \(\phi_1(x)\), and \(\phi_{M-1}(x)\). Arrows indicate connections between the layers, with weights \(w_0\), \(w_1\), and \(w_{M-1}\) assigned to each arrow. The topmost node outputs \(y(x, w)\), indicating the final prediction based on input \(x\) and weights \(w\). The arrangement suggests a feedforward structure, where nodes in each layer are fully connected to the next layer.


Before the advent of deep learning it was common practice in machine learning to use some form of fixed pre-processing of the input variables $\mathbf{x}$, also known as feature extraction, expressed in terms of a set of basis functions $\left\{\phi_{j}(\mathbf{x})\right\}$. The goal was to choose a sufficiently powerful set of basis functions that the resulting learning task could be solved using a simple network model. Unfortunately, it is very difficult to hand-craft suitable basis functions for anything but the simplest applications. Deep learning avoids this problem by learning the required nonlinear transformations of the data from the data set itself.

We have already encountered an example of a regression problem when we dis- cussed curve fitting using polynomials. The polynomial function (1.1) can be expressed in the form (4.3) if we consider a single input variable $x$ and if we choose basis functions defined by $\phi_{j}(x)=x^{j}$. There are many other possible choices for the basis functions, for example

$$
\phi_{j}(x)=\exp \left\{-\frac{\left(x-\mu_{j}\right)^{2}}{2 s^{2}}\right\}
$$
where the $\mu_{j}$ govern the locations of the basis functions in input space, and the parameter $s$ governs their spatial scale. These are usually referred to as 'Gaussian' basis functions, although it should be noted that they are not required to have a probabilistic interpretation. In particular the normalization coefficient is unimportant because these basis functions will be multiplied by learnable parameters $w_{j}$.

Another possibility is the sigmoidal basis function of the form

$$
\phi_{j}(x)=\sigma\left(\frac{x-\mu_{j}}{s}\right)
$$
where $\sigma(a)$ is the logistic sigmoid function defined by
$$
\sigma(a)=\frac{1}{1+\exp (-a)} .
$$

Equivalently, we can use the tanh function because this is related to the logistic sigmoid by $\tanh (a)=2 \sigma(2 a)-1$, and so a general linear combination of logistic sigmoid functions is equivalent to a general linear combination of tanh functions in the sense that they can represent the same class of input-output functions. These various choices of basis function are illustrated in Figure 4.2.

![](https://cdn.mathpix.com/cropped/2025_10_01_d90a4d82c88904700833g-133.jpg?height=434&width=1471&top_left_y=241&top_left_x=144)

**Image Description:** The image consists of three adjacent plots. 

1. The first plot (left) features a two-dimensional graph of a function with the x-axis ranging from -1 to 1 and the y-axis from -1 to 1. It displays multiple curves, indicating different values or iterations of a parameter, with distinctive color coding.

2. The second plot (middle) presents a more complex waveform that oscillates between the y-axis limits, again color-coded.

3. The third plot (right) displays closely spaced oscillatory curves, indicating higher frequency or multiple phase shifts of the function. All axes are uniform across the plots, aiding comparative analysis.

Figure 4.2 Examples of basis functions, showing polynomials on the left, Gaussians of the form (4.4) in the centre, and sigmoidal basis functions of the form (4.5) on the right.

Yet another possible choice of basis function is the Fourier basis, which leads to an expansion in sinusoidal functions. Each basis function represents a specific frequency and has infinite spatial extent. By contrast, basis functions that are localized to finite regions of input space necessarily comprise a spectrum of different spatial frequencies. In signal processing applications, it is often of interest to consider basis functions that are localized in both space and frequency, leading to a class of functions known as wavelets (Ogden, 1997; Mallat, 1999; Vidakovic, 1999). These are also defined to be mutually orthogonal, to simplify their application. Wavelets are most applicable when the input values live on a regular lattice, such as the successive time points in a temporal sequence or the pixels in an image.

Most of the discussion in this chapter, however, is independent of the choice of basis function set, and so we will not specify the particular form of the basis functions, except for numerical illustration. Furthermore, to keep the notation simple, we will focus on the case of a single target variable $t$, although we will briefly outline

Section 4.1.7 the modifications needed to deal with multiple target variables.

### 4.1.2 Likelihood function

We solved the problem of fitting a polynomial function to data by minimizing a sum-of-squares error function, and we also showed that this error function could be motivated as the maximum likelihood solution under an assumed Gaussian noise model. We now return to this discussion and consider the least-squares approach, and its relation to maximum likelihood, in more detail.

As before, we assume that the target variable $t$ is given by a deterministic function $y(\mathbf{x}, \mathbf{w})$ with additive Gaussian noise so that

$$
t=y(\mathbf{x}, \mathbf{w})+\epsilon
$$
where $\epsilon$ is a zero-mean Gaussian random variable with variance $\sigma^{2}$. Thus, we can write
$$
p\left(t \mid \mathbf{x}, \mathbf{w}, \sigma^{2}\right)=\mathcal{N}\left(t \mid y(\mathbf{x}, \mathbf{w}), \sigma^{2}\right) .
$$

Now consider a data set of inputs $\mathbf{X}=\left\{\mathbf{x}_{1}, \ldots, \mathbf{x}_{N}\right\}$ with corresponding target values $t_{1}, \ldots, t_{N}$. We group the target variables \{ $t_{n}$ \} into a column vector that we denote by $\mathbf{t}$ where the typeface is chosen to distinguish it from a single observation of a multivariate target, which would be denoted $\mathbf{t}$. Making the assumption that these data points are drawn independently from the distribution (4.8), we obtain an expression for the likelihood function, which is a function of the adjustable parameters $\mathbf{w}$ and $\sigma^{2}$ :

$$
p\left(\mathbf{t} \mid \mathbf{X}, \mathbf{w}, \sigma^{2}\right)=\prod_{n=1}^{N} \mathcal{N}\left(t_{n} \mid \mathbf{w}^{\mathrm{T}} \boldsymbol{\phi}\left(\mathbf{x}_{n}\right), \sigma^{2}\right)
$$
where we have used (4.3). Taking the logarithm of the likelihood function and making use of the standard form (2.49) for the univariate Gaussian, we have
\$\$\begin{aligned}

\ln p\left(\mathbf{t} \mid \mathbf{X}, \mathbf{w}, \sigma^{2}\right) \& =\sum_{n=1}^{N} \ln \mathcal{N}\left(t_{n} \mid \mathbf{w}^{\mathrm{T}} \boldsymbol{\phi}\left(\mathbf{x}_{n}\right), \sigma^{2}\right) <br>
\& =-\frac{N}{2} \ln \sigma^{2}-\frac{N}{2} \ln (2 \pi)-\frac{1}{\sigma^{2}} E_{D}(\mathbf{w})

\end{aligned}$$
where the sum-of-squares error function is defined by
$$E_{D}(\mathbf{w})=\frac{1}{2} \sum_{n=1}^{N}\left\{t_{n}-\mathbf{w}^{\mathrm{T}} \boldsymbol{\phi}\left(\mathbf{x}_{n}\right)\right\}^{2} .\$\$

## Section 2.3.4

The first two terms in (4.10) can be treated as constants when determining $\mathbf{w}$ because they are independent of $\mathbf{w}$. Therefore, as we saw previously, maximizing the likelihood function under a Gaussian noise distribution is equivalent to minimizing the sum-of-squares error function (4.11).

### 4.1.3 Maximum likelihood

Having written down the likelihood function, we can use maximum likelihood to determine $\mathbf{w}$ and $\sigma^{2}$. Consider first the maximization with respect to $\mathbf{w}$. The gradient of the log likelihood function (4.10) with respect to $\mathbf{w}$ takes the form

$$
\nabla_{\mathbf{w}} \ln p\left(\mathbf{t} \mid \mathbf{X}, \mathbf{w}, \sigma^{2}\right)=\frac{1}{\sigma^{2}} \sum_{n=1}^{N}\left\{t_{n}-\mathbf{w}^{\mathrm{T}} \boldsymbol{\phi}\left(\mathbf{x}_{n}\right)\right\} \boldsymbol{\phi}\left(\mathbf{x}_{n}\right)^{\mathrm{T}} .
$$

Setting this gradient to zero gives

$$
0=\sum_{n=1}^{N} t_{n} \boldsymbol{\phi}\left(\mathbf{x}_{n}\right)^{\mathrm{T}}-\mathbf{w}^{\mathrm{T}}\left(\sum_{n=1}^{N} \boldsymbol{\phi}\left(\mathbf{x}_{n}\right) \boldsymbol{\phi}\left(\mathbf{x}_{n}\right)^{\mathrm{T}}\right) .
$$

Solving for $\mathbf{w}$ we obtain

$$
\mathbf{w}_{\mathrm{ML}}=\left(\boldsymbol{\Phi}^{\mathrm{T}} \boldsymbol{\Phi}\right)^{-1} \boldsymbol{\Phi}^{\mathrm{T}} \mathbf{t},
$$
which are known as the normal equations for the least-squares problem. Here $\boldsymbol{\Phi}$ is an $N \times M$ matrix, called the design matrix, whose elements are given by $\Phi_{n j}=\phi_{j}\left(\mathbf{x}_{n}\right)$, so that
\$\$\mathbf{\Phi}=\left(\begin{array}{cccc}

\phi_{0}\left(\mathbf{x}_{1}\right) \& \phi_{1}\left(\mathbf{x}_{1}\right) \& \cdots \& \phi_{M-1}\left(\mathbf{x}_{1}\right) <br>
\phi_{0}\left(\mathbf{x}_{2}\right) \& \phi_{1}\left(\mathbf{x}_{2}\right) \& \cdots \& \phi_{M-1}\left(\mathbf{x}_{2}\right) <br>
\vdots \& \vdots \& \ddots \& \vdots <br>
\phi_{0}\left(\mathbf{x}_{N}\right) \& \phi_{1}\left(\mathbf{x}_{N}\right) \& \cdots \& \phi_{M-1}\left(\mathbf{x}_{N}\right)

\end{array}\right)\$\$

The quantity

$$
\boldsymbol{\Phi}^{\dagger} \equiv\left(\boldsymbol{\Phi}^{\mathrm{T}} \boldsymbol{\Phi}\right)^{-1} \boldsymbol{\Phi}^{\mathrm{T}}
$$
is known as the Moore-Penrose pseudo-inverse of the matrix $\boldsymbol{\Phi}$ (Rao and Mitra, 1971; Golub and Van Loan, 1996). It can be regarded as a generalization of the notion of a matrix inverse to non-square matrices. Indeed, if $\boldsymbol{\Phi}$ is square and invertible, then using the property $(\mathbf{A B})^{-1}=\mathbf{B}^{-1} \mathbf{A}^{-1}$ we see that $\boldsymbol{\Phi}^{\dagger} \equiv \boldsymbol{\Phi}^{-1}$.

At this point, we can gain some insight into the role of the bias parameter $w_{0}$. If we make the bias parameter explicit, then the error function (4.11) becomes

$$
E_{D}(\mathbf{w})=\frac{1}{2} \sum_{n=1}^{N}\left\{t_{n}-w_{0}-\sum_{j=1}^{M-1} w_{j} \phi_{j}\left(\mathbf{x}_{n}\right)\right\}^{2}
$$

Setting the derivative with respect to $w_{0}$ equal to zero and solving for $w_{0}$, we obtain

$$
w_{0}=\bar{t}-\sum_{j=1}^{M-1} w_{j} \overline{\phi_{j}}
$$
where we have defined
$$
\bar{t}=\frac{1}{N} \sum_{n=1}^{N} t_{n}, \quad \overline{\phi_{j}}=\frac{1}{N} \sum_{n=1}^{N} \phi_{j}\left(\mathbf{x}_{n}\right)
$$

Thus, the bias $w_{0}$ compensates for the difference between the averages (over the training set) of the target values and the weighted sum of the averages of the basis function values.

We can also maximize the log likelihood function (4.10) with respect to the variance $\sigma^{2}$, giving

$$
\sigma_{\mathrm{ML}}^{2}=\frac{1}{N} \sum_{n=1}^{N}\left\{t_{n}-\mathbf{w}_{\mathrm{ML}}^{\mathrm{T}} \boldsymbol{\phi}\left(\mathbf{x}_{n}\right)\right\}^{2}
$$
and so we see that the maximum likelihood value of the variance parameter is given by the residual variance of the target values around the regression function.

Figure 4.3 Geometrical interpretation of the leastsquares solution in an $N$-dimensional space whose axes are the values of $t_{1}, \ldots, t_{N}$. The least-squares regression function is obtained by finding the orthogonal projection of the data vector $\mathbf{t}$ onto the subspace spanned by the basis functions $\phi_{j}(x)$ in which each basis function is viewed as a vector $\varphi_{j}$ of length $N$ with elements $\phi_{j}\left(\mathbf{x}_{n}\right)$.
![](https://cdn.mathpix.com/cropped/2025_10_01_d90a4d82c88904700833g-136.jpg?height=372&width=551&top_left_y=217&top_left_x=1103)

**Image Description:** The image depicts a geometric diagram illustrating a vector space. It features a plane labeled \( S \) outlined in red. Within this plane, two red arrows represent vectors \( \varphi_1 \) and \( \varphi_2 \), while a blue arrow indicates vector \( y \), and a black arrow denotes vector \( t \). The diagram visually emphasizes the relationships between these vectors within the defined plane, showcasing how \( y \) and \( t \) correlate to the vectors originating from the plane. The arrangement suggests a focus on vector analysis or linear transformations.


### 4.1.4 Geometry of least squares

At this point, it is instructive to consider the geometrical interpretation of the least-squares solution. To do this, we consider an $N$-dimensional space whose axes are given by the $t_{n}$, so that $\mathbf{t}=\left(t_{1}, \ldots, t_{N}\right)^{\mathrm{T}}$ is a vector in this space. Each basis function $\phi_{j}\left(\mathbf{x}_{n}\right)$, evaluated at the $N$ data points, can also be represented as a vector in the same space, denoted by $\varphi_{j}$, as illustrated in Figure 4.3. Note that $\varphi_{j}$ corresponds to the $j$ th column of $\boldsymbol{\Phi}$, whereas $\phi\left(\mathrm{x}_{n}\right)$ corresponds to the transpose of the $n$th row of $\boldsymbol{\Phi}$. If the number $M$ of basis functions is smaller than the number $N$ of data points, then the $M$ vectors $\phi_{j}\left(\mathbf{x}_{n}\right)$ will span a linear subspace $\mathcal{S}$ of dimensionality $M$. We define $\mathbf{y}$ to be an $N$-dimensional vector whose $n$th element is given by $y\left(\mathbf{x}_{n}, \mathbf{w}\right)$, where $n=1, \ldots, N$. Because $\mathbf{y}$ is an arbitrary linear combination of the vectors $\varphi_{j}$, it can live anywhere in the $M$-dimensional subspace. The sum-of-squares error (4.11) is then equal (up to a factor of $1 / 2$ ) to the squared Euclidean distance between $\mathbf{y}$ and $\mathbf{t}$. Thus, the least-squares solution for $\mathbf{w}$ corresponds to that choice of $\mathbf{y}$ that lies in subspace $\mathcal{S}$ and is closest to $\mathbf{t}$. Intuitively, from Figure 4.3, we anticipate that this solution corresponds to the orthogonal projection of $\mathbf{t}$ onto the subspace $\mathcal{S}$. This is indeed the case, as can easily be verified by noting that the solution for $\mathbf{y}$ is given by $\boldsymbol{\Phi} \mathbf{w}_{\mathrm{ML}}$ and then confirming that this takes the form of an orthogonal projection.

In practice, a direct solution of the normal equations can lead to numerical difficulties when $\boldsymbol{\Phi}^{\mathrm{T}} \boldsymbol{\Phi}$ is close to singular. In particular, when two or more of the basis vectors $\varphi_{j}$ are co-linear, or nearly so, the resulting parameter values can have large magnitudes. Such near degeneracies will not be uncommon when dealing with real data sets. The resulting numerical difficulties can be addressed using the technique of singular value decomposition, or SVD (Deisenroth, Faisal, and Ong, 2020). Note that the addition of a regularization term ensures that the matrix is non-singular, even in the presence of degeneracies.

### 4.1.5 Sequential learning

The maximum likelihood solution (4.14) involves processing the entire training set in one go and is known as a batch method. This can become computationally costly for large data sets. If the data set is sufficiently large, it may be worthwhile to use sequential algorithms, also known as online algorithms, in which the data points are considered one at a time and the model parameters updated after each such presentation. Sequential learning is also appropriate for real-time applications in which the data observations arrive in a continuous stream and predictions must be
made before all the data points are seen.
We can obtain a sequential learning algorithm by applying the technique of stochastic gradient descent, also known as sequential gradient descent, as follows. If the error function comprises a sum over data points $E=\sum_{n} E_{n}$, then after presentation of data point $n$, the stochastic gradient descent algorithm updates the parameter vector $\mathbf{w}$ using

$$
\mathbf{w}^{(\tau+1)}=\mathbf{w}^{(\tau)}-\eta \nabla E_{n}
$$
where $\tau$ denotes the iteration number, and $\eta$ is a suitably chosen learning rate parameter. The value of $\mathbf{w}$ is initialized to some starting vector $\mathbf{w}^{(0)}$. For the sum-ofsquares error function (4.11), this gives
$$
\mathbf{w}^{(\tau+1)}=\mathbf{w}^{(\tau)}+\eta\left(t_{n}-\mathbf{w}^{(\tau)^{\mathrm{T}}} \boldsymbol{\phi}_{n}\right) \boldsymbol{\phi}_{n}
$$

where $\phi_{n}=\phi\left(\mathrm{x}_{n}\right)$. This is known as the least-mean-squares or the LMS algorithm.

### 4.1.6 Regularized least squares

## Exercise 4.6

We have previously introduced the idea of adding a regularization term to an error function to control over-fitting, so that the total error function to be minimized takes the form

$$
E_{D}(\mathbf{w})+\lambda E_{W}(\mathbf{w})
$$
where $\lambda$ is the regularization coefficient that controls the relative importance of the data-dependent error $E_{D}(\mathbf{w})$ and the regularization term $E_{W}(\mathbf{w})$. One of the simplest forms of regularizer is given by the sum of the squares of the weight vector elements:
$$
E_{W}(\mathbf{w})=\frac{1}{2} \sum_{j} w_{j}^{2}=\frac{1}{2} \mathbf{w}^{\mathrm{T}} \mathbf{w}
$$

If we also consider the sum-of-squares error function given by

$$
E_{D}(\mathbf{w})=\frac{1}{2} \sum_{n=1}^{N}\left\{t_{n}-\mathbf{w}^{\mathrm{T}} \boldsymbol{\phi}\left(\mathbf{x}_{n}\right)\right\}^{2},
$$
then the total error function becomes
$$
\frac{1}{2} \sum_{n=1}^{N}\left\{t_{n}-\mathbf{w}^{\mathrm{T}} \boldsymbol{\phi}\left(\mathbf{x}_{n}\right)\right\}^{2}+\frac{\lambda}{2} \mathbf{w}^{\mathrm{T}} \mathbf{w} .
$$

In statistics, this regularizer provides an example of a parameter shrinkage method because it shrinks parameter values towards zero. It has the advantage that the error function remains a quadratic function of $\mathbf{w}$, and so its exact minimizer can be found in closed form. Specifically, setting the gradient of (4.26) with respect to w to zero and solving for $\mathbf{w}$ as before, we obtain

$$
\mathbf{w}=\left(\lambda \mathbf{I}+\boldsymbol{\Phi}^{\mathrm{T}} \boldsymbol{\Phi}\right)^{-1} \boldsymbol{\Phi}^{\mathrm{T}} \mathbf{t} .
$$

This represents a simple extension of the least-squares solution (4.14).

Figure 4.4 Representation of a linear regression model as a neural network having a single layer of connections. Each basis function is represented by a node, with the solid node representing the 'bias' basis function $\phi_{0}$. Likewise each output $y_{1}, \ldots, y_{K}$ is represented by a node. The links between the nodes represent the corresponding weight and bias parameters.
![](https://cdn.mathpix.com/cropped/2025_10_01_d90a4d82c88904700833g-138.jpg?height=323&width=641&top_left_y=222&top_left_x=998)

**Image Description:** The diagram presents a neural network architecture, specifically a feedforward structure with layers represented in nodes. The leftmost layer includes a dark blue node labeled \( \phi_0(x) \), while subsequent layers have light blue nodes, \( \phi_1(x) \), \( \phi_2(x) \), ..., up to \( \phi_{M-1}(x) \). Arrows indicate connections, with each light blue node connected to \( K \) output nodes \( y_k(x, w) \). The diagram illustrates information flow from input through multiple layers to the output, highlighting the feedforward mechanism of the neural network.


### 4.1.7 Multiple outputs

So far, we have considered situations with a single target variable $t$. In some applications, we may wish to predict $K>1$ target variables, which we denote collectively by the target vector $\mathbf{t}=\left(t_{1}, \ldots, t_{K}\right)^{\mathrm{T}}$. This could be done by introducing a different set of basis functions for each component of $\mathbf{t}$, leading to multiple, independent regression problems. However, a more common approach is to use the same set of basis functions to model all of the components the target vector so that

$$
\mathbf{y}(\mathbf{x}, \mathbf{w})=\mathbf{W}^{\mathrm{T}} \boldsymbol{\phi}(\mathbf{x})
$$
where $\mathbf{y}$ is a $K$-dimensional column vector, $\mathbf{W}$ is an $M \times K$ matrix of parameters, and $\phi(\mathbf{x})$ is an $M$-dimensional column vector with elements $\phi_{j}(\mathbf{x})$ with $\phi_{0}(\mathbf{x})=1$ as before. Again, this can be represented as a neural network having a single layer of parameters, as shown in Figure 4.4.

Suppose we take the conditional distribution of the target vector to be an isotropic Gaussian of the form

$$
p\left(\mathbf{t} \mid \mathbf{x}, \mathbf{W}, \sigma^{2}\right)=\mathcal{N}\left(\mathbf{t} \mid \mathbf{W}^{\mathrm{T}} \boldsymbol{\phi}(\mathbf{x}), \sigma^{2} \mathbf{I}\right) .
$$

If we have a set of observations $\mathbf{t}_{1}, \ldots, \mathbf{t}_{N}$, we can combine these into a matrix $\mathbf{T}$ of size $N \times K$ such that the $n$th row is given by $\mathbf{t}_{n}^{\mathrm{T}}$. Similarly, we can combine the input vectors $\mathbf{x}_{1}, \ldots, \mathbf{x}_{N}$ into a matrix $\mathbf{X}$. The log likelihood function is then given by

$$
\begin{aligned}
\ln p\left(\mathbf{T} \mid \mathbf{X}, \mathbf{W}, \sigma^{2}\right) & =\sum_{n=1}^{N} \ln \mathcal{N}\left(\mathbf{t}_{n} \mid \mathbf{W}^{\mathrm{T}} \boldsymbol{\phi}\left(\mathbf{x}_{n}\right), \sigma^{2} \mathbf{I}\right) \\
& =-\frac{N K}{2} \ln \left(2 \pi \sigma^{2}\right)-\frac{1}{2 \sigma^{2}} \sum_{n=1}^{N}\left\|\mathbf{t}_{n}-\mathbf{W}^{\mathrm{T}} \boldsymbol{\phi}\left(\mathbf{x}_{n}\right)\right\|^{2} .
\end{aligned}
$$

As before, we can maximize this function with respect to $\mathbf{W}$, giving

$$
\mathbf{W}_{\mathrm{ML}}=\left(\boldsymbol{\Phi}^{\mathrm{T}} \boldsymbol{\Phi}\right)^{-1} \boldsymbol{\Phi}^{\mathrm{T}} \mathbf{T}
$$
where we have combined the input feature vectors $\phi\left(\mathrm{x}_{1}\right), \ldots, \phi\left(\mathrm{x}_{N}\right)$ into a matrix $\boldsymbol{\Phi}$. If we examine this result for each target variable $t_{k}$, we have
$$
\mathbf{w}_{k}=\left(\boldsymbol{\Phi}^{\mathrm{T}} \boldsymbol{\Phi}\right)^{-1} \boldsymbol{\Phi}^{\mathrm{T}} \mathbf{t}_{k}=\boldsymbol{\Phi}^{\dagger} \mathbf{t}_{k}
$$

where $\mathbf{t}_{k}$ is an $N$-dimensional column vector with components $t_{n k}$ for $n=1, \ldots N$. Thus, the solution to the regression problem decouples between the different target variables, and we need compute only a single pseudo-inverse matrix $\boldsymbol{\Phi}^{\dagger}$, which is shared by all the vectors $\mathbf{w}_{k}$.

The extension to general Gaussian noise distributions having arbitrary covari-

Exercise 4.7

Section 3.2.7 ance matrices is straightforward. Again, this leads to a decoupling into $K$ independent regression problems. This result is unsurprising because the parameters $\mathbf{W}$ define only the mean of the Gaussian noise distribution, and we know that the maximum likelihood solution for the mean of a multivariate Gaussian is independent of the covariance. From now on, we will therefore consider a single target variable $t$ for simplicity.

### 4.2. Decision theory

We have formulated the regression task as one of modelling a conditional probability distribution $p(t \mid \mathbf{x})$, and we have chosen a specific form for the conditional probability, namely a Gaussian (4.8) with an $\mathbf{x}$-dependent mean $y(\mathbf{x}, \mathbf{w})$ governed by parameters $\mathbf{w}$ and with variance given by the parameter $\sigma^{2}$. Both $\mathbf{w}$ and $\sigma^{2}$ can be learned from data using maximum likelihood. The result is a predictive distribution given by

$$
p\left(t \mid \mathbf{x}, \mathbf{w}_{\mathrm{ML}}, \sigma_{\mathrm{ML}}^{2}\right)=\mathcal{N}\left(t \mid y\left(\mathbf{x}, \mathbf{w}_{\mathrm{ML}}\right), \sigma_{\mathrm{ML}}^{2}\right)
$$

The predictive distribution expresses our uncertainty over the value of $t$ for some new input x. However, for many practical applications we need to predict a specific value for $t$ rather than returning an entire distribution, particularly where we must take a specific action. For example, if our goal is to determine the optimal level of radiation to use for treating a tumour and our model predicts a probability distribution over radiation dose, then we must use that distribution to decide the specific dose to be administered. Our task therefore breaks down into two stages. In the first stage, called the inference stage, we use the training data to determine a predictive distribution $p(t \mid \mathbf{x})$. In the second stage, known as the decision stage, we use this predictive distribution to determine a specific value $f(\mathbf{x})$, which will be dependent on the input vector $\mathbf{x}$, that is optimal according to some criterion. We can do this by minimizing a loss function that depends on both the predictive distribution $p(t \mid \mathbf{x})$ and on $f$.

Intuitively we might choose the mean of the conditional distribution, so that we would use $f(\mathbf{x})=y\left(\mathbf{x}, \mathbf{w}_{\text {ML }}\right)$. In some cases this intuition will be correct, but in other situations it can give very poor results. It is therefore useful to formalize this so that we can understand when it applies and under what assumptions, and the framework for doing this is called decision theory.

Suppose that we choose a value $f(\mathrm{x})$ for our prediction when the true value is $t$. In doing so, we incur some form of penalty or cost. This is determined by a loss, which we denote $L(t, f(\mathbf{x}))$. Of course, we do not know the true value of $t$, so instead of minimizing $L$ itself, we minimize the average, or expected, loss which is

Figure 4.5 The regression function $f^{\star}(x)$, which minimizes the expected squared loss, is given by the mean of the conditional distribution $p(t \mid x)$.
![](https://cdn.mathpix.com/cropped/2025_10_01_d90a4d82c88904700833g-140.jpg?height=539&width=699&top_left_y=219&top_left_x=950)

**Image Description:** The diagram depicts a two-dimensional Cartesian plot with an x-axis labeled as \( x \) and a y-axis labeled as \( t \). There are two curves: a red curve representing \( f^*(x) \) and a blue curve indicating \( p(t|x_0, w, \sigma^2) \). The curves intersect at a vertical line, suggesting a significant point where the two functions are related. The diagram illustrates a relationship between the variables \( t \) and \( x \), potentially indicating a dynamic system or model in a statistical context.

given by

$$
\mathbb{E}[L]=\iint L(t, f(\mathbf{x})) p(\mathbf{x}, t) \mathrm{d} \mathbf{x} \mathrm{~d} t
$$
where we are averaging over the distribution of both input and target variables, weighted by their joint distribution $p(\mathbf{x}, t)$. A common choice of loss function in regression problems is the squared loss given by $L(t, f(\mathbf{x}))=\{f(\mathbf{x})-t\}^{2}$. In this case, the expected loss can be written
$$
\mathbb{E}[L]=\iint\{f(\mathbf{x})-t\}^{2} p(\mathbf{x}, t) \mathrm{d} \mathbf{x} \mathrm{~d} t
$$

It is important not to confuse the squared-loss function with the sum-of-squares error function introduced earlier. The error function is used to set the parameters during training in order to determine the conditional probability distribution $p(t \mid \mathbf{x})$, whereas the loss function governs how the conditional distribution is used to arrive at a predictive function $f(\mathbf{x})$ specifying a prediction for each value of $\mathbf{x}$.

Our goal is to choose $f(\mathbf{x})$ so as to minimize $\mathbb{E}[L]$. If we assume a completely flexible function $f(\mathbf{x})$, we can do this formally using the calculus of variations to give

$$
\frac{\delta \mathbb{E}[L]}{\delta f(\mathbf{x})}=2 \int\{f(\mathbf{x})-t\} p(\mathbf{x}, t) \mathrm{d} t=0
$$

Solving for $f(\mathbf{x})$ and using the sum and product rules of probability, we obtain

$$
f^{\star}(\mathbf{x})=\frac{1}{p(\mathbf{x})} \int t p(\mathbf{x}, t) \mathrm{d} t=\int t p(t \mid \mathbf{x}) \mathrm{d} t=\mathbb{E}_{t}[t \mid \mathbf{x}]
$$
which is the conditional average of $t$ conditioned on $\mathbf{x}$ and is known as the regression function. This result is illustrated in Figure 4.5. It can readily be extended to multiple target variables represented by the vector $\mathbf{t}$, in which case the optimal solution is the

Exercise 4.8 conditional average $\mathbf{f}^{\star}(\mathbf{x})=\mathbb{E}_{t}[\mathbf{t} \mid \mathbf{x}]$. For a Gaussian conditional distribution of the
form (4.8), the conditional mean will be simply

$$
\mathbb{E}[t \mid \mathbf{x}]=\int t p(t \mid \mathbf{x}) \mathrm{d} t=y(\mathbf{x}, \mathbf{w})
$$

The use of calculus of variations to derive (4.37) implies that we are optimizing over all possible functions $f(\mathbf{x})$. Although any parametric model that we can implement in practice is limited in the range of functions that it can represent, the framework of deep neural networks, discussed extensively in later chapters, provides a highly flexible class of functions that, for many practical purposes, can approximate any desired function to high accuracy.

We can derive this result in a slightly different way, which will also shed light on the nature of the regression problem. Armed with the knowledge that the optimal solution is the conditional expectation, we can expand the square term as follows

$$
\begin{aligned}
& \{f(\mathbf{x})-t\}^{2}=\{f(\mathbf{x})-\mathbb{E}[t \mid \mathbf{x}]+\mathbb{E}[t \mid \mathbf{x}]-t\}^{2} \\
& \quad=\{f(\mathbf{x})-\mathbb{E}[t \mid \mathbf{x}]\}^{2}+2\{f(\mathbf{x})-\mathbb{E}[t \mid \mathbf{x}]\}\{\mathbb{E}[t \mid \mathbf{x}]-t\}+\{\mathbb{E}[t \mid \mathbf{x}]-t\}^{2}
\end{aligned}
$$

where, to keep the notation uncluttered, we use $\mathbb{E}[t \mid \mathbf{x}]$ to denote $\mathbb{E}_{t}[t \mid \mathbf{x}]$. Substituting into the loss function (4.35) and performing the integral over $t$, we see that the crossterm vanishes and we obtain an expression for the loss function in the form

$$
\mathbb{E}[L]=\int\{f(\mathbf{x})-\mathbb{E}[t \mid \mathbf{x}]\}^{2} p(\mathbf{x}) \mathrm{d} \mathbf{x}+\int \operatorname{var}[t \mid \mathbf{x}] p(\mathbf{x}) \mathrm{d} \mathbf{x}
$$

The function $f(\mathbf{x})$ we seek to determine appears only in the first term, which will be minimized when $f(\mathbf{x})$ is equal to $\mathbb{E}[t \mid \mathbf{x}]$, in which case this term will vanish. This is simply the result that we derived previously, and shows that the optimal least-squares predictor is given by the conditional mean. The second term is the variance of the distribution of $t$, averaged over $\mathbf{x}$, and represents the intrinsic variability of the target data and can be regarded as noise. Because it is independent of $f(x)$, it represents the irreducible minimum value of the loss function.

The squared loss is not the only possible choice of loss function for regression. Here we consider briefly one simple generalization of the squared loss, called the Minkowski loss, whose expectation is given by

$$
\mathbb{E}\left[L_{q}\right]=\iint|f(\mathbf{x})-t|^{q} p(\mathbf{x}, t) \mathrm{d} \mathbf{x} \mathrm{~d} t
$$
which reduces to the expected squared loss for $q=2$. The function $|f-t|^{q}$ is plotted against $f-t$ for various values of $q$ in Figure 4.6. The minimum of $\mathbb{E}\left[L_{q}\right]$ is given by the conditional mean for $q=2$, the conditional median for $q=1$, and the conditional mode for $q \rightarrow 0$.

Note that the Gaussian noise assumption implies that the conditional distribution of $t$ given $\mathbf{x}$ is unimodal, which may be inappropriate for some applications. In this case a squared loss can lead to very poor results and we need to develop more sophisticated approaches. For example, we can extend this model by using mixtures

![](https://cdn.mathpix.com/cropped/2025_10_01_d90a4d82c88904700833g-142.jpg?height=1083&width=1506&top_left_y=217&top_left_x=148)

**Image Description:** The image presents four plots of the function \(|f - t|^q\) as a function of the variable \(f - t\). Each plot is labeled with different values of the parameter \(q\) (0.3, 1, 2, and 10). The x-axis represents \(f - t\), ranging from -2 to 2. The y-axis represents \(|f - t|^q\), with values extending up to 2. The curves exhibit different shapes based on the value of \(q\), showing a transition from a sharp V-shape to steeper parabolic forms as \(q\) increases.

Figure 4.6 Plots of the quantity $L_{q}=|f-t|^{q}$ for various values of $q$.

Section 6.5

Section 5.2
of Gaussians to give multimodal conditional distributions, which often arise in the solution of inverse problems. Our focus in this section has been on decision theory for regression problems, and in the next chapter we shall develop analogous concepts for classification tasks.

### 4.3. The Bias-Variance Trade-off

## Section 1.2

So far in our discussion of linear models for regression, we have assumed that the form and number of basis functions are both given. We have also seen that the use of maximum likelihood can lead to severe over-fitting if complex models are trained using data sets of limited size. However, limiting the number of basis functions to avoid over-fitting has the side effect of limiting the flexibility of the model to capture interesting and important trends in the data. Although a regularization term can control over-fitting for models with many parameters, this raises the question of how to determine a suitable value for the regularization coefficient $\lambda$. Seeking the
solution that minimizes the regularized error function with respect to both the weight vector $\mathbf{w}$ and the regularization coefficient $\lambda$ is clearly not the right approach, since this leads to the unregularized solution with $\lambda=0$.

It is instructive to consider a frequentist viewpoint of the model complexity issue, known as the bias-variance trade-off. Although we will introduce this concept in the context of linear basis function models, where it is easy to illustrate the ideas using simple examples, the discussion has very general applicability. Note, however, that over-fitting is really an unfortunate property of maximum likelihood and does not arise when we marginalize over parameters in a Bayesian setting (Bishop, 2006).

## Section 4.2

When we discussed decision theory for regression problems, we considered various loss functions, each of which leads to a corresponding optimal prediction once we are given the conditional distribution $p(t \mid \mathbf{x})$. A popular choice is the squared-loss function, for which the optimal prediction is given by the conditional expectation, which we denote by $h(\mathbf{x})$ and is given by

$$
h(\mathbf{x})=\mathbb{E}[t \mid \mathbf{x}]=\int t p(t \mid \mathbf{x}) \mathrm{d} t
$$

We have also seen that the expected squared loss can be written in the form

$$
\mathbb{E}[L]=\int\{f(\mathbf{x})-h(\mathbf{x})\}^{2} p(\mathbf{x}) \mathrm{d} \mathbf{x}+\iint\{h(\mathbf{x})-t\}^{2} p(\mathbf{x}, t) \mathrm{d} \mathbf{x} \mathrm{~d} t
$$

Recall that the second term, which is independent of $f(x)$, arises from the intrinsic noise on the data and represents the minimum achievable value of the expected loss. The first term depends on our choice for the function $f(\mathbf{x})$, and we will seek a solution for $f(\mathbf{x})$ that makes this term a minimum. Because it is non-negative, the smallest value that we can hope to achieve for this term is zero. If we had an unlimited supply of data (and unlimited computational resources), we could in principle find the regression function $h(x)$ to any desired degree of accuracy, and this would represent the optimal choice for $f(\mathbf{x})$. However, in practice we have a data set $\mathcal{D}$ containing only a finite number $N$ of data points, and consequently, we cannot know the regression function $h(\mathbf{x})$ exactly.

If we were to model $h(\mathbf{x})$ using a function governed by a parameter vector $\mathbf{w}$, then from a Bayesian perspective, the uncertainty in our model would be expressed through a posterior distribution over $\mathbf{w}$. A frequentist treatment, however, involves making a point estimate of $\mathbf{w}$ based on the data set $\mathcal{D}$ and tries instead to interpret the uncertainty of this estimate through the following thought experiment. Suppose we had a large number of data sets each of size $N$ and each drawn independently from the distribution $p(t, \mathbf{x})$. For any given data set $\mathcal{D}$, we can run our learning algorithm and obtain a prediction function $f(\mathbf{x} ; \mathcal{D})$. Different data sets from the ensemble will give different functions and consequently different values of the squared loss. The performance of a particular learning algorithm is then assessed by taking the average over this ensemble of data sets.

Consider the integrand of the first term in (4.42), which for a particular data set $\mathcal{D}$ takes the form

$$
\{f(\mathbf{x} ; \mathcal{D})-h(\mathbf{x})\}^{2}
$$

Because this quantity will be dependent on the particular data set $\mathcal{D}$, we take its average over the ensemble of data sets. If we add and subtract the quantity $\mathbb{E}_{\mathcal{D}}[f(\mathbf{x} ; \mathcal{D})]$ inside the braces, and then expand, we obtain

$$
\begin{aligned}
& \left\{f(\mathbf{x} ; \mathcal{D})-\mathbb{E}_{\mathcal{D}}[f(\mathbf{x} ; \mathcal{D})]+\mathbb{E}_{\mathcal{D}}[f(\mathbf{x} ; \mathcal{D})]-h(\mathbf{x})\right\}^{2} \\
& =\left\{f(\mathbf{x} ; \mathcal{D})-\mathbb{E}_{\mathcal{D}}[f(\mathbf{x} ; \mathcal{D})]\right\}^{2}+\left\{\mathbb{E}_{\mathcal{D}}[f(\mathbf{x} ; \mathcal{D})]-h(\mathbf{x})\right\}^{2} \\
& +2\left\{f(\mathbf{x} ; \mathcal{D})-\mathbb{E}_{\mathcal{D}}[f(\mathbf{x} ; \mathcal{D})]\right\}\left\{\mathbb{E}_{\mathcal{D}}[f(\mathbf{x} ; \mathcal{D})]-h(\mathbf{x})\right\}
\end{aligned}
$$

We now take the expectation of this expression with respect to $\mathcal{D}$ and note that the final term will vanish, giving

$$
\begin{aligned}
& \mathbb{E}_{\mathcal{D}}\left[\{f(\mathbf{x} ; \mathcal{D})-h(\mathbf{x})\}^{2}\right] \\
& =\underbrace{\left\{\mathbb{E}_{\mathcal{D}}[f(\mathbf{x} ; \mathcal{D})]-h(\mathbf{x})\right\}^{2}}_{(\text {bias })^{2}}+\underbrace{\mathbb{E}_{\mathcal{D}}\left[\left\{f(\mathbf{x} ; \mathcal{D})-\mathbb{E}_{\mathcal{D}}[f(\mathbf{x} ; \mathcal{D})]\right\}^{2}\right]}_{\text {variance }}
\end{aligned}
$$

We see that the expected squared difference between $f(\mathbf{x} ; \mathcal{D})$ and the regression function $h(\mathbf{x})$ can be expressed as the sum of two terms. The first term, called the squared bias, represents the extent to which the average prediction over all data sets differs from the desired regression function. The second term, called the variance, measures the extent to which the solutions for individual data sets vary around their average, and hence, this measures the extent to which the function $f(\mathbf{x} ; \mathcal{D})$ is sensitive to the particular choice of data set. We will provide some intuition to support these definitions shortly when we consider a simple example.

So far, we have considered a single input value $\mathbf{x}$. If we substitute this expansion back into (4.42), we obtain the following decomposition of the expected squared loss:

$$
\text { expected loss }=(\text { bias })^{2}+\text { variance }+ \text { noise }
$$
where
\$\$\begin{aligned}

(bias)^{2} \& =\int\left\{\mathbb{E}_{\mathcal{D}}[f(\mathbf{x} ; \mathcal{D})]-h(\mathbf{x})\right\}^{2} p(\mathbf{x}) \mathrm{d} \mathbf{x} <br>
variance \& =\int \mathbb{E}_{\mathcal{D}}\left[\left\{f(\mathbf{x} ; \mathcal{D})-\mathbb{E}_{\mathcal{D}}[f(\mathbf{x} ; \mathcal{D})]\right\}^{2}\right] p(\mathbf{x}) \mathrm{d} \mathbf{x} <br>
noise \& =\iint\{h(\mathbf{x})-t\}^{2} p(\mathbf{x}, t) \mathrm{d} \mathbf{x} \mathrm{~d} t

\end{aligned}\$\$
and the bias and variance terms now refer to integrated quantities.
Our goal is to minimize the expected loss, which we have decomposed into the sum of a (squared) bias, a variance, and a constant noise term. As we will see, there is a trade-off between bias and variance, with very flexible models having low bias and high variance, and relatively rigid models having high bias and low variance. The model with the optimal predictive capability is the one that leads to the best balance between bias and variance. This is illustrated by considering the sinusoidal data set introduced earlier. Here we independently generate 100 data sets, each containing
$N=25$ data points, from the sinusoidal curve $h(x)=\sin (2 \pi x)$. The data sets are indexed by $l=1, \ldots, L$, where $L=100$. For each data set $\mathcal{D}^{(l)}$, we fit a model with $M=24$ Gaussian basis functions along with a constant 'bias' basis function to give a total of 25 parameters. By minimizing the regularized error function (4.26), we obtain a prediction function $f^{(l)}(x)$, as shown in Figure 4.7.

The top row corresponds to a large value of the regularization coefficient $\lambda$ that gives low variance (because the red curves in the left plot look similar) but high bias (because the two curves in the right plot are very different). Conversely on the bottom row, for which $\lambda$ is small, there is large variance (shown by the high variability between the red curves in the left plot) but low bias (shown by the good fit between the average model fit and the original sinusoidal function). Note that the result of averaging many solutions for the complex model with $M=25$ is a very good fit to the regression function, which suggests that averaging may be a beneficial procedure. Indeed, a weighted averaging of multiple solutions lies at the heart of a Bayesian approach, although the averaging is with respect to the posterior distribution of parameters, not with respect to multiple data sets.

We can also examine the bias-variance trade-off quantitatively for this example. The average prediction is estimated from

$$
\bar{f}(x)=\frac{1}{L} \sum_{l=1}^{L} f^{(l)}(x)
$$
and the integrated squared bias and integrated variance are then given by
\$\$\begin{aligned}

(bias)^{2} \& =\frac{1}{N} \sum_{n=1}^{N}\left\{\bar{f}\left(x_{n}\right)-h\left(x_{n}\right)\right\}^{2} <br>
variance \& =\frac{1}{N} \sum_{n=1}^{N} \frac{1}{L} \sum_{l=1}^{L}\left\{f^{(l)}\left(x_{n}\right)-\bar{f}\left(x_{n}\right)\right\}^{2}

\end{aligned}\$\$
where the integral over $x$, weighted by the distribution $p(x)$, is approximated by a finite sum over data points drawn from that distribution. These quantities, along with their sum, are plotted as a function of $\ln \lambda$ in Figure 4.8. We see that small values of $\lambda$ allow the model to become finely tuned to the noise on each individual data set leading to large variance. Conversely, a large value of $\lambda$ pulls the weight parameters towards zero leading to large bias.

Note that the bias-variance decomposition is of limited practical value because it is based on averages with respect to ensembles of data sets, whereas in practice we have only the single observed data set. If we had a large number of independent training sets of a given size, we would be better off combining them into a single larger training set, which of course would reduce the level of over-fitting for a given model complexity. Nevertheless, the bias-variance decomposition often provides useful insights into the model complexity issue, and although we have introduced it in this chapter from the perspective of regression problems, the underlying intuition has broad applicability.

![](https://cdn.mathpix.com/cropped/2025_10_01_d90a4d82c88904700833g-146.jpg?height=1471&width=1506&top_left_y=315&top_left_x=148)

**Image Description:** The image consists of six subplots arranged in a 2x3 grid, each displaying curves representing solutions to a differential equation. The x-axis (labeled \(x\)) ranges from 0 to 1, while the y-axis (labeled \(t\)) ranges from -1 to 1. Each subplot is annotated with the value of \(\ln \lambda\): 3, 1, and -3 across the top and bottom rows. The curves vary in complexity, showing oscillatory behavior for different \(\ln \lambda\) values, with red and green lines illustrating distinct solution behaviors for selected parameters.

Figure 4.7 Illustration of the dependence of bias and variance on model complexity governed by a regularization parameter $\lambda$, using the sinusoidal data from Chapter 1. There are $L=100$ data sets, each having $N=25$ data points, and there are 24 Gaussian basis functions in the model so that the total number of parameters is $M=25$ including the bias parameter. The left column shows the result of fitting the model to the data sets for various values of $\ln \lambda$ (for clarity, only 20 of the 100 fits are shown). The right column shows the corresponding average of the 100 fits (red) along with the sinusoidal function from which the data sets were generated (green).

Figure 4.8 Plot of squared bias and variance, together with their sum, corresponding to the results shown in Figure 4.7. Also shown is the average test set error for a test data set size of 1,000 points. The minimum value of (bias) ${ }^{2}+$ variance occurs around $\ln \lambda=0.43$, which is close to the value that gives the minimum error on the test data.
![](https://cdn.mathpix.com/cropped/2025_10_01_d90a4d82c88904700833g-147.jpg?height=539&width=900&top_left_y=219&top_left_x=754)

**Image Description:** The image is a graph depicting bias, variance, and test error as functions of the logarithm of lambda (ln λ). The x-axis represents ln λ, ranging from -3 to 3. The y-axis indicates the values of bias² (red), variance (blue), and the sum of bias² and variance (green), alongside the test error (magenta). Each curve demonstrates the relationship between the model complexity parameter (λ) and the respective statistical components, highlighting their behavior relative to changes in λ. The legend in the right corner assigns colors to each curve for clarity.


## Exercises

$4.1(\star)$ Consider the sum-of-squares error function given by (1.2) in which the function $y(x, \mathbf{w})$ is given by the polynomial (1.1). Show that the coefficients $\mathbf{w}=\left\{w_{i}\right\}$ that minimize this error function are given by the solution to the following set of linear equations:

$$
\sum_{j=0}^{M} A_{i j} w_{j}=T_{i}
$$
where
$$
A_{i j}=\sum_{n=1}^{N}\left(x_{n}\right)^{i+j}, \quad T_{i}=\sum_{n=1}^{N}\left(x_{n}\right)^{i} t_{n} .
$$

Here a suffix $i$ or $j$ denotes the index of a component, whereas $(x)^{i}$ denotes $x$ raised to the power of $i$.
$4.2(\star)$ Write down the set of coupled linear equations, analogous to (4.53), satisfied by the coefficients $w_{i}$ that minimize the regularized sum-of-squares error function given by (1.4).
$4.3(\star)$ Show that the tanh function defined by

$$
\tanh (a)=\frac{e^{a}-e^{-a}}{e^{a}+e^{-a}}
$$
and the logistic sigmoid function defined by (4.6) are related by
$$
\tanh (a)=2 \sigma(2 a)-1 .
$$

Hence, show that a general linear combination of logistic sigmoid functions of the form

$$
y(x, \mathbf{w})=w_{0}+\sum_{j=1}^{M} w_{j} \sigma\left(\frac{x-\mu_{j}}{s}\right)
$$
is equivalent to a linear combination of tanh functions of the form
$$
y(x, \mathbf{u})=u_{0}+\sum_{j=1}^{M} u_{j} \tanh \left(\frac{x-\mu_{j}}{2 s}\right)
$$

and find expressions to relate the new parameters $\left\{u_{1}, \ldots, u_{M}\right\}$ to the original parameters $\left\{w_{1}, \ldots, w_{M}\right\}$.
$4.4(\star \star \star)$ Show that the matrix

$$
\boldsymbol{\Phi}\left(\boldsymbol{\Phi}^{\mathrm{T}} \boldsymbol{\Phi}\right)^{-1} \boldsymbol{\Phi}^{\mathrm{T}}
$$
takes any vector $\mathbf{v}$ and projects it onto the space spanned by the columns of $\boldsymbol{\Phi}$. Use this result to show that the least-squares solution (4.14) corresponds to an orthogonal projection of the vector $\mathbf{t}$ onto the manifold $\mathcal{S}$, as shown in Figure 4.3.
$4.5(\star)$ Consider a data set in which each data point $t_{n}$ is associated with a weighting factor $r_{n}>0$, so that the sum-of-squares error function becomes
$$
E_{D}(\mathbf{w})=\frac{1}{2} \sum_{n=1}^{N} r_{n}\left\{t_{n}-\mathbf{w}^{\mathrm{T}} \boldsymbol{\phi}\left(\mathbf{x}_{n}\right)\right\}^{2}
$$

Find an expression for the solution $\mathbf{w}^{\star}$ that minimizes this error function. Give two alternative interpretations of the weighted sum-of-squares error function in terms of (i) data-dependent noise variance and (ii) replicated data points.
4.6 ( $\star$ ) By setting the gradient of (4.26) with respect to $\mathbf{w}$ to zero, show that the exact minimum of the regularized sum-of-squares error function for linear regression is given by (4.27).
$4.7(\star \star)$ Consider a linear basis function regression model for a multivariate target variable $\mathbf{t}$ having a Gaussian distribution of the form

$$
p(\mathbf{t} \mid \mathbf{W}, \boldsymbol{\Sigma})=\mathcal{N}(\mathbf{t} \mid \mathbf{y}(\mathbf{x}, \mathbf{W}), \boldsymbol{\Sigma})
$$
where
$$
\mathbf{y}(\mathbf{x}, \mathbf{W})=\mathbf{W}^{\mathrm{T}} \boldsymbol{\phi}(\mathbf{x})
$$

together with a training data set comprising input basis vectors $\phi\left(\mathrm{x}_{n}\right)$ and corresponding target vectors $\mathbf{t}_{n}$, with $n=1, \ldots, N$. Show that the maximum likelihood solution $\mathbf{W}_{\text {ML }}$ for the parameter matrix $\mathbf{W}$ has the property that each column is given by an expression of the form (4.14), which was the solution for an isotropic noise distribution. Note that this is independent of the covariance matrix $\boldsymbol{\Sigma}$. Show that the maximum likelihood solution for $\boldsymbol{\Sigma}$ is given by

$$
\boldsymbol{\Sigma}=\frac{1}{N} \sum_{n=1}^{N}\left(\mathbf{t}_{n}-\mathbf{W}_{\mathrm{ML}}^{\mathrm{T}} \boldsymbol{\phi}\left(\mathbf{x}_{n}\right)\right)\left(\mathbf{t}_{n}-\mathbf{W}_{\mathrm{ML}}^{\mathrm{T}} \boldsymbol{\phi}\left(\mathbf{x}_{n}\right)\right)^{\mathrm{T}} .
$$
$4.8(\star)$ Consider the generalization of the squared-loss function (4.35) for a single target variable $t$ to multiple target variables described by the vector $\mathbf{t}$ given by
$$
\mathbb{E}[L(\mathbf{t}, \mathbf{f}(\mathbf{x}))]=\iint\|\mathbf{f}(\mathbf{x})-\mathbf{t}\|^{2} p(\mathbf{x}, \mathbf{t}) \mathrm{d} \mathbf{x} \mathrm{~d} \mathbf{t}
$$

Using the calculus of variations, show that the function $\mathbf{f}(\mathbf{x})$ for which this expected loss is minimized is given by

$$
\mathbf{f}(\mathbf{x})=\mathbb{E}_{t}[\mathbf{t} \mid \mathbf{x}]
$$
4.9 (*) By expansion of the square in (4.64), derive a result analogous to (4.39) and, hence, show that the function $\mathbf{f}(\mathbf{x})$ that minimizes the expected squared loss for a vector $\mathbf{t}$ of target variables is again given by the conditional expectation of $\mathbf{t}$ in the form (4.65).
$4.10(\star \star)$ Rederive the result (4.65) by first expanding (4.64) analogous to (4.39).
4.11 ( $* \star$ ) The following distribution
$$
p\left(x \mid \sigma^{2}, q\right)=\frac{q}{2\left(2 \sigma^{2}\right)^{1 / q} \Gamma(1 / q)} \exp \left(-\frac{|x|^{q}}{2 \sigma^{2}}\right)
$$

is a generalization of the univariate Gaussian distribution. Here $\Gamma(x)$ is the gamma function defined by

$$
\Gamma(x)=\int_{-\infty}^{\infty} u^{x-1} e^{-u} \mathrm{~d} u
$$

Show that this distribution is normalized so that

$$
\int_{-\infty}^{\infty} p\left(x \mid \sigma^{2}, q\right) \mathrm{d} x=1
$$
and that it reduces to the Gaussian when $q=2$. Consider a regression model in which the target variable is given by $t=y(\mathbf{x}, \mathbf{w})+\epsilon$ and $\epsilon$ is a random noise variable drawn from the distribution (4.66). Show that the log likelihood function over $\mathbf{w}$ and $\sigma^{2}$, for an observed data set of input vectors $\mathbf{X}=\left\{\mathbf{x}_{1}, \ldots, \mathbf{x}_{N}\right\}$ and corresponding target variables $\mathbf{t}=\left(t_{1}, \ldots, t_{N}\right)^{\mathrm{T}}$, is given by
$$
\ln p\left(\mathbf{t} \mid \mathbf{X}, \mathbf{w}, \sigma^{2}\right)=-\frac{1}{2 \sigma^{2}} \sum_{n=1}^{N}\left|y\left(\mathbf{x}_{n}, \mathbf{w}\right)-t_{n}\right|^{q}-\frac{N}{q} \ln \left(2 \sigma^{2}\right)+\mathrm{const}
$$

where 'const' denotes terms independent of both $\mathbf{w}$ and $\sigma^{2}$. Note that, as a function of $\mathbf{w}$, this is the $L_{q}$ error function considered in Section 4.2.
$4.12(\star \star)$ Consider the expected loss for regression problems under the $L_{q}$ loss function given by (4.40). Write down the condition that $y(\mathbf{x})$ must satisfy to minimize $\mathbb{E}\left[L_{q}\right]$. Show that, for $q=1$, this solution represents the conditional median, i.e., the function $y(\mathbf{x})$ such that the probability mass for $t<y(\mathbf{x})$ is the same as for $t \geqslant y(\mathbf{x})$. Also show that the minimum expected $L_{q}$ loss for $q \rightarrow 0$ is given by the conditional mode, i.e., by the function $y(\mathbf{x})$ being equal to the value of $t$ that maximizes $p(t \mid \mathbf{x})$ for each $\mathbf{x}$.
![](https://cdn.mathpix.com/cropped/2025_10_01_d90a4d82c88904700833g-150.jpg?height=1244&width=1228&top_left_y=224&top_left_x=424)

**Image Description:** This image is a lecture slide titled "5 Single-layer Networks: Classification." The background appears abstract with soft, colorful gradients, likely to enhance visual engagement rather than convey data. There are no diagrams, equations, or graphs present; it primarily serves as a visual title card for the lecture content on single-layer networks and their classification.


In the previous chapter, we explored a class of regression models in which the output variables were linear functions of the model parameters and which can therefore be expressed as simple neural networks having a single layer of weight and bias parameters. We turn now to a discussion of classification problems, and in this chapter, we will focus on an analogous class of models that again can be expressed as single-layer neural networks. These will allow us to introduce many of the key concepts of classification before dealing with more general deep neural networks in later chapters.

The goal in classification is to take an input vector $\mathbf{x} \in \mathbb{R}^{D}$ and assign it to one of $K$ discrete classes $\mathcal{C}_{k}$ where $k=1, \ldots, K$. In the most common scenario, the classes are taken to be disjoint, so that each input is assigned to one and only one class. The input space is thereby divided into decision regions whose boundaries are called decision boundaries or decision surfaces. In this chapter, we consider linear
models for classification, by which we mean that the decision surfaces are linear functions of the input vector $\mathbf{x}$ and, hence, are defined by ( $D-1$ )-dimensional hyperplanes within the $D$-dimensional input space. Data sets whose classes can be separated exactly by linear decision surfaces are said to be linearly separable. Linear classification models can be applied to data sets that are not linearly separable, although not all inputs will be correctly classified.

We can broadly identify three distinct approaches to solving classification problems. The simplest involves constructing a discriminant function that directly assigns each vector $\mathbf{x}$ to a specific class. A more powerful approach, however, models the conditional probability distributions $p\left(\mathcal{C}_{k} \mid \mathbf{x}\right)$ in an inference stage and subsequently uses these distributions to make optimal decisions. Separating inference and deci-
Section 5.2.4 sion brings numerous benefits. There are two different approaches to determining the conditional probabilities $p\left(\mathcal{C}_{k} \mid \mathbf{x}\right)$. One technique is to model them directly, for example by representing them as parametric models and then optimizing the parameters using a training set. This will be called a discriminative probabilistic model. Alternatively, we can model the class-conditional densities $p\left(\mathbf{x} \mid \mathcal{C}_{k}\right)$, together with the prior probabilities $p\left(\mathcal{C}_{k}\right)$ for the classes, and then compute the required posterior probabilities using Bayes' theorem:

$$
p\left(\mathcal{C}_{k} \mid \mathbf{x}\right)=\frac{p\left(\mathbf{x} \mid \mathcal{C}_{k}\right) p\left(\mathcal{C}_{k}\right)}{p(\mathbf{x})}
$$

This will be called a generative probabilistic model because it offers the opportunity to generate samples from each of the class-conditional densities $p\left(\mathbf{x} \mid \mathcal{C}_{k}\right)$. In this chapter, we will discuss examples of all three approaches: discriminant functions, generative probabilistic models, and discriminative probabilistic models.

### 5.1. Discriminant Functions

A discriminant is a function that takes an input vector $\mathbf{x}$ and assigns it to one of $K$ classes, denoted $\mathcal{C}_{k}$. In this chapter, we will restrict attention to linear discriminants, namely those for which the decision surfaces are hyperplanes. To simplify the discussion, we consider first two classes and then investigate the extension to $K>2$ classes.

### 5.1.1 Two classes

The simplest representation of a linear discriminant function is obtained by taking a linear function of the input vector so that

$$
y(\mathbf{x})=\mathbf{w}^{\mathrm{T}} \mathbf{x}+w_{0}
$$
where $\mathbf{w}$ is called a weight vector, and $w_{0}$ is a bias (not to be confused with bias in the statistical sense). An input vector $\mathbf{x}$ is assigned to class $\mathcal{C}_{1}$ if $y(\mathbf{x}) \geqslant 0$ and to class $\mathcal{C}_{2}$ otherwise. The corresponding decision boundary is therefore defined by the relation $y(\mathbf{x})=0$, which corresponds to a ( $D-1$ )-dimensional hyperplane within

Figure 5.1 Illustration of the geometry of a linear discriminant function in two dimensions. The decision surface, shown in red, is perpendicular to $\mathbf{w}$, and its displacement from the origin is controlled by the bias parameter $w_{0}$. Also, the signed orthogonal distance of a general point $x$ from the decision surface is given by $y(\mathbf{x}) /\|\mathbf{w}\|$.
![](https://cdn.mathpix.com/cropped/2025_10_01_d90a4d82c88904700833g-152.jpg?height=709&width=900&top_left_y=217&top_left_x=764)

**Image Description:** The image depicts a two-dimensional diagram with axes labeled \(x_1\) and \(x_2\). A red line illustrates a boundary defined by the equation \(y = 0\), separating the regions \(R_1\) (where \(y > 0\)) and \(R_2\) (where \(y < 0\)). A green vector \(w\) is displayed, indicating direction. A blue vector represents the variable \(x\), with a dashed line indicating the projection \(x_\perp\). The equation \(-\frac{w_0}{\|w\|}\) aligns vertically. The diagram effectively visualizes the relationship between the vectors and the defined regions.


the $D$-dimensional input space. Consider two points $\mathbf{x}_{\mathrm{A}}$ and $\mathbf{x}_{\mathrm{B}}$ both of which lie on the decision surface. Because $y\left(\mathbf{x}_{\mathrm{A}}\right)=y\left(\mathbf{x}_{\mathrm{B}}\right)=0$, we have $\mathbf{w}^{\mathrm{T}}\left(\mathbf{x}_{\mathrm{A}}-\mathbf{x}_{\mathrm{B}}\right)=0$ and hence the vector $\mathbf{w}$ is orthogonal to every vector lying within the decision surface, and so $\mathbf{w}$ determines the orientation of the decision surface. Similarly, if $\mathbf{x}$ is a point on the decision surface, then $y(x)=0$, and so the normal distance from the origin to the decision surface is given by

$$
\frac{\mathbf{w}^{\mathrm{T}} \mathbf{x}}{\|\mathbf{w}\|}=-\frac{w_{0}}{\|\mathbf{w}\|}
$$

We therefore see that the bias parameter $w_{0}$ determines the location of the decision surface. These properties are illustrated for the case of $D=2$ in Figure 5.1.

Furthermore, note that the value of $y(x)$ gives a signed measure of the perpendicular distance $r$ of the point $\mathbf{x}$ from the decision surface. To see this, consider an arbitrary point $\mathbf{x}$ and let $\mathbf{x}_{\perp}$ be its orthogonal projection onto the decision surface, so that

$$
\mathbf{x}=\mathbf{x}_{\perp}+r \frac{\mathbf{w}}{\|\mathbf{w}\|}
$$

Multiplying both sides of this result by $\mathbf{w}^{\mathrm{T}}$ and adding $w_{0}$, and making use of $y(\mathbf{x})= \mathbf{w}^{\mathrm{T}} \mathbf{x}+w_{0}$ and $y\left(\mathbf{x}_{\perp}\right)=\mathbf{w}^{\mathrm{T}} \mathbf{x}_{\perp}+w_{0}=0$, we have

$$
r=\frac{y(\mathbf{x})}{\|\mathbf{w}\|}
$$

This result is illustrated in Figure 5.1.
Section 4.1.1
As with linear regression models, it is sometimes convenient to use a more compact notation in which we introduce an additional dummy 'input' value $x_{0}=1$ and then define $\widetilde{\mathbf{w}}=\left(w_{0}, \mathbf{w}\right)$ and $\widetilde{\mathbf{x}}=\left(x_{0}, \mathbf{x}\right)$ so that

$$
y(\mathbf{x})=\widetilde{\mathbf{w}}^{\mathrm{T}} \widetilde{\mathbf{x}}
$$

![](https://cdn.mathpix.com/cropped/2025_10_01_d90a4d82c88904700833g-153.jpg?height=671&width=1462&top_left_y=214&top_left_x=146)

**Image Description:** The image consists of two geometric diagrams illustrating the intersection of lines and regions defined by them. In the left diagram, two red straight lines (R1 and R2) intersect to form a green shaded triangle, with lines labeled C1 and C2. The right diagram displays three lines (R1, R2, and R3) with potential shading indicated by a question mark, line C1 intersects R1 and R3 but is not identical to either. The relationships of the lines provide context for understanding geometric angles and regions. Both diagrams explore concepts in geometry related to intersections.

Figure 5.2 Attempting to construct a $K$-class discriminant from a set of two-class discriminant functions leads to ambiguous regions, as shown in green. On the left is an example with two discriminant functions designed to distinguish points in class $\mathcal{C}_{k}$ from points not in class $\mathcal{C}_{k}$. On the right is an example involving three discriminant functions each of which is used to separate a pair of classes $\mathcal{C}_{k}$ and $\mathcal{C}_{j}$.

In this case, the decision surfaces are $D$-dimensional hyperplanes passing through the origin of the ( $D+1$ )-dimensional expanded input space.

### 5.1.2 Multiple classes

Now consider the extension of linear discriminant functions to $K>2$ classes. We might be tempted to build a $K$-class discriminant by combining a number of two-class discriminant functions. However, this leads to some serious difficulties (Duda and Hart, 1973), as we now show.

Consider a model with $K-1$ classifiers, each of which solves a two-class problem of separating points in a particular class $\mathcal{C}_{k}$ from points not in that class. This is known as a one-versus-the-rest classifier. The left-hand example in Figure 5.2 shows an example involving three classes where this approach leads to regions of input space that are ambiguously classified.

An alternative is to introduce $K(K-1) / 2$ binary discriminant functions, one for every possible pair of classes. This is known as a one-versus-one classifier. Each point is then classified according to a majority vote amongst the discriminant functions. However, this too runs into the problem of ambiguous regions, as illustrated in the right-hand diagram of Figure 5.2.

We can avoid these difficulties by considering a single $K$-class discriminant comprising $K$ linear functions of the form

$$
y_{k}(\mathbf{x})=\mathbf{w}_{k}^{\mathrm{T}} \mathbf{x}+w_{k 0}
$$
and then assigning a point $\mathbf{x}$ to class $\mathcal{C}_{k}$ if $y_{k}(\mathbf{x})>y_{j}(\mathbf{x})$ for all $j \neq k$. The decision boundary between class $\mathcal{C}_{k}$ and class $\mathcal{C}_{j}$ is therefore given by $y_{k}(\mathbf{x})=y_{j}(\mathbf{x})$ and

Figure 5.3 Illustration of the decision regions for a multi-class linear discriminant, with the decision boundaries shown in red. If two points $\mathrm{x}_{\mathrm{A}}$ and $\mathrm{x}_{\mathrm{B}}$ both lie inside the same decision region $\mathcal{R}_{k}$, then any point $\widehat{\mathbf{x}}$ that lies on the line connecting these two points must also lie in $\mathcal{R}_{k}$, and hence, the decision region must be singly connected and convex.
![](https://cdn.mathpix.com/cropped/2025_10_01_d90a4d82c88904700833g-154.jpg?height=414&width=587&top_left_y=219&top_left_x=1062)

**Image Description:** The diagram illustrates a geometric representation involving three vectors denoted as \( R_i \), \( R_j \), and \( R_k \) in red, converging at a point. It displays a horizontal blue line segment between \( X_A \) and \( X_B \), with \( \hat{X} \) positioned at its midpoint. The axes are implied rather than explicitly labeled, focusing on the spatial relationship and directional vectors emerging from the point of convergence. This visual aids in understanding vector relationships in a multi-dimensional space.


hence corresponds to a ( $D-1$ )-dimensional hyperplane defined by

$$
\left(\mathbf{w}_{k}-\mathbf{w}_{j}\right)^{\mathrm{T}} \mathbf{x}+\left(w_{k 0}-w_{j 0}\right)=0 .
$$

This has the same form as the decision boundary for the two-class case discussed in Section 5.1.1, and so analogous geometrical properties apply.

The decision regions of such a discriminant are always singly connected and convex. To see this, consider two points $\mathbf{x}_{\mathrm{A}}$ and $\mathbf{x}_{\mathrm{B}}$ both of which lie inside decision region $\mathcal{R}_{k}$, as illustrated in Figure 5.3. Any point $\widehat{\mathbf{x}}$ that lies on the line connecting $\mathrm{x}_{\mathrm{A}}$ and $\mathrm{x}_{\mathrm{B}}$ can be expressed in the form

$$
\widehat{\mathbf{x}}=\lambda \mathbf{x}_{\mathrm{A}}+(1-\lambda) \mathbf{x}_{\mathrm{B}}
$$
where $0 \leqslant \lambda \leqslant 1$. From the linearity of the discriminant functions, it follows that
$$
y_{k}(\widehat{\mathbf{x}})=\lambda y_{k}\left(\mathbf{x}_{\mathrm{A}}\right)+(1-\lambda) y_{k}\left(\mathbf{x}_{\mathrm{B}}\right) .
$$

Because both $\mathrm{x}_{\mathrm{A}}$ and $\mathrm{x}_{\mathrm{B}}$ lie inside $\mathcal{R}_{k}$, it follows that $y_{k}\left(\mathrm{x}_{\mathrm{A}}\right)>y_{j}\left(\mathrm{x}_{\mathrm{A}}\right)$ and that $y_{k}\left(\mathbf{x}_{\mathrm{B}}\right)>y_{j}\left(\mathbf{x}_{\mathrm{B}}\right)$, for all $j \neq k$, and hence $y_{k}(\widehat{\mathbf{x}})>y_{j}(\widehat{\mathbf{x}})$, and so $\widehat{\mathbf{x}}$ also lies inside $\mathcal{R}_{k}$. Thus, $\mathcal{R}_{k}$ is singly connected and convex.

Note that for two classes, we can either employ the formalism discussed here, based on two discriminant functions $y_{1}(\mathbf{x})$ and $y_{2}(\mathbf{x})$, or else use the simpler but
Section 5.1.1 essentially equivalent formulation based on a single discriminant function $y(\mathbf{x})$.

### 5.1.3 1-of- $K$ coding

For regression problems, the target variable $\mathbf{t}$ was simply the vector of real numbers whose values we wish to predict. In classification, there are various ways of using target values to represent class labels. For two-class problems, the most convenient is the binary representation in which there is a single target variable $t \in\{0,1\}$ such that $t=1$ represents class $\mathcal{C}_{1}$ and $t=0$ represents class $\mathcal{C}_{2}$. We can interpret the value of $t$ as the probability that the class is $\mathcal{C}_{1}$, with the probability values taking only the extreme values of 0 and 1 . For $K>2$ classes, it is convenient to use a 1 -of- $K$ coding scheme, also known as the one-hot encoding scheme, in which $\mathbf{t}$ is a vector of length $K$ such that if the class is $\mathcal{C}_{j}$, then all elements $t_{k}$ of $\mathbf{t}$ are zero
except element $t_{j}$, which takes the value 1 . For instance, if we have $K=5$ classes, then a data point from class 2 would be given the target vector

$$
\mathbf{t}=(0,1,0,0,0)^{\mathrm{T}}
$$

Again, we can interpret the value of $t_{k}$ as the probability that the class is $\mathcal{C}_{k}$ in which the probabilities take only the values 0 and 1 .

### 5.1.4 Least squares for classification

With linear regression models, the minimization of a sum-of-squares error func-

Section 4.1.3

## Exercise 5.1

Section 4.1.3 tion leads to a simple closed-form solution for the parameter values. It is therefore tempting to see if we can apply the same least-squares formalism to classification problems. Consider a general classification problem with $K$ classes and a 1 -of- $K$ binary coding scheme for the target vector $\mathbf{t}$. One justification for using least squares in such a context is that it approximates the conditional expectation $\mathbb{E}[\mathbf{t} \mid \mathbf{x}]$ of the target values given the input vector. For a binary coding scheme, this conditional expectation is given by the vector of posterior class probabilities. Unfortunately, these probabilities are typically approximated rather poorly, and indeed the approximations can have values outside the range $(0,1)$. However, it is instructional to explore these simple models and to understand how these limitations arise.

Each class $\mathcal{C}_{k}$ is described by its own linear model so that

$$
y_{k}(\mathbf{x})=\mathbf{w}_{k}^{\mathrm{T}} \mathbf{x}+w_{k 0}
$$
where $k=1, \ldots, K$. We can conveniently group these together using vector notation so that
$$
\mathbf{y}(\mathbf{x})=\widetilde{\mathbf{W}}^{\mathrm{T}} \widetilde{\mathbf{x}}
$$

where $\widetilde{\mathbf{W}}$ is a matrix whose $k$ th column comprises the ( $D+1$ )-dimensional vector $\widetilde{\mathbf{w}}_{k}=\left(w_{k 0}, \mathbf{w}_{k}^{\mathrm{T}}\right)^{\mathrm{T}}$ and $\widetilde{\mathbf{x}}$ is the corresponding augmented input vector $\left(1, \mathbf{x}^{\mathrm{T}}\right)^{\mathrm{T}}$ with a dummy input $x_{0}=1$. A new input $\mathbf{x}$ is then assigned to the class for which the output $y_{k}=\widetilde{\mathbf{w}}_{k}^{\mathrm{T}} \widetilde{\mathbf{x}}$ is largest.

We now determine the parameter matrix $\widetilde{\mathbf{W}}$ by minimizing a sum-of-squares error function. Consider a training data set $\left\{\mathbf{x}_{n}, \mathbf{t}_{n}\right\}$ where $n=1, \ldots, N$, and define a matrix $\mathbf{T}$ whose $n$th row is the vector $\mathbf{t}_{n}^{T}$, together with a matrix $\mathbf{X}$ whose $n$th row is $\widetilde{\mathbf{x}}_{n}^{\mathrm{T}}$. The sum-of-squares error function can then be written as

$$
E_{D}(\widetilde{\mathbf{W}})=\frac{1}{2} \operatorname{Tr}\left\{(\widetilde{\mathbf{X}} \widetilde{\mathbf{W}}-\mathbf{T})^{\mathrm{T}}(\widetilde{\mathbf{X}} \widetilde{\mathbf{W}}-\mathbf{T})\right\} .
$$

Setting the derivative with respect to $\widetilde{\mathbf{W}}$ to zero and rearranging, we obtain the solution for $\widetilde{\mathbf{W}}$ in the form

$$
\widetilde{\mathbf{W}}=\left(\widetilde{\mathbf{X}}^{\mathrm{T}} \widetilde{\mathbf{X}}\right)^{-1} \widetilde{\mathbf{X}}^{\mathrm{T}} \mathbf{T}=\widetilde{\mathbf{X}}^{\dagger} \mathbf{T}
$$
where $\widetilde{\mathbf{X}}^{\dagger}$ is the pseudo-inverse of the matrix $\widetilde{\mathbf{X}}$. We then obtain the discriminant
function in the form
$$
\mathbf{y}(\mathbf{x})=\widetilde{\mathbf{W}}^{\mathrm{T}} \widetilde{\mathbf{x}}=\mathbf{T}^{\mathrm{T}}\left(\widetilde{\mathbf{X}}^{\dagger}\right)^{\mathrm{T}} \widetilde{\mathbf{x}}
$$

An interesting property of least-squares solutions with multiple target variables is that if every target vector in the training set satisfies some linear constraint

$$
\mathbf{a}^{\mathrm{T}} \mathbf{t}_{n}+b=0
$$
for some constants $\mathbf{a}$ and $b$, then the model prediction for any value of $\mathbf{x}$ will satisfy the same constraint so that
$$
\mathbf{a}^{\mathrm{T}} \mathbf{y}(\mathbf{x})+b=0
$$

Thus, if we use a 1 -of- $K$ coding scheme for $K$ classes, then the predictions made by the model will have the property that the elements of $\mathbf{y}(\mathbf{x})$ will sum to 1 for any value of $\mathbf{x}$. However, this summation constraint alone is not sufficient to allow the model outputs to be interpreted as probabilities because they are not constrained to lie within the interval $(0,1)$.

The least-squares approach gives an exact closed-form solution for the discriminant function parameters. However, even as a discriminant function (where we use it to make decisions directly and dispense with any probabilistic interpretation), it suffers from some severe problems. We have seen that the sum-of-squares error function can be viewed as the negative log likelihood under the assumption of a
Section 2.3.4 Gaussian noise distribution. If the true distribution of the data is markedly different from being Gaussian, then least squares can give poor results. In particular, least squares is very sensitive to the presence of outliers, which are data points located a long way from the bulk of the data. This is illustrated in Figure 5.4. Here we see that the additional data points in the right-hand figure produce a significant change in the location of the decision boundary, even though these points would be correctly classified by the original decision boundary in the left-hand figure. The sum-of-squares error function gives too much weight to data points that are a long way from the decision boundary, even though they are correctly classified. Outliers can arise due to rare events or may simply be due to mistakes in the data set. Techniques that are sensitive to a very few data points are said to lack robustness. For comparison, Figure 5.4 also shows results from a technique called logistic regression, which is more robust to outliers.

The failure of least squares should not surprise us when we recall that it corresponds to maximum likelihood under the assumption of a Gaussian conditional distribution, whereas binary target vectors clearly have a distribution that is far from Gaussian. By adopting more appropriate probabilistic models, we can obtain classification techniques with much better properties than least squares, and which can also be generalized to give flexible nonlinear neural network models, as we will see in later chapters.

![](https://cdn.mathpix.com/cropped/2025_10_01_d90a4d82c88904700833g-157.jpg?height=709&width=1471&top_left_y=239&top_left_x=146)

**Image Description:** The image consists of two scatter plot diagrams, each with a different linear decision boundary. The left plot shows red 'x' markers representing one class and blue circles representing another. The green line denotes the decision boundary separating the two classes. The right plot features a different boundary represented by a purple line, showing a potentially more complex division between the classes. Both plots utilize a Cartesian coordinate system with the horizontal axis ranging from -4 to 8 and the vertical axis from -4 to 4, displaying data points for classification analysis.

Figure 5.4 The left-hand plot shows data from two classes, denoted by red crosses and blue circles, together with the decision boundaries found by least squares (magenta curve) and by a logistic regression model (green curve). The right-hand plot shows the corresponding results obtained when extra data points are added at the bottom right of the diagram, showing that least squares is highly sensitive to outliers, unlike logistic regression.

### 5.2. Decision Theory

When we discussed linear regression we saw how the process of making predictions in machine learning can be broken down into the two stages of inference and decision. We now explore this perspective in much greater depth specifically in the context of classifiers.

Suppose we have an input vector $\mathbf{x}$ together with a corresponding vector $\mathbf{t}$ of target variables, and our goal is to predict $\mathbf{t}$ given a new value for $\mathbf{x}$. For regression problems, $\mathbf{t}$ will comprise continuous variables and in general will be a vector as we may wish to predict several related quantities. For classification problems, $\mathbf{t}$ will represent class labels. Again, $\mathbf{t}$ will in general be a vector if we have more than two classes. The joint probability distribution $p(\mathbf{x}, \mathbf{t})$ provides a complete summary of the uncertainty associated with these variables. Determining $p(\mathbf{x}, \mathbf{t})$ from a set of training data is an example of inference and is typically a very difficult problem whose solution forms the subject of much of this book. In a practical application, however, we must often make a specific prediction for the value of $\mathbf{t}$ or more generally take a specific action based on our understanding of the values $\mathbf{t}$ is likely to take, and this aspect is the subject of decision theory.

Consider, for example, our earlier medical diagnosis problem in which we have taken an image of a skin lesion on a patient, and we wish to determine whether the patient has cancer. In this case, the input vector $\mathbf{x}$ is the set of pixel intensities in
the image, and the output variable $t$ will represent the absence of cancer, which we denote by the class $\mathcal{C}_{1}$, or the presence of cancer, which we denote by the class $\mathcal{C}_{2}$. We might, for instance, choose $t$ to be a binary variable such that $t=0$ corresponds to class $\mathcal{C}_{1}$ and $t=1$ corresponds to class $\mathcal{C}_{2}$. We will see later that this choice of label values is particularly convenient when working with probabilities. The general inference problem then involves determining the joint distribution $p\left(\mathbf{x}, \mathcal{C}_{k}\right)$, or equivalently $p(\mathbf{x}, t)$, which gives us the most complete probabilistic description of the variables. Although this can be a very useful and informative quantity, ultimately, we must decide either to give treatment to the patient or not, and we would like this choice to be optimal according to some appropriate criterion (Duda and Hart, 1973). This is the decision step, and the aim of decision theory is that it should tell us how to make optimal decisions given the appropriate probabilities. We will see that the decision stage is generally very simple, even trivial, once we have solved the inference problem. Here we give an introduction to the key ideas of decision theory as required for the rest of the book. Further background, as well as more detailed accounts, can be found in Berger (1985) and Bather (2000).

Before giving a more detailed analysis, let us first consider informally how we might expect probabilities to play a role in making decisions. When we obtain the skin image x for a new patient, our goal is to decide which of the two classes to assign the image to. We are therefore interested in the probabilities of the two classes, given the image, which are given by $p\left(\mathcal{C}_{k} \mid \mathbf{x}\right)$. Using Bayes' theorem, these probabilities can be expressed in the form

$$
p\left(\mathcal{C}_{k} \mid \mathbf{x}\right)=\frac{p\left(\mathbf{x} \mid \mathcal{C}_{k}\right) p\left(\mathcal{C}_{k}\right)}{p(\mathbf{x})}
$$

Note that any of the quantities appearing in Bayes' theorem can be obtained from the joint distribution $p\left(\mathbf{x}, \mathcal{C}_{k}\right)$ by either marginalizing or conditioning with respect to the appropriate variables. We can now interpret $p\left(\mathcal{C}_{k}\right)$ as the prior probability for the class $\mathcal{C}_{k}$ and $p\left(\mathcal{C}_{k} \mid \mathbf{x}\right)$ as the corresponding posterior probability. Thus, $p\left(\mathcal{C}_{1}\right)$ represents the probability that a person has cancer, before the image is taken. Similarly, $p\left(\mathcal{C}_{1} \mid \mathbf{x}\right)$ is the posterior probability, revised using Bayes' theorem in light of the information contained in the image. If our aim is to minimize the chance of assigning $\mathbf{x}$ to the wrong class, then intuitively we would choose the class having the higher posterior probability. We now show that this intuition is correct, and we also discuss more general criteria for making decisions.

### 5.2.1 Misclassification rate

Suppose that our goal is simply to make as few misclassifications as possible. We need a rule that assigns each value of $\mathbf{x}$ to one of the available classes. Such a rule will divide the input space into regions $\mathcal{R}_{k}$ called decision regions, one for each class, such that all points in $\mathcal{R}_{k}$ are assigned to class $\mathcal{C}_{k}$. The boundaries between decision regions are called decision boundaries or decision surfaces. Note that each decision region need not be contiguous but could comprise some number of disjoint regions. To find the optimal decision rule, consider first the case of two classes, as in the cancer problem, for instance. A mistake occurs when an input vector belonging
to class $\mathcal{C}_{1}$ is assigned to class $\mathcal{C}_{2}$ or vice versa. The probability of this occurring is given by

$$
\begin{aligned}
p(\text { mistake }) & =p\left(\mathbf{x} \in \mathcal{R}_{1}, \mathcal{C}_{2}\right)+p\left(\mathbf{x} \in \mathcal{R}_{2}, \mathcal{C}_{1}\right) \\
& =\int_{\mathcal{R}_{1}} p\left(\mathbf{x}, \mathcal{C}_{2}\right) \mathrm{d} \mathbf{x}+\int_{\mathcal{R}_{2}} p\left(\mathbf{x}, \mathcal{C}_{1}\right) \mathrm{d} \mathbf{x}
\end{aligned}
$$

We are free to choose the decision rule that assigns each point $\mathbf{x}$ to one of the two classes. Clearly, to minimize $p$ (mistake) we should arrange that each $\mathbf{x}$ is assigned to whichever class has the smaller value of the integrand in (5.20). Thus, if $p\left(\mathbf{x}, \mathcal{C}_{1}\right)>p\left(\mathbf{x}, \mathcal{C}_{2}\right)$ for a given value of $\mathbf{x}$, then we should assign that $\mathbf{x}$ to class $\mathcal{C}_{1}$. From the product rule of probability, we have $p\left(\mathbf{x}, \mathcal{C}_{k}\right)=p\left(\mathcal{C}_{k} \mid \mathbf{x}\right) p(\mathbf{x})$. Because the factor $p(\mathbf{x})$ is common to both terms, we can restate this result as saying that the minimum probability of making a mistake is obtained if each value of $\mathbf{x}$ is assigned to the class for which the posterior probability $p\left(\mathcal{C}_{k} \mid \mathbf{x}\right)$ is largest. This result is illustrated for two classes and a single input variable $x$ in Figure 5.5.

For the more general case of $K$ classes, it is slightly easier to maximize the probability of being correct, which is given by

$$
\begin{aligned}
p(\text { correct }) & =\sum_{k=1}^{K} p\left(\mathbf{x} \in \mathcal{R}_{k}, \mathcal{C}_{k}\right) \\
& =\sum_{k=1}^{K} \int_{\mathcal{R}_{k}} p\left(\mathbf{x}, \mathcal{C}_{k}\right) \mathrm{d} \mathbf{x}
\end{aligned}
$$

which is maximized when the regions $\mathcal{R}_{k}$ are chosen such that each $\mathbf{x}$ is assigned to the class for which $p\left(\mathbf{x}, \mathcal{C}_{k}\right)$ is largest. Again, using the product rule $p\left(\mathbf{x}, \mathcal{C}_{k}\right)= p\left(\mathcal{C}_{k} \mid \mathbf{x}\right) p(\mathbf{x})$, and noting that the factor of $p(\mathbf{x})$ is common to all terms, we see that each $\mathbf{x}$ should be assigned to the class having the largest posterior probability $p\left(\mathcal{C}_{k} \mid \mathbf{x}\right)$.

### 5.2.2 Expected loss

For many applications, our objective will be more complex than simply minimizing the number of misclassifications. Let us consider again the medical diagnosis problem. We note that, if a patient who does not have cancer is incorrectly diagnosed as having cancer, the consequences may be that they experience some distress plus there is the need for further investigations. Conversely, if a patient with cancer is diagnosed as healthy, the result may be premature death due to lack of treatment. Thus, the consequences of these two types of mistake can be dramatically different. It would clearly be better to make fewer mistakes of the second kind, even if this was at the expense of making more mistakes of the first kind.

We can formalize such issues through the introduction of a loss function, also called a cost function, which is a single, overall measure of loss incurred in taking any of the available decisions or actions. Our goal is then to minimize the total loss incurred. Note that some authors consider instead a utility function, whose value

![](https://cdn.mathpix.com/cropped/2025_10_01_d90a4d82c88904700833g-160.jpg?height=1445&width=1262&top_left_y=229&top_left_x=275)

**Image Description:** The image consists of two panels (a and b) depicting probability density functions for two classes, \( C_1 \) and \( C_2 \), across two regions \( R_1 \) and \( R_2 \). 

In panel (a), the x-axis represents the data values and includes two vertical lines at \( x_0 \) and \( \hat{x} \), with shaded areas indicating the probabilities \( p(x, C_1) \) (green) and \( p(x, C_2) \) (red). 

Panel (b) shows a similar distribution with overlapping shaded areas, indicating the revised probabilities after a certain modification. The x-axis remains the same, while the shading orientation reflects the changes in class distributions.

Figure 5.5 Schematic illustration of the joint probabilities $p\left(x, \mathcal{C}_{k}\right)$ for each of two classes plotted against $x$, together with the decision boundary $x=\widehat{x}$. Values of $x \geqslant \widehat{x}$ are classified as class $\mathcal{C}_{2}$ and hence belong to decision region $\mathcal{R}_{2}$, whereas points $x<\widehat{x}$ are classified as $\mathcal{C}_{1}$ and belong to $\mathcal{R}_{1}$. Errors arise from the blue, green, and red regions, so that for $x<\widehat{x}$, the errors are due to points from class $\mathcal{C}_{2}$ being misclassified as $\mathcal{C}_{1}$ (represented by the sum of the red and green regions). Conversely for points in the region $x \geqslant \widehat{x}$, the errors are due to points from class $\mathcal{C}_{1}$ being misclassified as $\mathcal{C}_{2}$ (represented by the blue region). By varying the location $\widehat{x}$ of the decision boundary, as indicated by the red double-headed arrow in (a), the combined areas of the blue and green regions remains constant, whereas the size of the red region varies. The optimal choice for $\widehat{x}$ is where the curves for $p\left(x, \mathcal{C}_{1}\right)$ and $p\left(x, \mathcal{C}_{2}\right)$ cross, as shown in (b) and corresponding to $\widehat{x}=x_{0}$, because in this case the red region disappears. This is equivalent to the minimum misclassification rate decision rule, which assigns each value of $x$ to the class having the higher posterior probability $p\left(\mathcal{C}_{k} \mid x\right)$.

Figure 5.6 An example of a loss matrix with elements $L_{k j}$ for the cancer treatment problem. The rows correspond to the true class, whereas the columns correspond to the assignment of class made by our decision

$$
\begin{aligned}
& \text { normal } \\
& \text { normal } \\
& \text { cancer }
\end{aligned}\left(\begin{array}{cc}
0 & 1 \\
100 & 0
\end{array}\right)
$$

criterion.
they aim to maximize. These are equivalent concepts if we take the utility to be simply the negative of the loss. Throughout this text we will use the loss function convention. Suppose that, for a new value of $\mathbf{x}$, the true class is $\mathcal{C}_{k}$ and that we assign $\mathbf{x}$ to class $\mathcal{C}_{j}$ (where $j$ may or may not be equal to $k$ ). In so doing, we incur some level of loss that we denote by $L_{k j}$, which we can view as the $k, j$ element of a loss matrix. For instance, in our cancer example, we might have a loss matrix of the form shown in Figure 5.6. This particular loss matrix says that there is no loss incurred if the correct decision is made, there is a loss of 1 if a healthy patient is diagnosed as having cancer, whereas there is a loss of 100 if a patient having cancer is diagnosed as healthy.

The optimal solution is the one that minimizes the loss function. However, the loss function depends on the true class, which is unknown. For a given input vector $\mathbf{x}$, our uncertainty in the true class is expressed through the joint probability distribution $p\left(\mathbf{x}, \mathcal{C}_{k}\right)$, and so we seek instead to minimize the average loss, where the average is computed with respect to this distribution and is given by

$$
\mathbb{E}[L]=\sum_{k} \sum_{j} \int_{\mathcal{R}_{j}} L_{k j} p\left(\mathbf{x}, \mathcal{C}_{k}\right) \mathrm{d} \mathbf{x}
$$

Each $\mathbf{x}$ can be assigned independently to one of the decision regions $\mathcal{R}_{j}$. Our goal is to choose the regions $\mathcal{R}_{j}$ to minimize the expected loss (5.22), which implies that for each $\mathbf{x}$, we should minimize $\sum_{k} L_{k j} p\left(\mathbf{x}, \mathcal{C}_{k}\right)$. As before, we can use the product rule $p\left(\mathbf{x}, \mathcal{C}_{k}\right)=p\left(\mathcal{C}_{k} \mid \mathbf{x}\right) p(\mathbf{x})$ to eliminate the common factor of $p(\mathbf{x})$. Thus, the decision rule that minimizes the expected loss assigns each new $\mathbf{x}$ to the class $j$ for which the quantity

$$
\sum_{k} L_{k j} p\left(\mathcal{C}_{k} \mid \mathbf{x}\right)
$$
is a minimum. Once we have chosen values for the loss matrix elements $L_{k j}$, this is clearly trivial to do.

### 5.2.3 The reject option

We have seen that classification errors arise from the regions of input space where the largest of the posterior probabilities $p\left(\mathcal{C}_{k} \mid \mathbf{x}\right)$ is significantly less than unity or equivalently where the joint distributions $p\left(\mathbf{x}, \mathcal{C}_{k}\right)$ have comparable values. These are the regions where we are relatively uncertain about class membership. In some applications, it will be appropriate to avoid making decisions on the difficult cases in anticipation of obtaining a lower error rate on those examples for which a classification decision is made. This is known as the reject option. For example, in our hypothetical cancer screening example, it may be appropriate to use an automatic

Figure 5.7 Illustration of the reject option. Inputs $x$ such that the larger of the two posterior probabilities is less than or equal to some threshold $\theta$ will be rejected.
![](https://cdn.mathpix.com/cropped/2025_10_01_d90a4d82c88904700833g-162.jpg?height=519&width=677&top_left_y=219&top_left_x=972)

**Image Description:** The diagram is a probability graph displaying conditional probabilities \( p(C_1 | x) \) and \( p(C_2 | x) \) across a range of values for \( x \). The x-axis represents the variable \( x \), while the y-axis indicates probabilities ranging from 0 to 1. The blue curve depicts \( p(C_1 | x) \) and the red curve represents \( p(C_2 | x) \). The green vertical lines denote the boundaries of a "reject region," where classification is uncertain, and the horizontal dashed line indicates a threshold probability level.


system to classify those images for which there is little doubt as to the correct class, while requesting a biopsy to classify the more ambiguous cases. We can achieve this by introducing a threshold $\theta$ and rejecting those inputs $\mathbf{x}$ for which the largest of the posterior probabilities $p\left(\mathcal{C}_{k} \mid \mathbf{x}\right)$ is less than or equal to $\theta$. This is illustrated for two classes and a single continuous input variable $x$ in Figure 5.7. Note that setting $\theta=1$ will ensure that all examples are rejected, whereas if there are $K$ classes, then setting $\theta<1 / K$ will ensure that no examples are rejected. Thus, the fraction of examples that are rejected is controlled by the value of $\theta$.

We can easily extend the reject criterion to minimize the expected loss, when a loss matrix is given, by taking account of the loss incurred when a reject decision is
Exercise 5.10 made.

### 5.2.4 Inference and decision

We have broken the classification problem down into two separate stages, the inference stage in which we use training data to learn a model for $p\left(\mathcal{C}_{k} \mid \mathbf{x}\right)$ and the subsequent decision stage in which we use these posterior probabilities to make optimal class assignments. An alternative possibility would be to solve both problems together and simply learn a function that maps inputs $\mathbf{x}$ directly into decisions. Such a function is called a discriminant function.

In fact, we can identify three distinct approaches to solving decision problems, all of which have been used in practical applications. These are, in decreasing order of complexity, as follows:
(a) First, solve the inference problem of determining the class-conditional densities $p\left(\mathbf{x} \mid \mathcal{C}_{k}\right)$ for each class $\mathcal{C}_{k}$ individually. Separately infer the prior class probabilities $p\left(\mathcal{C}_{k}\right)$. Then use Bayes' theorem in the form

$$
p\left(\mathcal{C}_{k} \mid \mathbf{x}\right)=\frac{p\left(\mathbf{x} \mid \mathcal{C}_{k}\right) p\left(\mathcal{C}_{k}\right)}{p(\mathbf{x})}
$$
to find the posterior class probabilities $p\left(\mathcal{C}_{k} \mid \mathbf{x}\right)$. As usual, the denominator in

Bayes' theorem can be found in terms of the quantities in the numerator, using

$$
p(\mathbf{x})=\sum_{k} p\left(\mathbf{x} \mid \mathcal{C}_{k}\right) p\left(\mathcal{C}_{k}\right)
$$

Equivalently, we can model the joint distribution $p\left(\mathbf{x}, \mathcal{C}_{k}\right)$ directly and then normalize to obtain the posterior probabilities. Having found the posterior probabilities, we use decision theory to determine the class membership for each new input $\mathbf{x}$. Approaches that explicitly or implicitly model the distribution of inputs as well as outputs are known as generative models, because by sampling from them, it is possible to generate synthetic data points in the input space.
(b) First, solve the inference problem of determining the posterior class probabilities $p\left(\mathcal{C}_{k} \mid \mathbf{x}\right)$, and then subsequently use decision theory to assign each new $\mathbf{x}$ to one of the classes. Approaches that model the posterior probabilities directly are called discriminative models.
(c) Find a function $f(\mathbf{x})$, called a discriminant function, that maps each input $\mathbf{x}$ directly onto a class label. For instance, for two-class problems, $f(\cdot)$ might be binary valued and such that $f=0$ represents class $\mathcal{C}_{1}$ and $f=1$ represents class $\mathcal{C}_{2}$. In this case, probabilities play no role.

Let us consider the relative merits of these three alternatives. Approach (a) is the most demanding because it involves finding the joint distribution over both $\mathbf{x}$ and $\mathcal{C}_{k}$. For many applications, $\mathbf{x}$ will have high dimensionality, and consequently, we may need a large training set to be able to determine the class-conditional densities to reasonable accuracy. Note that the class priors $p\left(\mathcal{C}_{k}\right)$ can often be estimated simply from the fractions of the training set data points in each of the classes. One advantage of approach (a), however, is that it also allows the marginal density of data $p(\mathbf{x})$ to be determined from (5.25). This can be useful for detecting new data points that have low probability under the model and for which the predictions may be of low accuracy, which is known as outlier detection or novelty detection (Bishop, 1994; Tarassenko, 1995).

However, if we wish only to make classification decisions, then it can be wasteful of computational resources and excessively demanding of data to find the joint distribution $p\left(\mathbf{x}, \mathcal{C}_{k}\right)$ when in fact we really need only the posterior probabilities $p\left(\mathcal{C}_{k} \mid \mathbf{x}\right)$, which can be obtained directly through approach (b). Indeed, the classconditional densities may contain a significant amount of structure that has little effect on the posterior probabilities, as illustrated in Figure 5.8. There has been much interest in exploring the relative merits of generative and discriminative approaches to machine learning and in finding ways to combine them (Jebara, 2004; Lasserre, Bishop, and Minka, 2006).

An even simpler approach is (c) in which we use the training data to find a discriminant function $f(\mathbf{x})$ that maps each $\mathbf{x}$ directly onto a class label, thereby combining the inference and decision stages into a single learning problem. In the example of Figure 5.8, this would correspond to finding the value of $x$ shown by

![](https://cdn.mathpix.com/cropped/2025_10_01_d90a4d82c88904700833g-164.jpg?height=694&width=1484&top_left_y=239&top_left_x=146)

**Image Description:** The image consists of two subplots. 

1. The left graph illustrates two class density functions, \( p(x|C_1) \) (in blue) and \( p(x|C_2) \) (in red), plotted against the variable \( x \), ranging from 0 to 1. The y-axis represents the density values, showing peaks at different \( x \) values for the two classes.

2. The right graph displays the posterior probability \( p(C_1|x) \) (blue) and \( p(C_2|x) \) (red) as functions of \( x \). The y-axis indicates probability, with a vertical green line at a specific \( x \) value, indicating a decision boundary.

Figure 5.8 Example of the class-conditional densities for two classes having a single input variable $x$ (left plot) together with the corresponding posterior probabilities (right plot). Note that the left-hand mode of the class-conditional density $p\left(\mathbf{x} \mid \mathcal{C}_{1}\right)$, shown in blue on the left plot, has no effect on the posterior probabilities. The vertical green line in the right plot shows the decision boundary in $x$ that gives the minimum misclassification rate, assuming the prior class probabilities, $p\left(\mathcal{C}_{1}\right)$ and $p\left(\mathcal{C}_{2}\right)$, are equal.

the vertical green line, because this is the decision boundary giving the minimum probability of misclassification.

With option (c), however, we no longer have access to the posterior probabilities $p\left(\mathcal{C}_{k} \mid \mathbf{x}\right)$. There are many powerful reasons for wanting to compute the posterior probabilities, even if we subsequently use them to make decisions. These include:

Minimizing risk. Consider a problem in which the elements of the loss matrix are subjected to revision from time to time (such as might occur in a financial application). If we know the posterior probabilities, we can trivially revise the minimum risk decision criterion by modifying (5.23) appropriately. If we have only a discriminant function, then any change to the loss matrix would require that we return to the training data and solve the inference problem afresh.

Reject option. Posterior probabilities allow us to determine a rejection criterion that will minimize the misclassification rate, or more generally the expected loss, for a given fraction of rejected data points.

Section 2.1.1
Compensating for class priors. Consider our cancer screening example again, and suppose that we have collected a large number of images from the general population for use as training data, which we use to build an automated screening system. Because cancer is rare amongst the general population, we might find that, say, only 1 in every 1,000 examples corresponds to the presence of cancer.

If we used such a data set to train an adaptive model, we could run into severe difficulties due to the small proportion of those in the cancer class. For instance, a classifier that assigned every point to the normal class would achieve $99.9 \%$ accuracy, and it may be difficult to avoid this trivial solution. Also, even a large data set will contain very few examples of skin images corresponding to cancer, and so the learning algorithm will not be exposed to a broad range of examples of such images and hence is not likely to generalize well. A balanced data set with equal numbers of examples from each of the classes would allow us to find a more accurate model. However, we then have to compensate for the effects of our modifications to the training data. Suppose we have used such a modified data set and found models for the posterior probabilities. From Bayes' theorem (5.24), we see that the posterior probabilities are proportional to the prior probabilities, which we can interpret as the fractions of points in each class. We can therefore simply take the posterior probabilities obtained from our artificially balanced data set, divide by the class fractions in that data set, and then multiply by the class fractions in the population to which we wish to apply the model. Finally, we need to normalize to ensure that the new posterior probabilities sum to one. Note that this procedure cannot be applied if we have learned a discriminant function directly instead of determining posterior probabilities.

Combining models. For complex applications, we may wish to break the problem into a number of smaller sub-problems each of which can be tackled by a separate module. For example, in our hypothetical medical diagnosis problem, we may have information available from, say, blood tests as well as skin images. Rather than combine all of this heterogeneous information into one huge input space, it may be more effective to build one system to interpret the images and a different one to interpret the blood data. If each of the two models gives posterior probabilities for the classes, then we can combine the outputs systematically using the rules of probability. One simple way to do this is to assume that, for each class separately, the distributions of inputs for the images, denoted by $\mathrm{x}_{\mathrm{I}}$, and the blood data, denoted by $\mathrm{x}_{\mathrm{B}}$, are independent, so that

$$
p\left(\mathbf{x}_{\mathrm{I}}, \mathbf{x}_{\mathrm{B}} \mid \mathcal{C}_{k}\right)=p\left(\mathbf{x}_{\mathrm{I}} \mid \mathcal{C}_{k}\right) p\left(\mathbf{x}_{\mathrm{B}} \mid \mathcal{C}_{k}\right)
$$

This is an example of a conditional independence property, because the independence holds when the distribution is conditioned on the class $\mathcal{C}_{k}$. The posterior probability, given both the image and blood data, is then given by

$$
\begin{aligned}
p\left(\mathcal{C}_{k} \mid \mathbf{x}_{\mathrm{I}}, \mathbf{x}_{\mathrm{B}}\right) & \propto p\left(\mathbf{x}_{\mathrm{I}}, \mathbf{x}_{\mathrm{B}} \mid \mathcal{C}_{k}\right) p\left(\mathcal{C}_{k}\right) \\
& \propto p\left(\mathbf{x}_{\mathrm{I}} \mid \mathcal{C}_{k}\right) p\left(\mathbf{x}_{\mathrm{B}} \mid \mathcal{C}_{k}\right) p\left(\mathcal{C}_{k}\right) \\
& \propto \frac{p\left(\mathcal{C}_{k} \mid \mathbf{x}_{\mathrm{I}}\right) p\left(\mathcal{C}_{k} \mid \mathbf{x}_{\mathrm{B}}\right)}{p\left(\mathcal{C}_{k}\right)}
\end{aligned}
$$

Thus, we need the class prior probabilities $p\left(\mathcal{C}_{k}\right)$, which we can easily estimate from the fractions of data points in each class, and then we need to normalize

Figure 5.9 The confusion matrix for the cancer treatment problem, in which the rows correspond to the true class and the columns correspond to the assignment of class made by our decision criterion. The elements of the matrix show the numbers of true negatives, false positives, false negatives, and true positives.

$$
\begin{aligned}
& \text { normal } \\
& \text { cancer }
\end{aligned}\left(\begin{array}{cc}
\text { normal } & \text { cancer } \\
N_{\mathrm{TN}} & N_{\mathrm{FP}} \\
N_{\mathrm{FN}} & N_{\mathrm{TP}}
\end{array}\right)
$$

## Section 11.2.3

## Chapter 7

the resulting posterior probabilities so they sum to one. The particular conditional independence assumption (5.26) is an example of a naive Bayes model. Note that the joint marginal distribution $p\left(\mathbf{x}_{\mathrm{I}}, \mathbf{x}_{\mathrm{B}}\right)$ will typically not factorize under this model. We will see in later chapters how to construct models for combining data that do not require the conditional independence assumption (5.26). A further advantage of using models that output probabilities rather than decisions is that they can easily be made differentiable with respect to any adjustable parameters (such as the weight coefficients in the polynomial regression example), which allows them to be composed and trained jointly using gradient-based optimization methods.

### 5.2.5 Classifier accuracy

The simplest measure of performance for a classifier is the fraction of test set points that are correctly classified. However, we have seen that different types of error can have different consequences, as expressed through the loss matrix, and often we therefore do not simply wish to minimize the number of misclassifications. By changing the location of the decision boundary, we can make trade-offs between different kinds of error, for example with the goal of minimizing an expected loss. Because this is such an important concept, we will introduce some definitions and terminology so that the performance of a classifier can be better characterized.

## Section 2.1.1

We will consider again our cancer screening example. For each person tested, there is a 'true label' of whether they have cancer or not, and there is also the prediction made by the classifier. If, for a particular person, the classifier predicts cancer and this is in fact the true label, then the prediction is called a true positive. However, if the person does not have cancer it is a false positive. Likewise, if the classifier predicts that a person does not have cancer and this is correct, then the prediction is called a true negative, otherwise it is a false negative. The false positives are also known as type 1 errors whereas the false negatives are called type 2 errors. If $N$ is the total number of people taking the test, then $N_{\mathrm{TP}}$ is the number of true positives, $N_{\mathrm{FP}}$ is the number of false positives, $N_{\mathrm{TN}}$ is the number of true negatives, and $N_{\mathrm{FN}}$ is the number of false negatives, where

$$
N=N_{\mathrm{TP}}+N_{\mathrm{FP}}+N_{\mathrm{TN}}+N_{\mathrm{FN}} .
$$

This can be represented as a confusion matrix as shown in Figure 5.9. Accuracy, measured by the fraction of correct classifications, is then given by

$$
\text { Accuracy }=\frac{N_{\mathrm{TP}}+N_{\mathrm{TN}}}{N_{\mathrm{TP}}+N_{\mathrm{FP}}+N_{\mathrm{TN}}+N_{\mathrm{FN}}} .
$$

We can see that accuracy can be misleading if there are strongly imbalanced classes. In our cancer screening example, for instance, where only 1 person in 1,000 has cancer, a naive classifier that simply decides that nobody has cancer will achieve $99.9 \%$ accuracy and yet is completely useless.

Several other quantities can be defined in terms of these numbers, of which the most commonly encountered are

$$
\begin{aligned}
\text { Precision } & =\frac{N_{\mathrm{TP}}}{N_{\mathrm{TP}}+N_{\mathrm{FP}}} \\
\text { Recall } & =\frac{N_{\mathrm{TP}}}{N_{\mathrm{TP}}+N_{\mathrm{FN}}} \\
\text { False positive rate } & =\frac{N_{\mathrm{FP}}}{N_{\mathrm{FP}}+N_{\mathrm{TN}}} \\
\text { False discovery rate } & =\frac{N_{\mathrm{FP}}}{N_{\mathrm{FP}}+N_{\mathrm{TP}}}
\end{aligned}
$$

In our cancer screening example, precision represents an estimate of the probability that a person who has a positive test does indeed have cancer, whereas recall is an estimate of the probability that a person who has cancer is correctly detected by the test. The false positive rate is an estimate of the probability that a person who is normal will be classified as having cancer, whereas the false discovery rate represents the fraction of those testing positive who do not in fact have cancer.

By altering the location of the decision boundary, we can change the trade-offs between the two kinds of errors. To understand this trade-off, we revisit Figure 5.5, but now we label the various regions as shown in Figure 5.10. We can relate the labelled regions to the various true and false rates as follows:

$$
\begin{aligned}
& N_{\mathrm{FP}} / N=E \\
& N_{\mathrm{TP}} / N=D+E \\
& N_{\mathrm{FN}} / N=B+C \\
& N_{\mathrm{TN}} / N=A+C
\end{aligned}
$$

where we are implicitly considering the limit $N \rightarrow \infty$ so that we can relate number of observations to probabilities.

### 5.2.6 ROC curve

A probabilistic classifier will output a posterior probability, which can be converted to a decision by setting a threshold. As the value of the threshold is varied, we can reduce type 1 errors at the expense of increasing type 2 errors, or vice versa. To better understand this trade-off, it is useful to plot the receiver operating characteristic or ROC curve (Fawcett, 2006), a name that originates from procedures to measure the performance of radar receivers. This is a graph of true positive rate versus false positive rate, as shown in Figure 5.11.

As the decision boundary in Figure 5.10 is moved from $-\infty$ to $\infty$, the ROC curve is traced out and can then be generated by plotting the cumulative fraction of

![](https://cdn.mathpix.com/cropped/2025_10_01_d90a4d82c88904700833g-168.jpg?height=650&width=1252&top_left_y=222&top_left_x=280)

**Image Description:** The image presents a probability distribution diagram displaying two regions, \( R_1 \) and \( R_2 \), along a horizontal axis. The vertical axis represents the probability density function \( p(x, C) \). The curve has peaks indicating different probabilities, with areas labeled \( A, B, C, D, \) and \( E \). A vertical dashed line at \( x_0 \) separates regions corresponding to different classifications, with a horizontal double-headed arrow indicating the distance to \( \hat{x} \). The diagram visually interprets classification boundaries and the distribution of data points across given classes.

Figure 5.10 As in Figure 5.5, with the various regions labelled. In the cancer classification problem, region $\mathcal{R}_{1}$ is assigned to the normal class whereas region $\mathcal{R}_{2}$ is assigned to the cancer class.

correct detection of cancer on the $y$-axis versus the cumulative fraction of incorrect detection on the $x$-axis. Note that a specific confusion matrix represents one point along the ROC curve. The best possible classifier would be represented by a point at the top left corner of the ROC diagram. The bottom left corner represents a simple classifier that assigns every point to the normal class and therefore has no true positives but also no false positives. Similarly, the top right corner represents a classifier that assigns everything to the cancer class and therefore has no false negatives but also no true negatives. In Figure 5.11, the classifiers represented by the blue curve are better than those of the red curve for any choice of, say, false positive rate. It is also possible, however, for such curves to cross over, in which case the choice of which is better will depend on the choice of operating point.

As a baseline, we can consider a random classifier that simply assigns each data point to cancer with probability $\rho$ and to normal with probability $1-\rho$. As we vary the value of $\rho$ it will trace out an ROC curve given by a diagonal straight line, as shown in Figure 5.11. Any classifier below the diagonal line performs worse than random guessing.

Sometimes it is useful to have a single number that characterises the whole ROC curve. One approach is to measure the area under the curve (AUC). A value of 0.5 for the AUC represents random guessing whereas a value of 1.0 represents a perfect classifier.

Another measure is the $F$-score, which is the geometric mean of precision and

Figure 5.11 The receiver operator characteristic (ROC) curve is a plot of true positive rate against false positive rate, and it characterizes the trade-off between type 1 and type 2 errors in a classification problem. The upper blue curve represents a better classifier than the lower red curve. Here the dashed curve represents the performance of a simple random classifier.
![](https://cdn.mathpix.com/cropped/2025_10_01_d90a4d82c88904700833g-169.jpg?height=695&width=699&top_left_y=219&top_left_x=950)

**Image Description:** The image is a graph depicting a Receiver Operating Characteristic (ROC) curve. The x-axis represents the False Positive Rate ranging from 0 to 1, while the y-axis denotes the True Positive Rate, also ranging from 0 to 1. Two cumulative distribution functions are illustrated with blue and red lines, indicating different model performances. A dashed diagonal line serves as a baseline (random classifier) for comparison. The area under the curves (AUC) quantifies model effectiveness, with curves closer to the top-left corner indicating better performance.

recall, and is therefore defined by

$$
\begin{aligned}
F & =\frac{2 \times \text { precision } \times \text { recall }}{\text { precision }+ \text { recall }} \\
& =\frac{2 N_{\mathrm{TP}}}{2 N_{\mathrm{TP}}+N_{\mathrm{FP}}+N_{\mathrm{FN}}}
\end{aligned}
$$

Of course, we can also combine the confusion matrix in Figure 5.9 with the loss matrix in Figure 5.6 to compute the expected loss by multiplying the elements pointwise and summing the resulting products.

Although the ROC curve can be extended to more than two classes, it rapidly becomes cumbersome as the number of classes increases.

### 5.3. Generative Classifiers

We turn next to a probabilistic view of classification and show how models with linear decision boundaries arise from simple assumptions about the distribution of the data. We have already discussed the distinction between the discriminative and

## Section 5.2.4

the generative approaches to classification. Here we will adopt a generative approach in which we model the class-conditional densities $p\left(\mathbf{x} \mid \mathcal{C}_{k}\right)$ as well as the class priors $p\left(\mathcal{C}_{k}\right)$ and then use these to compute posterior probabilities $p\left(\mathcal{C}_{k} \mid \mathbf{x}\right)$ through Bayes' theorem.

First, consider problems having two classes. The posterior probability for class

Figure 5.12 Plot of the logistic sigmoid function $\sigma(a)$ defined by (5.42), shown in red, together with the scaled probit function $\Phi(\lambda a)$, for $\lambda^{2}=\pi / 8$, shown in dashed blue, where $\Phi(a)$ is defined by (5.86). The scaling factor $\pi / 8$ is chosen so that the derivatives of the two curves are equal for $a=0$.
![](https://cdn.mathpix.com/cropped/2025_10_01_d90a4d82c88904700833g-170.jpg?height=489&width=716&top_left_y=239&top_left_x=923)

**Image Description:** The image depicts a graph illustrating a logistic function. The x-axis ranges from approximately -5 to 5, while the y-axis ranges from 0 to 1. Two curves are plotted: a solid red line and a dashed blue line, indicating different models or data fits. The curve asymptotically approaches 0 as x approaches -5 and approaches 1 as x approaches 5, with an inflection point around x=0. The graph may represent model comparisons or data fitting in a statistical context.


$\mathcal{C}_{1}$ can be written as

$$
\begin{aligned}
p\left(\mathcal{C}_{1} \mid \mathbf{x}\right) & =\frac{p\left(\mathbf{x} \mid \mathcal{C}_{1}\right) p\left(\mathcal{C}_{1}\right)}{p\left(\mathbf{x} \mid \mathcal{C}_{1}\right) p\left(\mathcal{C}_{1}\right)+p\left(\mathbf{x} \mid \mathcal{C}_{2}\right) p\left(\mathcal{C}_{2}\right)} \\
& =\frac{1}{1+\exp (-a)}=\sigma(a)
\end{aligned}
$$

where we have defined

$$
a=\ln \frac{p\left(\mathbf{x} \mid \mathcal{C}_{1}\right) p\left(\mathcal{C}_{1}\right)}{p\left(\mathbf{x} \mid \mathcal{C}_{2}\right) p\left(\mathcal{C}_{2}\right)}
$$
and $\sigma(a)$ is the logistic sigmoid function defined by
$$
\sigma(a)=\frac{1}{1+\exp (-a)},
$$

which is plotted in Figure 5.12. The term 'sigmoid' means S-shaped. This type of function is sometimes also called a 'squashing function' because it maps the whole real axis into a finite interval. The logistic sigmoid has been encountered already in earlier chapters and plays an important role in many classification algorithms. It satisfies the following symmetry property:

$$
\sigma(-a)=1-\sigma(a)
$$
as is easily verified. The inverse of the logistic sigmoid is given by
$$
a=\ln \left(\frac{\sigma}{1-\sigma}\right)
$$

and is known as the logit function. It represents the log of the ratio of probabilities $\ln \left[p\left(\mathcal{C}_{1} \mid \mathbf{x}\right) / p\left(\mathcal{C}_{2} \mid \mathbf{x}\right)\right]$ for the two classes, also known as the log odds.

Note that in (5.40), we have simply rewritten the posterior probabilities in an equivalent form, and so the appearance of the logistic sigmoid may seem artificial.

However, it will have significance provided $a(\mathbf{x})$ has a constrained functional form. We will shortly consider situations in which $a(\mathbf{x})$ is a linear function of $\mathbf{x}$, in which case the posterior probability is governed by a generalized linear model.

If there are $K>2$ classes, we have

$$
\begin{aligned}
p\left(\mathcal{C}_{k} \mid \mathbf{x}\right) & =\frac{p\left(\mathbf{x} \mid \mathcal{C}_{k}\right) p\left(\mathcal{C}_{k}\right)}{\sum_{j} p\left(\mathbf{x} \mid \mathcal{C}_{j}\right) p\left(\mathcal{C}_{j}\right)} \\
& =\frac{\exp \left(a_{k}\right)}{\sum_{j} \exp \left(a_{j}\right)}
\end{aligned}
$$

which is known as the normalized exponential and can be regarded as a multi-class generalization of the logistic sigmoid. Here the quantities $a_{k}$ are defined by

$$
a_{k}=\ln \left(p\left(\mathbf{x} \mid \mathcal{C}_{k}\right) p\left(\mathcal{C}_{k}\right)\right) .
$$

The normalized exponential is also known as the softmax function, as it represents a smoothed version of the 'max' function because, if $a_{k} \gg a_{j}$ for all $j \neq k$, then $p\left(\mathcal{C}_{k} \mid \mathbf{x}\right) \simeq 1$, and $p\left(\mathcal{C}_{j} \mid \mathbf{x}\right) \simeq 0$.

We now investigate the consequences of choosing specific forms for the classconditional densities, looking first at continuous input variables $\mathbf{x}$ and then discussing briefly discrete inputs.

### 5.3.1 Continuous inputs

Let us assume that the class-conditional densities are Gaussian. We will then explore the resulting form for the posterior probabilities. To start with, we will assume that all classes share the same covariance matrix $\boldsymbol{\Sigma}$. Thus, the density for class $\mathcal{C}_{k}$ is given by

$$
p\left(\mathbf{x} \mid \mathcal{C}_{k}\right)=\frac{1}{(2 \pi)^{D / 2}} \frac{1}{|\boldsymbol{\Sigma}|^{1 / 2}} \exp \left\{-\frac{1}{2}\left(\mathbf{x}-\boldsymbol{\mu}_{k}\right)^{\mathrm{T}} \boldsymbol{\Sigma}^{-1}\left(\mathbf{x}-\boldsymbol{\mu}_{k}\right)\right\} .
$$

First, suppose that we have two classes. From (5.40) and (5.41), we have

$$
p\left(\mathcal{C}_{1} \mid \mathbf{x}\right)=\sigma\left(\mathbf{w}^{\mathrm{T}} \mathbf{x}+w_{0}\right)
$$
where we have defined
\$\$\begin{aligned}

\mathbf{w} \& =\boldsymbol{\Sigma}^{-1}\left(\boldsymbol{\mu}_{1}-\boldsymbol{\mu}_{2}\right) <br>
w_{0} \& =-\frac{1}{2} \boldsymbol{\mu}_{1}^{\mathrm{T}} \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}_{1}+\frac{1}{2} \boldsymbol{\mu}_{2}^{\mathrm{T}} \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}_{2}+\ln \frac{p\left(\mathcal{C}_{1}\right)}{p\left(\mathcal{C}_{2}\right)} .

\end{aligned}\$\$

We see that the quadratic terms in $\mathbf{x}$ from the exponents of the Gaussian densities have cancelled (due to the assumption of common covariance matrices), leading to a linear function of $\mathbf{x}$ in the argument of the logistic sigmoid. This result is illustrated for a two-dimensional input space $\mathbf{x}$ in Figure 5.13. The resulting decision boundaries correspond to surfaces along which the posterior probabilities $p\left(\mathcal{C}_{k} \mid \mathbf{x}\right)$

