---
course: CS 189
semester: Fall 2025
type: lecture
title: Gradient Descent
source_type: slides
source_file: Lecture 12 -- Gradient Descent.pptx
processed_date: '2025-10-07'
processor: mathpix
number: 12
slug: lec12-gradient-descent-processed
---

## Lecture 12

## Optimization and Gradient Descent

## The core algorithm of modern machine learning

## EECS 189/289, Fall 2025 @ UC Berkeley

Joseph E. Gonzalez and Narges Norouzi

## Key Factors in Generative AI Revolution

![The image features two logos: the first represents Wikipedia, depicted as a globe made of jigsaw puz](https://cdn.mathpix.com/cropped/2025_10_07_8dde35e8ca2175918b4dg-02.jpg?height=303&width=592&top_left_y=484&top_left_x=153)

**Image description:** The image features two logos: the first represents Wikipedia, depicted as a globe made of jigsaw puzzle pieces with various language symbols. The globe illustrates the collaborative nature of Wikipedia across diverse languages. The second logo is for Common Crawl, marked by a blue hexagon with the text "COMMON CRAWL" in bold, juxtaposed against a red circular background. This signifies a data repository for web crawling. Both elements emphasize collaborative information sharing and web data collection in an academic context.

Data

$\ln p\left(\operatorname{Word}_{n} \mid\left\{\operatorname{Word}_{i}\right\}_{i=1}^{n-1}\right)$
Loss Functions
![The image depicts a high-performance computing hardware system, specifically NVIDIA's GPU architectu](https://cdn.mathpix.com/cropped/2025_10_07_8dde35e8ca2175918b4dg-02.jpg?height=384&width=507&top_left_y=382&top_left_x=1913)

**Image description:** The image depicts a high-performance computing hardware system, specifically NVIDIA's GPU architecture. It features multiple stacked GPU units, each labeled with the NVIDIA logo. The arrangement shows thermal management components, including heatsinks and fans, indicative of a design optimized for processing power and cooling efficiency. The layout is compact, emphasizing parallel processing capabilities crucial for tasks in machine learning and deep neural networks. This kind of setup is utilized in data centers for heavy computational workloads. The purpose is to illustrate advancements in GPU technology for academic discussions on AI and computational efficiency.


Compute
![The diagram illustrates a transformer architecture used for natural language processing. It features](https://cdn.mathpix.com/cropped/2025_10_07_8dde35e8ca2175918b4dg-02.jpg?height=821&width=507&top_left_y=0&top_left_x=2585)

**Image description:** The diagram illustrates a transformer architecture used for natural language processing. It features two main sections: the encoder (left) and decoder (right). Each section contains layers labeled "Add & Norm," "Feed Forward," and "Multi-Head Attention." The encoder processes "Input Embedding" with "Positional Encoding," while the decoder's output is shown as "Output (shifted right)." The output probabilities are derived from a "Softmax" function applied after the linear transformation. The arrows indicate data flow, and the repetitive structure signifies stacking multiple layers (denoted as \(N_x\)).

![The image displays logos of three popular open-source machine learning frameworks: PyTorch, TensorFl](https://cdn.mathpix.com/cropped/2025_10_07_8dde35e8ca2175918b4dg-02.jpg?height=400&width=1170&top_left_y=1165&top_left_x=2160)

**Image description:** The image displays logos of three popular open-source machine learning frameworks: PyTorch, TensorFlow, and JAX. PyTorch is represented with a flame icon at the top, indicating its emphasis on dynamic computation. TensorFlow’s logo, featuring a stylized "TF" in orange, is positioned centrally. JAX's logo consists of colorful 3D blocks forming the letters "JAX," illustrating its focus on composable functions and NumPy integration. The arrangement emphasizes comparative importance in the machine learning landscape, serving as a visual guide for frameworks used in academic discussions and applications.

![The image features the word "Automatic" presented in a bold, prominent font. There are no diagrams, ](https://cdn.mathpix.com/cropped/2025_10_07_8dde35e8ca2175918b4dg-02.jpg?height=193&width=584&top_left_y=1581&top_left_x=2236)

**Image description:** The image features the word "Automatic" presented in a bold, prominent font. There are no diagrams, axes, or equations visible. Its purpose appears to emphasize the concept of automation, likely as a key theme of the lecture. The stark typography draws attention, suggesting a focus on clarity and impact in conveying the idea of automatic processes or systems.

![The diagram illustrates the concept of Stochastic Gradient Descent (SGD), featuring a contour plot t](https://cdn.mathpix.com/cropped/2025_10_07_8dde35e8ca2175918b4dg-02.jpg?height=885&width=1863&top_left_y=953&top_left_x=176)

**Image description:** The diagram illustrates the concept of Stochastic Gradient Descent (SGD), featuring a contour plot that represents a function's level curves. The axes are labeled, although specific labels are not visible. The black line traces the path taken by the optimization process, with discrete points indicating iterations. The title “Robbins-Monro 1951” signifies the foundational work on SGD. This graphical representation serves to demonstrate the convergence behavior of the algorithm over iterations in minimizing a function, showcasing how SGD updates parameters based on randomness in data selection.


Architectures
![The image appears to be a blank or partially blank slide, lacking any discernible diagrams, equation](https://cdn.mathpix.com/cropped/2025_10_07_8dde35e8ca2175918b4dg-02.jpg?height=60&width=99&top_left_y=1815&top_left_x=1994)

**Image description:** The image appears to be a blank or partially blank slide, lacking any discernible diagrams, equations, or images. As a result, there are no axes, labels, or significant visual content to describe. This absence suggests the slide may serve as a placeholder or introduction to forthcoming material in the academic lecture. Further context or content is needed to provide a detailed analysis of the intended academic concepts.


## The General Optimization Problem

## The General Optimization Problem

Most problems in ML can be reduced to the following general optimization problem
![The slide presents an equation showcasing the optimal solution in a mathematical optimization contex](https://cdn.mathpix.com/cropped/2025_10_07_8dde35e8ca2175918b4dg-04.jpg?height=1008&width=2912&top_left_y=656&top_left_x=109)

**Image description:** The slide presents an equation showcasing the optimal solution in a mathematical optimization context. The equation is $$ w^* = \arg \min f(w) $$, where \( w \) represents the learned parameters constrained by \( w \in \mathbb{R}^\Theta \). The diagram includes labeled boxes: "Objective Function" in blue, indicating the function to be minimized, "Solution" in orange, referring to the learned parameters, and "Constraints" in green, specifying the permissible range for \( w \). The layout visually emphasizes the relationship between the objective function, solution, and constraints.


## Optimization Constraints

The constraints define the valid set of parameters.

## $w^{*}=\arg \min f(w) w \in \Theta$ <br> Constraints

For the rest of this lecture, we will focus on the unconstrained setting.

$$
\Theta=\mathbb{R}^{d}
$$

The most common case in ML is unconstrained optimization in which $\Theta= \mathbb{R}^{d}$.

In some cases, there may be integer (e.g., $\Theta=\mathbb{Z}^{d}$ ) or probability constraints:

$$
\Theta=\left\{w \mid w \in \mathbb{R}^{d}, 0 \leq w \leq 1, \sum w_{i}=1\right\}
$$

## The Objective Function

The shape of the objective function plays a big role in the complexity of solving the optimization problem.

## Objective Function

$$
w^{*}=\underset{w \in \Theta}{\arg \min } f(w)
$$

The Objective Function in ML is often referred to as the loss or error function and depends on the data.

$$
f(w)=E[w ; \mathfrak{D}]
$$

## Examples Optimization Problems

$\underset{w \in \mathbb{R}}{\arg \min } w^{2}-3 w+4$
![The image features two labeled points on the left and right, each indicated by red circles. The left](https://cdn.mathpix.com/cropped/2025_10_07_8dde35e8ca2175918b4dg-07.jpg?height=660&width=1647&top_left_y=595&top_left_x=412)

**Image description:** The image features two labeled points on the left and right, each indicated by red circles. The left circle is labeled "Discrete" and the right is labeled "Multiple Minima," highlighted with speech bubbles. The diagram likely illustrates concepts in optimization or function characteristics, emphasizing the importance of discrete points and the presence of multiple local minima in a given context. The minimalist design effectively focuses on these concepts without additional details or axes.

$\underset{w \in \mathbb{R}}{\arg \min } w^{4}-5 w^{2}+w+4$
Local Minima
![The slide features a simple diagram illustrating the concept of "Global Minima." It contains two ora](https://cdn.mathpix.com/cropped/2025_10_07_8dde35e8ca2175918b4dg-07.jpg?height=486&width=801&top_left_y=786&top_left_x=2355)

**Image description:** The slide features a simple diagram illustrating the concept of "Global Minima." It contains two orange speech bubbles, one labeled "Global Minima," associated with two red circles positioned at the bottom of the image. This layout emphasizes the location of global minima in a function's graphical representation, showcasing their significance in optimization problems. The lack of axes and additional labels suggests a focus on qualitative understanding rather than quantitative analysis, serving educational purposes in communicating foundational concepts in mathematical optimization.


How do we solve these optimization problems?
What properties of the objective function make it "easier" to solve?

## Convex Functions

Definition: A function is convex on the domain $\Theta$ if for any pair of points $\forall w_{1}, w_{2} \in \Theta$ and for all $0 \leq t \leq 1$ :

$$
f\left(t w_{1}+(1-t) w_{2}\right) \leq t f\left(w_{1}\right)+(1-t) f\left(w_{2}\right)
$$

Convex:

$$
\begin{aligned}
& \text { Secant Line } \\
& t f\left(w_{1}\right)+(1-t) f\left(w_{2}\right) \\
& f\left(t w_{1}+(1-t) w_{2}\right)
\end{aligned} \quad \text { Non-Convex: }
$$

In general, there are efficient algorithms to reliably minimize convex functions.

## Convex (Constraint) Sets

Definition: A set $S$ is convex if it contains the line segment between any two points in the set:

$$
\forall w_{1}, w_{2} \in S, ; 0 \leq t \leq 1: t w_{1}+(1-t) w_{2} \in S
$$
![The image displays two shapes: a blue convex set on the left and a red non-convex set on the right. ](https://cdn.mathpix.com/cropped/2025_10_07_8dde35e8ca2175918b4dg-09.jpg?height=596&width=2839&top_left_y=914&top_left_x=288)

**Image description:** The image displays two shapes: a blue convex set on the left and a red non-convex set on the right. The convex set contains a dashed line segment connecting two points within the shape, illustrating that any line segment between two points in a convex set lies entirely within that set. The non-convex set reveals two points connected by a dashed line segment, which extends outside the shape, emphasizing that not all line segments between points in a non-convex set remain inside. The labels clearly distinguish between the two types of sets.


In general, there are efficient algorithms to reliably minimize convex functions over convex sets.

## Optimization Problems

Optimization of a convex objective function over a convex constraint set.

- ML problems that we can solve efficiently
- Primary focus of EECS-127 and books

Many classic ML problems are convex.

- Least Squares Regression and Logistic Regression

Most of modern ML (deep learning) is not convex.
-We use convex optimization techniques for non-convex problems

- In this class, introduce basic optimization concepts


## Minimizing the Error \& Optimizing in WeightSpace

## Points on the Error Surface

Each parameterization of a model corresponds to a point on the error surface $E(w ; D)$.
![The slide features a diagram illustrating data representation in a multi-dimensional space. The hori](https://cdn.mathpix.com/cropped/2025_10_07_8dde35e8ca2175918b4dg-12.jpg?height=558&width=2009&top_left_y=654&top_left_x=620)

**Image description:** The slide features a diagram illustrating data representation in a multi-dimensional space. The horizontal axis is labeled as a data feature, denoted by "Data" within a speech bubble. Several arrows indicate various data vectors: one is green (aligned with the axis), labeled as \( W_1 \), while others are colored red and blue, representing different directions or outcomes of data transformation. The arrows emphasize the relationship between the data points and their projected outputs, demonstrating vector directions in a geometric representation relevant to the lecture topic.


Goal: find parameterization with lowest error: $w^{*}=\underset{w \in \mathbb{R}^{d}}{\arg \min } E(w ; D)$

## Demo

Understanding the Error Surface
![The image consists of two sets of diagrams illustrating the loss surface and loss contours of a two-](https://cdn.mathpix.com/cropped/2025_10_07_8dde35e8ca2175918b4dg-13.jpg?height=1706&width=1970&top_left_y=68&top_left_x=1237)

**Image description:** The image consists of two sets of diagrams illustrating the loss surface and loss contours of a two-parameter optimization problem. The left includes a 3D loss surface with axes labeled as \( w_1 \), \( w_2 \), and loss values, displaying a topological shape with valleys and peaks. The right features 2D contour plots, with contours representing different loss levels; axes labeled as \( w_1 \) and \( w_2 \) indicate the relationship between the parameters. The color gradient represents varying loss values, useful for visualizing optimization landscapes in machine learning.

![The diagram illustrates a logistic regression model, labeled "Model 1." The horizontal axis (x) rang](https://cdn.mathpix.com/cropped/2025_10_07_8dde35e8ca2175918b4dg-14.jpg?height=915&width=996&top_left_y=68&top_left_x=29)

**Image description:** The diagram illustrates a logistic regression model, labeled "Model 1." The horizontal axis (x) ranges from 0 to 6, while the vertical axis (y) ranges from 0 to 1. Red dots represent positive class data points concentrated on the left, while blue dots represent negative class data points on the right. A black sigmoid curve demonstrates the model's predicted probabilities, starting near 1 for low x values and asymptotically approaching 0 as x increases. This visualizes the transition between classifications based on the logistic function.

![The image presents a graph labeled "A." It features a blue curve on a light blue background, illustr](https://cdn.mathpix.com/cropped/2025_10_07_8dde35e8ca2175918b4dg-14.jpg?height=770&width=834&top_left_y=1012&top_left_x=72)

**Image description:** The image presents a graph labeled "A." It features a blue curve on a light blue background, illustrating a function that approaches a vertical asymptote. The x-axis ranges from -20 to 0, while the y-axis appears to extend from 0 to approximately 2.5, though specific labels are not provided. A red point marks a specific location on the curve, likely indicating an important value or critical point of the function. The diagram serves to visually represent the behavior of the function, possibly in the context of limits or derivatives.

![The slide presents three main diagrams.   1. **Top Left (Model 2: Least Squares Regression)**: A sca](https://cdn.mathpix.com/cropped/2025_10_07_8dde35e8ca2175918b4dg-14.jpg?height=1787&width=1953&top_left_y=38&top_left_x=1037)

**Image description:** The slide presents three main diagrams. 

1. **Top Left (Model 2: Least Squares Regression)**: A scatter plot depicts cancer data, with the x-axis representing an independent variable and the y-axis representing a dependent variable. A linear regression line fits the data points.

2. **Top Right (Model 3: Logistic Regression)**: A logistic function curve demonstrates probability on the y-axis against a transformed independent variable on the x-axis. The data is jittered for clarity.

3. **Bottom Left (B)**: A contour plot illustrates a function's level curves, indicating gradient descent paths with a marked red point representing a local minimum.

4. **Bottom Right (C)**: A parabola shows a simple quadratic function, emphasizing its minimum point at the vertex.

![The image is a simplistic icon representing a checklist. It features a blue outline with a checkmark](https://cdn.mathpix.com/cropped/2025_10_07_8dde35e8ca2175918b4dg-15.jpg?height=478&width=417&top_left_y=569&top_left_x=238)

**Image description:** The image is a simplistic icon representing a checklist. It features a blue outline with a checkmark indicating completion of tasks. An empty circle to the left suggests options or selections within a list. This image serves to emphasize the importance of organization and task management in academic lectures, illustrating concepts such as to-do lists or assessment criteria. It is not a diagram, equation, or graph, but signifies the process of evaluating completed versus pending items.


## Match the model with the loss.

## Calculus and Optimization

## The Gradient

The gradient of a scalar-valued function $f: \mathbb{R}^{d} \rightarrow \mathbb{R}$ is the column vectorvalued function $\nabla f: \mathbb{R}^{d} \rightarrow \mathbb{R}^{d}$ of partial derivatives of $f$

$$
\nabla f(x)=\left[\begin{array}{c}
\frac{\partial f}{\partial x_{1}} \\
\vdots \\
\frac{\partial f}{\partial x_{d}}
\end{array}\right]
$$

![The slide features a diagram of a function \( f(x) \) plotted in blue, depicting a parabolic curve o](https://cdn.mathpix.com/cropped/2025_10_07_8dde35e8ca2175918b4dg-17.jpg?height=576&width=677&top_left_y=705&top_left_x=1845)

**Image description:** The slide features a diagram of a function \( f(x) \) plotted in blue, depicting a parabolic curve opening upwards on a Cartesian plane. The x-axis and y-axis are labeled, with the vertex marked by a red point. An orange arrow originates from this point, indicating the gradient \( \nabla f(x) \), which represents the direction of the steepest ascent. The y-axis reads \( f(x) \), while the x-axis is unlabelled. This diagram illustrates the relationship between a function's value and its gradient.


## Important Properties

- The gradient $\nabla f(x)$ points in the direction of steepest ascent.
- The gradient $\nabla f(x)=0$ at local minima, maxima, and saddle points.
- We already used this property to solve for the MLE (...many times...)


## Gradient Exercise

Consider the function:

$$
\left(\begin{array}{r}
\text { Gradient } \\
\nabla f(x)=\left[\begin{array}{c}
\frac{\partial f}{\partial x_{1}} \\
\vdots \\
\frac{\partial f}{\partial x_{d}}
\end{array}\right]
\end{array}\right]_{, y}
$$

$$
E(w)=\left(w_{0}-1\right)^{2}+\left(w_{1}-2\right)^{2}+1
$$

Compute the partial derivatives:

$$
\begin{aligned}
\frac{\partial}{\partial w_{0}} E(w) & =2\left(w_{0}-1\right) \\
\frac{\partial}{\partial w_{1}} E(w) & =2\left(w_{1}-2\right)
\end{aligned}
$$

Then the gradient is:

$$
\nabla E(w)=\left[\begin{array}{l}
2\left(w_{0}-1\right) \\
2\left(w_{1}-2\right)
\end{array}\right]
$$

![The image features a dialogue bubble with a yellow background and a pointed tail, resembling a speec](https://cdn.mathpix.com/cropped/2025_10_07_8dde35e8ca2175918b4dg-18.jpg?height=176&width=609&top_left_y=756&top_left_x=1607)

**Image description:** The image features a dialogue bubble with a yellow background and a pointed tail, resembling a speech bubble. Inside, it displays the word "Gradient" in a bold blue font. This design element likely serves to emphasize the concept of "gradient" in an academic context, suggesting its importance in the topic being discussed, possibly related to calculus, optimization, or machine learning. The visual format is intended to draw attention and facilitate understanding of the term within the lecture material.

![The slide features a speech bubble saying, "The gradient points in the direction of greatest ascent,](https://cdn.mathpix.com/cropped/2025_10_07_8dde35e8ca2175918b4dg-18.jpg?height=673&width=1846&top_left_y=1194&top_left_x=1484)

**Image description:** The slide features a speech bubble saying, "The gradient points in the direction of greatest ascent," emphasizing the concept of gradient direction in optimization. A second speech bubble states "Zero Gradient at \( w = (1,2) \)," indicating a critical point where the gradient's magnitude is zero. The layout showcases the relationships between gradients and optimization, and a QR code is included for additional resources or reference. The slide effectively illustrates the theoretical significance of gradients in determining ascent in multidimensional spaces.


## Demo

The Gradient of the Error Function
![The image features two sets of diagrams depicting loss surfaces and contours in machine learning opt](https://cdn.mathpix.com/cropped/2025_10_07_8dde35e8ca2175918b4dg-19.jpg?height=1778&width=2055&top_left_y=85&top_left_x=1258)

**Image description:** The image features two sets of diagrams depicting loss surfaces and contours in machine learning optimization. 

1. **3D Loss Surface**: Each plot displays a 3D representation with axes labeled \(w_1\) and \(w_2\), illustrating the loss function's landscape. The gradient colors indicate loss magnitude, with yellow to purple gradient showing regions of lower to higher loss.

2. **2D Loss Contours**: Contour plots beneath the 3D surfaces showcase the same loss function in 2D, with lines representing levels of loss, aiding visual understanding of convergence areas.

These diagrams are designed to demonstrate how optimization techniques traverse the loss landscape.

![The image depicts stylized clouds, likely symbolizing concepts related to cloud computing or atmosph](https://cdn.mathpix.com/cropped/2025_10_07_8dde35e8ca2175918b4dg-20.jpg?height=456&width=541&top_left_y=561&top_left_x=178)

**Image description:** The image depicts stylized clouds, likely symbolizing concepts related to cloud computing or atmospheric science. The larger cloud is positioned above a smaller one, illustrating hierarchical or comparative relationships. The purple outline may denote thematic relevance in an academic context, such as data storage or weather patterns. The simplicity of the design suggests it serves a decorative or conceptual purpose rather than presenting detailed technical information.


If $w$ is a d-dimensional vector and $E(w)$ is a scalar-valued function, what is the dimensionality of the gradient $\nabla E(w)$ with respect to $w$ ?

## Common Confusion

The gradient does not lie

$$
E(w)=\left(w_{0}-1\right)^{2}+\left(w_{1}-2\right)^{2}+1
$$
on the surface of the function.

Example: $E: \mathbb{R}^{2} \rightarrow \mathbb{R}$
The gradient $\nabla E: \mathbb{R}^{2} \rightarrow \mathbb{R}^{2}$ is 2-dimensional (not 3)

## Stationary Points and Zero Gradient

Anywhere the gradient is zero is a stationary point.

- Gradient is zero at local minima but zero gradient does not imply a minima

$$
E_{A}[w]=\left(w_{0}-1\right)^{2}+\left(w_{1}-2\right)^{2}+1 \quad E_{B}[w]=-\left(w_{0}-1\right)^{2}-\left(w_{1}-2\right)^{2}+12 \quad E_{C}[w]=-\left(w_{0}-1\right)^{2}+\left(w_{1}-2\right)^{2}+1
$$

Minimum
Maximum
Saddle Point

## Analytical Optimization using Gradient hast

In some situations, (e.g., ordinary least squares) we can find the stationary points by solving:

$$
\nabla E(w)=0
$$
- For these problems we often know that there is one stationary point and that it is the minima.

However, for many machine learning applications (e.g., logistic regression and neural networks) we cannot solve for the stationary points analytically.

Iterative optimization

## Gradient Descent

## Iterative Optimization Algorithms

Iterative optimization algorithms start with an initial estimate $\mathrm{w}^{(0)}$ and then iteratively refine that estimate:
![The image features a speech bubble with the text "Iteration τ" prominently displayed. The background](https://cdn.mathpix.com/cropped/2025_10_07_8dde35e8ca2175918b4dg-25.jpg?height=99&width=439&top_left_y=637&top_left_x=1190)

**Image description:** The image features a speech bubble with the text "Iteration τ" prominently displayed. The background is a solid blue, providing contrast to the white font, highlighting the concept of iteration in a mathematical or computational context. The term "iteration" often refers to the repeated application of a process or function, denoted by τ, which might be used to represent discrete time steps or cycles in algorithms. This image likely serves to emphasize a key concept in iterative methods or processes in an academic lecture.

![The slide presents a mathematical update for an estimate denoted as \( w(\tau) \). It shows the rela](https://cdn.mathpix.com/cropped/2025_10_07_8dde35e8ca2175918b4dg-25.jpg?height=414&width=2012&top_left_y=803&top_left_x=697)

**Image description:** The slide presents a mathematical update for an estimate denoted as \( w(\tau) \). It shows the relationship:

$$
w(\tau) = w(\tau - 1) + \Delta w(\tau - 1)
$$

with "New Estimate" and "Old Estimate" labels indicating the current and previous states, respectively. The update, represented in purple, is based on the gradient, suggesting an iterative approach for optimization. The visual structure aids in understanding the progression of estimates in a numerical method, illustrating how each step builds upon the last.

until stopping criterion is achieved.
The choice of initial estimate $\mathbf{w}^{(0)}$, how the update $\Delta \mathbf{w}^{(\tau-1)}$ is computed, and the stopping criterion define the iterative optimization algorithm.

## Intuition

![The slide features a block diagram depicting a model's error management. At the top center, a blue c](https://cdn.mathpix.com/cropped/2025_10_07_8dde35e8ca2175918b4dg-26.jpg?height=941&width=1446&top_left_y=374&top_left_x=978)

**Image description:** The slide features a block diagram depicting a model's error management. At the top center, a blue cube labeled "Error" signifies an error output. Below, a rectangular base displays three circular knobs labeled \( W_0 \), \( W_1 \), and \( W_2 \). Each knob is green with an upward-pointing arrow, indicating adjustable parameters or weights in a model. The arrangement suggests a flow from weights to error, highlighting the relationship between parameters and error correction in the learning process. This visual aids in understanding how weight adjustments influence model performance.


Goal: Minimize the loss by turning the knobs.

## Intuition

![The image depicts a three-dimensional diagram of a system with three control knobs labeled \( W_0 \)](https://cdn.mathpix.com/cropped/2025_10_07_8dde35e8ca2175918b4dg-27.jpg?height=818&width=1451&top_left_y=497&top_left_x=973)

**Image description:** The image depicts a three-dimensional diagram of a system with three control knobs labeled \( W_0 \), \( W_1 \), and \( W_2 \), situated on a horizontal surface. Each knob has a green circular base with a directional arrow indicating rotation. Above the knobs is a blue cube labeled "Error," suggesting a focus on error correction or feedback in the system. The purpose of the image likely illustrates the concept of adjusting parameters through user interaction to mitigate or understand error dynamics.


## Intuition

![The slide features a 3D bar labeled "Error" positioned above a rectangular base that includes three ](https://cdn.mathpix.com/cropped/2025_10_07_8dde35e8ca2175918b4dg-28.jpg?height=1205&width=1460&top_left_y=110&top_left_x=969)

**Image description:** The slide features a 3D bar labeled "Error" positioned above a rectangular base that includes three knobs labeled \(W_0\), \(W_1\), and \(W_2\). Each knob is circular with an arrow indicating a rotational movement, suggesting adjustment capabilities. The overall diagram likely represents a system where error adjustment is modulated by manipulating the weights \(W_0\), \(W_1\), and \(W_2\), often found in control systems or machine learning models. The arrangement visually conveys the concept of error correction through varying input parameters.


## Intuition

![The image depicts a conceptual model related to error measurement in a computational context. It fea](https://cdn.mathpix.com/cropped/2025_10_07_8dde35e8ca2175918b4dg-29.jpg?height=661&width=1442&top_left_y=654&top_left_x=978)

**Image description:** The image depicts a conceptual model related to error measurement in a computational context. It features three labeled green circles: \(W_0\), \(W_1\), and \(W_2\). Each circle contains an arrow, indicating a rotational mechanism or processing step. Above these, a blue rectangular box labeled "Error" signifies the central focus of the model, likely representing an error metric or outcome. This layout suggests a systematic approach to error analysis, with the circles potentially indicating different weight parameters or components involved in the error calculation process.


## Intuition

![The image depicts a model representing a system with three components: \( W_0 \), \( W_1 \), and \( ](https://cdn.mathpix.com/cropped/2025_10_07_8dde35e8ca2175918b4dg-30.jpg?height=584&width=1446&top_left_y=731&top_left_x=978)

**Image description:** The image depicts a model representing a system with three components: \( W_0 \), \( W_1 \), and \( W_2 \), each illustrated as green circular elements with directional indicators showing rotational motion. Above these components, a block labeled "Error" signifies the central connection or output of the system. The axes and specific labels for each component are not provided. This diagram likely aims to visualize the relationship between the components and the propagation of error in a mechanical or computational context.


## Intuition

![The image depicts a block diagram representing an error correction mechanism. It features three comp](https://cdn.mathpix.com/cropped/2025_10_07_8dde35e8ca2175918b4dg-31.jpg?height=690&width=1464&top_left_y=765&top_left_x=965)

**Image description:** The image depicts a block diagram representing an error correction mechanism. It features three components labeled \( W_0 \), \( W_1 \), and \( W_2 \), each illustrated with a circular dial featuring directional arrows. These indicators suggest the adjustment of weights in response to error signals. An "Error" label is positioned at the top center, indicating the block's overall function. Arrows at the bottom point left and right, symbolizing input and feedback, typical in control system diagrams. The image succinctly conveys the process of weight adjustment to minimize error.


What if we knew which way to turn the knob and an idea of how far?

## This is the Gradient!

## Intuition

![The image depicts a diagram of a neural network representation. It features a rectangular block labe](https://cdn.mathpix.com/cropped/2025_10_07_8dde35e8ca2175918b4dg-32.jpg?height=1081&width=1451&top_left_y=374&top_left_x=973)

**Image description:** The image depicts a diagram of a neural network representation. It features a rectangular block labeled "Error" positioned above three circular nodes labeled \( W_0 \), \( W_1 \), and \( W_2 \). Each node is illustrated with a triangle indicating a weight or activation. Horizontal arrows point towards the nodes from the left, suggesting input flow. The purpose of the diagram is to visually demonstrate the relationship between weights and error in a machine learning context, highlighting how adjustments to weights influence overall model accuracy.


## Intuition

![The diagram illustrates a neural network structure with three weight parameters labeled \( W_0 \), \](https://cdn.mathpix.com/cropped/2025_10_07_8dde35e8ca2175918b4dg-33.jpg?height=822&width=1451&top_left_y=637&top_left_x=973)

**Image description:** The diagram illustrates a neural network structure with three weight parameters labeled \( W_0 \), \( W_1 \), and \( W_2 \) represented by green circles. An error node is positioned above a rectangular block, indicating the central role of error measurement in the network. Arrows pointing left and right suggest input and output flows, respectively. This visual emphasizes the relationship between weights and error in adjusting network performance, confirming the importance of weight optimization in training algorithms.


Try the loss game (its free)!
(c) (1) (1)

## Intuition

![The image depicts a neural network component with three weighted inputs, \( W_0, W_1, W_2 \), illust](https://cdn.mathpix.com/cropped/2025_10_07_8dde35e8ca2175918b4dg-34.jpg?height=822&width=1451&top_left_y=637&top_left_x=973)

**Image description:** The image depicts a neural network component with three weighted inputs, \( W_0, W_1, W_2 \), illustrated as circular nodes. Each weight is shown with a directional arrow indicating input direction. Above them is a rectangular block labeled "Error," signifying an error signal or loss function. The arrows pointing downwards represent the flow of information towards the output layer. This diagram visually represents the error propagation process in backpropagation, emphasizing the role of weights in minimizing the error during training.


## Intuition

![The image depicts a diagram illustrating a neural network structure. It features three circular node](https://cdn.mathpix.com/cropped/2025_10_07_8dde35e8ca2175918b4dg-35.jpg?height=694&width=1451&top_left_y=761&top_left_x=973)

**Image description:** The image depicts a diagram illustrating a neural network structure. It features three circular nodes labeled \( W_0 \), \( W_1 \), and \( W_2 \), each likely representing weights in the network. An "Error" block is positioned above these nodes, indicating the output error related to the network's predictions. Arrows pointing left and down suggest the direction of data flow or adjustments during the training process. The layout visually emphasizes the relationship between weights and the resulting error, highlighting the need for error minimization in neural network training.


## Intuition

![The image depicts a model of a system with three weights, labeled \( W_0 \), \( W_1 \), and \( W_2 \](https://cdn.mathpix.com/cropped/2025_10_07_8dde35e8ca2175918b4dg-36.jpg?height=698&width=1451&top_left_y=761&top_left_x=973)

**Image description:** The image depicts a model of a system with three weights, labeled \( W_0 \), \( W_1 \), and \( W_2 \), represented by green circular indicators. Each weight seems to correspond to a component influencing the system's error, denoted by the blue box labeled "Error" above them. The setup likely illustrates how different weights contribute to the overall error in a computational or algorithmic context. The axes are not specifically visible, indicating a focus on the components rather than a quantitative graph.


## This is Gradient descent!

Try the loss game (its free)!
(c)(i)@(D)

## Gradient Descent Intuition

![The image presents a gradient descent visualization in a 2D space defined by axes \(W_1\) and \(W_2\](https://cdn.mathpix.com/cropped/2025_10_07_8dde35e8ca2175918b4dg-37.jpg?height=1629&width=3220&top_left_y=238&top_left_x=97)

**Image description:** The image presents a gradient descent visualization in a 2D space defined by axes \(W_1\) and \(W_2\). Contour lines depict the optimization landscape, with lower regions indicating better objective function values. A red dot marks the "Goal: \(w^*\)," the optimal solution, while a blue dot signifies the "Initial estimate: Often a random guess." Arrows show the direction of steepest descent, demonstrating how the algorithm iteratively converges towards the optimum. The color gradient transitions from green to yellow, indicating increasing function values, which reinforces the optimization goal visually.


## Gradient Descent Intuition

In high dimensional problems we can't
![The image represents a color gradient scale, likely indicating values along a spectrum. The scale ra](https://cdn.mathpix.com/cropped/2025_10_07_8dde35e8ca2175918b4dg-38.jpg?height=1595&width=273&top_left_y=272&top_left_x=3040)

**Image description:** The image represents a color gradient scale, likely indicating values along a spectrum. The scale ranges from a deep purple (1.515) at the top to a light green (0.276) at the bottom. Each row features distinct color transitions along the gradient, corresponding to decreasing numerical values. This visual representation can be utilized to illustrate the distribution of data values in various scientific contexts, such as heat maps or concentration gradients, facilitating a quick assessment of data trends or thresholds.
 plot the error surface.

Why?

1. Difficult to visualize ;)
2. Expensive! How Expensive?

- Grid resolution of $K$
- Weight dimension $D$

Requires $\boldsymbol{\mathcal { O }}\left(\boldsymbol{K}^{\boldsymbol{D}}\right)$ error evaluations!
![The image features a grid of dots representing discrete points in a 2D space. The x-axis is labeled ](https://cdn.mathpix.com/cropped/2025_10_07_8dde35e8ca2175918b4dg-38.jpg?height=652&width=656&top_left_y=701&top_left_x=1824)

**Image description:** The image features a grid of dots representing discrete points in a 2D space. The x-axis is labeled with \( K = 6 \), indicating a total of 6 evaluations, while the y-axis suggests that loss evaluations are conducted at varying levels labeled as \( s \). Additionally, \( D = 2 \) indicates a dimensionality measure. The labeled bubble highlights the focus on "Loss Evaluation," emphasizing the analysis of model performance across the specified evaluations. The arrangement suggests a systematic approach to evaluating loss across defined parameters in a computational study context.


Initial estimate:
Often a random guess

## Gradient Descent Intuition

At each estimate $\mathrm{w}^{(\tau)}$ we can compute

- The error - a scalar
- How good is the current estimate
- Doesn't say which way to go...
- The gradient - a vector
- Which direction increases error
- Go the opposite direction


## 1846379

Start with an initial estimate $\mathbf{w}^{(0)}$ and then iteratively refine.
Start with an initial estimate $\mathbf{w}^{(0)}$ and then iteratively refine.
$\bigcirc$ At each estimate $\mathbf{w}^{(\tau)}$ we can compute
![The image contains a blank space with no visible diagrams, equations, or labels. As a result, there ](https://cdn.mathpix.com/cropped/2025_10_07_8dde35e8ca2175918b4dg-39.jpg?height=141&width=103&top_left_y=867&top_left_x=344)

**Image description:** The image contains a blank space with no visible diagrams, equations, or labels. As a result, there is no information to describe regarding axes, labels, or any mathematical expressions. The absence of content suggests it may serve as a placeholder in the lecture slide layout, indicating a missing visual element or transition to the next topic.
 .

$\begin{array}{r}\text { Error } \\ \hline 1.515 \\ 1.391 \\ 1.267 \\ 1.143 \\ 1.019 \\ \hline 0.895 \\ 0.771 \\ 0.647 \\ \hline \hline \overline{{ }^{-}} 0.524 \\ \hline \overline{{ }^{-}} 0.4 \\ \hline \overline{{ }^{-}} 0.4 \\ \hline \overline{{ }^{-}} 0.276\end{array}$

## The (Batch) Gradient Descent Alg.

Update the weights by moving in the opposite direction of the gradient:

```
w (0) = Initialize()
for \tau in range(1, }\mp@subsup{\tau}{\operatorname{max}}{}\mathrm{ ):
    \mathbf { w } ^ { ( \tau ) } = \mathbf { w } ^ { ( \tau - 1 ) } - \eta \nabla E ( \mathbf { w } ^ { ( \tau - 1 ) } )
    if Converged( }\mp@subsup{\mathbf{w}}{}{(\tau)},\mp@subsup{\mathbf{w}}{}{(\tau-1)},\epsilon)\mathrm{ : terminate
```

Initialize() typically a small random value (more on this later)
Converged( $\mathbf{w}^{(\tau)}, \mathbf{w}^{(\tau-1)}, \epsilon$ ) stopping condition (e.g., $\epsilon$ change in $\Delta \mathbf{w}^{(\tau-1)}$ )
Hyper Parameters:

- $\eta$ is the learning rate (typically a small value $0<\eta<1$, more on this later)
- $\tau_{\text {max }}$ is the maximum number of iterations


## Demo

Batch Gradient Descent
![The image comprises two 3D diagrams displaying loss surfaces and contours. Each diagram has axes lab](https://cdn.mathpix.com/cropped/2025_10_07_8dde35e8ca2175918b4dg-41.jpg?height=1761&width=2063&top_left_y=102&top_left_x=1267)

**Image description:** The image comprises two 3D diagrams displaying loss surfaces and contours. Each diagram has axes labeled \( w_1 \) and \( w_2 \), representing weight parameters. The color gradient indicates loss values, with warmer colors denoting higher losses. Red dots signify the final solution points, while arrows depict the gradient descent paths. The contours illustrate how the loss changes across the parameter space, facilitating visualization of optimization techniques. The diagrams emphasize convergence and behavior of loss functions in a multidimensional parameter setting, aiding understanding of optimization processes.


## Loss (Error) Curves

The loss or error curve plots the error as a function of the number of iterations (steps) of gradient descent.
![The diagram illustrates training and validation errors over gradient steps. The x-axis represents "I](https://cdn.mathpix.com/cropped/2025_10_07_8dde35e8ca2175918b4dg-42.jpg?height=1180&width=3138&top_left_y=624&top_left_x=88)

**Image description:** The diagram illustrates training and validation errors over gradient steps. The x-axis represents "Iteration (Gradient Steps)," while the y-axis indicates "Loss (Error)." The blue curve depicts training error, generally decreasing, and the red curve shows validation error, initially decreasing but stalling and eventually increasing, indicating overfitting. A star marks the "Best Model" point corresponding to the lowest validation error. The diagram emphasizes choosing parameters associated with minimal validation error. Text notes underlie the concepts of overfitting and the typical positioning of validation curves above training curves.


# Convergence Assuming Quadratic Approximation 

![The image is an icon depicting a checklist or survey form. It features a rectangular shape with roun](https://cdn.mathpix.com/cropped/2025_10_07_8dde35e8ca2175918b4dg-44.jpg?height=464&width=405&top_left_y=578&top_left_x=246)

**Image description:** The image is an icon depicting a checklist or survey form. It features a rectangular shape with rounded corners, illustrating a simple checklist. The top section shows a checked box, indicating completion or confirmation of an item, while the lower part presents an empty circular option and a horizontal line, suggesting unselected choices or activities. The design is minimalistic, emphasizing clarity and organization, commonly used in academic presentations to highlight assessment tools or feedback mechanisms.


Assuming a simple quadratic equation (e.g., $x^{\wedge} 2$ ) is gradient descent guaranteed to converge? presenting from

## Quadratic Approximations

The $2^{\text {nd }}$ order (quadratic) Taylor approximation of a function function $f: \mathbb{R} \rightarrow \mathbb{R}$ evaluated at $\hat{x}$ is:

$$
\tilde{f}(x)=f(\hat{x})+\left(\left.\frac{\partial}{\partial x} f(x)\right|_{x=\hat{x}}\right)(x-\hat{x})+\frac{1}{2}\left(\left.\frac{\partial^{2}}{\partial x^{2}} f(x)\right|_{x=\hat{x}}\right)(x-\hat{x})^{2}
$$

For example: $f(x)=\sin (x)$

$$
\begin{array}{r}
\tilde{f}_{\hat{x}=1}(x)=\sin (1)+\cos (1)(x-1) \\
+\frac{1}{2}(-\sin (1))(x-1)^{2} \\
\tilde{f}_{\hat{x}=5}(x)=\sin (5)+\cos (5)(x-5) \\
+\frac{1}{2}(-\sin (5))(x-5)^{2}
\end{array}
$$

## Taylor Expansion of the Error Surface

The $2^{\text {nd }}$-order Taylor expansion of the error surface at $\widehat{w}$ :

$$
\tilde{E}_{\widehat{w}}(w)=E(\widehat{w})+(w-\widehat{w})^{T} b+\frac{1}{2}(w-\widehat{w})^{T} H(w-\widehat{w})
$$
- $b=\nabla E(\widehat{w})$ is the Gradient of $E$ evaluated at $\widehat{w}$ (vector)
- $H=\nabla \nabla E(\widehat{w})$ is the Hessian of $E$ evaluated at $\widehat{w}$ (a matrix)

## Hessians (the $2^{\text {nd }}$ derivative)

The Hessian of a scalar-valued function $E: \mathbb{R}^{D} \rightarrow \mathbb{R}$ is the matrix-valued function $\nabla^{2} E: \mathbb{R}^{D} \rightarrow \mathbb{R}^{D \times D}$ of the $2^{\text {nd }}$ partial derivatives of $f$

$$
H=\nabla^{2} E=\left[\begin{array}{ccc}
\frac{\partial^{2}}{\partial x_{1}^{2}} E(x) & \cdots & \frac{\partial^{2}}{\partial x_{1} x_{D}} E(x) \\
\vdots & \ddots & \vdots \\
\frac{\partial^{2}}{\partial x_{D} x_{1}} E(x) & \cdots & \frac{\partial^{2}}{\partial x_{D}^{2}} E(x)
\end{array}\right]
$$

The Hessian

- describes the local curvature of $E$
- determines the critical point type (minima, maxima, or saddle)


## Hessian Exercise

Consider the function:

$$
E(w)=\left(w_{0}-1\right)^{2}+\left(w_{1}-2\right)^{2}+1
$$

Then the gradient is: $\nabla E(w)=\left[\begin{array}{l}2\left(w_{0}-1\right) \\ 2\left(w_{1}-2\right)\end{array}\right]$
The Hessian is:

$$
\mathrm{H}(\mathrm{w})=\left[\begin{array}{cc}
\frac{\partial^{2}}{\partial w_{0}^{2}} E(w) & \frac{\partial^{2}}{\partial w_{0} \partial w_{1}} E(w) \\
\frac{\partial^{2}}{\partial w_{0} \partial w_{1}} E(w) & \frac{\partial^{2}}{\partial w_{1}^{2}} E(w)
\end{array}\right]=\left[\begin{array}{ll}
2 & 0 \\
0 & 2
\end{array}\right]
$$

Hessian

$$
\nabla^{2} E=\left[\begin{array}{ccc}
\frac{\partial^{2}}{\partial x_{1}^{2}} E(x) & \ldots & \frac{\partial^{2}}{\partial x_{1} x_{D}} E(x) \\
\vdots & \ddots & \vdots \\
\frac{\partial^{2}}{\partial x_{D} x_{1}} E(x) & \ldots & \frac{\partial^{2}}{\partial x_{D}^{2}} E(x)
\end{array}\right]
$$

## Taylor Expansion of the Error Surface

The $2^{\text {nd }}$-order Taylor expansion of the error surface at $\widehat{w}$ :

$$
\tilde{E}_{\widehat{w}}(w)=E(\widehat{w})+(w-\widehat{w})^{T} b+\frac{1}{2}(w-\widehat{w})^{T} H(w-\widehat{w})
$$
- $b=\nabla E(\widehat{w})$ is the Gradient of $E$ evaluated at $\widehat{w}$ (vector)
- $H=\nabla \nabla E(\widehat{w})$ is the Hessian of $E$ evaluated at $\widehat{w}$ (a matrix)

If $\widehat{w}$ is a stationary point $\left(w^{*}\right)$ then $\nabla E\left(w^{*}\right)=0=b$

$$
\tilde{E}_{w^{*}}(w)=E\left(w^{*}\right)+\frac{1}{2}\left(w-w^{*}\right)^{T} H\left(w-w^{*}\right)
$$

Use a change of variables to better understand this part.

## Eigen Decomposition of the Hessian

${ }^{1} \cdot$ Ve can construct the Eigen decomposition of the Hessian

$$
H u_{i}=\lambda_{i} u_{i}
$$
- $u_{i}$ are the complete set of orthonormal eigen vectors: $\forall_{i, j}: u_{i}^{T} u_{i}=\delta_{i j}$
- $\lambda_{i}$ are the real-valued eigen values

Then we apply a change of variables:

$$
w-w^{*}=\sum_{i} \alpha_{i} u_{i} \text { where } \alpha_{i}=\left(w-w^{*}\right)^{T} u_{i}
$$

This allows us to rewrite:

$$
\tilde{E}_{w^{*}}(w)=E\left(w^{*}\right)+\frac{1}{2}\left(w-w^{*}\right)^{T} H\left(w-w^{*}\right)=E\left(w^{*}\right)+\frac{1}{2} \sum_{i} \alpha_{i}^{2} \lambda_{i}
$$
- $u_{i}$ are the complete set of orthonormal eigen vectors: $\forall_{i, j}: u_{i}^{T} u_{i}=\delta_{i j}$
- $\lambda_{i}$ are the real-valued eigen values

Then we apply a change of variables:

$$
w-w^{*}=\sum_{i} \alpha_{i} u_{i} \text { where } \alpha_{i}=\left(w-w^{*}\right)^{T} u_{i}
$$

This allows us to rewrite:

$$
\tilde{E}_{w^{*}}(w)=E\left(w^{*}\right)+\frac{1}{2}\left(w-w^{*}\right)^{T} H\left(w-w^{*}\right)=E\left(w^{*}\right)+\frac{1}{2} \sum_{i} \alpha_{i}^{2} \lambda_{i}
$$

Show it!

$$
\begin{aligned}
& \left(w-w^{*}\right)^{T} H\left(w-w^{*}\right)=\left(\sum_{i} \alpha_{i} u_{i}\right)^{T} H\left(\sum_{i} \alpha_{i} u_{i}\right)=\left(\sum_{i} \alpha_{i} u_{i}\right)^{T}\left(\sum_{i} \alpha_{i} H u_{i}\right) \\
& =\left(\sum_{i} \alpha_{i} u_{i}\right)^{T}\left(\sum_{i} \alpha_{i} \lambda_{i} u_{i}\right)=\sum_{i j} \alpha_{i} \alpha_{j} u_{i}^{T} u_{j} \lambda_{j}=\sum_{i j} \alpha_{i} \alpha_{j} \delta_{i j} \lambda_{j}=\sum_{i} \alpha_{i}^{2} \lambda_{i} \\
& \text { Eigenvector Defn. }
\end{aligned}
$$

## Taylor Expansion at the Stationary Pts.

If $\widehat{w}$ is a stationary point $\left(w^{*}\right)\left(\right.$ where $\left.b=\left.\nabla E\right|_{w=\widehat{w}}=0\right)$

$$
\widetilde{E}_{w^{*}}(\alpha)=E\left(w^{*}\right)+\frac{1}{2} \sum_{i} \alpha_{i}^{2} \lambda_{i}
$$

At $\alpha=0$ we get the error at the critical point $E\left(w^{*}\right)$.

- If we move in direction $\boldsymbol{i}$ (corresponding to $u_{i}$ ) then $\alpha_{i}^{2}$ increases.
- The change in error then depends on the sign of $\lambda_{i}$

Therefore, if the eigenvalues are

- all positive then $w^{*}$ is a local minimum.
- all negative then $w^{*}$ is a local maximum.
- mixed then $w^{*}$ is a saddle point.


## Stationary Points and the Hessian

Eigenvalues of the Hessian determine the curvature at the critical points

Minimum
Maximum
Saddle Point

## Eigenvector of the Hessian

Elliptical contours of constant error align with the eigenvectors of the Hessian matrix.

Loss Surface
![The diagram is a contour plot with axes labeled \( w_0 \) (horizontal) and \( w_1 \) (vertical). The](https://cdn.mathpix.com/cropped/2025_10_07_8dde35e8ca2175918b4dg-54.jpg?height=937&width=1268&top_left_y=824&top_left_x=280)

**Image description:** The diagram is a contour plot with axes labeled \( w_0 \) (horizontal) and \( w_1 \) (vertical). The contours represent levels of a bivariate function, indicating regions of equal function value. Color gradients range from purple (lowest values) to yellow (highest values), highlighting areas of increasing intensity. The contours provide insights into the function's behavior in the \( w_0 \) and \( w_1 \) space, useful for optimization and visualization of multivariable functions.


Taylor Approximation
![The diagram depicts a contour plot with axes labeled \( w_0 \) and \( w_1 \). Contours represent val](https://cdn.mathpix.com/cropped/2025_10_07_8dde35e8ca2175918b4dg-54.jpg?height=971&width=1473&top_left_y=803&top_left_x=1615)

**Image description:** The diagram depicts a contour plot with axes labeled \( w_0 \) and \( w_1 \). Contours represent values of a function (indicated by "trace 1"). Two eigenvectors are highlighted—\( u_1 \) corresponding to the smallest eigenvalue (s = 0.24) and \( u_2 \) for the largest eigenvalue (s = 19.33). The point \( w^* \) marks a critical point. Direction arrows indicate the trajectories of \( u_1 \) and \( u_2 \), illustrating how each eigenvector aligns with the function's gradients at \( w^* \).


## Demo

## Taylor Expansions and the Hessian

![The left diagram presents a 3D loss surface, depicting the relationship between weights \( w_1 \) an](https://cdn.mathpix.com/cropped/2025_10_07_8dde35e8ca2175918b4dg-55.jpg?height=1422&width=2093&top_left_y=280&top_left_x=1237)

**Image description:** The left diagram presents a 3D loss surface, depicting the relationship between weights \( w_1 \) and \( w_2 \) with the loss value represented by color gradients. The axes are labeled \( w_1 \) and \( w_2 \), while the vertical axis signifies the loss magnitude. Contour lines illustrate loss levels. The right diagram shows 2D loss contours, where the \( w_1 \) and \( w_2 \) axes are labeled similarly. Color gradients indicate loss values, and arrows suggest gradients. Together, these visualizations aid in understanding optimization landscapes in machine learning.


## Recap: Quadratic Approx. Error

We constructed the $\mathbf{2}^{\text {nd }}$-order Taylor expansion of the error surface at the stationary point $w^{*}$ (where $b=\left.\nabla E\right|_{w=w^{*}}=0$ ):

$$
\widetilde{E}_{w^{*}}(w)=E\left(w^{*}\right)+\frac{1}{2}\left(w-w^{*}\right)^{T} H\left(w-w^{*}\right)
$$

Using the Eigen-decomposition of the Hessian: $H u_{i}=\lambda_{i} u_{i}$ we applied a change of variables $\alpha_{i}=\left(w-w^{*}\right)^{T} u_{i}$ to obtain:

$$
\tilde{E}_{w^{*}}(\alpha)=E\left(w^{*}\right)+\frac{1}{2} \sum_{i} \alpha_{i}^{2} \lambda_{i}
$$
- This is minimized at $\alpha=0$

We can evaluate gradient descent on this simple quadratic approximation to better understand the convergence properties.
![The diagram depicts a contour plot with \( w_1 \) on the vertical axis and \( w_2 \) on the horizont](https://cdn.mathpix.com/cropped/2025_10_07_8dde35e8ca2175918b4dg-56.jpg?height=724&width=843&top_left_y=1135&top_left_x=2415)

**Image description:** The diagram depicts a contour plot with \( w_1 \) on the vertical axis and \( w_2 \) on the horizontal axis. Contours represent levels of a function, indicated by gradient colors from purple to yellow. The point \( U_1 \) is marked in blue, \( U_2 \) in orange, and \( w^* \) in black. A blue line connects \( U_1 \) to \( w^* \), illustrating an optimization path. This visual aids in understanding optimization processes in multivariable functions, highlighting convergence from initial points to a critical point.


## Grad. Descent on the Quadratic Appro鸡繼

Quadratic Approx. Error
Compute the gradient with respect to $\alpha_{i}$ :

$$
\frac{\partial}{\partial \alpha_{i}} \tilde{E}_{w^{*}}(\alpha)=\alpha_{i} \lambda_{i}
$$

$$
\tilde{E}_{w^{*}}(\alpha)=E\left(w^{*}\right)+\frac{1}{2} \sum_{i} \alpha_{i}^{2} \lambda_{i}
$$

Which gives the gradient descent update:

$$
\alpha_{i}^{(\tau)}=\alpha_{i}^{(\tau-1)}-\eta\left(\alpha_{i}^{(\tau-1)} \lambda_{i}\right)=\left(1-\eta \lambda_{i}\right) \alpha_{i}^{(\tau-1)}
$$

If we iteratively apply the gradient descent update, we obtain:

$$
\begin{gathered}
\alpha_{i}^{(1)}=\left(1-\eta \lambda_{i}\right) \alpha_{i}^{(0)} \rightarrow \alpha_{i}^{(2)}=\left(1-\eta \lambda_{i}\right)\left(\left(1-\eta \lambda_{i}\right) \alpha_{i}^{(0)}\right) \rightarrow \ldots \rightarrow \\
\alpha_{i}^{(\tau)}=\left(1-\eta \lambda_{i}\right)^{\tau} \alpha_{i}^{(0)}
\end{gathered}
$$

## Grad. Descent on the Quadratic Appro

Assuming a quadratic approximation to the error function around the minima, the $\tau^{\text {th }}$ iteration of gradient descent we have:

$$
\alpha_{i}^{(\tau)}=\left(1-\eta \lambda_{i}\right)^{\tau} \alpha_{i}^{(0)}
$$
and the error is minimized when $\boldsymbol{\alpha}_{\boldsymbol{i}}=\mathbf{0}$.
To ensure $\alpha_{i}^{(\tau)} \rightarrow 0$ as $\tau \rightarrow \infty$ the learning rate should satisfy:
$$
\left|1-\eta \lambda_{i}\right|<1 \text { (Convergence Condition) }
$$

- Smaller $\left|\mathbf{1}-\boldsymbol{\eta} \lambda_{i}\right|$ is better (converges faster)
- Negative $1-\eta \lambda_{i}<0$ oscillatory behavior


## Learning Rate Impact on Convergence

To maximize the progress on each step of gradient descent we want to make $\boldsymbol{\eta}$ as large as possible (take bigger steps)

- We want $\left(1-\eta \lambda_{i}\right)^{\tau} \rightarrow 0$ quickly as $\tau$ increases (larger $\eta \rightarrow$ smaller $1-\eta \lambda_{i}$ ).
- However, to ensure convergence we require $\left|1-\boldsymbol{\eta} \boldsymbol{\lambda}_{\boldsymbol{i}}\right|<1$ for all $\boldsymbol{\lambda}_{\boldsymbol{i}}$

Let $\lambda_{\text {min }}$ and $\lambda_{\text {max }}$ be the smallest and largest eigenvalues of the Hessian

- Then $\eta<\frac{2}{\lambda_{\max }}$ because if $\eta \geq \frac{2}{\lambda_{\max }}$ then $\left|1-\eta \lambda_{\max }\right| \geq 1$

The slowest converging dimension correspond to $\boldsymbol{\lambda}_{\text {min }}$ so the smallest we can make ( $1-\eta \lambda_{i}$ ):

$$
\left(1-\frac{2}{\lambda_{\max }} \lambda_{\min }\right)<\left(1-\eta \lambda_{i}\right)
$$

## Condition Number and Loss Surface

The slowest converging dimension (i) is the dimension corresponding to $\lambda_{\text {min }}$ and its convergence to 0 is lower bounded by:

$$
\left(1-\frac{2}{\lambda_{\max }} \lambda_{\min }\right)^{\tau} \alpha_{i}^{(0)}
$$

The ratio of $\frac{\lambda_{\text {min }}}{\lambda_{\text {max }}}$ is the condition number.

- A problem is said to be well conditioned if it has a large condition number.

Gradient descent converges quickly on well conditioned problems.
![The slide features a contour plot representing a two-dimensional function, with axes labeled \(w_1\)](https://cdn.mathpix.com/cropped/2025_10_07_8dde35e8ca2175918b4dg-60.jpg?height=916&width=1136&top_left_y=824&top_left_x=2058)

**Image description:** The slide features a contour plot representing a two-dimensional function, with axes labeled \(w_1\) (horizontal) and \(w_2\) (vertical). The contours indicate levels of a function where the gradients are lowest along the blue line, denoting the direction with the smallest eigenvalue \(U_{\text{min}}\) and highest along the red line labeled \(U_{\text{max}}\). The point \(W^*\) is marked, indicating an optimal solution in this context. The diagram illustrates the relationship between eigenvalues and directions of curvature in optimization problems.


## Issues with Gradient Descent (GD)

GD converges slowly when the error surface is relatively flat:

> Step size
> (learning rate $\eta$ ) is too small.

GD oscillates when the error surface is poorly conditioned Step size (learning rate $\eta$ ) is too big.

## Demo

Batch Gradient Descent

## Momentum

Momentum can be used improve convergence in these settings by accumulating common update directions

$$
\begin{gathered}
\Delta \mathbf{w}^{(\tau-1)}=-\eta \nabla E\left(\mathbf{w}^{(\tau-1)}\right)+\mu \Delta \mathbf{w}^{(\tau-2)} \\
\mathbf{w}^{(\tau)}=\mathbf{w}^{(\tau-1)}+\Delta \mathbf{w}^{(\tau-1)}
\end{gathered}
$$

- $0 \leq \mu<1(\mu=0.9)$ is the momentum hyperparameter

We now need to maintain the previous $\Delta \mathbf{w}^{(\tau-2)}$ update direction
Intuition: The solution "picks up speed" along the direction of repeated updates.

## How Momentum Helps: Flat Directions

In directions that are relatively flat (e.g., is approx. constant $\nabla E$ ):

$$
\begin{gathered}
\Delta \mathbf{w}^{(\infty)}=-\eta \nabla E+\mu(-\eta \nabla E+\mu(-\eta \nabla E+\mu(\ldots))) \\
\Delta \mathbf{w}^{(\infty)}=-\eta \nabla E(\underbrace{\left(1+\mu+\mu^{2}+\mu^{3}+\cdots\right)}_{\text {Geometric Series }}=-\left(\frac{\eta}{1-\mu}\right) \nabla E
\end{gathered}
$$

The momentum parameter $\mu$ increases the effective learning rate

- Remember that $0 \leq \mu<1$ and therefore $\eta \leq\left(\frac{\eta}{1-\mu}\right)$


## How Momentum Helps: Oscillations

In directions that are steep (e.g., oscillating sign $\nabla E$ ):

$$
\begin{aligned}
\Delta \mathbf{w}^{(\tau)} & =-\eta \nabla E+\mu(-\eta(-\nabla E)+\mu(-\eta \nabla E+\mu(\ldots))) \\
\Delta \mathbf{w}^{(\infty)} & =-\eta \nabla E\left(\frac{\left.1-\mu+\mu^{2}-\mu^{3}+\cdots\right)}{\text { Alternating Géometric Series }}=-\left(\frac{\eta}{1+\mu}\right) \nabla E\right.
\end{aligned}
$$

The momentum parameter $\boldsymbol{\mu}$ decrease the effective learning rate

- Remember that $0 \leq \mu<1$ and therefore $\left(\frac{\eta}{1+\mu}\right) \leq \eta$


## Demo

Mini-batch Gradient Descent
![The left diagram illustrates a 3D loss surface with axes labeled as \( w_1 \) and \( w_2 \), showing](https://cdn.mathpix.com/cropped/2025_10_07_8dde35e8ca2175918b4dg-66.jpg?height=1095&width=2071&top_left_y=33&top_left_x=1263)

**Image description:** The left diagram illustrates a 3D loss surface with axes labeled as \( w_1 \) and \( w_2 \), showing a gradient descent path (in black) that approaches a local minimum (red dot). The color gradient indicates loss values, with purple representing lower losses. The right diagram presents the corresponding loss contours with vectors indicating the gradient direction. The contours reflect loss levels, with values indicated in a color gradient from yellow (lower loss) to blue (higher loss). The final solution is marked in red, emphasizing the convergence point of the gradient descent.

![The image displays contour plots of a function labeled "w1" with a color gradient indicating increas](https://cdn.mathpix.com/cropped/2025_10_07_8dde35e8ca2175918b4dg-66.jpg?height=669&width=1927&top_left_y=1143&top_left_x=1365)

**Image description:** The image displays contour plots of a function labeled "w1" with a color gradient indicating increasing function values. The axes represent a two-dimensional plane, likely X and Y, although specific labels are not visible. Black contour lines form concentric ellipses, resembling wave patterns. Overlaid are black points connected by a line, suggesting a trajectory or optimization path within the contour field. This visualization aims to illustrate the behavior of a function in a multi-dimensional space, highlighting optimal paths or critical points. The gradient scale on the right shows values corresponding to the contours.


## Lecture 12

# Optimization and Gradient Descent 

Credit: Joseph E. Gonzalez and Narges Norouzi Reference Book Chapters: Chapter 7

