---
course: CS 189
semester: Fall 2025
type: discussion
title: Discussion 1
source_type: slides
source_file: Discussion Mini Lecture 01.pptx
processed_date: '2025-10-01'
processor: mathpix
---

## Discussion Mini Lecture 1

## Math for Machine Learning

Review of Pre-Req Material and Connection to ML

## CS 189/289A, Fall 2025 @ UC Berkeley

Sara Pohland

## Notes about Discussion Mini Lectures

-Discussions are offset from lectures

- E.g., Discussion 1 takes place in Week 2 and covers concepts presented in Lecture 1 during Week 1
- You can utilize course resources in whatever way works best for you, but we recommend that you:

1. Attend/watch lectures
2. Watch the corresponding discussion mini lecture
3. Attend a discussion section
4. Review uncovered discussion problems on your own*
5. Optionally, look at additional resources at end of slides

* We will likely not cover all of the discussion material during the 50 min discussion session!

Additional problems are provided for you as a way to review concepts and get extra practice.

## Concepts Covered

1. ML as Function Approximation
2. Linear Algebra
3. Multivariate Calculus
4. Probability Theory

## ML as Function Approximation

1. ML as Function Approximation
2. Linear Algebra
3. Multivariate Calculus
4. Probability Theory

## Supervised Learning

![](https://cdn.mathpix.com/cropped/2025_10_01_08f7a2606343847f9617g-05.jpg?height=1434&width=2272&top_left_y=395&top_left_x=12)

**Image Description:** The slide features a mixed-media layout combining images and flow diagrams. At the top left, there is a photograph of puppies amidst flowers. Below it, a flowchart outlines three machine learning (ML) models, each represented by a dark blue triangle connected to their outputs via arrows: the first model predicts "dog," the second categorizes a message as "spam," and the third estimates a price of "$1,200.00" based on features like "Bed," "Bath," "Location," and additional input info. This structure illustrates various applications of ML in different contexts.


## ML as Function Approximation

![](https://cdn.mathpix.com/cropped/2025_10_01_08f7a2606343847f9617g-06.jpg?height=1434&width=2271&top_left_y=412&top_left_x=0)

**Image Description:** The image consists of a combination of elements including a pet photo (puppies), text that mimics a scam message, and a table with property details. 

1. The diagram features three blocks labeled \( f? \), \( g? \), and \( h? \) arranged vertically, with arrows indicating a flow of information from the left image and text into each block. The left side contains an image of puppies, while the right side includes textual elements indicating a classification task related to "dog," "spam," and a property listing with price "$1,200.00".

2. There are no equations in the image. 

3. Miscelaneous elements include the photo of puppies, a sample scam text, and a property table.


## How do we Represent our Data?

![](https://cdn.mathpix.com/cropped/2025_10_01_08f7a2606343847f9617g-07.jpg?height=447&width=600&top_left_y=425&top_left_x=85)

**Image Description:** This image features a golden retriever puppy and a gray kitten sitting together on grass, surrounded by colorful flowers. The puppy is on the left, with fluffy fur and soft features, while the kitten is positioned on the right, looking slightly away. Both animals appear calm and content, enhancing the theme of companionship. The background is blurred, focusing attention on the pets, which creates a warm and inviting scene that symbolizes friendship and affection between different species.


$$
\longrightarrow \begin{gathered}
H \times W \times 3 \\
\text { array of pixels }
\end{gathered} \xrightarrow{\text { flatten }} x=\left[\begin{array}{c}
120 \\
50 \\
240 \\
\vdots \\
15
\end{array}\right] \in \mathbb{R}^{D} \underset{\substack{\text { vector of } \\
\text { pixels }}}{H \times W}
$$

HI, It's your boss. Im stuck in Nigeria with none money. Please wire to TRWIGB2LXXX SOON.

$$
\longrightarrow x=\left[\begin{array}{c}
H I, \\
I t^{\prime} s \\
\vdots \\
S O O N .
\end{array}\right] \in \mathbb{R}^{D} \quad \begin{gathered}
\text { vector } \\
\text { of } \\
\text { words }
\end{gathered}
$$

![](https://cdn.mathpix.com/cropped/2025_10_01_08f7a2606343847f9617g-07.jpg?height=427&width=2233&top_left_y=1448&top_left_x=0)

**Image Description:** The image features a structured layout with a table on the left displaying columns labeled # Bed, # Bath, and Location, with specific values of 4, 3, and Berkeley. An arrow points to a mathematical expression indicating a vector \( \mathbf{x} \) defined as:

$$
\mathbf{x} = \begin{bmatrix} 4 \\ 3 \\ \vdots \\ \text{Berkeley} \end{bmatrix} \in \mathbb{R}^D
$$

This signifies a representation of features in a multi-dimensional space, indicating the dimensionality \( D \) of the feature vector, incorporating both numerical and categorical data.


Practice with data manipulation: Discussion 1 notebook

## ML as Function Approximation

![](https://cdn.mathpix.com/cropped/2025_10_01_08f7a2606343847f9617g-08.jpg?height=447&width=600&top_left_y=425&top_left_x=85)

**Image Description:** This is a miscellaneous image depicting a light golden retriever puppy and a gray tabby kitten resting side by side in a grassy area adorned with pink flowers. The puppy is looking directly at the camera, while the kitten is slightly leaning against the puppy, appearing calm and relaxed. The background is softly focused, emphasizing the animals in the foreground.

$\longrightarrow \boldsymbol{x}=\left[\begin{array}{c}120 \\ 50 \\ 240 \\ \vdots \\ 15\end{array}\right] \in \mathbb{R}^{D} \quad \begin{gathered}\text { vector of } \\ \text { pixels }\end{gathered} \quad \operatorname{dog}=f(\boldsymbol{x})$

HI, It's your boss. Im stuck in Nigeria with none money. Please wire to TRWIGB2LXXX
$\longrightarrow x=\left[\begin{array}{c}H I, \\ I t^{\prime} s \\ \vdots \\ S O O N .\end{array}\right] \in \mathbb{R}^{D} \quad \begin{gathered}\text { vector } \\ \text { of } \\ \text { words }\end{gathered} \quad$ spam $=g(\boldsymbol{x})$ SOON.

| $\#$ <br> Bed | $\#$ <br> Bath | $\cdots$ | Location |
| :---: | :---: | :---: | :---: |
| 4 | 3 | $\cdots$ | Berkeley |

$$
\longrightarrow \boldsymbol{x}=\left[\begin{array}{c}
4 \\
3 \\
\vdots \\
\text { Berkeley }
\end{array}\right] \in \mathbb{R}^{D}
$$

vector of feature

$$
\$ 1,200,000=h(x)
$$

Linear Algebra Review

1. ML as Function Approximation
2. Linear Algebra
3. Multivariate Calculus
4. Probability Theory

## ML as Function Approximation

![](https://cdn.mathpix.com/cropped/2025_10_01_08f7a2606343847f9617g-10.jpg?height=447&width=600&top_left_y=425&top_left_x=85)

**Image Description:** This is a miscellaneous image depicting a golden retriever puppy and a gray kitten resting on grass surrounded by pink flowers. The puppy is positioned on the left, appearing fluffy and friendly, while the kitten sits closely beside it, appearing curious yet calm. The vibrant colors of the flowers contrast with the animals' fur, emphasizing the warm, serene atmosphere of the scene.

$\longrightarrow \boldsymbol{x}=\left[\begin{array}{c}120 \\ 50 \\ 240 \\ \vdots \\ 15\end{array}\right] \in \mathbb{R}^{D} \quad \begin{gathered}\text { vector of } \\ \text { pixels }\end{gathered} \quad \operatorname{dog}=f(\boldsymbol{x})$

HI, It's your boss. Im stuck in Nigeria with none money. Please wire to TRWIGB2LXXX
$\longrightarrow x=\left[\begin{array}{c}H I, \\ I t^{\prime} s \\ \vdots \\ S O O N .\end{array}\right] \in \mathbb{R}^{D} \quad \begin{gathered}\text { vector } \\ \text { of } \\ \text { words }\end{gathered} \quad$ spam $=g(\boldsymbol{x})$ SOON.

| $\#$ <br> Bed | $\#$ <br> Bath | $\cdots$ | Location |
| :---: | :---: | :---: | :---: |
| 4 | 3 | $\cdots$ | Berkeley |

$$
\longrightarrow \boldsymbol{x}=\left[\begin{array}{c}
4 \\
3 \\
\vdots \\
\text { Berkeley }
\end{array}\right] \in \mathbb{R}^{D}
$$

vector of feature

$$
\$ 1,200,000=h(x)
$$

## Linear Function Approximation

![](https://cdn.mathpix.com/cropped/2025_10_01_08f7a2606343847f9617g-11.jpg?height=447&width=600&top_left_y=425&top_left_x=85)

**Image Description:** The image is a misc. image featuring a golden retriever puppy and a gray kitten sitting together amidst flowers in a grassy area. The puppy is on the left, appearing fluffy and alert, while the kitten is on the right, looking slightly aloof. The background includes pink flowers, enhancing the tranquil and playful ambiance of the scene.

$\longrightarrow x=\left[\begin{array}{c}120 \\ 50 \\ 240 \\ \vdots \\ 15\end{array}\right] \in \mathbb{R}^{D} \quad \begin{gathered}\text { vector of } \\ \text { pixels }\end{gathered}$

$$
\begin{aligned}
\operatorname{dog}=f(\boldsymbol{x})= & \boldsymbol{A} \boldsymbol{x} \\
& \boldsymbol{A} \in \mathbb{R}^{K \times D}
\end{aligned}
$$

HI, It's your boss. Im stuck in Nigeria with none money. Please wire to TRWIGB2LXXX SOON.

$$
\longrightarrow \boldsymbol{x}=\left[\begin{array}{c}
H I, \\
I t^{\prime} s \\
\vdots \\
S O O N .
\end{array}\right] \in \mathbb{R}^{D} \quad \begin{array}{cr}
\text { vector } & \text { spam }=g(\boldsymbol{x})=\boldsymbol{B} \boldsymbol{x} \\
\text { of } & \text { words }
\end{array} \quad \begin{aligned}
\mathbf{B} \in \mathbb{R}^{K \times D} &
\end{aligned}
$$

![](https://cdn.mathpix.com/cropped/2025_10_01_08f7a2606343847f9617g-11.jpg?height=427&width=2234&top_left_y=1448&top_left_x=0)

**Image Description:** The image presents a diagram illustrating a feature vector, denoted as \( \mathbf{x} = \begin{bmatrix} 4 \\ 3 \\ \vdots \\ \text{Berkeley} \end{bmatrix} \in \mathbb{R}^D \). It includes a table with columns labeled "# Bed", "# Bath", and "Location" showing numerical values (4, 3) for features related to real estate in Berkeley. An arrow indicates the transformation of these features into a vector representation. The diagram transitions from a structured table format to a mathematical vector notation, highlighting the correlation between raw data and feature representation in a multidimensional space.


$$
\begin{aligned}
\$ 1,200,000= & h(\boldsymbol{x})=\boldsymbol{C} \boldsymbol{x} \\
& \mathbf{C} \in \mathbb{R}^{K \times D}
\end{aligned}
$$

## Properties of Functions

* If an input maps to more than one output, we do not have a valid function.
injective - every input in our domain maps to one* output in our range
![](https://cdn.mathpix.com/cropped/2025_10_01_08f7a2606343847f9617g-12.jpg?height=388&width=1073&top_left_y=480&top_left_x=480)

**Image Description:** The image depicts a diagram representing a function in mathematics. It consists of two labeled boxes: on the left is “domain” and on the right is “range.” Each box contains a series of filled circles, indicating elements in the domain and range. Arrows connect specific elements from the domain to the range, illustrating the mapping of inputs to outputs. The color scheme is light blue, enhancing the visual clarity of the connections. This diagram effectively illustrates the concept of functions in mathematics.



## Every image has a label within our set of labels.

surjective - every output in our range maps to at least one input in domain

![](https://cdn.mathpix.com/cropped/2025_10_01_08f7a2606343847f9617g-12.jpg?height=366&width=1077&top_left_y=978&top_left_x=476)

**Image Description:** The image is a diagram illustrating the concept of a function in mathematics. It consists of two sets: one labeled "domain" on the left and the other labeled "range" on the right. Each set contains several circular nodes. Arrows connect each node in the domain to specific nodes in the range, indicating the mapping relationships between the two sets. This diagram visually represents how elements from the domain are related to elements in the range, emphasizing the directed nature of the function.

Every label is associated with at least one image.

bijective ${ }^{\mathrm{n}}$ - there exists exactly one output in range for every input in domain
domai
![](https://cdn.mathpix.com/cropped/2025_10_01_08f7a2606343847f9617g-12.jpg?height=337&width=554&top_left_y=1496&top_left_x=748)

**Image Description:** The image depicts a neural network architecture. It consists of two layers of circular nodes, representing neurons, arranged vertically. The left layer has four neurons, while the right layer has five neurons. There are directed connections (arrows) from each neuron in the left layer to every neuron in the right layer, indicating a fully connected feedforward network. The background areas of the layers are shaded lightly to differentiate them and emphasize the structure of the network. The diagram illustrates the flow of information from input to output through the neurons.

range of image-label pairs.

## Properties of Linear Functions

![](https://cdn.mathpix.com/cropped/2025_10_01_08f7a2606343847f9617g-13.jpg?height=451&width=2922&top_left_y=421&top_left_x=89)

**Image Description:** The image consists of a diagram that illustrates the relationship between pixel values of an image and a mathematical representation. On the left, a photo of two puppies is depicted, emphasizing the subject of the analysis. On the right, a mathematical equation is presented. The equation states:

$$
x = \begin{bmatrix}
120 \\
50 \\
240 \\
\vdots \\
15
\end{bmatrix} \in \mathbb{R}^D
$$

with the function \( \text{dog} = f(x) = Ax \), indicating a transformation of the pixel vector \( x \) using matrix \( A \), where \( A \in \mathbb{R}^{m \times d} \).

f is injective - if $A$ is full column rank $\quad \longrightarrow \mathrm{R}(\boldsymbol{A})=\mathbb{R}^{D} \quad \operatorname{rank}(\boldsymbol{A})=D$
f is surjective - if $A$ is full row rank $\quad \longrightarrow \mathrm{R}(\boldsymbol{A})=\mathbb{R}^{K} \quad \operatorname{rank}(\boldsymbol{A})=K$
f is bijective - if $A$ is full rank/invertible $\longrightarrow \mathrm{R}(\boldsymbol{A})=\mathbb{R}^{D} \quad \operatorname{rank}(\boldsymbol{A})=D=K$

$$
\mathrm{R}(\boldsymbol{A})=\mathbb{R}^{K}
$$

## Determining Matrix Rank

$$
A=\left[\begin{array}{lll}
1 & 2 & 3 \\
0 & 1 & 2 \\
2 & 4 & 6
\end{array}\right] \in \mathbb{R}^{3 \times 3}
$$

## Option 1: Python

import numpy as np

A = np.array([[1, 2, 3],
[0, 1, 2],
[2, 4, 6]])
print(np.linalg.matrix_rank(A))
0.0s

## Option 2:

Inspectóion ${ }_{\text {inearly }}$ independent rows/relymps? zuo; notice that:

$$
\begin{aligned}
& \text { row } 3=2 * \text { row } 1 \\
& \operatorname{col} 3=2 * \operatorname{col} 2-\operatorname{col} 1
\end{aligned}
$$

Option 3: Singular
Values
How many non-zero singular values?

## Multivariate Calculus Review

1. ML as Function Approximation
2. Linear Algebra
3. Multivariate Calculus
4. Probability Theory

## ML as Function Approximation

![](https://cdn.mathpix.com/cropped/2025_10_01_08f7a2606343847f9617g-16.jpg?height=447&width=600&top_left_y=425&top_left_x=85)

**Image Description:** The image features a golden retriever puppy and a gray kitten nestled together amidst green grass and blooming pink flowers. The animals are positioned closely, with the puppy resting its head slightly against the kitten, highlighting a sense of companionship. The background is soft-focused to emphasize the subjects, while the flowers provide a colorful, vibrant contrast to their fur. This image evokes themes of friendship and harmony in nature.

$\longrightarrow \boldsymbol{x}=\left[\begin{array}{c}120 \\ 50 \\ 240 \\ \vdots \\ 15\end{array}\right] \in \mathbb{R}^{D} \quad \begin{gathered}\text { vector of } \\ \text { pixels }\end{gathered} \quad \operatorname{dog}=f(\boldsymbol{x})$

HI, It's your boss. Im stuck in Nigeria with none money. Please wire to TRWIGB2LXXX
$\longrightarrow x=\left[\begin{array}{c}H I, \\ I t^{\prime} s \\ \vdots \\ S O O N .\end{array}\right] \in \mathbb{R}^{D} \quad \begin{gathered}\text { vector } \\ \text { of } \\ \text { words }\end{gathered} \quad$ spam $=g(\boldsymbol{x})$ SOON.

| $\#$ <br> Bed | $\#$ <br> Bath | $\cdots$ | Location |
| :---: | :---: | :---: | :---: |
| 4 | 3 | $\cdots$ | Berkeley |

$$
\longrightarrow \boldsymbol{x}=\left[\begin{array}{c}
4 \\
3 \\
\vdots \\
\text { Berkeley }
\end{array}\right] \in \mathbb{R}^{D}
$$

vector of feature

$$
\$ 1,200,000=h(x)
$$

## Sensitivity of Function

How sensitive is my output to my input variables?
How will my output change if I change my input by a small amount?

$$
\begin{aligned}
& \boldsymbol{x}=\left[\begin{array}{c}
x_{1} \\
x_{2} \\
\vdots \\
x_{D}
\end{array}\right] \in \mathbb{R}^{D} \\
& \mathrm{y}=f(\boldsymbol{x}) \in \mathbb{R}
\end{aligned}
$$

If I change the first input variable ( $x_{1}$ ) by a small amount ( $\Delta x_{1}$ ) by how much will my output ( $y$ ) change?

$$
\Delta \mathrm{y} \approx \frac{\partial f}{\partial x_{1}}(\boldsymbol{x}) * \Delta x_{1}
$$

## partial derivative!

## Learning New Features

![](https://cdn.mathpix.com/cropped/2025_10_01_08f7a2606343847f9617g-18.jpg?height=464&width=596&top_left_y=425&top_left_x=85)

**Image Description:** This is a miscellaneous image depicting a golden retriever puppy and a gray kitten sitting together in a flower-filled garden. The puppy is on the left, with a light golden coat, and appears playful, while the kitten, on the right, has a fluffy gray coat and a calm demeanor. Both animals are surrounded by vibrant pink flowers and green grass, creating a serene and warm atmosphere. The overall composition emphasizes companionship and innocence in a natural setting.

![](https://cdn.mathpix.com/cropped/2025_10_01_08f7a2606343847f9617g-18.jpg?height=193&width=434&top_left_y=484&top_left_x=795)

**Image Description:** The image presents a simple diagram with a horizontal arrow indicating directionality. At the left end of the diagram, the symbol \( f_1 \) is displayed prominently. The arrow extends to the right, suggesting a progression or transformation related to the function \( f_1 \). There are no other labels or details provided, leaving the function's context or meaning unspecified. This design is minimalist and evokes clues about analysis or function transformations in a mathematical or theoretical framework.

![](https://cdn.mathpix.com/cropped/2025_10_01_08f7a2606343847f9617g-18.jpg?height=277&width=1400&top_left_y=476&top_left_x=1369)

**Image Description:** The image is a diagram representing a conceptual flow in a machine learning model. On the left, "Image Features" is labeled, indicating the input features derived from an image. An arrow points to the right, suggesting a transformation by a function, denoted as \( f_2 \). This function is querying the relationship between input features and the output, labeled "dog." The diagram illustrates the process of mapping image features to a classification output in a machine learning context.


HI, It's your boss. Im stuck in Nigeria with none money. Please
![](https://cdn.mathpix.com/cropped/2025_10_01_08f7a2606343847f9617g-18.jpg?height=184&width=439&top_left_y=969&top_left_x=790)

**Image Description:** The image features a text-based query. It shows the symbol \( g_1 \) followed by a question mark, indicating uncertainty or inquiry regarding the function or variable \( g_1 \). There is a horizontal arrow extending to the right, which may symbolize progression or directionality in a conceptual or theoretical context. The overall design suggests that \( g_1 \) is a focal point for discussion or analysis within an academic framework.

![](https://cdn.mathpix.com/cropped/2025_10_01_08f7a2606343847f9617g-18.jpg?height=234&width=971&top_left_y=961&top_left_x=1373)

**Image Description:** The image consists of two main elements: on the left, bold text states "Text Features," and on the right, there is a mathematical symbol denoting \( g_2 \) followed by a question mark. The layout emphasizes the relationship between text features and the mathematical representation, implying an exploration of the significance or value of \( g_2 \) in the context of text analysis or related field. There are no graphs or equations presented; therefore, it is a miscellaneous image focusing on the conceptual link between text and a mathematical variable.

spam wire to TRWIGB2LXXX SOON.

| $\#$ <br> Bed | $\#$ <br> Bath | $\ldots$ | Location |
| :---: | :---: | :---: | :---: |
| 4 | 3 | $\ldots$ | Berkeley |

![](https://cdn.mathpix.com/cropped/2025_10_01_08f7a2606343847f9617g-18.jpg?height=184&width=434&top_left_y=1445&top_left_x=795)

**Image Description:** The image appears to be a simple academic slide featuring a labeled variable $h_1$ followed by a question mark. It includes a horizontal arrow pointing to the right, suggesting a concept or value to be determined or discussed. The slide likely represents an inquiry into the significance or value of $h_1$ in a particular context, possibly in relation to a diagram, model, or theoretical framework in an academic subject.

![](https://cdn.mathpix.com/cropped/2025_10_01_08f7a2606343847f9617g-18.jpg?height=222&width=977&top_left_y=1444&top_left_x=1376)

**Image Description:** This image appears to be a text-based lecture slide that emphasizes "Tabular Features" on the left in bold, large font. On the right, there is a directional arrow pointing to the right, accompanied by a smaller text "h₂?" in a different font style. The overall layout suggests a focus on linking tabular features with the variable or concept represented by "h₂." There are no diagrams, equations, or complex graphical elements present in the image; it primarily conveys a conceptual link between the two textual elements.

\$1,200,00
0

## Chain Rule

How sensitive is my output to my input variables?
How will my output change if I change my input by a small amount?

$$
\begin{gathered}
\boldsymbol{x}=\left[\begin{array}{c}
x_{1} \\
x_{2} \\
\vdots \\
x_{D}
\end{array}\right] \in \mathbb{R}^{D} \quad \begin{array}{c}
\text { If I change the first input variable }\left(x_{1}\right. \\
\text { amount }\left(\Delta x_{1}\right) \text { by how much will my ou }
\end{array} \\
\Delta \mathrm{y} \approx \frac{\partial}{\partial x_{1}} f_{2}\left(f_{1}(\boldsymbol{x})\right) * \Delta x_{1} \\
\mathrm{y}=f_{2}\left(f_{1}(\boldsymbol{x})\right) \in \mathbb{R} \quad \Delta \mathrm{y} \approx \frac{\partial f_{2}}{\partial x_{1}}\left(f_{1}(\boldsymbol{x})\right) * \frac{\partial f_{1}}{\partial x_{1}}(\boldsymbol{x}) * \Delta x_{1}
\end{gathered}
$$

## Probability Theory Review

1. ML as Function Approximation
2. Linear Algebra
3. Multivariate Calculus
4. Probability Theory

## ML as Function Approximation

![](https://cdn.mathpix.com/cropped/2025_10_01_08f7a2606343847f9617g-21.jpg?height=447&width=600&top_left_y=425&top_left_x=85)

**Image Description:** This image features a golden retriever puppy and a gray kitten sitting together in a grassy area adorned with colorful flowers. The puppy is positioned on the left, showcasing its soft fur and playful expression, while the kitten is on the right, appearing more reserved with its sleek fur. Both animals exhibit a sense of companionship against the vibrant floral backdrop, suggesting themes of friendship and innocence in nature. The overall composition conveys a warm, inviting ambiance, emphasizing the bond between different species.

$\longrightarrow \boldsymbol{x}=\left[\begin{array}{c}120 \\ 50 \\ 240 \\ \vdots \\ 15\end{array}\right] \in \mathbb{R}^{D} \quad \begin{gathered}\text { vector of } \\ \text { pixels }\end{gathered} \quad \operatorname{dog}=f(\boldsymbol{x})$

HI, It's your boss. Im stuck in Nigeria with none money. Please wire to TRWIGB2LXXX
$\longrightarrow x=\left[\begin{array}{c}H I, \\ I t^{\prime} s \\ \vdots \\ S O O N .\end{array}\right] \in \mathbb{R}^{D} \quad \begin{gathered}\text { vector } \\ \text { of } \\ \text { words }\end{gathered} \quad$ spam $=g(\boldsymbol{x})$ SOON.

| $\#$ <br> Bed | $\#$ <br> Bath | $\cdots$ | Location |
| :---: | :---: | :---: | :---: |
| 4 | 3 | $\cdots$ | Berkeley |

$$
\longrightarrow \boldsymbol{x}=\left[\begin{array}{c}
4 \\
3 \\
\vdots \\
\text { Berkeley }
\end{array}\right] \in \mathbb{R}^{D}
$$

vector of feature

$$
\$ 1,200,000=h(x)
$$

## Probabilistic Interpretation of ML

![](https://cdn.mathpix.com/cropped/2025_10_01_08f7a2606343847f9617g-22.jpg?height=447&width=600&top_left_y=425&top_left_x=85)

**Image Description:** This is a miscellaneous image featuring a golden retriever puppy and a gray kitten sitting against a backdrop of green grass and pink flowers. The puppy is on the left, looking toward the viewer with a soft expression, while the kitten is on the right, resting against the puppy. The overall composition is bright and cheerful, conveying a sense of companionship and playfulness between the two animals in a natural outdoor setting.

$\longrightarrow \boldsymbol{x}=\left[\begin{array}{c}120 \\ 50 \\ 240 \\ \vdots \\ 15\end{array}\right] \in \mathbb{R}^{D} \quad \begin{gathered}\text { vector of } \\ \text { pixels }\end{gathered} \quad P(\mathbf{Y}=\operatorname{dog} \mid \boldsymbol{X}=\boldsymbol{x})$ ?
HI, It's your boss. Im stuck in Nigeria with none money. Please wire to TRWIGB2LXXX

$$
\longrightarrow \boldsymbol{x}=\left[\begin{array}{c}
H I, \\
I t^{\prime} s \\
\vdots \\
S O O N .
\end{array}\right] \in \mathbb{R}^{D} \quad \begin{gathered}
\text { vector } \\
\text { of } \\
\text { words }
\end{gathered} \quad P(\mathbf{Y}=\operatorname{spam} \mid \boldsymbol{X}=\boldsymbol{x}) ?
$$

SOON.
![](https://cdn.mathpix.com/cropped/2025_10_01_08f7a2606343847f9617g-22.jpg?height=418&width=3288&top_left_y=1454&top_left_x=0)

**Image Description:** The image contains a diagram showcasing a feature vector representation in the context of a dataset. On the left, a table lists three attributes: the number of beds (4), baths (3), and location ("Berkeley"). The feature vector \( \mathbf{x} = \begin{bmatrix} 4 \\ 3 \\ \text{Berkeley} \end{bmatrix} \in \mathbb{R}^D \) is depicted, indicating it is a vector in a multidimensional space. To the right, a probability function \( P(Y = (k, 2 \text{ mil.} \mid X = \mathbf{x}) \) is shown, illustrating a probabilistic relationship based on the feature vector.


## Bayes' Theorem

$$
\begin{array}{rlr}
P(\boldsymbol{Y}=\boldsymbol{k} \mid \boldsymbol{X}=\boldsymbol{x})=\frac{P(\boldsymbol{X}=\boldsymbol{x} \mid \boldsymbol{Y}=\boldsymbol{k}) P(\boldsymbol{Y}=\boldsymbol{k})}{P(\boldsymbol{X}=\boldsymbol{x})} & \text { Law of total probability: } \\
& =\frac{P(\boldsymbol{X}=\boldsymbol{x} \mid \boldsymbol{Y}=\boldsymbol{k}) P(\boldsymbol{Y}=\boldsymbol{k})}{\sum_{\boldsymbol{y}} P(\boldsymbol{X}=\boldsymbol{x} \mid \boldsymbol{Y}=\boldsymbol{y}) P(\boldsymbol{Y}=\boldsymbol{y})} & =\sum_{\boldsymbol{y}} P(\boldsymbol{X}=\boldsymbol{x})=\sum_{\boldsymbol{y}} P(\boldsymbol{X}=\boldsymbol{x}, \boldsymbol{Y}=\boldsymbol{y})
\end{array}
$$

It might be hard to model $P(\boldsymbol{Y}=\boldsymbol{k} \mid \boldsymbol{X}=\boldsymbol{x})$ directly. In some cases, it's easier to estimate $P(\boldsymbol{X}=\boldsymbol{x} \mid \boldsymbol{Y}=\boldsymbol{y})$ and $P(\boldsymbol{Y}=\boldsymbol{y})$, then use Bayes' theorem to estimate $P(\boldsymbol{Y}=\boldsymbol{k} \mid \boldsymbol{X}=\boldsymbol{x})$.

## Discussion Mini Lecture 1

## Math for Machine Learning

Contributors: Sara Pohland

## Additional Resources

1. ML as Function Approximation

- Lecture 1 and 3

2. Linear Algebra

- Deep Learning Foundations and Concepts - Appendix A
- Mathematics for Machine Learning - Section 3

3. Multivariate Calculus

- Calculus Early Transcendentals - Chapter 14
- Mathematics for Machine Learning - Section 4

4. Probability Theory

- Lecture 4
- Deep Learning Foundations and Concepts - Chapter 2
- Mathematics for Machine Learning - Section 5

