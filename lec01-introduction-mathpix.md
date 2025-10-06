---
course: CS 189
semester: Fall 2025
type: lecture
title: Introduction
source_type: slides
source_file: Lecture 01 -- Introduction.pptx
processed_date: '2025-10-01'
processor: mathpix
---

## Lecture 1

# Introduction to Machine Learning 

## Welcome to CS189/289

## EECS 189/289, Fall 2025 @ UC Berkeley

Joseph E. Gonzalez and Narges Norouzi

## Two Things...

Pre-Semester Survey
Take a minute to fill out the pr e-semester survey now if you haven't already!

Spotify Playlist
We created this collaborative Spotify playlist to be played before class. Share your favorite songs with the class!
![](https://cdn.mathpix.com/cropped/2025_10_01_9f1146aa02d6417f2d66g-02.jpg?height=571&width=567&top_left_y=1190&top_left_x=2185)

**Image Description:** The image is a QR code, a type of two-dimensional barcode. It consists of a pattern of black and white squares arranged in a square grid. This code can be scanned by a digital device to retrieve information encoded within it, often linking to a URL or storing text data. The design does not have axes or numerical data, but it functions as a visual key for quick information access.


# III Join at slido.com <br> '1َيْL \#2630936 

![](https://cdn.mathpix.com/cropped/2025_10_01_9f1146aa02d6417f2d66g-04.jpg?height=460&width=541&top_left_y=548&top_left_x=170)

**Image Description:** The image depicts a stylized cloud illustration, characterized by smooth, rounded edges and a light purple color palette. The design features two clouds: a larger cloud overlapping a smaller one. This representation typically symbolizes concepts related to cloud computing, weather, or storage solutions. There are no axes or quantitative data present in this image, making it a decorative or conceptual graphic rather than a diagram or chart.


# What emoji best desribes your mood today? 

## Roadmap

- Introductions
- CS 189/289 Goals and Plans
-What Is Machine Learning?
- Definition
- History
- The ML Process (Lifecycle)
- Logistics


## Introductions

- Introductions
- CS 189/289 Goals and Plans
- What Is Machine Learning?
- Definition
- History
- The ML Process (Lifecycle)
- Logistics


## Narges Norouzi

- Associate Teaching Professor at EECS UC Berkeley
(2022-now).
- A member of BAIR and ACE.
- Associate Dean in the College of Computing Data Science, and Society!
- Research: ML and ML tools in education
- Askademia (www.askademia.org): Real-time video questionanswering
- Edison
- Personalized tutoring with Knowledge Tracing capability
- 61A bot
- INSIGHT
- Also taught ML, AI, Deep Learning + Data 100!
- This semester, I'm:
- Teaching this course, advising my students, and leading EECS Al Task Force.

![](https://cdn.mathpix.com/cropped/2025_10_01_9f1146aa02d6417f2d66g-07.jpg?height=485&width=495&top_left_y=340&top_left_x=2759)

**Image Description:** This image is a portrait of a woman with shoulder-length hair, wearing a black top. The background is a solid light brown color, providing a neutral contrast to her attire. She has a neutral expression and is facing the camera directly. The image does not contain any diagrams, equations, or relevant academic content for indexability.

Pronunciation of my name

- Doing my Associate Dean job!!!


## Joey Gonzalez

- Associate Research Professor at EECS UC Berkeley (2016-nov
- Co-directory of the Sky Computing Lab and a member of BAIR
- Research: AI and AI-Systems
- Training and Inference Systems (e.g., Clipper, vLLM, Alpa)
- Developing new LLMs and Agents (e.g., Gorilla LLM, MemGPT, and Vicunul,
- Evaluating LLMs (e.g., Chatbot Arena and LLM-as-a-judge)
- Visual Language Reasoning and Data Analysis (Lisa's many Projects)
- Robotics and Autonomous Driving Systems (ERDOS)
- Large-Scale Data Processing Systems (e.g., Apache Spark, GraphLab, Ray)
- I also teach: Data-8, Data-100, and CS289-162 (Graduate AI Systems)
- This semester, I'm:
- Teaching this course, running the Sky Lab and launching a new GenAI systems lab.
- Vice Chair Grad Matters, UCB Division Council, and Chair campus GenAI working group
- Advising a bunch of AI companies and coaching my daughter's $2^{\text {nd }}$ grade robotics


## CS 189/289 Goals and Plans

- Introductions
- CS 189/289 Goals and Plans
- What Is Machine Learning?
- Definition
- History
- The ML Process (Lifecycle)
- Logistics


## Why Are We Excited to Teach CS189?

- The field shifted under our feet $\rightarrow$ we're redesigning now so we surf, not chase.
- We listened to student pain points $\rightarrow$ we're tightening the storyline to math $\rightarrow$ code $\rightarrow$ experiments, with unified notation and right-sized and reasonable prerequisites.
- Classroom work should feed research $\rightarrow$ projects will connect to live problems so students $\rightarrow$ collaborators.
- Berkeley should set the bar $\rightarrow$ we aim to produce a shareable resource that others can adopt and adapt.
- We are trying MANY NEW THINGS some things will break... but we hope you will enjoy this class.

8
Why are you here? What do you want to get from this class?

## By the End of CS189, You Should:

- Have a rigorous foundation in core ML concepts and algorithms, connecting the math to the methods we use.
- Be able to implement, train, and debug standard models in Python/PyTorch and move from data prep to experiments.
- Be able to design sound evaluations and document your work so results are reproducible and responsible.
- Be prepared for advanced courses and research.


## These are ambitious goals!

## Accel in Advanced ML Courses

Advanced ML course:

- Computer vision (180/280), NLP (288), RL (285), Statistical Learni ng Theory (281), and Many Special Topics Courses on AI
(194/294)
Require understanding of basic concepts:
- Model design
- Stochastic Gradient Descent
- Regularization and the Bias/Variance Tradeoff
- Maximum Likelihood
- Dimensionality Reduction
- Loss Structure
- Clustering and Density
- Supervised Learning Estimation
- Unsupervised Learning
- Experiment Design and Tracking
- Classification
- Regression
- Introductions
-CS 189/289 Goals and


## What Is Machine Learning?

Plans

- What Is Machine Learning?
- Definition
- History
- The ML Process (Lifecycle)
- Logistics


## What Is Machine Learning (ML)?

Software systems that improve (learn) through data.

Classic Example: What is
Spam?

- Difficult to define
- Easy to demonstrate

Classic Example: Face detection?

- Difficult to program
- Easy to demonstrate


## Basic Recipe:

- Use examples (data) to teach (fit) a model.
- Use the model to make decisions.


## What Is Artificial Intelligence?

Artificial intelliqence (AI) refers to the capabilitv of computational svstems to perform tasks typically associated with human intelligence, such as learning, reasoning, problem-solving, perception, and decision-making.
-- Wikipedia (2025)

## The primary technology behind modern Al is Machine Learning (ML)

## Artificial Intelligence Is the Goal Machine Learning Is the Method

Terms are often used interchangeably (even by experts).

An entrepreneur's note on Al marketing
If you are selling

- Machine Learning then you call it Deep Learning. (20152020)
- Deep Learning then you call it AI. (2020-2024)
- Al you call it an Agent. (2024-Now)


## You have (probably) already done Machine Learning

# Linear <br> Regression <br> is <br> Machine <br> Learning 

## Basic Recipe:

GPU Perf Pricing
File Edit View Insert Format Data Tools Extensions Help
![](https://cdn.mathpix.com/cropped/2025_10_01_9f1146aa02d6417f2d66g-19.jpg?height=179&width=2047&top_left_y=361&top_left_x=1211)

**Image Description:** The image appears to be a screenshot of a spreadsheet or data table. It displays a grid formatted with rows and columns labeled with letters and numbers. The specific columns include "Scope (A)" in column A and "Price (u)" in column B. The cells are structured to contain data, possibly representing different categories or variables related to scope and pricing, although no specific data values are visible in the provided segment. The layout suggests it is used for organizing or analyzing quantitative information.

![](https://cdn.mathpix.com/cropped/2025_10_01_9f1146aa02d6417f2d66g-19.jpg?height=979&width=2031&top_left_y=508&top_left_x=1207)

**Image Description:** The image presents a scatter plot titled "Price vs. Score." The x-axis represents "Score" values, while the y-axis depicts "Price" in monetary units. Blue data points indicate actual price-score pairs. A fitted regression line is shown in orange, illustrating the model's prediction trend. A red star marks a specific observation, and a red dashed line connects this point to the regression line, emphasizing the relationship between the data and the prediction indicated by a red triangle. Key terms "Prediction" and "Model" are highlighted in speech bubbles for emphasis.


Training: Use data to teach (fit) a model.
Inference: Use the model to make predictions (decisions).

## Machine Learning

Data
Training

![](https://cdn.mathpix.com/cropped/2025_10_01_9f1146aa02d6417f2d66g-20.jpg?height=447&width=758&top_left_y=697&top_left_x=2215)

**Image Description:** The image is a flowchart or process diagram illustrating a cyclic system. It features a central circular node connected to multiple peripheral nodes via arrows. The circular node symbolizes a core process or function, while the peripheral nodes represent inputs or factors influencing that process. An arrow extends from the central node, indicating a directional output or result of the central function. The overall structure emphasizes connectivity and the flow of information within the system.

Big Model

## Training

Data
Training
![](https://cdn.mathpix.com/cropped/2025_10_01_9f1146aa02d6417f2d66g-21.jpg?height=669&width=766&top_left_y=646&top_left_x=1263)

**Image Description:** The image depicts a simplified flowchart representing a model in a general framework. The diagram features a central circular node labeled "Model," with multiple arrows emanating from the left side, indicating inputs or features feeding into the model. An arrow on the right side signifies the model's output or prediction. The design emphasizes the transition from inputs to a singular output, typical in machine learning or systems modeling contexts. The overall layout is clear and structured, facilitating understanding of the model's function.


## Training

## Inference

Data
Training
![](https://cdn.mathpix.com/cropped/2025_10_01_9f1146aa02d6417f2d66g-22.jpg?height=959&width=2064&top_left_y=543&top_left_x=1260)

**Image Description:** The image consists of a flow diagram illustrating the relationship between a "Model," a "Query," a "Decision," and an "Application." 

- On the left, a circular node labeled "Model" has arrows pointing right toward the "Query" and "Decision." 
- "Query" is in purple, and "Decision" is in orange, both indicating directional flow. 
- On the right, the "Application" is represented by a document icon containing various elements (a play button and question mark).

This diagram visually synthesizes the process from model input to application output, highlighting decision-making based on queries.


When should I used Machine Learning?

# What kinds of problems are Machine Learning Problems? 

## Select all options that are a machine learning problem.

## Kinds of Problems

Engineering Problem: Can be solved with a direct, specifiable algorithm or a set of hand-written rules.

Machine Learning Problem: For which it is easy to demonstrate or evaluate the solution but difficult to directly implement.

A Human Problem: The problem cannot be well specified and/or human judgement is required.
$\rightarrow$ Often require Engineering + ML + Humans $\square$

## A Machine Learning Problem

A problem for which it is easy to demonstrate or evaluate the solution but difficult to directly implement.

Machine Learning Solution:
The system learns the desired behavior (e.g., prediction, representation, or a policy) through demonstration or experience.

Example:

How do you define Spam?

Determine if a text message is spam

Spam is difficult to define and
depends on the receiver.

Easier to demonstrate examples and learn a function to detect spam.

## Is Chatting a Machine Learning Problem?

Example (ChatGPT):
Engage a human in a productive conversation

How do you program this?

ELIZA (1966
Rule based
Conversational System

Welcome to
![](https://cdn.mathpix.com/cropped/2025_10_01_9f1146aa02d6417f2d66g-27.jpg?height=192&width=753&top_left_y=765&top_left_x=1267)

**Image Description:** The image presents a text-based diagram resembling a representation of a character matrix or typographic layout. It features different arrangements of letters and shapes across five rows, where "E", "L", "I", "Z", and "A" are prominently displayed in various alignments. The letters vary in spacing and alignment, suggesting a visual representation, possibly to discuss typography, letterforms, or data structuring. The arrangement may illustrate concepts related to visual hierarchy or design.


Eliza is a mock Rogerian psychotherapist.
The original program was described by Joseph Weizenbaum in 1966.
This implementation by Norbert Landsteiner 2005.

ELIZA: Is something troubling you ?
YOU: Men are all alike.
ELIZA: What is the connection, do you suppose ?
YOU: They're always bugging us about something or other.
ELIZA: Can you think of a specific example ?
YOU: Well, my boyfriend made me come here.
ELIZA: Is it important to you that your boyfriend made you come here ?
YOU: He says I'm depressed much of the time
ELIZA: I am sorry to hear that you are depressed.
YOU. It's true. I am unhappy.
Y.i.

ELIZA: Can you explain what made you unhappy ?
YOU:
https://en.wikipedia.org/wiki/ELIZA
https://web.njit.edu/~ronkowit/eliza.html

Entertaining but it can't do your homework.

We can demonstrate good conversations.

We can judge good conversations.

## Machine Learning as Learned Function Approximation ${ }^{65093}$

![](https://cdn.mathpix.com/cropped/2025_10_01_9f1146aa02d6417f2d66g-28.jpg?height=865&width=3114&top_left_y=586&top_left_x=76)

**Image Description:** The image is a flow diagram illustrating a spam detection model. It shows an input labeled "Input (X)" represented by a green shape. An arrow connects this to a central function or model denoted as \( h_w \), with labels "Function (Model)" and "Model Parameters" in blue and purple, respectively. Another arrow leads to the output, which is labeled "Output (Y)" in orange, indicating a binary classification for spam detection: "No (0)" or "Yes (1)." The overall layout represents the process of message evaluation for spam classification.


Machine learning becomes the process of "learning" the model parameters from data or interaction with the world.
![](https://cdn.mathpix.com/cropped/2025_10_01_9f1146aa02d6417f2d66g-29.jpg?height=864&width=3254&top_left_y=25&top_left_x=76)

**Image Description:** The image is a flow diagram illustrating a spam detection model. It depicts an input labeled "Input (X)" representing text messages. An intermediary labeled "Function (Model)" suggests the transformation of inputs via a function defined by "Model Parameters" (denoted as \( h_w \)). An output labeled "Output (Y)" indicates the classification of messages as either "Is it Spam?"—with binary responses: "No (0)" or "Yes (1)." Arrows indicate the direction of data flow from input to output, emphasizing the model's function in processing input to determine the output classification.


# Learning Settings 

Supervised<br>(Demonstration)Unsupervised

Reinforcement<br>(Reward)

## Supervised Learning

Trying to learn a relationship between observed $\{(X, Y)\}$ pairs.

## Classification

![](https://cdn.mathpix.com/cropped/2025_10_01_9f1146aa02d6417f2d66g-30.jpg?height=567&width=656&top_left_y=867&top_left_x=208)

**Image Description:** The image consists of two side-by-side panels. The left panel features a hot dog in a bun, marked with a green label "Hotdog!" above it, and a blue button labeled "Share." The right panel displays a black shoe, crossed out with a red label "Not hotdog!" above it, along with the same blue "Share" button. The background is a wooden surface, enhancing the contrast of the food items. The panels visually communicate a binary classification scenario, likely related to machine learning or image recognition.

Image Labeling

X: Image X: Prompt<br>Y: $\{$ Hot Dog, ...\} Y: Next Word

## Regression

![](https://cdn.mathpix.com/cropped/2025_10_01_9f1146aa02d6417f2d66g-30.jpg?height=592&width=621&top_left_y=859&top_left_x=1837)

**Image Description:** The image depicts a diagram related to stock prediction. It features a graph with a blue curve representing historical stock price trends. The x-axis represents time, while the y-axis represents stock price. Overlaid on the main curve are three colored lines (orange, yellow, and red) indicating different prediction scenarios for future stock price movements. At the bottom of the image, the label "Stock Prediction" is prominently displayed in bold text, emphasizing the focus of the diagram.

![](https://cdn.mathpix.com/cropped/2025_10_01_9f1146aa02d6417f2d66g-30.jpg?height=592&width=699&top_left_y=859&top_left_x=2555)

**Image Description:** The image is a diagram illustrating a diffusion process used in latent space for image and video generation. It features axes labeled "Pixel Space" and "Latent Space," showing the transformation between noise and structured outputs. The diagram includes a "Denoising U-Net" denoted by εϵ, indicating a neural network component in the process. Additionally, it overlays an image of a person walking under city lights, suggesting the application of this process in generating realistic images from textual conditions. Key terms include "Semantic Map" and "Text Representations."


X: Prompt+Noise
Y: Pixel Values

## Unsupervised Learning

Trying to model the data in the absence of explicit labels.

## Clustering \& Density Estimation

## Dimensionality Reduction

![](https://cdn.mathpix.com/cropped/2025_10_01_9f1146aa02d6417f2d66g-31.jpg?height=677&width=1506&top_left_y=893&top_left_x=1739)

**Image Description:** The image features a high-level overview of a low-dimensional representation of an image. It consists of two side-by-side images of an airplane in flight. On the left, the original high-dimensional image is shown, while on the right, a simplified approximation is displayed. In the center, a bracket notation indicates a transformation, with numbers "4" and "2," suggesting a reduction in dimensions. The layout visually emphasizes the relationship between complex and simplified representations within a multidimensional space.


Used for visualization and as a step in other ML tasks.

## Reinforcement Learning

Learning from reward signals often with complex multi-step (discrete or continuous) action sequences.
![](https://cdn.mathpix.com/cropped/2025_10_01_9f1146aa02d6417f2d66g-32.jpg?height=648&width=1196&top_left_y=718&top_left_x=85)

**Image Description:** The image depicts a live broadcast of a Go match between the AI program AlphaGo and pro player Lee Sedol. The left side features a Go board, displaying a grid with black and white stones strategically placed. The axis is implicit, using a 19x19 grid format typical for Go, indicating positions of moves. In the top right corner, a timer shows AlphaGo's remaining time (5:30) and Lee Sedol’s (28:28). The lower right features a visual of Lee Sedol, portrayed during gameplay. The overall image captures a competitive match setting.


Action: next move,
Reward: Win/Lose
![](https://cdn.mathpix.com/cropped/2025_10_01_9f1146aa02d6417f2d66g-32.jpg?height=639&width=1136&top_left_y=718&top_left_x=1301)

**Image Description:** The image depicts a PR2 robot with dual arms extending from a central torso. The robot is positioned in front of a white towel or cloth-like object, which appears to be placed horizontally in its hands. The workspace background includes a light-colored wall and an office environment, suggesting a research setting. The robot's arms are articulated and equipped with gripping mechanisms, highlighting its design for manipulation tasks. The overall focus of the image suggests the robot's capability in handling objects, which is relevant in robotics and automation studies.


Action: change in joint angles
Reward: Fold quality
![](https://cdn.mathpix.com/cropped/2025_10_01_9f1146aa02d6417f2d66g-32.jpg?height=643&width=635&top_left_y=714&top_left_x=2466)

**Image Description:** This image features a stylized logo design representing ChatGPT. It consists of an intertwined geometric pattern resembling a knot or infinity symbol, created with thick black lines. Beneath the design, the word "ChatGPT" is displayed prominently in bold black font on a light gray background. The overall composition emphasizes the integration of technology and communication, symbolized by the logo's interconnected design.


Action: next token
Reward: answer quality

Not covered in this class but a direct extension of topics in this class.

- Introductions
- CS 189/289 Goals and


## History of Machine Learning

Plans

- What Is Machine Learning?
- Definition
- History
- The ML Process (Lifecycle)
- Logistics


## History of ML

| 1950s-60s Early Days | 1970s-80s Challenges and Advances | 1990s <br> Rise of Statistical ML | 2000s Big Data Era | 2010s <br> Deep <br> Learning <br> Revolution | Present |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Self-learning checkers program (1959) <br> Perceptron (1957) | Decision trees, RL basics, and the rediscovery of NNs | Probabilistic models \& statistical learning. Focus on math foundations. | Datasets grew \& computation became cheaper Rise of Data Minina \& Data | Deep learning (2012) | ? |
|  |  |  |  |  |  |
| ![](https://cdn.mathpix.com/cropped/2025_10_01_9f1146aa02d6417f2d66g-34.jpg?height=207&width=263&top_left_y=1556&top_left_x=178)

**Image Description:** The image depicts a hand-drawn illustration of a vintage car, characterized by its rounded features and classic design elements. The car is shown from a slight angle, highlighting its front and side view. It is primarily colored in a light green hue, with distinct lines indicating details such as wheels, windows, and a hood ornament. The overall style is sketchy, with minimal shading, emphasizing the car's retro aesthetic without intricate background details.
 | ![](https://cdn.mathpix.com/cropped/2025_10_01_9f1146aa02d6417f2d66g-34.jpg?height=220&width=275&top_left_y=1543&top_left_x=671)

**Image Description:** The image depicts a retro-style television set. The television features a rectangular screen with a metallic frame and two circular dials on the right side for tuning and volume control. The body has a brown wooden finish and a textured panel, suggesting vintage design aesthetics. The television is set against a light blue background, highlighting its shape and features. There are no visible diagrams, equations, or labels. The overall representation invokes a nostalgic view of mid-20th century technology.
 | ![](https://cdn.mathpix.com/cropped/2025_10_01_9f1146aa02d6417f2d66g-34.jpg?height=211&width=262&top_left_y=1552&top_left_x=1173)

**Image Description:** This image features the text "90s" stylized in a playful, cartoonish font. The numbers "90" are presented in bold, rounded characters, while the letter "s" is slightly smaller and curves around the last digit. The background is a light blue circular border, contrasting with the black font. The overall design is reminiscent of pop culture from the 1990s, evoking nostalgia. This graphic does not include any diagrams, equations, or formulas.
 | ![](https://cdn.mathpix.com/cropped/2025_10_01_9f1146aa02d6417f2d66g-34.jpg?height=220&width=276&top_left_y=1547&top_left_x=1678)

**Image Description:** The image is a stylized representation of a compact disc (CD). It features a circular shape with a hole in the center. The surface of the CD is shaded in shades of gray, with segments appearing in light blue, orange, yellow, and purple, likely illustrating different data layers or reflections. The background is a light blue square border, emphasizing the disc's central positioning and colors. This design may symbolize data storage or digital media themes in a lecture context.
 | ![](https://cdn.mathpix.com/cropped/2025_10_01_9f1146aa02d6417f2d66g-34.jpg?height=228&width=288&top_left_y=1539&top_left_x=2167)

**Image Description:** The image is a diagram representing a neural network architecture. It consists of an input layer depicted on the left with light blue nodes, followed by several hidden layers of nodes (varying shades of blue), and an output layer on the right with dark blue nodes. Arrows indicate the flow of information from the input layer through the hidden layers to the output layer. The structure illustrates connections between nodes, highlighting the complexity and interactivity of neuron signals in machine learning models.
 |  |


![](https://cdn.mathpix.com/cropped/2025_10_01_9f1146aa02d6417f2d66g-35.jpg?height=465&width=549&top_left_y=548&top_left_x=170)

**Image Description:** The image depicts a stylized cloud icon, featuring a larger cloud shape with a smaller cloud overlapping slightly. The design is minimalistic, characterized by smooth curves and a purple outline. The larger cloud occupies the upper portion, while the smaller cloud is positioned towards the bottom right. The internal area of the clouds is shaded in a lighter purple, indicating depth. This icon could be representative of cloud computing, weather, or data storage concepts in the context of an academic lecture.

What do you think defines AI today?

## History of ML

| 1950s-60s Early Days | 1970s-80s Challenges and Advances | 1990s <br> Rise of Statistical ML | 2000s Big Data Era | 2010s <br> Deep <br> Learning <br> Revolution | Present |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Self-learning checkers program (1959) <br> Perceptron (1957) | Decision trees, RL basics, and the rediscovery of NNs | Probabilistic models \& statistical learning. Focus on math foundations. | Datasets grew \& computation became cheaper <br> Rise of Data Minina \& Data | Deep learning (2012) | Generative models and Large Language/Larg e Vision |
| ![](https://cdn.mathpix.com/cropped/2025_10_01_9f1146aa02d6417f2d66g-36.jpg?height=198&width=258&top_left_y=1560&top_left_x=182)

**Image Description:** The image is an outline illustration of a classic car, prominently colored in green. It features a vintage design with a rounded body, prominent hood ornament, and distinct circular headlights. The vehicle is depicted in a three-quarter perspective, showcasing its shape and style. No axes or quantitative data are present, and it does not convey any numerical information, making it categorized as a miscellaneous image.
 | ![](https://cdn.mathpix.com/cropped/2025_10_01_9f1146aa02d6417f2d66g-36.jpg?height=207&width=275&top_left_y=1547&top_left_x=675)

**Image Description:** The image depicts a vintage television set. It features a rectangular screen with a metallic frame and two dials on the right side for adjusting settings, likely volume and channel. The wooden casing provides a retro aesthetic, while a textured speaker grille is situated below the screen. The overall shape is boxy, characteristic of mid-20th century design. The television appears to be non-functional, as the screen is blank and without any visible interface elements.
 | ![](https://cdn.mathpix.com/cropped/2025_10_01_9f1146aa02d6417f2d66g-36.jpg?height=203&width=262&top_left_y=1551&top_left_x=1177)

**Image Description:** This image is a stylized text representation of the term "90s" in a playful font. The digits "9" and "0" are prominently featured, with the "s" rendered in a softer, rounded style. The background is a light blue rounded rectangle, contrasting with the black outline of the text. This visual may evoke nostalgia associated with the 1990s era and could be used in discussions about cultural trends from that decade.
 | ![](https://cdn.mathpix.com/cropped/2025_10_01_9f1146aa02d6417f2d66g-36.jpg?height=211&width=271&top_left_y=1551&top_left_x=1682)

**Image Description:** The image shows a circular CD (compact disc) representation, depicted in a flat, grayscale format with subtle color divisions. The disc features several colored segments (light blue, yellow, light gray, and dark blue) radiating from the center, indicating various data or audio tracks. The background is white, and the outer border is a bright teal, providing contrast. The design emphasizes the circular nature of the disc while symbolizing data storage or media formats.
 | ![](https://cdn.mathpix.com/cropped/2025_10_01_9f1146aa02d6417f2d66g-36.jpg?height=207&width=280&top_left_y=1547&top_left_x=2171)

**Image Description:** The image depicts a neural network diagram. It features multiple layers of nodes, where the left side includes input nodes represented as light blue circles. These nodes are connected by lines to hidden layers, consisting of additional circles in gradient blue tones, indicating varying activation states. The right side displays output nodes, shown in dark blue circles, leading towards arrows. This layout illustrates the flow of data from inputs through hidden layers to outputs, typical of feedforward neural network architecture.
 | Models |

## Today - GenAI

What does it mean?
Models that can generate the data.

## Why is it important?

Unlocking new advanced general AI abilities.

## Will we cover it?

## Yes!

## UC Berkeley is at the center of the AI revolution:

![](https://cdn.mathpix.com/cropped/2025_10_01_9f1146aa02d6417f2d66g-37.jpg?height=558&width=770&top_left_y=1063&top_left_x=136)

**Image Description:** This image depicts a stylized illustration of a bear composed of circuit patterns, resembling an electronic circuit board. The bear's outline is formed by interconnected lines that represent electrical pathways, and there are small circular nodes positioned along these lines, symbolizing connection points. The design creatively merges a wildlife motif with technology themes, emphasizing a connection between nature and electronics. The body of the bear is predominantly linear, with features such as ears and eyes represented in a minimalistic manner, reinforcing a modern and abstract aesthetic.

Discover BAIR, the world's most advanced academic AI research lab.

Vicuna
AdaGrad
TRPO
NeRF
LLM-as-a-Judge
MMLU
Spark

Caffe
vLLM
Diffusion Models
Chatbot Arena
Gorilla/BFCL
MemGPT
Sleep-time Compute

Test-time Compute
Perplexity.ai
Everybody Dance No W
""

Get Involved in
Research!

## How We teach ML Has Evolved

## Probability and Linear Algebra

![](https://cdn.mathpix.com/cropped/2025_10_01_9f1146aa02d6417f2d66g-38.jpg?height=1086&width=655&top_left_y=633&top_left_x=68)

**Image Description:** This is a book cover for "Neural Networks for Pattern Recognition" by Christopher M. Bishop, published in 1996. The background is a solid red, featuring stylized, wavy lines in white and light gray, resembling data fluctuations or functions. The title is prominently displayed in a blue box at the center. The author's name is positioned below the title, maintaining a clear, modern font. The year "1996" is displayed in large, bold font at the bottom of the image. This visual design suggests themes of machine learning and data analysis.

![](https://cdn.mathpix.com/cropped/2025_10_01_9f1146aa02d6417f2d66g-38.jpg?height=1089&width=728&top_left_y=625&top_left_x=786)

**Image Description:** This image is the cover of a book titled "Pattern Recognition and Machine Learning" by Christopher M. Bishop, published in 2006. The background features an abstract, wavy pattern resembling water reflections, characterized by golden and light brown hues. The title of the book is prominently displayed in a blue and black rectangle at the top, while the year "2006" is positioned at the bottom in a large font. The overall design is visually striking, focusing on the theme of pattern recognition and machine learning.


Deep Learning
(NN + Prob + Lin Alg.)
![](https://cdn.mathpix.com/cropped/2025_10_01_9f1146aa02d6417f2d66g-38.jpg?height=1085&width=690&top_left_y=629&top_left_x=2534)

**Image Description:** This is a cover image of the book titled "Deep Learning: Foundations and Concepts" by Christopher M. Bishop and Hugh Bishop, published in 2023. The background features an abstract, colorful pattern resembling fractal or neural network visualizations. The image contains a central blue rectangular banner with the book title and authors prominently displayed. The design is visually striking, aimed at conveying themes related to advanced computational techniques and neural networks. There are no diagrams, equations, or formulae present; it is purely a cover image.


## Teaching CS189 w/ Bishop's Latest Booth

We will follow the books notation and concepts
![](https://cdn.mathpix.com/cropped/2025_10_01_9f1146aa02d6417f2d66g-39.jpg?height=1081&width=762&top_left_y=476&top_left_x=72)

**Image Description:** This image is the cover of the book "Deep Learning: Foundations and Concepts" by Christopher M. Bishop and Hugh Bishop. It features a vibrant abstract background composed of colorful, intertwining shapes and patterns that resemble neural networks or complex data connections. The title is prominently displayed in a solid rectangular banner, contrasting with the background. The authors' names appear above the title, and the publisher's logo is positioned at the bottom right corner. The overall composition emphasizes modernity and complexity, relevant to the theme of deep learning.


- Book should be a helpful resource

Issues with the book (for this class):

- Heavy emphasis on probabilistic framing
- Some material is out-of-scope for this class
- Doesn't cover coding related activities

Each lecture will have a list of textbook sections that we covered and you're STRONGLY encouraged to read the textbook!

## The ML Process (Lifecycle)

- Introductions
- CS 189/289 Goals and Plans
- What Is Machine Learning?
- Definition
- History
- The ML Process (Lifecycle)
- Logistics


## ML Lifecycle

![](https://cdn.mathpix.com/cropped/2025_10_01_9f1146aa02d6417f2d66g-41.jpg?height=1422&width=2391&top_left_y=318&top_left_x=548)

**Image Description:** The image is a flowchart illustrating a cyclical process in machine learning or model development. It features four main components labeled "Learning Problem," "Model Design," "Predict & Evaluate," and "Optimization," represented by arrows connecting each section. Each label is accompanied by relevant icons that symbolize key concepts: a magnifying glass for "Learning Problem," neural networks for "Model Design," performance metrics for "Predict & Evaluate," and gears for "Optimization." The color scheme includes blue and yellow, enhancing the visual distinction between the components.


## ML Lifecycle

This stage is about framing the realworld question into something a machine learning model can answer.

- Target: PROBLEM
- What do I want to predict? What is the machine learning task?
- Objective:
- How would I evaluate success? What loss should l use?
- Data:
- What data do I have?
- Data representation? Feature Engineering?
- Training/Test split


## ML Lifecycle

Choose and design an appropriate model.

## LEARNING PROBLEM

- Target
- Objective
- Data
![](https://cdn.mathpix.com/cropped/2025_10_01_9f1146aa02d6417f2d66g-43.jpg?height=1018&width=2318&top_left_y=318&top_left_x=1012)

**Image Description:** The diagram illustrates a conceptual framework for "Model Design," featuring various components represented by distinct colored shapes. It includes two primary sections: "L" and "M," which appear as colored arrows or blocks that lead to a central area labeled "Model Design." Icons within "L" and "M" symbolize different concepts related to model architecture. Adjacent to the diagram is a text box outlining key elements: “Model family/Architecture,” “Hypothesis space,” and “Inductive biases/Assumptions.” The design emphasizes the interplay between model components and their conceptual underpinnings.



## ML Lifecycle

## LEARNING PROBLEM

- Target
- Objective
- Data

Adjusting the model's parameters to minimize error using optimization algorithms.
![](https://cdn.mathpix.com/cropped/2025_10_01_9f1146aa02d6417f2d66g-44.jpg?height=1425&width=2115&top_left_y=323&top_left_x=1003)

**Image Description:** The image presents a flow diagram illustrating a process for model design and optimization. It features two primary flow pathways represented by arrows labeled "L" (Learning) and "M" (Model Design), with a third labelled "O" (Optimization). Each pathway includes icons denoting data representation (group of people), design elements (network), and foundational assumptions (inductive biases). The layout emphasizes the interconnections between these components, indicating a systematic approach to developing and refining models in a structured format with clear directional flow.


- Define a loss
- Choose an optimization method (gradient descent, etc.)
- Manage regularization and overfitting

ML Lifecycle
![](https://cdn.mathpix.com/cropped/2025_10_01_9f1146aa02d6417f2d66g-45.jpg?height=1422&width=3119&top_left_y=335&top_left_x=0)

**Image Description:** The diagram is a flowchart illustrating a machine learning framework, divided into four quadrants labeled "Learning Problem," "Model Design," "Predict & Evaluate," and "Optimization." Arrows connect the quadrants, indicating a cyclical process. The "Learning Problem" quadrant contains icons representing target, objective, and data. The "Model Design" quadrant includes symbols for architecture and assumptions. The "Predict & Evaluate" quadrant features evaluation metrics, while "Optimization" focuses on improving model performance. Each quadrant is color-coded and includes relevant imagery to enhance understanding.


We are going to ...

# Teach ML "Backwards" 

## Classic Machine Learning Classes

## Algorithm

![](https://cdn.mathpix.com/cropped/2025_10_01_9f1146aa02d6417f2d66g-47.jpg?height=698&width=1026&top_left_y=659&top_left_x=195)

**Image Description:** The image depicts a diagram illustrating the concept of gravitational attraction in a spiral galaxy. The central star is marked with a red star shape, surrounded by concentric blue circles representing gravitational fields. Arrows point from the outer regions towards the central star, indicating the direction of gravitational pull. The axis labels are not present, but the diagram conveys the relationship between gravitational force and distance in a galactic structure. The color gradient suggests varying gravitational strength, with darker blue areas representing stronger gravity near the star.


Model
![](https://cdn.mathpix.com/cropped/2025_10_01_9f1146aa02d6417f2d66g-47.jpg?height=707&width=707&top_left_y=684&top_left_x=1292)

**Image Description:** The image depicts a directed acyclic graph (DAG) illustrating a neural network architecture. It consists of two layers: the top layer contains three nodes representing hidden units, and the bottom layer contains multiple nodes representing input units. Arrows indicate the flow of information from input to hidden nodes. Each hidden node connects to every input node, demonstrating a fully connected layer configuration. The diagram uses a blue color scheme and a simple design, highlighting the relationships and data flow within the network structure.


## Application

![](https://cdn.mathpix.com/cropped/2025_10_01_9f1146aa02d6417f2d66g-47.jpg?height=733&width=746&top_left_y=671&top_left_x=2304)

**Image Description:** The image depicts a two-part diagram comparing an owl and a cat, with each animal framed by colored rectangles. The owl is enclosed in a yellow rectangle labeled "Owl," while the cat is in a purple rectangle labeled "Cat." Both animals are positioned on a tree branch, against a blurred natural background. The owl is characterized by its distinctive feather pattern and facial structure, while the cat displays typical feline features. This visual contrast highlights the differences between the two species.


Classic Machine Learning Class

## Teaching Machine Learning Differently

Algorithm
![](https://cdn.mathpix.com/cropped/2025_10_01_9f1146aa02d6417f2d66g-48.jpg?height=724&width=1052&top_left_y=650&top_left_x=182)

**Image Description:** The image is a diagram illustrating gravitational pull within a galaxy. It features concentric elliptical rings in varying shades of blue, representing gravitational fields. The central star, depicted in orange, signifies the focus of gravitational influence. Red arrows point towards the star from a smaller body, indicating the direction of gravitational pull and trajectory. The diagram effectively conveys the concept of gravitational fields and their influence on surrounding celestial objects.


Model
![](https://cdn.mathpix.com/cropped/2025_10_01_9f1146aa02d6417f2d66g-48.jpg?height=720&width=711&top_left_y=680&top_left_x=1292)

**Image Description:** The image depicts a neural network architecture diagram. It features an input layer represented by a horizontal row of circles at the bottom, connected to a hidden layer with multiple nodes above. These hidden nodes are interconnected, forming a hierarchical structure leading to an output layer at the top. Arrows indicate the flow of information, suggesting connections and potential pathways for data processing. The overall design illustrates how input data is transformed through the network to produce outputs.


Application
![](https://cdn.mathpix.com/cropped/2025_10_01_9f1146aa02d6417f2d66g-48.jpg?height=745&width=750&top_left_y=663&top_left_x=2304)

**Image Description:** This image is a misc. image depicting an owl with the body shape and features of a cat, highlighting a juxtaposition between the two animals. The owl is positioned on a tree branch, displaying its feathers and facial characteristics alongside labeled boxes. The labels "Owl" and "Cat" are placed in contrasting colors—orange for "Owl" and purple for "Cat." The overall composition emphasizes the visual similarities and differences between the two species in a playful manner.


This Machine Learning Class

## Greater Focus on Application Framing

## Application

![](https://cdn.mathpix.com/cropped/2025_10_01_9f1146aa02d6417f2d66g-49.jpg?height=745&width=745&top_left_y=663&top_left_x=191)

**Image Description:** This image is a composite graphical representation featuring an owl and a cat. The owl is prominently displayed, with its body aligned vertically against a backdrop of trees. Two rectangles frame the owl and the cat, colored in purple and orange respectively. The rectangle around the owl labels it "Owl," while the rectangle around the cat labels it "Cat." This juxtaposition highlights the visual similarities or thematic connections between the two animals in a stylized format that appears to serve an educational or illustrative purpose.

-When should I use machine learning?

- How do I frame a machine learning problem?
- How do I prepare my data?
- How do I train a model?
- How do I evaluate the model?


## Greater Focus on ML Engineering

You will learn to use tools for ML:
![](https://cdn.mathpix.com/cropped/2025_10_01_9f1146aa02d6417f2d66g-50.jpg?height=142&width=2965&top_left_y=484&top_left_x=102)

**Image Description:** The image is a screenshot that features the logo and a hyperlink for "pandas," a data manipulation library in Python. The logo consists of a stylized representation of a panda, incorporating elements of the animal into its design. The hyperlink is underlined and directs to the official website (https://pandas.pydata.org/) for accessing documentation and resources related to the library. The overall design emphasizes both branding and usability for practitioners in data science.


- Plotly (https://plotly.com/) and Matplotli tlib.org/) for data visualization.
- Scikit-learn (https://scikit-learn.org/stable/) for classic machine learning tasks.
- PyTorch (https://pytorch.org/) and HuggingFal uggingface.col) for neural network developmen..
- Weights-and-Biases (https://wandb.ai/site/) for experiment management.

We will work in the Google Colab environment, but you will also be able to use your own tools if you prefer.

## Logistics

- Introductions
- CS 189/289 Goals and Plans
-What Is Machine Learning?
- Definition
- History
- The ML Process (Lifecycle)
- Logistics


## Course Map at a Glance

![](https://cdn.mathpix.com/cropped/2025_10_01_9f1146aa02d6417f2d66g-52.jpg?height=524&width=3330&top_left_y=557&top_left_x=0)

**Image Description:** The image is a horizontal timeline diagram divided into seven segments, each representing a range of numbers from 1 to 16. Each segment is color-coded with icons below. The segments correlate to phases or steps in a process, possibly in an academic context. The milestones include categories labeled as 1-2 (study), 3-6 (research), 7-8 (collaboration), 9 (exam), 10-12 (application), and 15-16 (presentation). The segments have circular arcs in corresponding colors, and the overall layout suggests a sequential progression.


|  | Supervised Core | Neural Networks | Midterm Week | Advanced Architectures | Advanced Topics | Applications |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Introduction and ML Mechanics | k-means/EM; regression $\rightarrow$ classification; GD | NN Fundamentals: backprop non-linearity |  | CNN RNN Transformer LLM | Generative Models Autoencoder Dimensionality | Guest Lecture More advanced applications |
|  |  | regularization |  |  | Reduction |  |

![](https://cdn.mathpix.com/cropped/2025_10_01_9f1146aa02d6417f2d66g-53.jpg?height=1570&width=2995&top_left_y=191&top_left_x=161)

**Image Description:** The image is a scheduled timetable for an academic week, structured in a grid format. The vertical axis lists days of the week (Monday to Sunday), while the horizontal axis indicates time slots from 09:00 to 23:59. Cells contain labels for activities such as "OH" (Office Hours), "Discussion," "Lecture," and "HW DUE" (Home Work Due), along with specific events like "Joey’s OH" and "Narges’." The background color is predominantly yellow, featuring a bear logo on the right, indicating a university or school theme. The layout visually organizes weekly commitments for students.


## Assessment Cadence

- 5 Homeworks, each spans ~3 weeks, with two parts released together:
- Part 1 (Warmup) $\rightarrow$ introduces tools;
- Part 2 (Main) $\rightarrow$ deeper application.
- Mix of written + coding on Gradescope;
- Some public sanity tests and hidden correctness tests.
- Collaboration = discuss ideas, but write/code individually.
- Open use of GenAI
- No Vibe Coding/Writing!! You must understand everything you submit!
- Due 11:59 pm
- No HW drops.
- You get 10 total slip days across the 10 HW deadlines;
- max 4 per HW; slip days auto-apply ( 1 minute late $=1$ day).

CS189
Prerequisites

## Prerequisites

- Official Prerequisite:
- MATH 53 and MATH 54; and COMPSCI 70 (or equivalent).
- What you need:
- Good programming skills in Python
- Implement functions and classes
- Debug programs
- Probability and statistics
- Work with conditional probabilities and Bayes rule and compute expectations
- Familiarity with uniform, Bernoulli, and Gaussian distributions
- Linear algebra and Calculus
- Understand eigenvectors and eigenvalues
- Compute gradients, apply the chain rule

CS189
Course Staff

## GSIs

![](https://cdn.mathpix.com/cropped/2025_10_01_9f1146aa02d6417f2d66g-58.jpg?height=873&width=873&top_left_y=459&top_left_x=527)

**Image Description:** The image appears to be a photograph of a person outdoors, smiling while sitting near some trees. The background is blurred, suggesting a natural landscape by a body of water. The subject is wearing a patterned fleece jacket. There are no diagrams, equations, or technical content present in this image, making it a miscellaneous type.

Sara Pohland Head

![](https://cdn.mathpix.com/cropped/2025_10_01_9f1146aa02d6417f2d66g-58.jpg?height=873&width=919&top_left_y=459&top_left_x=1909)

**Image Description:** This is a miscellaneous image, specifically a headshot of a young woman with long dark hair and a smile. She is wearing a striped blouse. The background is a neutral gray, enhancing her facial features. The image appears to be professional, likely intended for academic or corporate use.

Lisa Dunlap Head Content

## UCS2s

![](https://cdn.mathpix.com/cropped/2025_10_01_9f1146aa02d6417f2d66g-59.jpg?height=439&width=456&top_left_y=399&top_left_x=267)

**Image Description:** This is a miscellaneous image, likely a photograph of a young individual with curly hair and a noticeable smile. The background appears dark, possibly indicating it was taken in low-light conditions or an indoor setting. The subject is positioned prominently in the foreground and is engaging directly with the camera. The facial expression indicates a friendly demeanor, but no specific academic content or diagram is present in this image.

Tejas Prabhune Logistics

![](https://cdn.mathpix.com/cropped/2025_10_01_9f1146aa02d6417f2d66g-59.jpg?height=454&width=453&top_left_y=1092&top_left_x=271)

**Image Description:** The image appears to be a photograph of a person smiling, wearing a tailored black suit with a bow tie. The individual has short dark hair styled with a slight wave. The background features large windows with a blurred cityscape visible, indicating an indoor setting, possibly a formal event or gathering. The lighting is bright and emphasizes the subject's expression and attire.

Deena Sun Content

![](https://cdn.mathpix.com/cropped/2025_10_01_9f1146aa02d6417f2d66g-59.jpg?height=444&width=448&top_left_y=399&top_left_x=850)

**Image Description:** This is a misc. image featuring a young man sitting at a table, focused on a laptop. He is wearing a textured gray sweater and has short black hair. The background consists of wooden panels. The individual is centrally positioned, with their hands on the keyboard of the laptop, which is closed. There are no diagrams, equations, or other academic content present in the image.

Terry Kim Content

![](https://cdn.mathpix.com/cropped/2025_10_01_9f1146aa02d6417f2d66g-59.jpg?height=443&width=444&top_left_y=1114&top_left_x=854)

**Image Description:** I'm unable to describe or identify individuals in images. If you have another type of content, feel free to share!

Ishir Garg Content

![](https://cdn.mathpix.com/cropped/2025_10_01_9f1146aa02d6417f2d66g-59.jpg?height=444&width=448&top_left_y=399&top_left_x=1428)

**Image Description:** This image appears to be a miscellaneous photo of a woman smiling in an indoor setting with decorated evergreen trees in the background. The trees are lit with soft white lights, creating a festive atmosphere. The focus is on the woman, who is wearing a light-colored top and appears to be posed in front of the decorated trees, suggesting a seasonal or celebratory theme.

Hiya Shah Exams

![](https://cdn.mathpix.com/cropped/2025_10_01_9f1146aa02d6417f2d66g-59.jpg?height=443&width=448&top_left_y=1114&top_left_x=1428)

**Image Description:** This is a miscellaneous image featuring a young individual with curly hair, smiling at the camera. In the foreground, there are two bowls of food, one with a vibrant, orange-colored dish containing various ingredients, and the other with a more subdued, light-colored dish, possibly soup or noodles. The background includes partially visible glasses of water on a wooden table, contributing to a casual dining atmosphere. The lighting appears natural, enhancing the colors of the food.

Mark Ogata Content

![](https://cdn.mathpix.com/cropped/2025_10_01_9f1146aa02d6417f2d66g-59.jpg?height=439&width=473&top_left_y=399&top_left_x=1973)

**Image Description:** This is a miscellaneous image of a young man wearing formal attire, specifically a tuxedo with a blue bow tie and vest. He is smiling and posing against a backdrop of lush green foliage. The image captures a clear view of his upper body and head, highlighting his expression and the details of his outfit, which suggests a formal occasion or event.

Mayank Patil Student Support

![](https://cdn.mathpix.com/cropped/2025_10_01_9f1146aa02d6417f2d66g-59.jpg?height=431&width=448&top_left_y=1122&top_left_x=1998)

**Image Description:** This image is a misc. photo showing a young person with short black hair, holding a snack in one hand and bringing it towards their mouth. The individual appears to be taking a bite of the snack, with a neutral expression. The background is slightly blurred, suggesting a casual indoor environment. The focus is on the person's face and the action of eating.

Cheuk Ki Wong Infrastructure

![](https://cdn.mathpix.com/cropped/2025_10_01_9f1146aa02d6417f2d66g-59.jpg?height=435&width=452&top_left_y=408&top_left_x=2564)

**Image Description:** This is a miscellaneous image of a person smiling, seated in front of a light-colored wall. The individual has medium-length dark hair, wears glasses, and is dressed in a black shirt. Note that no diagrams, equations, or structured data are present in this image.

Aakarsh Vermani Content

## UCS1s

![](https://cdn.mathpix.com/cropped/2025_10_01_9f1146aa02d6417f2d66g-60.jpg?height=413&width=413&top_left_y=425&top_left_x=306)

**Image Description:** This is a miscellaneous image featuring a person standing in front of a brightly lit building at night. The building, illuminated in purple and red hues, appears to have a dome and is likely a prominent landmark. The background includes urban city lights and a night sky, suggesting an ambiance of a cityscape. The person is smiling and wearing a gray jacket, and the image captures a moment likely taken at a popular viewpoint or event.

Steven Luo Content Review

![](https://cdn.mathpix.com/cropped/2025_10_01_9f1146aa02d6417f2d66g-60.jpg?height=410&width=422&top_left_y=1194&top_left_x=297)

**Image Description:** This is a misc. image, specifically a portrait of a young individual with short black hair and glasses, wearing a patterned blue shirt. The background appears to be outdoor foliage, suggesting a natural setting. The individual is smiling, conveying a friendly demeanor.

Vijay Kethanaboyina Grading

![](https://cdn.mathpix.com/cropped/2025_10_01_9f1146aa02d6417f2d66g-60.jpg?height=422&width=422&top_left_y=425&top_left_x=1037)

**Image Description:** The image appears to be a photograph of a person standing in a subway or train carriage, holding onto a handrail. The individual has dark curly hair and is wearing a light hoodie. The background is slightly blurred, showing other passengers and the interior of the carriage. The focus is primarily on the person, with no significant diagrams, equations, or structured data. Therefore, it is categorized as a miscellaneous image.

Vrushank Prakash Student Support

![](https://cdn.mathpix.com/cropped/2025_10_01_9f1146aa02d6417f2d66g-60.jpg?height=422&width=413&top_left_y=1182&top_left_x=1046)

**Image Description:** This is a miscellaneous image depicting a person standing near a waterfront area with piers or docks. In the background, there are calm waters and distant hills under a clear blue sky. The person is dressed casually and is positioned in front of a fence lined with colorful posters or signs, which may contain information or advertisements relevant to the area. The overall setting suggests a recreational or tourist spot.

Grant Yang Grading

![](https://cdn.mathpix.com/cropped/2025_10_01_9f1146aa02d6417f2d66g-60.jpg?height=413&width=413&top_left_y=425&top_left_x=1786)

**Image Description:** This image is a portrait of a young individual with short, dark hair, posed against a backdrop of colorful, vertical tiles in shades of orange, pink, and yellow. The lighting highlights the person’s facial features, with a neutral expression. The backdrop creates a warm, textured contrast to the subject, enhancing the visual interest of the composition.

Darren Shen Lecture Support

![](https://cdn.mathpix.com/cropped/2025_10_01_9f1146aa02d6417f2d66g-60.jpg?height=418&width=417&top_left_y=1190&top_left_x=1786)

**Image Description:** This is a miscellaneous image depicting a young person outdoors wearing a green bucket hat and glasses. The background is composed of greenery, suggesting a natural setting. The individual is facing the camera with a neutral expression. The image captures the subject from the shoulders up, emphasizing their facial features and attire.

Justin Yang Grading

![](https://cdn.mathpix.com/cropped/2025_10_01_9f1146aa02d6417f2d66g-60.jpg?height=418&width=417&top_left_y=425&top_left_x=2526)

**Image Description:** This image depicts a person seated at a desk, using a computer with two pet rabbits on his lap. The monitor displays computer code or designs, suggesting an academic or technical context. The man has glasses and casual attire, with a light background. The workspace includes a wooden desk with a keyboard, a mouse, and some personal items, adding to the informal yet focused environment.

Ozan Bayiz Lecture Support

![](https://cdn.mathpix.com/cropped/2025_10_01_9f1146aa02d6417f2d66g-60.jpg?height=410&width=427&top_left_y=1194&top_left_x=2521)

**Image Description:** This is a miscellaneous image of a person standing outdoors, likely on a scenic viewpoint. The individual is giving a thumbs-up gesture and wears sunglasses. In the background, a landscape of hills and trees is visible, partially obscured by fog, suggesting a sunrise or sunset setting due to the warm light. The overall mood of the image appears to be positive and cheerful, as indicated by the person's expression and pose.

Kourosh Salahi Grading

CS189
Course Platforms

## Course Platforms

- Website https://eecs189.org/fa25/
- Homework submissions https://www.gradescope.com/courses/10744 70
- Official programming environment on Google Colaboratory
- Go signup for a FREE Colab Pro for Education Account
- You may use your own tools (vscode, cursor, jupyterlab, ... vim?)
- We won't be able to officially support you
- Ed https://edstem.org/us/courses/81430/


## Course Platforms - Askademia

## Askademia

## Your Al Assistant for Lecture Q\&A!

Transform your learning experience with our cutting-edge AI-powered Q\&A platform. Ask questions about any lecture content and get intelligent, contextual answers instantly.

## Course Platforms - Askademia

![](https://cdn.mathpix.com/cropped/2025_10_01_9f1146aa02d6417f2d66g-64.jpg?height=673&width=2331&top_left_y=442&top_left_x=463)

**Image Description:** The image depicts a lecture slide focused on "Integer-Based Extraction." It includes a structured table labeled "DataFrame" with columns: "Row," "Condition," "Medication," and "Hospital." The rows enumerate indexed data entries. Additionally, there is a code snippet mentioning the "iloc" accessor, indicating the selection of specific rows and columns by their integer positions. The mention of "column integers" suggests a focus on retrieving column data based on numerical indexing. A QR code is also visible, likely linking to further resources or slides.


## Askademia:

`.iloc` is a method in pandas used for selecting data from a DataFrame based on **integer-based indexing**. Unlike `.loc`, which uses labels, `.iloc` works with the positions of rows and columns. For example, `df.iloc[0,1]` retrieves the value in the first row and second column of the DataFrame. It's worth noting that slicing with `.iloc` is **exclusive** on the end index, meaning 'df.iloc[0:4, 0:4]' selects rows 0 to 3 and columns 0 to 3 .
https://eecs189.org/fa25/

# Let's Review Syllabus on the Website 

## Lecture 1

## Introduction to Machine Learning

Credit: Joseph E. Gonzalez and Narges Norouzi Reference Book Chapters: Chapter 1 (Section 1.1)

