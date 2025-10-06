---
course: CS 189
semester: Fall 2025
type: lecture
title: ML Mechanics - Techniques
source_type: slides
source_file: Lecture 03 -- ML Mechanics - Techniques.pptx
processed_date: '2025-10-01'
processor: mathpix
---

## Lecture 3

# Machine Learning Mechanics Terminology and Techniques 

Framing ML Problems and ML Techniques and Terminologies

EECS 189/289, Fall 2025 @ UC Berkeley
Joseph E. Gonzalez and Narges Norouzi

# III Join at slido.com <br> '1َيْL \#2826097 

## Goals For Today

- Introduce major concepts in using machine learning.
- Provide a high-level understanding but not rigorous
- We will revisit all the topics more formally later in the semester
- Show you how to do basic machine learning in Python.
- Visit each step of the Machine Learning Lifecycle.
- Prepare you for Homework 1.


## Today We Introduce Scikit-Learn

Widely used python package for:

- Data Prep
- Feature Engineering
- Classic Models
- Evaluation in Machine Learning.
scikit-learn
Machine Learning in Python
Getting Started
Release Highlights for 1.7
- Simple and efficient tools for predictive data analysis
- Accessible to everybody, and reusable in various contexts
- Built on NumPy, SciPy, and matplotlib
- Open source, commercially usable - BSD license

Classification
Identifying which category an object belongs to.
Applications: Spam detection, image recognition. Algorithms: Gradient boosting, nearest neighbors, random forest, logistic regression, and more...
![](https://cdn.mathpix.com/cropped/2025_10_01_11b3b7c6329b6de92897g-04.jpg?height=371&width=635&top_left_y=1105&top_left_x=1241)

**Image Description:** The image is a matrix of plots illustrating the decision boundaries of various machine learning algorithms: SVM (Support Vector Machine), Decision Trees, Random Forests, Neural Networks, and Adaboost. Each plot features blue and red points, representing two distinct classes, with the background color indicating the decision regions. The axes are not labeled, but each plot shows the classification performance metrics (e.g., accuracy) in the corners. The visualizations highlight how different models classify data, with variations in decision boundary shapes and complexity across the algorithms.


Examples
Dimensionality reduction
Reducing the number of random variables to consider.

Applications: Visualization, increased efficiency.
Algorithms: PCA, feature selection, non-negative

Regression
Predicting a continuous-valued attribute associated with an object.

Applications: Drug response, stock prices. Algorithms: Gradient boosting, nearest neighbors, random forest, ridge, and more...

![](https://cdn.mathpix.com/cropped/2025_10_01_11b3b7c6329b6de92897g-04.jpg?height=346&width=626&top_left_y=1126&top_left_x=1939)

**Image Description:** The diagram is a line graph depicting "Predicted average energy transfer during the week." The x-axis represents "Time of the week," spanning from Sunday to Saturday, while the y-axis denotes "Normalized energy transfer," with values ranging from 0 to 1. Three series are plotted: a blue line representing "recorded average," a green line for "max_iter=5," and an orange line for "max_iter=50," all illustrating variations in energy transfer throughout the week. The graph shows oscillating patterns, indicating fluctuations in energy transfer over the specified time period.

Examples

Model selection
Comparing, validating and choosing parameters and models.

Applications: Improved accuracy via parameter tuning.

Clustering
Automatic grouping of similar objects into sets.
Applications: Customer segmentation, grouping experiment outcomes.
Algorithms: k -Means, HDBSCAN , hierarchical clustering, and more...
![](https://cdn.mathpix.com/cropped/2025_10_01_11b3b7c6329b6de92897g-04.jpg?height=320&width=388&top_left_y=1114&top_left_x=2755)

**Image Description:** The image displays a K-means clustering diagram on a PCA-reduced dataset, specifically from the "digits" dataset. It shows multiple colored regions, each representing a distinct cluster of data points (black dots) in a two-dimensional space. The axes are not labeled but indicate the reduced principal components. The white crosses mark the centroids of each cluster. The diagram visually conveys how the data is partitioned into clusters, demonstrating the K-means algorithm's effectiveness in grouping similar data points.


Examples
Preprocessing
Feature extraction and normalization.
Applications: Transforming input data such as text for use with machine learning algorithms.
Algorithms: Preprocessing, feature extraction, and

## Which of these libraries have you used before?

## ML Lifecycle

![](https://cdn.mathpix.com/cropped/2025_10_01_11b3b7c6329b6de92897g-06.jpg?height=1426&width=2399&top_left_y=322&top_left_x=547)

**Image Description:** The image is a flowchart illustrating a cyclical process in machine learning. It consists of four quadrants labeled "Learning Problem," "Model Design," "Optimization," and "Predict & Evaluate." Each quadrant is associated with icons symbolizing their respective functions. Arrows connect the quadrants, indicating the iterative nature of the process. The color scheme includes blue and yellow, enhancing visual engagement. The overall layout suggests a collaborative workflow that highlights the interdependencies among the phases of machine learning development.


## ML Lifecycle

![](https://cdn.mathpix.com/cropped/2025_10_01_11b3b7c6329b6de92897g-07.jpg?height=745&width=1392&top_left_y=323&top_left_x=531)

**Image Description:** The image is a graphic representation of a "Learning Problem." It features a stylized curve that resembles a graph, indicating an upward trend, possibly related to performance or success in learning. A blue arrow points right, symbolizing direction or progression, and there is an icon depicting a chart or graph with data points, emphasizing quantitative analysis. The background is predominantly white, enhancing the visual clarity of the elements. The word "LEARNING PROBLEM" is prominently displayed in bold text, underlining the theme of the slide.


- Target:
- What do I want to predict?
- What is the machine learning task?
- Objective:
- How would I evaluate success?
- What loss should I use?
- Data:
- What data do I have?
- Data representation?
- Training/Test split


## Example: FashionHub

We are launching a new fashion trading website where people can upload pictures of clothing they want to trade.

- We want to automatically tag the clothing into categories based on what sellers upload.
-We have some example clothing pictures with category labels.

What do we want to predict?
Type of Clothing
What data do we have?
Labeled Pairs
How would we evaluate
Overall Accuracy? success?

## ML Lifecycle

- Target:


## LEARNING PROBLEM

- What do I want to predict?
- What is the machine learning task?
- Objective:
- How would I evaluate success?
- What loss should I use?
- Data:
- What data do I have?
- Data representation?
- Training/Test split

Understand the Data! (Look at the data!)

## How to Look at the Data

How much data do you have? ( $N=$ ?)
What are the Features?

- How many dimensions? ( $D=$ ?)
- What is the distribution?
- Are they all numeric?
- Are there missing values?

What are the labels (are there labels)?

- Are they discrete?
- What is the distribution?
- Are there missing labels or errors?
![](https://cdn.mathpix.com/cropped/2025_10_01_11b3b7c6329b6de92897g-10.jpg?height=1302&width=1226&top_left_y=378&top_left_x=2028)

**Image Description:** The image depicts a diagram illustrating a dataset structure, labeled "The Data." It consists of a rectangular matrix divided into two sections. The left section, labeled "Features," is green and represents \(D\) features, with \(N\) instances indicated along the vertical side. The right section, colored orange, contains "Labels (Optional)." The diagram emphasizes the optional nature of labels in relation to the features, visually presenting the organization of data in machine learning contexts.


Look at the data!! (sample, read, plot...)

## Demo

Understanding the
Machine Learning Problem
and
Looking at the Data
![](https://cdn.mathpix.com/cropped/2025_10_01_11b3b7c6329b6de92897g-11.jpg?height=1119&width=1724&top_left_y=76&top_left_x=1326)

**Image Description:** The image displays a grid of clothing items arranged in a 4x5 layout, with various types of apparel and footwear. Each item is placed in a square cell, labeled with its category such as "Ankle boot," "T-shirt/top," "Dress," "Pullover," "Sneaker," or "Coat." The x-axis represents different clothing categories, while the y-axis potentially denotes frequency or another quantitative measure, though numerical values are absent. This visual aids in comparing various clothing items within defined categories, illustrating diversity in styles and types.


Pixel value distribution
![](https://cdn.mathpix.com/cropped/2025_10_01_11b3b7c6329b6de92897g-11.jpg?height=388&width=1829&top_left_y=1369&top_left_x=1331)

**Image Description:** The image is a histogram displaying the distribution of a dataset. The x-axis represents the value range, spanning from 0 to 250, while the y-axis indicates the count of occurrences for each bin, ranging from 0 to over 1000. The bars are filled with a gradient color, representing varying frequencies across the value range. Peaks occur at certain intervals, suggesting clusters of values, while the distribution tails off towards the higher values. It visually summarizes how data points are spread out across the specified range.


## Taxonomy of Machine Learning

![](https://cdn.mathpix.com/cropped/2025_10_01_11b3b7c6329b6de92897g-12.jpg?height=1444&width=3192&top_left_y=416&top_left_x=39)

**Image Description:** The image is a diagram illustrating the classification of machine learning types. It has a tree-like structure with "Supervised Learning" and "Unsupervised Learning" as main branches. 

- **Supervised Learning** branches into "Regression" (showing a line graph) and "Classification" (depicting labeled images of objects). 
- **Unsupervised Learning** divides into "Dimensionality Reduction" and "Clustering," represented with scatter plots. 
- An additional section for "Reinforcement Learning" is included, featuring "Alpha Go." 

Axes are not explicitly labeled, but the context is clear regarding learning paradigms.


## Supervised Learning Learning from examples (demonstrations)

The "training data" consists of examples of a functional relationship.

- Most commonly-used learning process.
- The "fastest" way to learn - requires the least data.

Quantitative Label

- Requires examples of the relationship.
Regression
Classification
- May not always be easy to obtain.

Supervised
Learning
![](https://cdn.mathpix.com/cropped/2025_10_01_11b3b7c6329b6de92897g-13.jpg?height=413&width=613&top_left_y=1437&top_left_x=2088)

**Image Description:** The image is a line graph depicting stock prediction trends. The horizontal axis represents time, while the vertical axis indicates stock price. The graph features a blue line illustrating historical stock performance, with three forecast lines in varying colors (yellow, orange, and red) diverging from the blue line, suggesting potential future price paths. The label "Stock Prediction" is prominently displayed below the horizontal axis, indicating the focus of the diagram on forecasting stock price movements.


![](https://cdn.mathpix.com/cropped/2025_10_01_11b3b7c6329b6de92897g-13.jpg?height=346&width=400&top_left_y=1445&top_left_x=2849)

**Image Description:** The image presents a two-panel format commonly used for instructional or illustrative purposes. The left panel, labeled "Hotdog!" in green, shows an open takeout box containing two hot dogs. The right panel, labeled "Not hotdog!" in red, displays a black sneaker placed within a similar box. Each panel features a "Share" button at the bottom, with a "No thanks" prompt below it. This visual likely serves to differentiate between items in a humorous or educational context, potentially related to a classification task in machine learning.


## Classification Problems

The labels are discrete classes or categories.
![](https://cdn.mathpix.com/cropped/2025_10_01_11b3b7c6329b6de92897g-14.jpg?height=362&width=405&top_left_y=153&top_left_x=2568)

**Image Description:** The image depicts a comparison between two items in a side-by-side format. On the left, there is a hotdog in a box labeled "Hotdog!" with a green checkmark. On the right, there is a black shoe in a similar box labeled "Not hotdog!" with a red cross. Both sections have a "Share" button at the bottom. This image conveys the concept of classification, possibly in the context of machine learning or computer vision, distinguishing between food items and non-food items.


- Example: what type of clothing is in the image

Discrete labels are often "encoded" as numbers (but should be treated as classes)

- Example: $0=$ T-shirt/top, $1=$ Trousers, $2=$ Pullover

Two main types of classification problems:

- Binary Classification: two classes (spam vs. not spam)
- Multi-class Classification: more than two classes (e.g., a food prediction* - Hotdog, Pizza, Burrito, ...).

[^0]
## ML Lifecycle

- Target:


## \section*{LEARNING PROBLEM <br> <br> LEARNING PROBLEM

}![](https://cdn.mathpix.com/cropped/2025_10_01_11b3b7c6329b6de92897g-15.jpg?height=673&width=626&top_left_y=395&top_left_x=1046)

**Image Description:** The image features a stylized graphic related to data analysis. It includes a curved blue shape that resembles an abstract chart, along with a small circle icon depicting a bar graph and a magnifying glass. The curves suggest trends or growth, indicating analysis and interpretation of data. The overall design implies a focus on improving understanding of complex datasets in an academic or professional context. There are no quantitative axes present, as it is not a traditional diagram but a representational image.


- What do I want to predict?
- What is the machine learning task?
- Objective:
- How would I evaluate success?
- What loss should I use?
- Data:
- What data do I have?
- Data representation?
- Training/Test split


## Understan d the Data

Learning

## Train-Test Split

## How will we evaluate the model?

We are about to train a model using data.
How will we know if the model has "learned" from our data?
-Could we measure how well our model fits the data?

> The Exam Analogy:
> What would happen if we gave everyone access to the exam and solutions (the data) to study (train) for the exam?
> - Would everyone do well?
> - Does this mean they learned the material in the class?

Why is memorizing the data (exam) not good?

## Generalization

Generalization in machine learning is the ability of a model to perform well on new, unseen data sampled from the same distribution as its training data.

To evaluate generalization, we need a method to evaluate the model's performance on new data not used for training.
![](https://cdn.mathpix.com/cropped/2025_10_01_11b3b7c6329b6de92897g-17.jpg?height=1183&width=1829&top_left_y=612&top_left_x=1365)

**Image Description:** The image depicts a regression analysis diagram. The main elements include:

- **Axes**: The horizontal axis represents the independent variable (input data), while the vertical axis represents the dependent variable (output data).
- **Data Points**: Violet dots indicate the training data points.
- **Regression Lines**: A yellow curve represents a fitted model, while a green straight line shows a simpler model.
- **Annotations**: Speech bubbles highlight the training data and suggest the potential of a better model for new, unseen data, indicating a comparison of model performance.


## Evaluating generalization using the Train-Test Split

The train-test split is the standard technique we use to evaluate generalization in machine learning:

1. Shuffle the training data
2. Split into two parts:

- Larger Training Part (~80\%): used to develop and train the model.
- Smaller Testing Part (~20\%): used to evaluate generalization performance.
![](https://cdn.mathpix.com/cropped/2025_10_01_11b3b7c6329b6de92897g-18.jpg?height=1094&width=167&top_left_y=663&top_left_x=2075)

**Image Description:** The image is a vertical rectangular block with a blue background. The word "Data" is prominently displayed in large, bold, yellow font, centered in the middle of the block. The text is capitalized, conveying emphasis on the term. This image likely serves as a title or thematic header for a section of an academic presentation related to data or data analysis. The overall design is simple and focuses on visual clarity.

![](https://cdn.mathpix.com/cropped/2025_10_01_11b3b7c6329b6de92897g-18.jpg?height=1094&width=197&top_left_y=663&top_left_x=2942)

**Image Description:** The image features a vertical rectangular partition divided into two segments: the upper section labeled "Train" in large, bold yellow text against a purple background, and the lower section labeled "Test" in large, bold white text on a dark yellow background. The design emphasizes a clear distinction between training and testing phases in a machine learning context, indicating their sequential and separate roles in model development. The layout is simple, focusing on color contrast and typographic hierarchy to convey the concepts effectively.


You should only use the test dataset once after developing and training the model.

If you use the test data to tune the model, the test data no longer measures

Train - Test Split
![](https://cdn.mathpix.com/cropped/2025_10_01_11b3b7c6329b6de92897g-18.jpg?height=201&width=567&top_left_y=1088&top_left_x=2304)

**Image Description:** The image is a simple black arrow pointing to the right. It is designed with a thick, solid body and a triangular head, suggesting directionality. The arrow is placed against a blank, white background, enhancing contrast and visibility. This type of diagram is often used in presentations to indicate progression, causality, or movement from one point to another. Its simplicity makes it easily interpretable in various contexts.

generalization.

The train-test split is the standard technique we use to evaluate generalization in machine learning:

1. Shuffle the training data
2. But what if I want to peek at the test data to tune for better generalization?

## The Validation Split

The validation dataset is used to evaluate generalization performance during the model development process.
![](https://cdn.mathpix.com/cropped/2025_10_01_11b3b7c6329b6de92897g-20.jpg?height=1120&width=1068&top_left_y=688&top_left_x=187)

**Image Description:** The image is a diagram illustrating the concept of a "Train-Test Split" in data analysis. It features a vertical blue section labeled "Data" on the left, with a horizontal arrow pointing towards two adjacent sections on the right, labeled "Train" (in purple) and "Test" (in orange). This split indicates the division of a dataset into two subsets: one for training a model and the other for testing its performance. The diagram emphasizes the importance of separating data to evaluate model accuracy effectively.

![](https://cdn.mathpix.com/cropped/2025_10_01_11b3b7c6329b6de92897g-20.jpg?height=716&width=873&top_left_y=820&top_left_x=1301)

**Image Description:** The image depicts a diagram illustrating a "Train - Val. Split" process in machine learning. It features a vertical rectangular partition with two labeled sections: the left section labeled "Train" in purple and the right section labeled "Val." in orange. An arrow points from the "Train" section to the "Val." section, indicating the separation of training data from validation data during model training. The use of contrasting colors denotes the distinct phases of the machine learning workflow. The design emphasizes the importance of splitting datasets for effective model evaluation.


Fit (train) the model using the training
data.
![](https://cdn.mathpix.com/cropped/2025_10_01_11b3b7c6329b6de92897g-20.jpg?height=218&width=222&top_left_y=1046&top_left_x=2360)

**Image Description:** The image depicts a circular arrow design, representing a loop or continuous cycle. The circular shape is filled with a solid dark blue color. An arrow at the top indicates the direction of movement, suggesting flow or repetition. This graphic is commonly used to symbolize concepts such as feedback loops, iterative processes, or cyclical systems in academic presentations. There are no axes or numerical data within this image, making it a standalone visual without specific quantitative information.


Tune the model design using the validation data.

## The Train-Validation-Test Split Analogy

You can think of the train, validation, and test splits as how you might study for an exam.

Practice question and answer pairs that you use to study for the exam. (Go to discussion!)

Practice exam evaluates if your studying process is working (or if you need to study more).

The exam is how we evaluate if you should pass the class.

## Demo

## Train-Test-Validation Splits

```
    1 # use sklearn to construct a train test split
    2 from sklearn.model_selection import train_test_split
    3
        # Construct the train - test split
        images_tr, images_te, labels_tr, labels_te = train_test_split(
            images, labels, test_size=0.2, random_state=42)
    7
        # Construct the train - validation split
        images_tr, images_val, labels_tr, labels_val = train_test_split(
            images_tr, labels_tr, test_size=0.2, random_state=42)
    12 print("images_tr shape:", images_tr.shape)
    13 print("images_val shape:", images_val.shape)
    14 print("images_te shape:", images_te.shape)
    0.1s
images_tr shape: (38400, 28, 28)
images_val shape: (9600, 28, 28)
images_te shape: (12000, 28, 28)
```


## ML Lifecycle

## LEARNING PROBLEM

- Target
- Objective
- Data
![](https://cdn.mathpix.com/cropped/2025_10_01_11b3b7c6329b6de92897g-23.jpg?height=967&width=2149&top_left_y=314&top_left_x=1007)

**Image Description:** The image is a concept diagram illustrating the process of model design in a machine learning context. It features interconnected arrows in blue and yellow, indicating a flow or progression. The left side includes icons representing feature engineering and model family/architecture, accompanied by text. The axes are implied rather than explicitly defined; rather, the diagram emphasizes relationships and the iterative nature of design. Key terms include "Hypothesis space" and "Inductive biases/Assumptions." The visual elements support an understanding of the model development process.



## Feature Engineering

Feature engineering is the process of selecting and encoding input features from the raw features.

- Selecting Features: including the right features can help improve model performance.
- Adding new features from other data sources can improve performance
- Too many features can be harmful ... when data is limited
- Soon we will explore techniques to automatically select features
- Encoding Features: some features may need to be transformed into the appropriate numeric representation.
- Categorical data, text, images, ... often require transformations
- The core innovation in deep learning is learning features encodings.


## Encoding Numerical Data

Numerical features (numbers) are often used without modification.
However, there are a few important exceptions:

- Categorical Features: ZIP Code, a product SKU, or a numerical coding of a string (e.g., "red" $=1$, "blue" $=2, \ldots$ )
- These are typically one-hot-encoded


## One-Hot Encoding

One-hot Encoding takes a categorical feature (e.g., color, shape) and generates multiple binary features (one for each possible value) and assigns 1 to the feature column with the original value.

| Color |  | Color:Re d | Color:Blu e | Color:Green | Color:Yello w | Color:Missin g |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Red |  |  |  |  |  |  |
| Green |  | 1 | 0 | 0 | 0 | 0 |
| Red |  | 0 | 0 | 1 | 0 | 0 |
| Blue | $\rightarrow$ | 1 | 0 | 0 | 0 | 0 |
| Blue |  | 0 | 1 | 0 | 0 | 0 |
| Yellow |  | 0 | 1 | 0 | 0 | 0 |
| Missing |  | 0 | 0 | 0 | 1 | 0 |
|  |  | 0 | 0 | 0 | 0 | 1 |

## Encoding Numerical Data

Numerical features (numbers) are often used without modification.
However, there are a few important exceptions:

- Categorical Features: ZIP Code, a product SKU, or a numerical coding of a string (e.g., "red" $=1$, "blue" $=2, \ldots$ )
- These are typically one-hot-encoded
- Heavily Skewed Features: click counts, user content, pricing, or other situations where features can have extreme values.
- Often apply log transformations
- Feature Standardization: features with different magnitudes and variability can complicate modeling and optimization
- Typically apply standardization


## Feature Standardization

Using the training data to compute the mean ( $\mu$ ) and variance ( $\sigma^{2}$ ) of the feature $x$ and then apply the following transformation:

$$
z=\frac{x-\mu}{\sigma}
$$
to obtain the new feature $\boldsymbol{z}$ with zero mean and unit variance.
- We will see later that this can improve the model fitting process

Note: The same transformation must be applied at test-time using the mean ( $\mu$ ) and variance ( $\sigma^{2}$ ) from the training data.

## Encoding Text Data

There are several methods for encoding a string of text:

- One-hot-Encoding: For categorical strings like color, state, name, etc. we often use one-hot-encodings.
- Bag-of-Words Encoding: The classic method for encoding multi-word strings (e.g., email, messages, etc.) is to use a bag-of-words.
- Demonstrate in a moment.
- Learned Vector Embeddings: Today we often use large language models to convert strings of text to fixed vectors.
- We will see how to do this and how these methods encode text as tokens later in the course.
All techniques produce a high-dimensional vector representation of a string.


## Bag-of-words Encoding

Each word in the vocabulary is encoded as a separate column and the occurrence or count of that word is stored in each column:

"Learning about machine learning is fun."
![](https://cdn.mathpix.com/cropped/2025_10_01_11b3b7c6329b6de92897g-30.jpg?height=473&width=1489&top_left_y=756&top_left_x=1501)

**Image Description:** The image is a diagram representing a vector in a high-dimensional space, commonly used in machine learning or natural language processing. The vector is depicted with a horizontal axis labeled with various words (e.g., "aardvark," "aardwolf," "fun," "learning," "machine," "zyzzyva"). Each word corresponds to an index in the vector, with integer values (e.g., 0, 1, 2) indicating the vector's representation in a feature space. The diagram visually emphasizes the distribution of values across the dimensions, highlighting which words have active representations in the vector format.


Stop words (e.g., the, is, of...) that contain minimal information are often dropped from the vocabulary.

In graduate school, Prof. Gonzalez moved into a new building that had no art.

So, he secretly installed this "art piece" on the ML floor of the new building.

Do you see the stop word?

There used to be a dustbin and broom ... but the janitors got confused ...

New buildings need ML inspired art!
![](https://cdn.mathpix.com/cropped/2025_10_01_11b3b7c6329b6de92897g-31.jpg?height=1875&width=1469&top_left_y=0&top_left_x=1462)

**Image Description:** The image depicts a conceptual installation featuring a transparent bag filled with brightly colored plastic letters, suspended from the ceiling. Below, a square tile surface is scattered with black plastic letters, some oriented upright while others lay flat. The wall behind is a muted beige, contrasting with the vibrant colors of the letters, suggesting a theme of chaos versus order in language or communication. The arrangement emphasizes the randomness of language fragments, inviting interpretation related to linguistic concepts or artistic expression.


## Encoding Image Data

Images can be thought of as 3 dimensional tensor

- Flatten the Image Tensor: Probably the most naïve representation but can be effective for smaller monochromatic images. (homework 1)
- Transformations: Color space and pixel normalization
- Hand Craft Features: Edge detectors, texture descriptors
![](https://cdn.mathpix.com/cropped/2025_10_01_11b3b7c6329b6de92897g-32.jpg?height=996&width=1047&top_left_y=4&top_left_x=2211)

**Image Description:** The image depicts a multi-dimensional array structure, likely representing a convolutional neural network's input data format. It includes three axes labeled "Height," "Width," and "Color Channel," suggesting a three-dimensional matrix where each cell is a pixel in an image. Various colored squares (red, green, blue) within the grid signify different values in the color channel. An arrow pointing to a one-dimensional representation labeled "Flatten" indicates the process of transforming the multi-dimensional array into a single vector for further processing. A QR code is also present in the upper right corner.

- Deep Learning Representations: Use neural networks to learn embeddings
- More on this later in the course

All techniques produce a high-dimensional vector representation of an image.

## Featurization with Scikit-Learn

Scikit-Learn has a large collection of data transformations to aid in
7.1. Pipelines and composite estimators the feature engineering process
7.1.1. Pipeline: chaining estimators
7.1.2. Transforming target in regression
7.1.3. FeatureUnion: composite feature spaces
7.1.4. ColumnTransformer for heterogeneous data
7.1.5. Visualizing Composite Estimators
7.2. Feature extraction

Many of them have the form:
7.2.1. Loading features from dicts
7.2.2. Feature hashing
7.2.3. Text feature extraction
7.2.4. Image feature extraction
![](https://cdn.mathpix.com/cropped/2025_10_01_11b3b7c6329b6de92897g-33.jpg?height=812&width=3142&top_left_y=1063&top_left_x=171)

**Image Description:** The image depicts a code snippet related to data preprocessing in Python using the `sklearn` library. It showcases the use of the `OneHotEncoder` class. The code consists of three lines: the first initializes the encoder, the second fits it to the "color" column of a DataFrame `df`, and the third transforms this column. The accompanying text highlights the purpose of each line. On the right, there's a visible link to the Scikit-learn documentation. The slide aims to illustrate data transformation techniques in machine learning.


## Demo

Feature Engineering

```
def featurizer(images):
    flattened = flatten(images)
    return image_scaler.transform(flattened)
X_tr = featurizer(images_tr)
```

Variance image
![](https://cdn.mathpix.com/cropped/2025_10_01_11b3b7c6329b6de92897g-34.jpg?height=456&width=116&top_left_y=927&top_left_x=2143)

**Image Description:** The image is a vertical gradient bar representing a scale, likely for displaying intensity or a quantitative measure. The bar transitions from dark (150) at the top to light (0) at the bottom. The numbers on the right indicate values corresponding to the gradient, suggesting it could visualize data such as temperature, pressure, or concentration levels, with 150 being the maximum value and 0 the minimum.

![](https://cdn.mathpix.com/cropped/2025_10_01_11b3b7c6329b6de92897g-34.jpg?height=520&width=499&top_left_y=901&top_left_x=2517)

**Image Description:** The image appears to be a pixelated or blurred visual representation of a neural network output, likely depicting a segmentation or classification result. It features a grid layout with the x-axis ranging from 0 to 20 and the y-axis from 0 to 30. The color gradient suggests intensity values, with variations in shades indicating areas of potential object identification within the input data. The overall structure resembles an abstract or deconstructed form, possibly illustrating features extracted from an image dataset.


## ML Lifecycle

## LEARNING PROBLEM

- Target
- Objective
- Data
![](https://cdn.mathpix.com/cropped/2025_10_01_11b3b7c6329b6de92897g-35.jpg?height=963&width=2136&top_left_y=318&top_left_x=1020)

**Image Description:** The image is a schematic diagram illustrating the relationship between "Learning Algorithm" and "Model Design." It features colored arrows, with blue and yellow sections indicating different components of the process. The left section shows icons representing data and feature engineering, while the right outlines model architecture. Text elements highlight aspects like "Feature Engineering," "Model/Architecture," "Hypothesis Space," and "Inductive Biases/Assumptions." Overall, it emphasizes the flow and interaction between these key areas in machine learning.



## Feature Engineerin <br> g

## Model Family

## Machine Learning as Function Approximation

![](https://cdn.mathpix.com/cropped/2025_10_01_11b3b7c6329b6de92897g-36.jpg?height=801&width=2289&top_left_y=582&top_left_x=459)

**Image Description:** The diagram depicts a conceptual model illustrating a function \( h_w \) that processes an input \( X \) to produce an output \( Y \). The input is represented as a green irregular shape labeled "Input (X)," while the output is shown as a circular orange shape labeled "Output (Y)." An arrow directed from the input to the output indicates the transformation or mapping facilitated by the function \( h_w \). The focus is on the relationship between inputs and outputs in a machine learning context.


## Machine Learning as Function Approximation with (Learned) Features

$$
h_{w}=g_{w_{2}} \circ f_{w_{1}}
$$
$g_{W}$
Output
Feature (Y)
(Learned?)
Featurizatio Vectors

Learned
Function

## Generic Function Approximation

Enables the development of general model families for $g_{w_{2}}$ along with corresponding training algorithms.

- The model family determines the form of the function $g_{w_{2}}$, as well as the output space (e.g., classification or regression).
- The hypothesis space is the space of all possible models in the model family.
- Model families are characterized by the inductive biases they introduce and the complexity they can represent.
![](https://cdn.mathpix.com/cropped/2025_10_01_11b3b7c6329b6de92897g-38.jpg?height=800&width=1583&top_left_y=1029&top_left_x=1739)

**Image Description:** The image is a diagram illustrating the relationship between feature vectors and output in a machine learning context. It consists of three key elements: a yellow circle labeled "Feature Vectors," connected by an arrow to an orange oval labeled "Output (Y)." Below the feature vectors, a blue rectangle states "Learned Function," indicating the process mapping feature vectors to output. The arrow represents the transformation performed by the learned function from feature vectors to produce the output. The diagram visually communicates the flow of data and the learning process in machine learning models.



## The Linear Regression Model Family

The linear regression model family has the form:

$$
g_{w}(\mathbf{x})=w_{0}+\sum_{d=1}^{D} w_{d} x_{d}
$$

Different choices of the weight vector $w$ produce different models.

- Example:
- $g_{[1,1]}(\mathbf{x})=1+x_{1}$
- $g_{[1,-1]}(\mathrm{x})=1-x_{1}$
- $g_{[2,2]}(\mathrm{x})=2+2 x_{1}$

The hypothesis space is all lines parametrized by realvalued slopes and intercepts.

## Inductive Biases

The set of assumptions made in the model design to enable generalization beyond the training data.

Example:
Training data: $\{(-1,1),(1,1)\}$
Test Point: ( $x=0, y=$ ?)
Infinitely many models fit the data

- Any of them could be correct!

Choosing the model family (e.g., linear) is introducing an inductive bias.
No Free Lunch Theorem in ML: there is no universally best model - need to choose the model with the right inductive biases.

## Inductive Biases in Features and the Model

$$
h_{w}=g_{w_{2}} \circ f_{w_{1}}
$$

Input (X) $f_{W_{1}}$
$g_{W}$
Output
Feature (Y) Vectors

Featurizatio n n

Feature engineering introduces inductive biases in $h_{w}$.
$=$
What inductive bias assumptions are made with the bag-of-words encoding?

## Complexity: <br> Linear Models vs Non-Linear Models

Regression
![](https://cdn.mathpix.com/cropped/2025_10_01_11b3b7c6329b6de92897g-43.jpg?height=703&width=1234&top_left_y=531&top_left_x=34)

**Image Description:** The image is a two-dimensional graph depicting a comparison between linear and non-linear relationships. The x-axis represents the independent variable, while the y-axis represents the dependent variable. A straight orange line labeled "Linear" illustrates a linear relationship, characterized by constant slope. A purple curve labeled "Non-Linear" represents variable slopes, indicating changing relationships between the variables. Black dots on the graph suggest data points that may align with either relationship type. The overall design emphasizes the distinction between linear and non-linear models in data interpretation.


Classification (Decision Boundary)
![](https://cdn.mathpix.com/cropped/2025_10_01_11b3b7c6329b6de92897g-43.jpg?height=668&width=2004&top_left_y=527&top_left_x=1309)

**Image Description:** The image presents two diagrams comparing linear and non-linear decision boundaries in a 2D feature space. The left diagram shows a linear decision boundary (indicated by a yellow line) separating two classes, represented by '+' (positive class) and '−' (negative class) markers. The right diagram illustrates a non-linear decision boundary (indicated by a curved purple line) that also separates the two classes but in a more complex manner. Both diagrams use shaded regions (red and blue) to indicate the areas of classification for each decision boundary type.


Nonlinear models are more expressive and can represent more complex relationships.

Are non-linear models better?

## Complexity and Overfitting

Raw data

## More complex isn't always better.

## Fit models to samples of data

 $\square$
## $\searrow$

## Regularization

Regularization is the process of adding constraints or penalties to the learning process to improve generalization.

Many models and learning algorithms have methods to tune the regularization during the training process.

- We will see this in the optimization phase of the ML Lifecycle.


## Parametric vs. Non-Parametric Models

Parametric model families have a fixed number of parameters that does not depend on the size of the training data.

- Example: linear regression model $g_{w}(\mathbf{x})=w_{0}+\sum_{d=1}^{D} w_{d} x_{d}$

Non-parametric model families have "parameters" that grow with the training data.

- Example: Nearest Neighbor Model
![](https://cdn.mathpix.com/cropped/2025_10_01_11b3b7c6329b6de92897g-46.jpg?height=804&width=1255&top_left_y=1063&top_left_x=2075)

**Image Description:** The image presents a Voronoi diagram featuring various colored regions that illustrate the partitioning of space based on given training data points, represented by black dots. Each region corresponds to a point, with a highlighted "Query Point x" in green. The diagram visually shows how the space around the training data is divided; points within a specific region are closer to the corresponding training point than to any other. The label "Training Data" is positioned in a speech bubble to imply the source of the points.


$$
\begin{aligned}
& g_{\mathcal{D}=\left\{\left(x_{n}, y_{n}\right)\right\}_{n=1}^{N}}(\mathbf{x})=y_{i(\mathbf{x})} \\
& \quad \text { where } i(\mathbf{x})=\arg \min _{n}\left\|\mathbf{x}_{n}-\mathbf{x}\right\|
\end{aligned}
$$

The "parameters" are all the training data

## Choosing the Model Family

1. Determine the learning problem: Classification, Regression, Clustering, Dimensionality Reduction
2. Start with linear models and good features

- Linear models + features eng. is a common way to encode inductive biases

3. Trying increasingly complex models and check validation performance

- Today's lecture focuses on models in the Scikit-learn package.
- Future lectures we will explore deep learning model design.


## Logistic Regression Model Family

Logistic regression is a linear model for classification of the form:

$$
g_{w}(\mathbf{x})=\sigma\left(w_{0}+\sum_{d=1}^{D} w_{d} x_{d}\right)
$$
where $\sigma(t)=\frac{1}{\left(1+e^{-t}\right)}$ is a non-linear transformation
$$
\sigma(t)=\frac{1}{\left(1+e^{-t}\right)}
$$

used to model probabilities.

$$
\mathrm{P}(y=1 \mid \mathrm{x})=g_{w}(\mathrm{x})
$$
- Can be extended to multi-class classification Focus of future lectures.
![](https://cdn.mathpix.com/cropped/2025_10_01_11b3b7c6329b6de92897g-48.jpg?height=775&width=1140&top_left_y=1088&top_left_x=2190)

**Image Description:** The image depicts a two-dimensional diagram illustrating a linear decision boundary in a classification problem. The x-axis and y-axis represent feature values. Two distinct classes of data points are shown: one class marked with blue dots and the other with red crosses, separated by the yellow diagonal line, which indicates the decision boundary. The area above the line represents one class, while the area below represents the other. The gradient shading suggests the confidence in classification near the boundary.


## ML Lifecycle

## LEARNING PROBLEM

- Target
- Objective
- Data
![](https://cdn.mathpix.com/cropped/2025_10_01_11b3b7c6329b6de92897g-49.jpg?height=1433&width=1344&top_left_y=323&top_left_x=1024)

**Image Description:** The image is a conceptual diagram illustrating a framework or model related to a system. It consists of interconnected geometric shapes in blue and yellow. The shapes form a flow pattern, indicating movement or interaction between distinct components labeled with symbols, possibly representing data, analysis, or frameworks. The labels "L," "M," and "O" are prominently featured within the shapes, suggesting stages or categories within a process. Icons within the shapes depict various elements, possibly signifying statistics or networks. The overall design is modern and abstract, emphasizing the relationships between different components of the system.



# MODEL DESWGBAN <br> family/Architecture <br> - Hypothesis space <br> - Inductive biases / Assumptions 

OPTIMIZATIO<br>N Iterative Opt. Algorithms<br>- Hyperparameter Tuning

## Iterative Optimization Algorithms

We typically use iterative optimization algorithms to train (fit) the model by solving the following problem:
![](https://cdn.mathpix.com/cropped/2025_10_01_11b3b7c6329b6de92897g-50.jpg?height=516&width=2799&top_left_y=654&top_left_x=289)

**Image Description:** The image contains a mathematical equation used in optimization, presented as follows:

$$
\hat{w} = \arg \min \left( \text{Error}[h_w \cdot \mathcal{D}_{\text{training}}] + \lambda \text{Reg}[w] \right)
$$

It signifies the minimization of error on training data combined with a regularization term. The terms within the brackets represent error and regularization components, while the arguments indicate optimization parameters. The image also features colored text boxes labeling "Lectures 10-11," "Lecture 5," and "Lecture 6," likely indicating relevant educational contexts.


- These algorithms can be slow and require tuning
- Often approximate and return local minima
- Tuning can affect generalization performance

This slide will be the focus of a significant part of the class.

## Hyperparameters

The hyperparameters are the parameters that are constant during the optimization (training) algorithm.

- Example $\lambda$ in $\widehat{w}=\underset{w \in \Theta}{\arg \min } \operatorname{Error}\left[h_{w} ; \mathfrak{D}_{\text {training }}\right]+\lambda \operatorname{Reg}[w]$

The hyperparameters are chosen to improve generalization performance as measured on the validation dataset.

- Hyperparameters are often selected using a grid-search

```
for hp1 in [0.1, 1, 10]:
    for hp2 in [0.1, 1, 10]:
        model = MyModel(hp1=hp1, hp2=hp2)
        model.fit(X_train, y_train)
        error[hp1, hp2] = error(y_val, model.predict(X_val))
hp1_best, hp2_best = arg_min(error)
```

|  | $\mathbf{0 . 1}$ | $\mathbf{1}$ | $\mathbf{1 0}$ |
| :--- | :--- | :--- | :--- |
| $\mathbf{0 . 1}$ | .8 | .7 | .7 |
| $\mathbf{1}$ | .4 | .2 | .3 |
| $\mathbf{1 0}$ | .5 | .6 | .6 |

## Training vs Validation Accuracy

When tuning regularization hyperparameters against the validation dataset, it is common to see plots where

- training acc. is greater than validation accuracy
- acc. on the training dataset continues to increase.
- acc. on the validation dataset increases and then

LR Classifier Accuracy vs Reg. Parameter
![](https://cdn.mathpix.com/cropped/2025_10_01_11b3b7c6329b6de92897g-52.jpg?height=703&width=2557&top_left_y=1160&top_left_x=480)

**Image Description:** The image consists of two graphs side by side. The left graph plots "Avg. Log Prob." against "Reg. Parameter C," with the x-axis showing the regularization parameter (log scale, ranging from 0.001 to 1) and the y-axis displaying average log probability. It features two lines representing "Train" (blue) and "Val" (red) with their respective values. The right graph shows "Accuracy" versus "Reg. Parameter C" on similar axes. Both graphs highlight "Sweet Spot" areas with green arrows, indicating optimal regularization levels. Overall, the image emphasizes the relationship between regularization and model performance metrics.


## ML Lifecycle

## LEARNING PROBLEM

- Target
- Objective
- Data

PREDICT \&

- Makim/BAULUATSE
- Accuracy Metrics
![](https://cdn.mathpix.com/cropped/2025_10_01_11b3b7c6329b6de92897g-53.jpg?height=1449&width=2293&top_left_y=319&top_left_x=954)

**Image Description:** The image is a quadrant diagram illustrating a model design framework. It features four quadrants, labeled "L," "P," "M," and "O," represented by blue and yellow arrows. The left quadrant ("L") likely addresses "Learning System," while "P" refers to "Performance." The top quadrant ("M") is focused on "Model Design," listing aspects like family/architecture and hypothesis space. The bottom quadrant ("O") pertains to "Optimization," mentioning iterative optimization algorithms and hyperparameter tuning. Arrows flow between quadrants, indicating relationships among the components.



## Making Predictions

Inference is the process of making predictions with a model.

- Label Prediction (e.g., the class or regressor value represent the most likely value given the data but doesn't reflect the uncertainty in the prediction.
- sklearn: model.predict()
![](https://cdn.mathpix.com/cropped/2025_10_01_11b3b7c6329b6de92897g-54.jpg?height=388&width=758&top_left_y=514&top_left_x=2351)

**Image Description:** The image is a pixelated representation of a pullover sweater, accompanied by an arrow pointing to the word “Pullover” in bold, black text. The sweater is depicted in a simplified, low-resolution format, focusing on the general shape and outline. The left side features a schematic or abstract visualization of the garment, while the right side clearly labels the item. This format suggests a classification or identification context, possibly used for teaching purposes in a fashion or garment recognition module.

- Predicted Distribution (e.g., the class probabilities or mean and variance for regression) encodes the uncertainty in the predicted value.
- sklearn: model.predict_proba()
\# for classification models
![](https://cdn.mathpix.com/cropped/2025_10_01_11b3b7c6329b6de92897g-54.jpg?height=549&width=741&top_left_y=995&top_left_x=2347)

**Image Description:** The image features a diagram displaying a classification model's output for clothing item recognition. On the left, a pixelated image of a pullover is shown. On the right, a bar chart presents the probability scores for various clothing types, with the "Pullover" category highlighted and its probability slightly over 0.6, indicating the model's confidence. Other categories include "Trouser," "T-shirt/top," "Sneaker," and "Shirt," with lower probabilities ranging from 0 to 0.2. The x-axis represents probability (prob), while the y-axis lists the clothing types.


Many modern techniques predict distributions (e.g., ChatGPT).

## Evaluation Metrics

How do we measure success of a model at test time?

- Accuracy or some other error metric on the test data Are all types of error equal?

| Classification <br> Example | Spam | Not Spam |
| :--- | :--- | :--- |
| Predicted Spam | True Positive | False Positive |
| Predicted Not Spam | False Negative | True Negative |

- Falsely classityıng sometning as spam nas risks Often use decision theory to make decisions from probabilities.
- Need to quantify the costs of decisions not just the model.


## Demo

Scikit Learn Classification

```
1 from sklearn.linear_model import LogisticRegression
2
3 model = LogisticRegression()
4 model.fit(X=X_tr, y=y_tr)
```

![](https://cdn.mathpix.com/cropped/2025_10_01_11b3b7c6329b6de92897g-56.jpg?height=638&width=312&top_left_y=910&top_left_x=1292)

**Image Description:** The image consists of two gray-scale pixelated diagrams, each representing a distinct footwear category: "Ankle boot" and "Sandal." The diagrams are arranged vertically. Each diagram has a grid structure, with the x-axis and y-axis labeled with numerical values ranging from 0 to 20. The pixelation suggests that the images may depict feature maps or data clusters for the respective categories in a machine learning context, likely showing their visual features in a compressed format.

![](https://cdn.mathpix.com/cropped/2025_10_01_11b3b7c6329b6de92897g-56.jpg?height=297&width=284&top_left_y=920&top_left_x=1738)

**Image Description:** The image is a grayscale pixelated representation labeled "Bag." It appears to be a visual representation of an object or classification category, possibly related to image recognition or machine learning. The pixelation obscures detailed features, making it challenging to discern specific characteristics. The lack of color indicates the use of a monochromatic scheme, likely for emphasis on shape or texture rather than colorimetric data. No axes or numerical data are presented.

![](https://cdn.mathpix.com/cropped/2025_10_01_11b3b7c6329b6de92897g-56.jpg?height=320&width=277&top_left_y=1224&top_left_x=1739)

**Image Description:** The image depicts a pixelated representation of a shirt, likely used in a machine learning context to illustrate feature extraction or classification. The diagram does not include traditional axes, but it visually conveys an abstracted form of a shirt through varying grayscale pixel intensities. The lower axis is labeled with numerical values (0, 10, 20), suggesting a range that might pertain to pixel indices or feature dimensions. The overall effect is that of simplification for computational analysis.

![](https://cdn.mathpix.com/cropped/2025_10_01_11b3b7c6329b6de92897g-56.jpg?height=294&width=273&top_left_y=918&top_left_x=2156)

**Image Description:** The image appears to be a pixelated depiction of a coat, possibly representing a garment commonly used in academic discussions related to fashion, textiles, or material science. The absence of detailed features suggests it may be intended to highlight general shape and structure rather than specific design elements. The term "Coat" is prominently displayed above the image, indicating the subject matter. The pixelation implies a focus on abstraction rather than a clear visual representation.

![](https://cdn.mathpix.com/cropped/2025_10_01_11b3b7c6329b6de92897g-56.jpg?height=316&width=278&top_left_y=1224&top_left_x=2151)

**Image Description:** The image appears to be a low-resolution pixelated representation of a sneaker, labeled "Sneaker" at the top. It features a rectangular grid structure with x and y axes labeled with numerical values ranging from 0 to 20. The pixelation obscures detailed features, but suggests the outline or form of a sneaker in a monochromatic scheme. The image's primary focus is to represent the object category visually, likely for classification or identification purposes in a machine learning context.

![](https://cdn.mathpix.com/cropped/2025_10_01_11b3b7c6329b6de92897g-56.jpg?height=290&width=269&top_left_y=922&top_left_x=2568)

**Image Description:** The image appears to be a blurred or pixelated representation of a dress. It lacks clarity but suggests the silhouette of a garment, potentially indicating its style or design. The background is neutral, which emphasizes the dress. No axes or detailed diagrammatic elements are present, and the image does not convey explicit quantitative information or scientific data.

![](https://cdn.mathpix.com/cropped/2025_10_01_11b3b7c6329b6de92897g-56.jpg?height=316&width=269&top_left_y=1228&top_left_x=2568)

**Image Description:** The image is a pixelated representation of a T-shirt or top classified within a dataset. It appears to be a grayscale diagram that lacks clear details due to pixelation. The x-axis ranges from 0 to 20, likely indicating some form of measurement or categorization related to the T-shirt. The y-axis does not display numerical values but seems to be labeled with the item type "T-shirt/top" at the top. The overall visual indicates a classification task within a machine learning context, potentially for image recognition or object detection.

![](https://cdn.mathpix.com/cropped/2025_10_01_11b3b7c6329b6de92897g-56.jpg?height=285&width=273&top_left_y=927&top_left_x=2981)

**Image Description:** The image appears to be a blurred or pixelated diagram of a pullover garment. It likely features the front view of the pullover, with a focus on the neckline and shoulder areas. The diagram may serve to illustrate design elements or construction techniques relevant to garment design. The absence of clarity in the image means specific measurements or details are not discernible. The textual label "Pullover" indicates the subject of the illustration.

![](https://cdn.mathpix.com/cropped/2025_10_01_11b3b7c6329b6de92897g-56.jpg?height=320&width=278&top_left_y=1224&top_left_x=2976)

**Image Description:** The image is a grayscale diagram representing a pixelated output, likely from a convolutional neural network (CNN) classification task, identifying a "Trouser" item. The x-axis ranges from 0 to 20, possibly indicating a feature dimension or pixel width, while the y-axis is not clearly defined but suggests a height dimension related to the object being classified. The overall structure appears as a simplified representation of a trouser silhouette, though heavily pixelated, emphasizing areas of higher intensity.


# Machine Leaning Mechanics Terminology and Techniques 

Credit: Joseph E. Gonzalez and Narges Norouzi
Reference Book Chapters: Chapter 1

## Homework!

Berkeley's favorite pastime
![](https://cdn.mathpix.com/cropped/2025_10_01_11b3b7c6329b6de92897g-58.jpg?height=1875&width=1911&top_left_y=0&top_left_x=1173)

**Image Description:** This image depicts a whimsical illustration featuring a cartoon bear wearing glasses and studying at a table, with a robot beside it. The bear is writing in an open book, while the robot appears to be assisting or encouraging. A desk lamp illuminates the scene, and there are stacked books on the table, suggesting a learning environment. The background includes a shelf with books, reinforcing the academic theme. The overall tone is playful, emphasizing collaboration in education.


282609
7

## Homework Outlines

## Part 1: Lecture application

## Part 2: Paper implementation

- Written questions (similar to previous years)
- Coding: implementing concepts from lecture, setting up datasets used in part 2
- Paper reading + written questions
- We will teach you how to read papers before HW2
- Paper implementation + extensions of the paper


## Homework in the age of AGI

- Use AI - it is incredibly helpful
- Rule of thumb:
o Use AI when you can easily verify
o Visualizations (more about this in HW2)
o Documentation lookup
o When you cannot verify - do not use Al
o Writing functions or applications that you don't know how to approach
o If you cant write tests or a design doc, don't vibe code it
o Be careful with git/functions which have write access to your machine
modified: modified: modified: deleted: deleted: deleted: modified: modified:
examples/041-audio-extraction/aud examples/042-pdf-extraction/pdf-e examples/043-caching-responses/ca examples/044-parallel-extraction/ examples/044-parallel-extraction/ examples/044-parallel-extraction/
examples/046-hooks-and-callbacks/ examples/047-type-adapters/type-a

Untracked files:
(use "git add <file>..." to include in what will be example_cleanup_todo.md
no changes added to commit (use "git add" and/or "git

## Downloading Homeworks

- All content (discussions, lectures, homeworks) are in the BerkeleyML/fa25-
$\checkmark$ CS189 Content Downloader student repo
This notebook works on Google Colab and local machines. It fetches only the folder you ask for from
- Content downloader the public course repo and puts it under cs189/<repo>/<folder>

Where files go notebook will download

- Colab: /content/drive/MyDrive/cs189/<folder>
- Local: ./cs189/<folder> the content for you (either locally or in colab)
How to use

1. Run the next cell to define fetch_repo_folder .

- Content will also be in the

2. Call it with the folder you want from the repo (e.g., hw/hw2, lec/lec02/data). shared Google Drive
Example (homework 1) (including downloader
```
folder = "lec/lec02"
path = fetch_repo_folder(folder) # -> /.../cs189/fa25-student/lec/lec02
print("Ready at:", path)
```

notebook)

## Homework 1

## Part 1 - due Sep 19th

- Written questions - prereqs
o linear algebra, calculus, and probability
o Recommend not using AI - you will use these techniques throughout the class
- Coding: basic ML pipelines
- Pandas, plotly, scikit learn, image transformations
o Do not procrastinate - the later problems are much harder than the earlier problems


## Part 2: due Sep26th

- No paper this week
- we will provide some fun papers to look at on Ed
- Coding: improving models
o Your model from part 1 is now failing on a secret test set - how can we improve performance without any fancy model techniques?


## Fun paper about train/test splits

Do ImageNet Classifiers Generalize to ImageNet?

Benjamin Recht*<br>Rebecca Roelofs<br>Ludwig Schmidt<br>Vaishaal Shankar<br>UC Berkeley<br>UC Berkeley<br>UC Berkeley<br>UC Berkeley


#### Abstract

We build new test sets for the CIFAR-10 and ImageNet datasets. Both benchmarks have been the focus of intense research for almost a decade, raising the danger of overfitting to excessively re-used test sets. By closely following the original dataset creation processes, we test to what extent current classification models generalize to new data. We evauluate a broad range of models accuracy gains on the original test sets translate to larger gains on the new test sets. Our results suggest that the accuracy drops are not caused by adaptivity, but by the models' inability to generalize to slightly "harder" images than those found in the original test sets.


## 1 Introduction

The overarching goal of machine learning is to produce models that generalize. We usually quantify generalization by measuring the performance of a model on a held-out test set. What does good performance on the test set then imply? At the very least, one would hope that the model also performs well on a new test set assembled from the same data source by following the same data cleaning protocol.
In this paper, we realize this thought experiment by replicating the dataset creation process for two prominent benchmarks, CIFAR-10 and ImageNet [10, 35]. In contrast to the ideal outcome, we find that a wide range of classification models fail to reach their original accuracy scores. The accuracy drops range from $3 \%$ to $15 \%$ on CIFAR-10 and $11 \%$ to $14 \%$ on ImageNet. On ImageNet, the accuracy loss amounts to approximately five years of progress in a highly active period of machine learning research.
*Authors ordered alphabetically. Ben did none of the work.

ImageNetV2 was collected in the same way as ImageNet but ImageNet models achieve a lower accuracy

![](https://cdn.mathpix.com/cropped/2025_10_01_11b3b7c6329b6de92897g-63.jpg?height=606&width=1358&top_left_y=780&top_left_x=1712)

**Image Description:** The image comprises a 3x3 grid of photographs displaying various angles and perspectives of the yellow lady's slipper orchid (Cypripedium calceolus). Each photograph highlights the distinct shape and coloration of the orchid's blooms, characterized by a tubular structure and vibrant yellow pouch. The surrounding foliage, consisting of green leaves and forest floor elements, provides context to the orchid's natural habitat. This arrangement allows for a comparative study of morphological variations in the species.

ImageNet

ImageNetV2

visually very similar

## Fun paper about train/test splits

Do ImageNet Classifiers Generalize to ImageNet?

Benjamin Recht*
Rebecca Roelofs
Ludwig Schmidt
Vaishaal Shankar
UC Berkeley
UC Berkeley
UC Berkeley
UC Berkeley

## Abstract

We build new test sets for the CIFAR-10 and ImageNet datasets. Both benchmarks have been the focus of intense research for almost a decade, raising the danger of overfitting to excessively re-used test sets. By closely following the original dataset creation processes, we test to what extent current classification models generalize to new data. We evaluate a broad range of models and find accuracy drops of $3 \%-15 \%$ on CIFAR-10 and $11 \%-14 \%$ on ImageNet. However, accuracy gains on the original test sets translate to larger gains on the new test sets. Our results
suggest that the accuracy drops are not caused by adaptivity, but by the models' inability to generalize to slightly "harder" images than those found in the original test sets.

## 1 Introduction

The overarching goal of machine learning is to produce models that generalize. We usually quantify generalization by measuring the performance of a model on a held-out test set. What does good performance on the test set then imply? At the very least, one would hope that the model also performs well on a new test set assembled from the same data source by following the same data cleaning protocol.
In this paper, we realize this thought experiment by replicating the dataset creation process for two prominent benchmarks, CIFAR-10 and ImageNet [10, 35]. In contrast to the ideal outcome, we find that a wide range of classification models fail to reach their original accuracy scores. The accuracy drops range from $3 \%$ to $15 \%$ on CIFAR-10 and $11 \%$ to $14 \%$ on ImageNet. On ImageNet, the accuracy loss amounts to approximately five years of progress in a highly active period of machine learning research.
*Authors ordered alphabetically. Ben did none of the work.

ImageNetV2 was collected in the same way as ImageNet but ImageNet models achieve a lower accuracy
![](https://cdn.mathpix.com/cropped/2025_10_01_11b3b7c6329b6de92897g-64.jpg?height=801&width=1490&top_left_y=803&top_left_x=1679)

**Image Description:** The image presents a scatter plot illustrating the relationship between original test accuracy and new test accuracy on the ImageNet dataset. The X-axis represents "Original test accuracy (top-1, %)" ranging from 60 to 90, while the Y-axis denotes "New test accuracy (top-1, %)," also ranging from 40 to 90. Blue points indicate individual data points, with a fitted red line showing the actual trend versus a dashed black line representing the expected relationship. The plot highlights discrepancies between expected and actual new test accuracy, with annotations indicating both lines.


Have Fun!!!


[^0]:    *Funny scene in silicon valley where a binary classifier was trained for food predict

