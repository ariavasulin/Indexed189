---
course: CS 189
semester: Fall 2025
type: lecture
number: 4
title: Lecture 4
source_type: notebook
source_file: lec04.ipynb
processed_date: '2025-09-30'
---



<h1 class="cal cal-h1">Lecture 04: K-Means and Probability – CS 189, Fall 2025</h1>


In this lecture notebook, we will explore k-means clustering and some of the basic probability calculations from lecture.

```python
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly import figure_factory as ff
colors = px.colors.qualitative.Plotly
px.defaults.width = 800
from ipywidgets import HBox
import numpy as np
pd.set_option('plotting.backend', 'plotly')
```
```python
# make the images folder if it doesn't exist
import os
if not os.path.exists("images"):
    os.makedirs("images")

# # Uncomment for HTML Export
# import plotly.io as pio
# pio.renderers.default = "notebook_connected"
```


<h2 class="cal cal-h2">The Bike Dataset</h2>

Here we will apply k-means clustering to the bike dataset to explore the distribution of length and speed of Prof. Gonzales' bike rides.
```python
# bikes = pd.read_csv("speed_length_data.csv")
bikes = pd.read_csv("https://eecs189.org/fa25/resources/assets/lectures/lec04/speed_length_data.csv")
bikes.head()
```
```python
bikes.plot.scatter(x='Speed', y='Length', title='Speed vs Length of Bike Segments', 
                   height=800)
```


<h3 class="cal cal-h3">Scikit-Learn K-Means Clustering</h3>

We have data for 4 bikes. Let's try k-means clustering using scikit-learn with 4 clusters.


```python
from sklearn.cluster import KMeans
# Create a KMeans model with 4 clusters
kmeans = KMeans(n_clusters=4, random_state=42)
# Fit the model to the data
kmeans.fit(bikes[['Speed', 'Length']])
# Get the cluster labels
bikes['scikit k-means'] = kmeans.predict(bikes[['Speed', 'Length']]).astype(str)
```
We can visualize the clustering
```python
fig = px.scatter(
    bikes, x='Speed', y='Length', color='scikit k-means',
    title='K-Means Clustering of Bike Segments',
    height=800)
fig.add_scatter(
    x=kmeans.cluster_centers_[:,0],
    y=kmeans.cluster_centers_[:,1],
    mode='markers',
    marker=dict(color='black', size=10),
    name='Centroids'
)
#fig.write_image("images/bike_kmeans.pdf", scale=2, height=800, width=700)

```
<br><br><br>

---
Return to Lecture

---

<br><br><br>



<h2 class="cal cal-h2">Implementing the K-Means Clustering Algorithm</h2>

Lloyd's algorithm for K-means clustering has three key elements: initialization, assignment, and update. We will implement a function for each.


<h3 class="cal cal-h3">Initialization</h3>

Here we use the basic Forgy method of randomly selecting k data points as the initial cluster centers.
```python
def initialize_centers(x, k):
    """Randomly select k unique points from x to use as initial centers."""
    ind = np.random.choice(np.arange(x.shape[0]), k, replace=False)
    return x[ind]
```
```python
k = 4
x = bikes[['Speed', 'Length']].to_numpy()
centers = initialize_centers(x, k)
centers
```


<h3 class="cal cal-h3">Assignment</h3>



In this step, we assign each data point to the nearest cluster center.
```python
def compute_assignments(x, centers):
    """Assign each point in x to the nearest center."""
    distances = np.linalg.norm(x[:, np.newaxis] - centers, axis=2)
    return np.argmin(distances, axis=1)
```
```python
assignments = compute_assignments(x, centers)
assignments
```


<h3 class="cal cal-h3">Update Centers</h3>



In this step, we update the cluster centers by computing the mean of all points assigned to each center.

```python
def update_centers(x, assignments, k):
    """Update centers based on the current assignments."""
    return np.array([x[assignments == i].mean(axis=0) for i in range(k)])
```
```python
centers = update_centers(x, assignments, k)
centers
```


<h3 class="cal cal-h3">Lloyd's Algorithm</h3>



We put all these pieces together in a loop that continues until the centers no longer change.
```python
def k_means_clustering(x, k, max_iters=100):
    centers = initialize_centers(x, k)
    assignments_old = -np.ones(x.shape[0])
    soln_path = [centers]
    for _ in range(max_iters):
        assignments = compute_assignments(x, centers)
        centers = update_centers(x, assignments, k)
        soln_path.append(centers)
        if np.array_equal(assignments, assignments_old):
            break
        assignments_old = assignments
    return centers, assignments, soln_path
```
```python
np.random.seed(43)
centers, assignments, soln_path = k_means_clustering(x, k)
len(soln_path)
```
The following code visualizes the clustering process at each iteration.

```python
### Construct an animation of the k-means algorithm.
### You do not need to understand the code below for the class.
### It is just for making the animation.

## Prepare a giant table with all the data and centers labeled with the iteration.
pts = []
for i, centers in enumerate(soln_path):
    df = bikes[['Speed', 'Length']].copy()
    df['Class'] = compute_assignments(x, centers).astype(str)
    df2 = pd.DataFrame(centers, columns=['Speed', 'Length'])
    df2['Class'] = 'Center'
    df_combined = pd.concat([df, df2], ignore_index=True)
    # I also need the index of each point in center for the animation
    # the index acts as a unique identifier for each point across frames
    df_combined.reset_index(inplace=True)
    # The iteration number tracks the frame in the animation
    df_combined['Iteration'] = i
    pts.append(df_combined)
# I stack all the data into one big table.
frames = pd.concat(pts, ignore_index=True)

## Make the animation
fig = px.scatter(frames, x='Speed', y='Length', color='Class', 
                 animation_group='index',
                 animation_frame='Iteration', title='K-Means Clustering',
                 width=700, height=800)
## The aspect ratio of the plot is missleading.
fig.update_layout(
    xaxis=dict(scaleanchor="y", scaleratio=1),
    yaxis=dict(scaleanchor="x", scaleratio=1)
)
# fig.write_image("images/bike_kmeans_animation_0.pdf", height=800, width=700)

## Touchup the centers to make them more visible
fig.update_traces(marker=dict(size=12, symbol='x', color='black'), 
                  selector=dict(legendgroup='Center') )
for i, f in enumerate(fig.frames):
    for trace in f.data:
        if trace.name == 'Center':
            trace.update(marker=dict(size=12, symbol='x', color='black'))
    # go.Figure(f.data, f.layout).write_image(
    #     f"images/bike_kmeans_animation_{i+1}.pdf", height=800, width=700)
# fig.write_html("images/bike_kmeans_animation.html",include_plotlyjs='cdn', full_html=True)
fig

```
<br><br><br>

---
Return to Lecture

---

<br><br><br>



<h2 class="cal cal-h2">K-Means on Pixel Data</h2>

Let's load an image from a recent bike ride.
```python
from PIL import Image
import requests
from io import BytesIO
url = "https://eecs189.org/fa25/resources/assets/lectures/lec04/bike2.jpeg"
response = requests.get(url)
img = np.array(Image.open(BytesIO(response.content)))
# img = np.array(Image.open("bike2.jpeg"))
print(img.shape)
px.imshow(img)
```
We can think of an image as a collection of pixels, each represented by a color value. In the case of RGB images, each pixel is represented by three values corresponding to the red, green, and blue color channels. These are three dimensional vectors.  We can plot these vectors.
```python
image_df = pd.DataFrame(img.reshape(-1,3), columns=['R', 'G', 'B'])
image_df['color'] = ("rgb(" + 
                     image_df['R'].astype(str) + "," + 
                     image_df['G'].astype(str) + "," + 
                     image_df['B'].astype(str) + ")")
fig = go.Figure()
small_image_df = image_df.sample(100000, random_state=42)
fig.add_scatter3d(x=small_image_df['R'], y=small_image_df['G'], z=small_image_df['B'],
                   mode='markers', marker=dict(color=small_image_df['color'], opacity=0.5, size=2))
fig.update_layout(scene=dict(xaxis_title='R', yaxis_title='G', zaxis_title='B'), 
                  width=800, height=800,)
# fig.write_html("images/bike_color_space.html",include_plotlyjs='cdn', full_html=True)
fig
```


<h3 class="cal cal-h3">Applying K-Means</h3>

```python
from sklearn.cluster import KMeans
# Apply k-means clustering to the RGB columns
n_clusters = 8
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
image_df['cluster'] = kmeans.fit_predict(image_df[['R', 'G', 'B']])
image_df['cluster'].value_counts()
```
```python
from plotly.subplots import make_subplots
img_kmeans = (
    kmeans.cluster_centers_[image_df['cluster'].values]
    .reshape(img.shape)
)
# make two linked subplots
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Original Image", "K-Means Compressed Image"),
    specs=[[{"type": "xy"}, {"type": "xy"}]]
)
fig.add_trace(px.imshow(img).data[0], row=1, col=1)
fig.add_trace(px.imshow(img_kmeans).data[0], row=1, col=2)

# 1) Make both subplots share identical ranges
# (use half-pixel bounds to match how Image traces are rendered)
H, W = img.shape[:2]
xrange = [-0.5, W - 0.5]
yrange = [H - 0.5, -0.5]  # origin at top
for c in (1, 2):
    fig.update_xaxes(range=xrange, row=1, col=c)
    fig.update_yaxes(range=yrange, row=1, col=c)

# 2) Lock square pixels and prevent domain stretch
fig.update_yaxes(scaleanchor="x",  scaleratio=1, constrain="domain", row=1, col=1)
fig.update_yaxes(scaleanchor="x2", scaleratio=1, constrain="domain", row=1, col=2)

# Link panning/zooming between the two images (now safe)
fig.update_xaxes(matches="x")
fig.update_yaxes(matches="y")

# Cosmetic
# fig.write_html("images/bike_kmeans_compression.html",include_plotlyjs='cdn', full_html=True)
fig.update_layout(width=900, height=600, margin=dict(t=50, b=30, l=20, r=20))

```



<h3 class="cal cal-h3">Choosing the Number of Clusters</h3>

We can use the k-means objective to evaluate the quality of clustering for different values of k. The objective (called inertia) is the sum of squared distances from each point to its assigned cluster center. A lower objective indicates better clustering.

A standard method of choosing k is the "elbow method", where we plot the k-means score against k and look for an "elbow" point where the rate of improvement slows down.
```python
scores = pd.DataFrame(columns=['k'])
scores['k'] = [2, 4, 8, 16, 32, 64, 128, 256]
scores.set_index('k', inplace=True)

# Apply k-means clustering to the RGB columns
from sklearn.cluster import KMeans
for k in scores.index:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(image_df[['R', 'G', 'B']])
    # scores.loc[k, 'score'] = kmeans.score(image_df[['R', 'G', 'B']])
    scores.loc[k, 'score'] = kmeans.inertia_ # Negative score
```
```python
fig = px.line(
    scores, 
    title="K-Means Objective vs Number of Clusters",
    markers=True,
    labels={"index": "Number of Clusters (k)", 
            "value": "K-Means Objective"},
    width=700, height=400
)
fig.update_layout(xaxis_type="log", xaxis_exponentformat='power', showlegend=False)
fig.update_layout(margin=dict(t=50, b=30, l=20, r=20))
# fig.write_image("images/kmeans_score_vs_k.pdf", scale=2, height=400, width=700)
fig
```
<br><br><br>

---
Return to Lecture

---

<br><br><br>


<h2 class="cal cal-h2">Wake Word Detector</h2>


In lecture, we derived the probability that a wake word was spoken given the detector detected a wake word. We used Bayes' theorem to do this.

\begin{align*}
P(W = 1| D=1) &= \frac{P(D=1|W=1)P(W=1)}{P(D=1)}\\
&=\frac{P(D=1|W=1)P(W=1)}{P(D=1|W=1)P(W=1) + P(D=1|W=0)P(W=0)}
\end{align*}

In the following function, we will implement this calculation.
```python
def wake_word_detector(
        p_wake = 0.0001,               # P(wake) prior probability of wake word
        p_detect_g_wake = 0.99,        # P(detect | wake) likelihood of detection given wake
        p_detect_g_nowake = 0.001      # P(detect | no wake) likelihood of detection given no wake
):
    # P(wake | detect) = P(detect | wake) * P(wake) / P(detect)
    p_detect = p_wake * p_detect_g_wake + (1 - p_wake) * p_detect_g_nowake
    p_wake_g_detect = p_detect_g_wake * p_wake / p_detect
    return p_wake_g_detect

wake_word_detector()
```
We want to understand what happens if we vary the recall of the detector. The recall (also called sensitivity) is defined as $P(D=1|W=1)$, the probability that the detector detects a wake word when a wake word was actually spoken. We also vary the false positive rate, defined as $P(D=1|W=0)$, the probability that the detector detects a wake word when no wake word was spoken.

$$
P(W = 1| D=1) =\frac{\text{(Recall)} P(W=1)}{\text{(Recall)} P(W=1) + \text{(False Positive Rate)} P(D=1|W=0)P(W=0)}
$$

We can see from the above equation that even as recall approaches 1, if the false positive rate is high, the probability that a detected wake word was actually spoken can still be low.

```python
p_detect_g_nowake = np.logspace(-6, -4, 100)
p_wake_g_detect = wake_word_detector(p_detect_g_nowake=p_detect_g_nowake, 
                                     p_detect_g_wake=1.0)
fig = px.line(
    x=p_detect_g_nowake,
    y=p_wake_g_detect,
    title="P(Wake | Detect) vs False Positive Rate for Perfect Recall",
    labels={
        "x": "P(Detect | No Wake) (False Positive Rate)",
        "y": "P(Wake | Detect)"
    },
    log_x=True
)
fig.update_layout( xaxis_exponentformat='power')
#fig.write_image("images/wake_word_detector_fpr.pdf", scale=2, height=500, width=700)
fig
```
```python
p_detect_g_wake = np.logspace(-0.8, 0, 100)
p_wake_g_detect = wake_word_detector(p_detect_g_wake=p_detect_g_wake,
                                     p_detect_g_nowake=0.0001)
fig = px.line(
    x=p_detect_g_wake,
    y=p_wake_g_detect,
    title="P(Wake | Detect) vs Recall (Sensitivity) for FPR=0.001",
    labels={
        "x": "P(Detect | Wake) Recall",
        "y": "P(Wake | Detect)"
    },
    log_x=True
)
fig.update_layout( xaxis_exponentformat='power')
#fig.write_image("images/wake_word_detector_recall.pdf", scale=2, height=500, width=700)
fig
```
<br><br><br>

---
Return to Lecture

---

<br><br><br>

