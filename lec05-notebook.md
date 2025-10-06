---
course: CS 189
semester: Fall 2025
type: lecture
number: 5
title: Lecture 5
source_type: notebook
source_file: lec05.ipynb
processed_date: '2025-09-30'
---



<h1 class="cal cal-h1">Lecture 05: Density Estimation and GMMs – CS 189, Fall 2025</h1>


In this lecture we will explore the implementation of the Gaussian Mixture Model (GMM) using the Expectation-Maximization (EM) algorithm. 

```python
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly import figure_factory as ff
from plotly.subplots import make_subplots
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
```
```python
### Uncomment for HTML Export
# import plotly.io as pio
# pio.renderers.default = "notebook_connected"
```

<h2 class="cal cal-h2">The Gaussian Distribution</h2>


The probability density function of a univariate Gaussian distribution with mean $\mu$ and variance $\sigma^2$ is given by:

$$
f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}
$$
```python
from scipy.stats import norm
mean = 0 
variance = .7
x = np.linspace(-4, 4, 100)
p = norm.pdf(x, loc=mean, scale=np.sqrt(variance)) # scale = standard deviation, loc = mean
fig = px.line(x=x, y=p, title=f"Standard Normal Distribution (mean={mean}, variance={variance})",
        labels={"x": "x", "y": "p(x)"}, width = 700, height = 400)
# fig.write_image("images/standard_normal.pdf", scale=2, height=400, width=700)
fig

```

<h3 class="cal cal-h3">Multivariate Normal Distribution</h3>

The equation for the multivariate normal is given by:
$$
f(\mathbf{x}) = \frac{1}{\sqrt{(2\pi)^D|\Sigma|}} \exp\left(-\frac{1}{2}(\mathbf{x}-\mu)^T\Sigma^{-1}(\mathbf{x}-\mu)\right)
$$
where $\mathbf{x}$ is a $D$-dimensional random vector, $\mu$ is the mean vector, and $\Sigma$ is the covariance matrix. 
```python
def mv_normal_pdf(X, mu, Sigma):
    """Compute the multivariate normal density at points X."""
    d = X.shape[1]
    X_centered = X - mu
    Sigma_inv = np.linalg.inv(Sigma)
    norm_const = 1 / np.sqrt((2 * np.pi) ** d * np.linalg.det(Sigma))
    exp_term = np.exp(-0.5 * np.sum(X_centered @ Sigma_inv * X_centered, axis=1))
    return norm_const * exp_term
```
```python
mu = np.array([1, 0])
Sigma = np.array([[3, 0.4], [0.4, 2]])

from scipy.stats import multivariate_normal
normal = multivariate_normal(mean=mu, cov=Sigma)
normal.pdf(np.array([[1, 0.5]]))
```
```python
mv_normal_pdf(np.array([[1, 0.5]]), mu, Sigma)
```
```python
def plot_bivariate_normal(mu, Sigma, fig=None):
    from scipy.stats import multivariate_normal
    normal = multivariate_normal(mean=mu, cov=Sigma)
    u = np.linspace(-9, 9, 100)
    X = np.array(np.meshgrid(u,u)).reshape(2,-1).T
    Z = normal.pdf(X)
    if fig is None:
        fig = make_subplots(rows=1, cols=2,
                            specs=[[{'type': 'surface'}, {'type': 'contour'}]],)
    fig.add_surface(x=X[:,0].reshape(100,100), y=X[:,1].reshape(100,100), 
                    z=Z.reshape(100,100), colorscale='Viridis',
                    contours=dict(z=dict(show=True, size=.01, start=0, end=0.3)), row=1, col=1)
    fig.add_contour(x=u, y=u, z=Z.reshape(100,100), colorscale='Viridis',
                    line_smoothing=1.3,
                    #contours_coloring='lines',
                    showscale=False,
                    row=1, col=2
                    )
    fig.update_layout(width=900, height=500)
    return fig
```
```python
mu = np.array([1, 0])
Sigma = np.array([[3, 0.4], [0.4, 2]])
plot_bivariate_normal(mu, Sigma)
```
The following interactive plot will only work in a Jupyter notebook environment. It allows you to visualize how changing the mean and covariance matrix affects the shape of the bivariate normal distribution.

```python
from ipywidgets import interactive_output, FloatSlider, HBox, VBox, widgets

u = np.linspace(-9, 9, 100)
X = np.array(np.meshgrid(u,u)).reshape(2,-1).T
normal = multivariate_normal(mean=mu, cov=Sigma)
Z = normal.pdf(X)
fig1 = go.FigureWidget()
fig1.add_surface(x=X[:,0].reshape(100,100), y=X[:,1].reshape(100,100), 
                z=Z.reshape(100,100), colorscale='Viridis',
                contours=dict(z=dict(show=True, size=.01, start=0, end=0.3)))
fig1.update_layout(width=600, height=500)
fig2 = go.FigureWidget()
fig2.add_contour(x=u, y=u, z=Z.reshape(100,100), colorscale='Viridis',
                line_smoothing=1.3)
fig2.update_layout(width=400, height=500)

mu1 = FloatSlider(min=-5, max=5, step=0.1, value=1, description='mu1')
mu2 = FloatSlider(min=-5, max=5, step=0.1, value=0, description='mu2')
sigma11 = FloatSlider(min=0.1, max=5, step=0.1, value=3, description='sigma11')
sigma22 = FloatSlider(min=0.1, max=5, step=0.1, value=2, description='sigma22')
sigma12 = FloatSlider(min=-3, max=3, step=0.1, value=0.4, description='sigma12')


def update(mu1, mu2, sigma11, sigma22, sigma12):
    mu = np.array([mu1, mu2])
    Sigma = np.array([[sigma11, sigma12], [sigma12, sigma22]])
    normal = multivariate_normal(mean=mu, cov=Sigma)
    Z = normal.pdf(X).reshape(100,100)
    with fig1.batch_update():
        fig1.data[0].z = Z
    with fig2.batch_update():
        fig2.data[0].z = Z

interactive_output(update, {
    'mu1': mu1, 'mu2': mu2,
    'sigma11': sigma11, 'sigma22': sigma22, 'sigma12': sigma12
})

HBox([VBox([mu1, mu2, sigma11, sigma22, sigma12]), fig1, fig2],  
     layout=widgets.Layout(align_items='center'))
```
<br><br><br>

---
Return to Lecture

---

<br><br><br>



<h2 class="cal cal-h2">The Bike Dataset</h2>

As with the previous lecture, we will use Professor Gonzalez's bike ride dataset to illustrate the concepts. The dataset contains the speed and length of bike rides taken with different bikes.
```python
# bikes = pd.read_csv("speed_length_data.csv")
bikes = pd.read_csv("https://eecs189.org/fa25/resources/assets/lectures/lec04/speed_length_data.csv")
bikes.head()
```
```python
bikes.plot.scatter(x='Speed', y='Length', title='Speed vs Length of Bike Segments', 
                   height=800)
```

<h2 class="cal cal-h2">The Gaussian Mixture Model</h2>


A Gaussian Mixture Model (GMM) is a probabilistic model that assumes all the data points are generated from a mixture of several Gaussian distributions:

$$
p(x \, \vert \, \pi, \mu, \Sigma) = \sum_{k=1}^{K} \pi_k \, \mathcal{N}(x | \mu_k, \Sigma_k)
$$

Just as with the K-Means model, we can use the `GaussianMixture` class from `sklearn.mixture` to fit a GMM to our data.

```python
from sklearn.mixture import GaussianMixture
# Create a Gaussian Mixture Model with 4 components
gmm = GaussianMixture(n_components=4, random_state=42, )
# Fit the model to the data
gmm.fit(bikes[['Speed', 'Length']])
# Get the cluster labels
bikes['scikit gmm'] = gmm.predict(bikes[['Speed', 'Length']]).astype(str)
bikes['prob'] = gmm.predict_proba(bikes[['Speed', 'Length']]).max(axis=1)
bikes
```
```python
mu = gmm.means_
Sigma = [np.linalg.inv(p) for p in gmm.precisions_]
p = gmm.weights_
```
```python
def gmm_surface(mu, Sigma, p, u_pts, v_pts):
    from scipy.stats import multivariate_normal
    u, v = np.meshgrid(u_pts, v_pts)
    X_pts = np.array([u.flatten(), v.flatten()]).T
    Z = np.zeros(X_pts.shape[0])
    for k in range(len(p)):
        Z += p[k] * multivariate_normal(mu[k], Sigma[k]).pdf(X_pts)
    return go.Contour(x=u_pts, y=v_pts, z=Z.reshape(u.shape), 
                      colorscale='Viridis',
                      colorbar=dict(x=1.05, y=0.35, len=0.75)
                    )
```
```python
num_points = 100
speed_pts = np.linspace(bikes['Speed'].min()-3, bikes['Speed'].max()+3, num_points)
length_pts = np.linspace(bikes['Length'].min()-3, bikes['Length'].max()+3, num_points)

fig = go.Figure()
fig.add_trace(gmm_surface(mu, Sigma, p, speed_pts, length_pts))
fig.update_layout(width=800, height=800)
fig.add_traces(px.scatter(bikes, x='Speed', y='Length', color='scikit gmm').data)
```
```python
fig = px.scatter(bikes, x='Speed', y='Length', symbol='scikit gmm', 
           size='prob', color="scikit gmm", title='GMM Clustering',
           color_continuous_scale="Viridis_r", size_max=15)
fig.update_layout(width=800, height=800)
```
<br><br><br>

---
Return to Lecture

---

<br><br><br>


<h2 class="cal cal-h2">Ancestor Sampling for the GMM</h2>


```python
# Ancestor Sampling to create a synthetic dataset
np.random.seed(42)
N = 100

mu = np.array([-1, 2, 5])
pi = np.array([0.2, 0.5, 0.3])
Sigma = np.array([0.2, 0.5, .1])
```
```python
z = np.random.choice(len(mu), size=N, p=pi)
x = np.random.normal(mu[z], np.sqrt(Sigma[z]))
```
```python
log_likelihood = np.sum(np.log(np.sum(
    pi[z] * norm.pdf(x[:, None], loc=mu[z], scale=np.sqrt(Sigma[z])), 
    axis=1 )))
```
```python
# Sort for better visualization
ind = z.argsort()
z = z[ind]
x = x[ind]

fig = px.scatter(x=x, y=np.random.rand(N)/20,  
                 title=f'Synthetic Dataset from GMM (Log Likelihood: {log_likelihood:.2f})',
                 opacity = 0.7,
                 color=z.astype(str), labels={'color': 'True Cluster'}, height=400)
u = np.linspace(-4, 9, 1000)
df = pd.DataFrame({'x': u})
for k in range(len(mu)):
    df[f'p{k}'] = pi[k] * norm.pdf(u, loc=mu[k], scale=np.sqrt(Sigma[k]))
df['p'] = df[[f'p{k}' for k in range(len(mu))]].sum(axis=1)
fig.add_traces(px.line(df, x='x', y=df.columns[1:], labels={'y': 'Density'}).data)
fig.update_layout(width=800, height=400)

fig
```
<br><br><br>

---
Return to Lecture

---

<br><br><br>


<h2 class="cal cal-h2">Implementing the GMM using the EM Algorithm</h2>

Directly maximizing the log-likelihood function is challenging:

\begin{align*}
\log p\left(\mathcal{D} \,\vert\, \mu, \Sigma \right) 
& = \log \left( \prod_{n=1}^{N} \sum_{k=1}^{K} \pi_k \, \mathcal{N}(x_n | \mu_k, \Sigma_k) \right) \\
& = \sum_{n=1}^{N} \log \left(\sum_{k=1}^{K} \pi_k \, \mathcal{N}(x_n | \mu_k, \Sigma_k) \right) \\
\end{align*}

because of the summation inside of the logarithm. Instead, we use the Expectation-Maximization (EM) algorithm to iteratively optimize the parameters.


<h3 class="cal cal-h3">The Initialization Step</h3>


A typical way to initialize GMM models is to start with k-means clustering to find the initial means of the Gaussian components.
```python
from sklearn.cluster import KMeans
def initialize_gmm(x, K):
    N, D = x.shape
    kmeans = KMeans(n_clusters=K)
    kmeans.fit(x)
    mu = kmeans.cluster_centers_
    Sigma = np.array([np.eye(D) for _ in range(K)])
    p = np.ones(K) / K
    return mu, Sigma, p
```
```python
mu, Sigma, p = initialize_gmm(bikes[['Speed', 'Length']], 4)
display(mu, Sigma, p)
```

<h3 class="cal cal-h3">The (E)xpectation -Step</h3>

In the E-step, we compute the responsibilities, which represent the probability that each data point belongs to each Gaussian component given the current parameters.
```python
def E_step(x, mu, Sigma, p):
    """E-step of the EM algorithm.
    Computes the posterior probabilities of the latent variables (z) given the data.
    """
    N, D = x.shape
    K = len(p)
    assert(Sigma.shape == (K, D, D))
    assert(mu.shape == (K,D))        
    p_z_given_x = np.zeros((N, K))
    for k in range(K):
        p_z_given_x[:, k] = p[k] * multivariate_normal(mu[k], Sigma[k]).pdf(x)
    p_z_given_x /= p_z_given_x.sum(axis=1, keepdims=True) # Normalize to get probabilities
    return p_z_given_x  

```
```python
p_z_given_x = E_step(bikes[['Speed', 'Length']], mu, Sigma, p)
p_z_given_x.shape
```
```python
p_z_given_x.sum(axis=1)  # Each row should sum to 1
```

<h3 class="cal cal-h3">The (M)aximization -Step</h3>

In this step, we update the parameters of the Gaussian components based on the responsibilities computed in the E-step.
```python
def M_step(x, p_z_given_x):
    """M-step of the EM algorithm.
    Updates the parameters (mu, sigma, p) based on the posterior probabilities.
    """
    N, D = x.shape
    N, K = p_z_given_x.shape
    mu_new = np.zeros((K, D))
    Sigma_new = np.zeros((K, D, D))
    p_new = np.zeros(K)
    
    for k in range(K):
        N_k = p_z_given_x[:, k].sum()
        mu_new[k, :] = p_z_given_x[:, k] @ x / N_k
        Sigma_new[k, :, :] = (p_z_given_x[:, k] * (x - mu_new[k, :]).T @ (x - mu_new[k, :])) / N_k
        Sigma_new[k, :, :] += 1e-3 * np.eye(D)  # Regularization
        p_new[k] = N_k / N

    return mu_new, Sigma_new, p_new

```
```python
M_step(bikes[['Speed', 'Length']], p_z_given_x)
```
```python
def em_algorithm(x, K, max_iters=100, initial_variance=100):
    D = 2
    p = np.ones(K) / K
    # sample initial mu from data
    mu, Sigma, p = initialize_gmm(x, K)
    mu = mu + np.random.randn(*mu.shape) * 3
    soln_path = [(mu, Sigma, p)]
    for i in range(max_iters):
        p_z_given_x = E_step(x, mu, Sigma, p)
        mu, Sigma, p = M_step(x, p_z_given_x)
        soln_path.append((mu, Sigma, p))
    return mu, Sigma, p, soln_path
```
```python
mu, Sigma, p, soln_path = em_algorithm(bikes[['Speed', 'Length']].values, 
                                       K=4, 
                                       max_iters=50)
print("mu", mu)
print("Sigma", Sigma)
```
```python
num_points = 100
speed_pts = np.linspace(bikes['Speed'].min()-3, bikes['Speed'].max()+3, num_points)
length_pts = np.linspace(bikes['Length'].min()-3, bikes['Length'].max()+3, num_points)

mu, Sigma, p = soln_path[-1]
fig = go.Figure()
fig.add_trace(gmm_surface(mu, Sigma, p, speed_pts, length_pts))
fig.update_layout(width=800, height=800)
fig.add_traces(px.scatter(bikes, x='Speed', y='Length', color='scikit gmm').data)
fig.add_scatter(x=mu[:,0], y=mu[:,1], mode='markers', marker=dict(color='black', size=10), name='Centers')
```
```python
from ipywidgets import  IntSlider
np.random.seed(42)
mu, Sigma, p, soln_path = em_algorithm(bikes[['Speed', 'Length']].values, 
                                       K=4, 
                                       max_iters=100)
num_points = 100
speed_pts = np.linspace(bikes['Speed'].min()-3, bikes['Speed'].max()+3, num_points)
length_pts = np.linspace(bikes['Length'].min()-3, bikes['Length'].max()+3, num_points)

mu, Sigma, p = soln_path[0]
fig = go.FigureWidget()
fig.add_trace(gmm_surface(mu, Sigma, p, speed_pts, length_pts))
fig.update_layout(width=800, height=800)
fig.add_traces(px.scatter(bikes, x='Speed', y='Length', color='scikit gmm').data)
fig.add_scatter(x=mu[:,0], y=mu[:,1], mode='markers', marker=dict(color='black', size=10), name='Centers')

def update(step):
    mu, Sigma, p = soln_path[step]
    with fig.batch_update():
        fig.data[0].z = gmm_surface(mu, Sigma, p, speed_pts, length_pts).z
    with fig.batch_update():
        fig.data[-1].x = mu[:, 0]
        fig.data[-1].y = mu[:, 1]
step_slider = IntSlider(min=0, max=len(soln_path)-1, step=1, value=0, description='Step')
interactive_output(update, {'step': step_slider}) 
VBox([fig, step_slider])


```

<h2 class="cal cal-h2">Issue with the MLE of the GMM</h2>


```python
# Ancestor Sampling to create a synthetic dataset
np.random.seed(42)
N = 100

mu = np.array([-1, 2, 5])
pi = np.array([0.2, 0.5, 0.3])
Sigma = np.array([0.2, 0.5, .1])

z = np.random.choice(len(mu), size=N, p=pi)
x = np.random.normal(mu[z], np.sqrt(Sigma[z]))

log_likelihood = np.sum(np.log(np.sum(
    pi[z] * norm.pdf(x[:, None], loc=mu[z], scale=np.sqrt(Sigma[z])), 
    axis=1 )))
```
```python
# Sort for better visualization
ind = z.argsort()
z = z[ind]
x = x[ind]

fig = px.scatter(x=x, y=np.random.rand(N)/20,  
                 title=f'Synthetic Dataset from GMM (Log Likelihood: {log_likelihood:.2f})',
                 opacity = 0.7,
                 color=z.astype(str), labels={'color': 'True Cluster'}, height=400)
u = np.linspace(-4, 9, 1000)
df = pd.DataFrame({'x': u})
for k in range(len(mu)):
    df[f'p{k}'] = pi[k] * norm.pdf(u, loc=mu[k], scale=np.sqrt(Sigma[k]))
fig.add_traces(px.line(df, x='x', y=df.columns[1:], labels={'y': 'Density'}).data)
fig.update_layout(width=800, height=400)

fig
```
```python
mu = np.array([x.min(), x.mean(), x.max()])
Sigma = np.array([1e-100, 10, 1e-100])
pi = np.array([0.3, 0.4, 0.3])

log_likelihood = np.sum(np.log(np.sum(
    pi[z] * norm.pdf(x[:, None], loc=mu[z], scale=np.sqrt(Sigma[z])), 
    axis=1 )))

fig = px.scatter(x=x, y=np.random.rand(N)/20,  
                 title=f'Extreme Values from GMM (Log Likelihood: {log_likelihood:.2f})',
                 opacity = 0.7,
                 color=z.astype(str), labels={'color': 'True Cluster'}, height=400)
u = np.linspace(-4, 9, 100)
u = np.append(u, mu)
u.sort()
df = pd.DataFrame({'x': u})
for k in range(len(mu)):
    df[f'p{k}'] = pi[k] * norm.pdf(u, loc=mu[k], scale=np.sqrt(Sigma[k]))
fig.add_traces(px.line(df, x='x', y=df.columns[1:], labels={'y': 'Density'}).data)
fig.update_layout(width=800, height=400)
fig.update_layout(yaxis_range=[0, 1])

fig
```
