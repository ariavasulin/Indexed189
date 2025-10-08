---
course: CS 189
semester: Fall 2025
type: notebook
number: 12
title: latex for expression
source_type: jupyter_notebook
processed_date: '2025-10-07'
---

<link rel="stylesheet" href="berkeley.css">

<h1 class="cal cal-h1">Lecture 12: Gradient Descent – CS 189, Fall 2025</h1>




```python
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly import figure_factory as ff
colors = px.colors.qualitative.Plotly
px.defaults.width = 800
from ipywidgets import HBox
import numpy as np
```

```python
# make the images folder if it doesn't exist
import os
if not os.path.exists("images"):
    os.makedirs("images")
```

<link rel="stylesheet" href="berkeley.css">

<h2 class="cal cal-h2">Plotting Code you Can Safely Ignore (but need to run).</h2>

This lecture has many complex visualizations and to keep the rest of the code short I have put the visualization code here.  Much of this code is out-of-scope for the course, but feel free to look through it if you are interested.


```python
def make_plot_grid(figs, rows, cols):
    """Create a grid of figures with Plotly figures."""
    from plotly.subplots import make_subplots
    def get_trace_type(fig):
        for trace in fig.data:
            if trace.type == 'surface':
                return 'surface'  # required for go.Surface
            elif trace.type.startswith('scatter3d') or trace.type.startswith('mesh3d'):
                return 'scene'  # 3D scene
        return 'xy'  # default 2D
    specs = [[{'type': get_trace_type(fig)} for fig in figs[i:i+cols]] for i in range(0, len(figs), cols)]
    fig_grid = make_subplots(rows=rows, cols=cols, specs=specs,
                             subplot_titles=[fig.layout.title.text for fig in figs])
    for i, fig in enumerate(figs):
        fig_grid.add_traces(fig.data, rows=(i//cols) + 1, cols=(i%cols) + 1 )
    return fig_grid
```

```python
def plot_lr_predictions(w, cancer):
    """Plot predictions of the logistic model."""
    cancer = cancer.copy()
    cancer['logistic_pred'] = np.where(
        logistic_model(w, cancer[['mean radius', 'mean texture']].values) > 0.5,
        'Pred Malignant', 'Pred Benign'
    )
    # Create a scatter plot with logistic predictions
    fig = px.scatter(cancer, x='mean radius', y='mean texture',
                     symbol='logistic_pred', color='target',
                     symbol_sequence=[ "circle-open", "cross"])
    for (i,t) in enumerate(fig.data): t.legendgroup = str(i)
    # decision boundary
    xs = np.linspace(cancer['mean radius'].min(), cancer['mean radius'].max(), 100)
    decision_boundary = -(w[0] * xs ) / w[1]
    fig.add_scatter(x=xs, y=decision_boundary, mode='lines',
                    name='Decision Boundary', legendgroup='Decision Boundary',
                    line=dict(color='black', width=2, dash='dash', ))
    # probability surface
    ys = np.linspace(cancer['mean texture'].min(), cancer['mean texture'].max(), 100)
    X, Y = np.meshgrid(xs, ys)
    Z = logistic_model(w, np.c_[X.ravel(), Y.ravel()]).reshape(X.shape)
    fig.add_contour(x=xs, y=ys, z=Z,
                    colorscale='Matter_r', opacity=0.5,
                    name='Probability Surface',
                    colorbar=dict(x=1.05, y=0.3, len=0.75))

    fig.update_layout(title=f'w=({w[0]:.2f}, {w[1]:.2f})',
                      xaxis_range=[xs.min(), xs.max()], yaxis_range=[ys.min(), ys.max()],
                      xaxis_title='Mean Radius (scaled)', yaxis_title='Mean Texture (scaled)',
                      width=800, height=600)
    return fig
```

```python
def plot_loss(w1, w2, error, ncontours=50):
    surf_fig = go.Figure()
    surf_fig.add_surface(z=error, x=w1, y=w2,
                         colorscale='Viridis_r', opacity=0.7, showscale=False,
                         contours=dict(z=dict(show=True, highlightcolor="white",
                                          start=error.min(), end=error.max(),
                                          size=(error.max()-error.min())/ncontours)))
    surf_fig.update_layout(title="Loss Surface")
    contour_fig = go.Figure()
    contour_fig.add_contour(x=w1.flatten(), y=w2.flatten(), z=error.flatten(),
                            colorscale='Viridis_r', opacity=0.7,
                            contours=dict(start=error.min(), end=error.max(),
                                          size=(error.max()-error.min())/ncontours),
                            colorbar=dict(x=1.05, y=0.35, len=0.75))
    contour_fig.update_layout(title="Loss Contours")
    fig = make_plot_grid([surf_fig, contour_fig], 1, 2).update_layout(height=800)
    fig.update_layout(scene=dict(xaxis_title='w1', yaxis_title='w2', zaxis_title='Loss', aspectmode='cube'))
    fig.update_layout(xaxis_range=[w1.min(), w1.max()], yaxis_range=[w2.min(), w2.max()],
                      xaxis_title='w1', yaxis_title='w2')
    return fig
```

```python
def plot_gradient(w1, w2, error, dw1, dw2, scale=1.0):
    fig = plot_loss(w1, w2, error)
    fig.add_trace(
        go.Cone(
            x=w1.flatten(), y=w2.flatten(), z=np.zeros_like(error).flatten(),  # Ground plane
            u=dw1.flatten(), v=dw2.flatten(), w=np.zeros_like(error).flatten(),  # No vertical component
            sizeref=2, anchor="tail", showscale=False
        ), 1,1)
    contour_fig = ff.create_quiver(
        x=w1.flatten(), y=w2.flatten(), u=dw1.flatten(), v=dw2.flatten(),
        line_width=2, line_color="white",
        scale = scale, arrow_scale=.2, showlegend=False)
    fig.add_traces(contour_fig.data, rows=1, cols=2)
    return fig
```

```python
def add_solution_path(fig, errors, ws):
    s = np.linspace(0, 1, len(ws))
    fig.add_scatter3d(x=ws[:, 0], y=ws[:, 1], z=errors, marker_color=s, marker_size=5,
                    mode='lines+markers', line=dict(color='black', width=2), opacity=0.5,
                    name='Gradient Descent Path', legendgroup='Gradient Descent',
                    row=1, col=1)
    fig.add_scatter(x=ws[:, 0], y=ws[:, 1], marker_color=s,
                    mode='lines+markers', line=dict(color='black', width=2), opacity=0.5,
                    name='Gradient Descent Path', legendgroup='Gradient Descent',
                    showlegend=False,
                    row=1, col=2)
    fig.add_scatter3d(x=[ws[-1, 0]], y=[ws[-1, 1]], z=[errors[-1]],
                       mode='markers', marker=dict(color='red', size=10),
                       name='Final Solution', legendgroup='Final Solution',
                       row=1, col=1)
    fig.add_scatter(x=[ws[-1, 0]], y=[ws[-1, 1]],
                       mode='markers', marker=dict(color='red', size=20),
                       name='Final Solution', legendgroup='Final Solution',
                       showlegend=False,
                       row=1, col=2)
    return fig
```

<link rel="stylesheet" href="berkeley.css">

<h2 class="cal cal-h2">Optimization Basics</h2>

Consider the following optimization problems?  What is the solution? How would you find it?

**Problem 1**: Minimize the function
$$
\arg \min_{w \in \mathbb{R}}w^2 - 3w + 4
$$

```python
w = np.linspace(-2,7,100)
f = lambda w: w**2 - 3 * w + 4
fig = px.line(x=w, y=f(w), labels={'x': 'w', 'y': 'f(w)'})
fig.update_traces(line_width=5)
# fig.write_image("images/function1.pdf")
fig
```

**Problem 2**: Minimize the function this time with integer contraints
$$
\arg \min_{w \in \mathbb{Z}}w^2 - 3w + 4
$$

```python
w = np.arange(-2, 7)
f = lambda w: w**2 - 3 * w + 4
fig = px.scatter(x=w, y=f(w), labels={'x': 'w', 'y': 'f(w)'})
fig.update_traces(marker_size=7)
# fig.write_image("images/function2.pdf",)
fig
```

**Problem 3**: Minimize the function
\begin{align}
\arg \min_{w \in \mathbb{R}} w^4 - 5 w^2 + w + 4
\end{align}

<!---
$$
\arg \min_{w \in \mathbb{R}} (1-w)(1+w)(2-w)(2+w) + w
$$

```python
import sympy
w = sympy.Symbol('w')
exp = sympy.factor( (1-w)*(1+w)*(2-w)*(2+w) + w)
display(exp)
# latex for expression
sympy.latex(exp)
```
--->

```python
w = np.linspace(-3,3,100)
f = lambda w: w**4 - 5 * w**2 + w + 4
fig = px.line(x=w, y=f(w), labels={'x': 'w', 'y': 'f(w)'})
#increase the line width
fig.update_traces(line_width=5)
fig.update_layout(margin=dict(l=20, r=20, t=20, b=20))
# fig.write_image("images/function3.pdf")
fig
```

<link rel="stylesheet" href="berkeley.css">

<h3 class="cal cal-h3">Convexity</h3>

In this section, we will discuss convex optimization problems and how to solve them.

**Todo** add definitions of convexity.



```python
def plot_secant(w, f, x1, x2):
    """Plot the secant line between two points on the function."""
    y1, y2 = f(x1), f(x2)
    fig = px.line(x=w, y=f(w), labels={'x': 'w', 'y': 'f(w)'})
    fig.add_scatter(x=[x1, x2], y=[f(x1), f(x2)],
                mode='markers+lines', name='Secant Line',
                marker=dict(size=20, color='green'),
                line=dict(color='green', dash="dash"))
    fig.update_traces(line_width=5)
    fig.update_layout(showlegend=False)
    return fig
```

```python
w = np.linspace(-2,7,100)
f = lambda w: w**2 - 3 * w + 4
fig_convex = plot_secant(w, f, -1, 5)
fig_convex.update_layout(title="Convex Function")
# fig_convex.write_image("images/convex_function.pdf")

w = np.linspace(-3,3,100)
f = lambda w: w**4 - 5 * w**2 + w + 4
fig_nonconvex = plot_secant(w, f, -1, 1)
fig_nonconvex.update_layout(title="Non-convex Function")
# fig_nonconvex.write_image("images/non_convex_function.pdf")

fig = make_plot_grid([fig_convex, fig_nonconvex], 1, 2)
fig.update_layout(showlegend=False)
```

<link rel="stylesheet" href="berkeley.css">

<h2 class="cal cal-h2">Understanding the Error Surface</h2>

In this part of the lecture we will visualize the error surface for a few problems.  This will help us understand why gradient descent works and how it finds the optimal solution.



<link rel="stylesheet" href="berkeley.css">

<h3 class="cal cal-h3">The Cross Entropy Loss Surface for Logistic Regression</h3>

In the previous lecture we studied the Logistic Regression model.  The cross entropy loss function is convex.  Here we will visualize a simple two dimensional case of the loss surface.

For this demo, we will use the breast cancer dataset. The dataset is available in the `datasets` module of `sklearn`. To keep things simple, we will focus on just two features of the dataset `"mean radius"`, `"mean texture"`.  We will normalize these features to keep the model simple.  




```python
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
cancer_dict = datasets.load_breast_cancer(as_frame=True)
cancer_df = pd.DataFrame(cancer_dict.data, columns=cancer_dict.feature_names)
cancer_df['target'] = cancer_dict.target.astype(str)
cancer_df = cancer_df[['mean radius', 'mean texture', 'target']].dropna()
scaler = StandardScaler()
cancer_df[['mean radius', 'mean texture']] = scaler.fit_transform(cancer_df[['mean radius', 'mean texture']])
print("The dataset:", cancer_df.shape)
display(cancer_df.head())
px.scatter(cancer_df, x='mean radius', y='mean texture', color='target', opacity=0.7)
```

Here we see there is likely a simple linear decision boundary that separates the two classes.  Let's visualize the decision boundary for a few model parameters.



<link rel="stylesheet" href="berkeley.css">

<h4 class="cal cal-h4">Logistic Regression Model</h4>

Recall that the logistic regression model is given by:

\begin{align}
p(t=1|x) = \sigma(w^T x + b) = \frac{1}{1 + e^{-(w^T x + b)}}
\end{align}

To be able to plot the loss surface we will drop the bias term and consider just two weights $w_1$ and $w_2$.  The resulting model is:

\begin{align}
p(t=1|x) = \sigma(w_1 x_1 + w_2 x_2) = \frac{1}{1 + e^{-(w_1 x_1 + w_2 x_2)}}
\end{align}


```python
def sigmoid(z):
    """Sigmoid function."""
    return 1 / (1 + np.exp(-z))

def logistic_model(w, x):
    """Logistic model for binary classification."""
    z = w @ x.T
    return sigmoid(z)
```

The following code plots the decision boundary for a few different values of $w_1$ and $w_2$.

```python
guesses = np.array([[-1., -1.],[-2., -4.]])
figs = [plot_lr_predictions(w, cancer_df) for w in guesses]
figs[1].update_traces(showlegend=False)
fig = make_plot_grid(figs, 1, 2)
# minor fixup of plots
fig.update_layout(height=600, xaxis_range=figs[0].layout.xaxis.range,
                  yaxis_range=figs[0].layout.yaxis.range)
fig.update_layout(height=600, xaxis2_range=figs[1].layout.xaxis.range,
                  yaxis2_range=figs[1].layout.yaxis.range)
fig
```

<link rel="stylesheet" href="berkeley.css">

<h4 class="cal cal-h4">Plotting the Cross Entropy Loss Surface</h4>

Recall that the average negative log likelihood loss function (cross entropy loss) for logistic regression is given by:

\begin{align}
L(w) = -\frac{1}{N} \sum_{i=1}^N \left( t_i \log(p(t_i|x_i)) + (1 - t_i) \log(1 - p(t_i|x_i)) \right)
\end{align}


```python
def neg_log_likelihood(w):
    """Negative log-likelihood for logistic regression."""
    x = cancer['x']
    t = cancer['t']
    z = w @ x.T
    # return -np.mean(t * np.log(logistic_model(w, x)) + (1 - t) * np.log(1 - logistic_model(w, x)))
    # more numerically stable version
    # np.mean(np.log(1+ np.exp(z)) - t * z)
    # stable softplus: log(1+exp(z))
    softplus_z = np.logaddexp(0, z)
    return np.mean(softplus_z - t * z)
```

For visualization purposes I am going to create a dictionary to track all the variables I am creating.

```python
cancer = dict()
cancer['npts'] = 30
cancer['w1'], cancer['w2'] = np.meshgrid(
    np.linspace(-10, 1, cancer['npts']),
    np.linspace(-5, 1.3, cancer['npts'])
)
cancer['ws'] = np.stack([cancer['w1'].flatten(), cancer['w2'].flatten()]).T
cancer['x'] = cancer_df[['mean radius', 'mean texture']].values
cancer['t'] = cancer_df['target'].values.astype(float)
```

```python
cancer['error'] = np.array([neg_log_likelihood(w)
                for w in cancer['ws']])
cancer['error'] = cancer['error'].reshape(cancer['w1'].shape)
fig = plot_loss(cancer['w1'], cancer['w2'], cancer['error'])
for i, g in enumerate(guesses):
    fig.add_scatter3d(x=[g[0]], y=[g[1]], z=[neg_log_likelihood(g)],
                      mode='markers', marker=dict(size=10, color=colors[i + 2]),
                      name=f'Weight {g}', legendgroup=str(i), row=1, col=1)
    fig.add_scatter(x=[g[0]], y=[g[1]], mode='markers',
                    marker=dict(size=10, color=colors[i + 2]),
                    name=f'Weight {g}', legendgroup=str(i), showlegend=False,
                    row=1, col=2)
fig
```

<link rel="stylesheet" href="berkeley.css">

<h4 class="cal cal-h4">Choosing the Best Parameters</h4>

We have computed the negative log likelihood loss surface for the logistic regression model for many parameter values.  Here we can simply choose the parameters with the lowest loss value.

```python
best_ind = np.argmin(cancer['error'])
cancer['grid_best'] = np.array([cancer['w1'].flatten()[best_ind],
                                cancer['w2'].flatten()[best_ind]])

fig = plot_loss(cancer['w1'], cancer['w2'], cancer['error'])
for i, g in enumerate(guesses):
    fig.add_scatter3d(x=[g[0]], y=[g[1]], z=[neg_log_likelihood(g)],
                      mode='markers', marker=dict(size=10, color=colors[i + 2]),
                      name=f'Weight {g}', legendgroup=str(i), row=1, col=1)
    fig.add_scatter(x=[g[0]], y=[g[1]], mode='markers',
                    marker=dict(size=10, color=colors[i + 2]),
                    name=f'Weight {g}', legendgroup=str(i), showlegend=False,
                    row=1, col=2)
fig.add_scatter3d(x=[cancer['grid_best'][0]], y=[cancer['grid_best'][1]], z=[cancer['error'].min()],
    mode='markers', marker=dict(size=10, color='red'),
    name=f"Best Weight [{cancer['grid_best'][0]:0.2f}, {cancer['grid_best'][1]:0.2f}]",
    legendgroup=f"Best",
    row=1, col=1)
fig.add_scatter(x=[cancer['grid_best'][0]], y=[cancer['grid_best'][1]],
    mode='markers', marker=dict(size=10, color='red'),
    name=f"Best Weight [{cancer['grid_best'][0]:0.2f}, {cancer['grid_best'][1]:0.2f}]",
    legendgroup=f"Best",
    showlegend=False, row=1, col=2)

fig
```

```python
plot_lr_predictions(cancer['grid_best'], cancer_df)
```

<link rel="stylesheet" href="berkeley.css">

<h3 class="cal cal-h3">The Squared Error Loss Squared Loss on a Non-Linear Model</h3>

In the following we will visualize a more complex loss surface.  We will use a non-linear model and the squared error loss function.  

```python
sine = dict()
np.random.seed(42)
sine['n'] = 200
# Generate some random data
sine['x'] = np.random.rand(sine['n']) * 2.5 * np.pi
sine['x'] = np.sort(sine['x']) # sort for easier plotting
sine['y'] = np.sin(1.1 + 2.5 * sine['x']) + 0.5 * np.random.randn(sine['n'])
sine_df = pd.DataFrame({'x': sine['x'], 'y': sine['y']})
```

Because we made this data we know that true underlying function that determines the data is a sine wave.

$$
\hat{y} = f_w(x) = \sin\left(1.1 + 2.5 x \right)
$$

We can visualize the data.

```python
fig = px.scatter(sine_df, x='x', y='y')
fig.update_traces(marker_color='black')
data_trace = fig.data[0]
# fig.write_image("images/sine_data.pdf", width=800, height=400)
fig
```

<link rel="stylesheet" href="berkeley.css">

<h4 class="cal cal-h4">Non-linear Sine Regression Model</h4>

Here we will use a model of the form:
\begin{align}
y = f(x) = \sin(w_0  + w_1 x)
\end{align}



```python
def sine_model(w, x):
    return np.sin(w[0] + x * w[1])
```

Let's try a few different parameter values and see how this model looks on our data.  We will try the following three possible models:

```python
sine['guesses'] = np.array([[0, 2], [2, 3], [0, 3.5]])
```

Here we make predictions at 100 test points for each model and plot the results.

```python
sine['xhat'] = np.linspace(sine_df['x'].min(), sine_df['x'].max(), 100)
sine['pred_df'] = pd.DataFrame({'x': sine['xhat']})
for w in sine['guesses']:
    sine['pred_df'][f'yhat(w={w})'] = sine_model(w, sine['xhat'])
sine['pred_df'].head()
```

```python
fig = go.Figure()
for i, w in enumerate(sine['pred_df'].columns[1:]):
    fig.add_trace(go.Scatter(x=sine['pred_df']['x'], y=sine['pred_df'][w],
                             mode='lines', name=w,
                             line=dict(width=4, color=colors[i+2])))
fig.update_traces(line_width=4)
fig.add_trace(data_trace)
fig.update_layout(margin=dict(l=20, r=20, t=20, b=20),
                  xaxis_title='x', yaxis_title='y')
# fig.write_image("images/sine_with_3_models.pdf", width=800, height=400)
fig
```

None of the above really match the data.  We would like to find a parameterization that is closer to the data. To do this we need a loss function.

<link rel="stylesheet" href="berkeley.css">

<h4 class="cal cal-h4">Visualizing Squared Loss</h4>

To really fit the data we need a measure of how good our model is relative to the data.  This is a loss function.  For this exercise we will use the **average squared loss** which is often just called the squared loss.

$$
E(w;\mathcal{D}) = L\left(f_w; \mathcal{D} = \left\{(x_i, y_i \right\}_{i=1}^n\right) = \frac{1}{n} \sum_{i=1}^n\left(y_i - f_w\left(x_i\right)\right)^2
= \frac{1}{n} \sum_{i=1}^n\left(y_i - \sin\left(w_0 + w_1 x_i \right)\right)^2
$$

```python
def sine_MSE(w):
    x = sine['x']
    y = sine['y']
    y_hat = sine_model(w, x)
    return np.mean((y - y_hat) ** 2)
```

Here, $w_0$ and $w_1$ are the parameters of our model.  We can visualize this loss function as a surface in 3D.  To do this we will create a grid of points in the parameter space and evaluate the loss function at each point.

```python
sine['npts'] = 30
sine['w0'], sine['w1'] = np.meshgrid(
    np.linspace(-1.5, 3, sine['npts']), np.linspace(1, 4, sine['npts']))
# combine w1 and w2 into a single tensor
sine['ws'] = np.stack([sine['w0'].flatten(), sine['w1'].flatten()]).T

sine['error'] = np.array([sine_MSE(w) for w in sine['ws']]).reshape(sine['w0'].shape)

fig = plot_loss(sine['w0'], sine['w1'], sine['error'])
for i, w in enumerate(sine['guesses']):
    fig.add_trace(go.Scatter3d(x=[w[0]], y=[w[1]], z=[sine_MSE(w)],
                               mode='markers', marker=dict(size=5, color=colors[i+2]),
                               name=f'w=({w[0]}, {w[1]})', legendgroup=str(i)),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=[w[0]], y=[w[1]],
                               mode='markers', marker=dict(size=20, color=colors[i+2]),
                               name=f'w=({w[0]}, {w[1]})', legendgroup=str(i), showlegend=False),
                  row=1, col=2)

fig
```

<link rel="stylesheet" href="berkeley.css">

<h4 class="cal cal-h4">Choosing the Best Parameters</h4>

Just as before, we can simply choose the parameters with the lowest loss value.

```python
ind = np.argmin(sine['error'])
sine['grid_best'] = sine['ws'][ind,:]
sine['grid_best_error'] = sine['error'].flatten()[ind]
print(f"Best weights: {sine['grid_best']}, with error: {sine['grid_best_error']}")
```

```python
fig = plot_loss(sine['w0'], sine['w1'], sine['error'])
for i, w in enumerate(sine['guesses']):
    fig.add_trace(go.Scatter3d(x=[w[0]], y=[w[1]], z=[sine_MSE(w)],
                               mode='markers', marker=dict(size=5, color=colors[i+2]),
                               name=f'w=({w[0]}, {w[1]})', legendgroup=str(i)),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=[w[0]], y=[w[1]],
                               mode='markers', marker=dict(size=20, color=colors[i+2]),
                               name=f'w=({w[0]}, {w[1]})', legendgroup=str(i), showlegend=False),
                  row=1, col=2)
fig.add_scatter3d(x=[sine['grid_best'][0]], y=[sine['grid_best'][1]], z=[sine['grid_best_error']],
    mode='markers', marker=dict(size=10, color='red'),
    name=f"Best Weight [{sine['grid_best'][0]:0.2f}, {sine['grid_best'][1]:0.2f}]", legendgroup=f"Best",
    row=1, col=1)
fig.add_scatter(x=[sine['grid_best'][0]], y=[sine['grid_best'][1]],
    mode='markers', marker=dict(size=20, color='red'),
    name=f"Best Weight [{sine['grid_best'][0]:0.2f}, {sine['grid_best'][1]:0.2f}]", legendgroup=f"Best",
    showlegend=False, row=1, col=2)

fig
```

Plotting the best fit model on the data we see that it is a much better fit than our previous guesses.

```python
fig = go.Figure()
for i, w in enumerate(sine['pred_df'].columns[1:]):
    fig.add_trace(go.Scatter(x=sine['pred_df']['x'], y=sine['pred_df'][w], mode='lines', name=w,
                             line=dict(width=4, color=colors[i+2])))
fig.update_traces(line_width=4)
fig.add_trace(data_trace)
fig.update_layout(margin=dict(l=20, r=20, t=20, b=20),
                  xaxis_title='x', yaxis_title='y')
sine['pred_df']['grid_best'] = sine_model(sine['grid_best'], sine['xhat'])
fig.add_scatter(x=sine['pred_df']['x'], y=sine['pred_df']['grid_best'],
                mode='lines',
                name=f"Best w=({sine['grid_best'][0]:.2f}, {sine['grid_best'][1]:.2f})",
                line=dict(width=8, color='red')
)
# fig.write_image("images/sine_with_3_models.pdf", width=800, height=400)
fig
```

<link rel="stylesheet" href="berkeley.css">

<h3 class="cal cal-h3">Slido Visualization</h3>


```python
fig1 = px.scatter(x=cancer['x'][:,0],
                  y=cancer['t'] + 0.01*np.random.normal(size=cancer['t'].shape),
                  color=cancer['t'].astype(str), title="Cancer Data (Jittered)")

ws = np.linspace(-20, 3, 100)
nll = np.array([
    np.mean(np.log1p(np.exp(-w*cancer['x'][:,0])) - cancer['t'] * (w*cancer['x'][:,0]))
    for w in ws])

fig2 = px.line(x=ws, y=nll, labels={'x': 'w', 'y': 'Negative Log-Likelihood'})
ind = np.argmin(nll)
best_w = ws[ind]
fig2.add_vline(x=best_w, line=dict(color='red', dash='dash'),
               annotation_text=f"Best w={best_w:.2f}",
               annotation_position="top right")
fig2.add_scatter(x=[best_w], y=[nll[ind]], mode='markers',
                 marker=dict(color='red', size=10),
                 name="Best w")
xtest = np.linspace(cancer['x'][:,0].min(), cancer['x'][:,0].max(), 100)
fig1.add_scatter(x=xtest,
                 y=sigmoid(best_w * xtest),
                 mode='lines', line=dict(color='black', width=4),
                 name=f'Logistic Model (w={best_w:.2f})')
fig = make_plot_grid([fig1, fig2], 1, 2)
fig.update_layout(height=600)
```

```python
fig1 = px.scatter(x=cancer['x'][:,0],
                  y=cancer['x'][:,1], title="Cancer Data")
ws = np.linspace(-5, 5, 100)
sqloss = [np.mean((cancer['x'][:,1] - (w * cancer['x'][:,0]))**2) for w in ws]
fig2 = px.line(x=ws, y=sqloss, labels={'x': 'w', 'y': 'Squared Loss'})
ind = np.argmin(sqloss)
best_w = ws[ind]
xtest = np.linspace(cancer['x'][:,0].min(), cancer['x'][:,0].max(), 100)
fig1.add_scatter(x=xtest,
                 y=best_w * xtest,
                 mode='lines', line=dict(color='black', width=4),
                 name=f'Linear Model (w={best_w:.2f})')
fig2.add_vline(x=best_w, line=dict(color='red', dash='dash'),
               annotation_text=f"Best w={best_w:.2f}",
               annotation_position="top right")
fig2.add_scatter(x=[best_w], y=[sqloss[ind]], mode='markers',
                 marker=dict(color='red', size=10),
                 name="Best w")
fig = make_plot_grid([fig1, fig2], 1, 2)
fig.update_layout(height=600)
```

```python
offset = 3
fig1 = px.scatter(x=cancer['x'][:,0]+offset,
                  y=cancer['t'] + 0.01*np.random.normal(size=cancer['t'].shape),
                  color=cancer['t'].astype(str), title="Cancer Data (Jittered)")
#fig1.show()
w1,w2 = np.meshgrid(np.linspace(5,15, 40), np.linspace(-8, 0, 40))
ws = np.stack([w1.flatten(), w2.flatten()]).T
nll = np.array([
    np.mean(np.mean(np.logaddexp(0, w[1]*(cancer['x'][:,0]+offset) + w[0])) -
    cancer['t'] * (w[1]*(cancer['x'][:,0]+offset) + w[0]))
    for w in ws]).reshape(w1.shape)

fig2 = plot_loss(w1, w2, nll)
ind = np.argmin(nll)
best_w = ws[ind,:]
fig2.add_scatter3d(x=[best_w[0]], y=[best_w[1]], z=[nll.flatten()[ind]],
                   mode='markers', marker=dict(size=10, color='red'),
                   name=f'Best w=({best_w[0]:.2f}, {best_w[1]:.2f})', legendgroup='Best',
                   row=1, col=1)
fig2.add_scatter(x=[best_w[0]], y=[best_w[1]],
                   mode='markers', marker=dict(size=20, color='red'),
                   name=f'Best w=({best_w[0]:.2f}, {best_w[1]:.2f})', legendgroup='Best',
                   showlegend=False,
                   row=1, col=2)
fig2.show()
xtest = np.linspace(cancer['x'][:,0].min()+offset, cancer['x'][:,0].max()+offset, 100)
fig1.add_scatter(x=xtest,
                 y=sigmoid(best_w[1] * xtest + best_w[0]),
                 mode='lines', line=dict(color='black', width=4),
                 name=f'Logistic Model (w=({best_w[0]:.2f}, {best_w[1]:.2f}))')
fig1.update_layout(height=600)
```

<link rel="stylesheet" href="berkeley.css">

<h2 class="cal cal-h2">Visualizing the Gradient of the Error Function</h2>

In the following, we will visualize the gradient of the error function on top of the error surface.  


<link rel="stylesheet" href="berkeley.css">

<h3 class="cal cal-h3">The Gradient of the Cross Entropy Loss for Logistic Regression</h3>

Recall that the logistic regression model is given by (here we drop the bias term for simplicity):
$$
p(t=1|x) = \sigma(w^T x) = \frac{1}{1 + e^{-(w^\top x )}}
$$
To compute the gradient of the negative log likelihood loss function (cross entropy loss), we need to first define the loss function. To simplify some of the future steps we will use the **average** negative log likelihood loss function.

\begin{align*}
L(w) &= -\frac{1}{N} \sum_{i=1}^N \left( t_i \log(p(t_i|w, x_i)) + (1 - t_i) \log(1 - p(t_i|w, x_i)) \right) \\
L(w) &= -\frac{1}{N} \sum_{i=1}^N \left( t_i \log(\sigma(w^\top x_i)) + (1 - t_i) \log(1 - \sigma(w^\top x_i)) \right) \\
\end{align*}


There is a useful identity that we can use to simplify the computation of the gradient and Hessian.  The derivative of the sigmoid function is given by:
$$
\frac{\partial}{\partial z} \sigma(z) = \sigma(z)(1 - \sigma(z)) = \sigma(z)\sigma(-z)$$

So we can compute the $j^{th}$ term of the gradient as follows:
\begin{align*}
\frac{\partial L}{\partial w_j}
&= -\frac{1}{N} \sum_{i=1}^N \left( t_i \frac{\partial }{\partial w_j}\log(\sigma(w^T x_i)) + (1 - t_i) \frac{\partial }{\partial w_j}\log(1 - \sigma(w^T x_i)) \right) \\
&= -\frac{1}{N} \sum_{i=1}^N \left( t_i \frac{1}{\sigma(w^T x_i)} \frac{\partial }{\partial w_j}\sigma(w^T x_i) + (1 - t_i) \frac{1}{1 - \sigma(w^T x_i)} \frac{\partial }{\partial w_j}(1 - \sigma(w^T x_i)) \right)\\
&= -\frac{1}{N} \sum_{i=1}^N \left( t_i \frac{1}{\sigma(w^T x_i)} \sigma(w^T x_i)(1 - \sigma(w^T x_i)) x_i + (1 - t_i) \frac{1}{1 - \sigma(w^T x_i)} (-\sigma(w^T x_i)(1 - \sigma(w^T x_i))) x_{ij} \right)\\
&= -\frac{1}{N} \sum_{i=1}^N \left( t_i (1 - \sigma(w^T x_i)) x_{ij} - (1 - t_i) \sigma(w^T x_i) x_{ij} \right)\\
&= -\frac{1}{N} \sum_{i=1}^N \left( t_i x_{ij} - \sigma(w^T x_i) x_{ij} \right)\\
&= -\frac{1}{N} \sum_{i=1}^N \left( t_i x_{ij} - \sigma(w^T x_i) x_{ij} \right)\\
&= \frac{1}{N} \sum_{i=1}^N \left(\sigma(w^T x_i) - t_i \right) x_{ij}
\end{align*}

```python
def grad_NLL(w):
    """Compute the gradient of the negative log-likelihood."""
    p = logistic_model(w, cancer['x'])
    grad = np.mean((p - cancer['t']).reshape(-1, 1) * cancer['x'], 0)
    return grad

grad_NLL(np.array([-1, 2]))
```

We can now visualize the gradient of the loss function for this more complex model.

```python
(cancer['dw1'], cancer['dw2']) = np.array(
    [grad_NLL(w) for w in cancer['ws']]).T
cancer['dw1'] = cancer['dw1'].reshape(cancer['w1'].shape)
cancer['dw2'] = cancer['dw2'].reshape(cancer['w1'].shape)
# fig.write_image("images/loss_surface_3d_with_gradients.pdf", width=800, height=800)
fig = plot_gradient(cancer['w1'], cancer['w2'], cancer['error'], cancer['dw1'], cancer['dw2'], scale=2)
fig
```

<link rel="stylesheet" href="berkeley.css">

<h3 class="cal cal-h3">The Gradient Squared Error for our Sine Model</h3>

Recall that our sine model is given by:
\begin{align}
y = f(x) = \sin(w_0  + w_1 x)
\end{align}
and our squared loss function is given by:

$$
E(w;\mathcal{D}) = L\left(f_w; \mathcal{D} = \left\{(x_n, y_n) \right\}_{n=1}^N\right) = \frac{1}{N} \sum_{n=1}^N\left(y_n - f_w\left(x_n\right)\right)^2
= \frac{1}{N} \sum_{n=1}^N\left(y_n - \sin\left(w_0 + w_1 x_n \right)\right)^2
$$

Taking the gradient we get:

\begin{align*}
\nabla E(w;\mathcal{D}) =
\begin{bmatrix}
\frac{\partial E}{\partial w_0} \\
\frac{\partial E}{\partial w_1}
\end{bmatrix}
\end{align*}

Calculating each term we get:

\begin{align*}
\frac{\partial E}{\partial w_0}
&= \frac{1}{N} \sum_{n=1}^N 2\left(y_n - \sin\left(w_0 + w_1 x_n \right)\right) \left(-\cos\left(w_0 + w_1 x_n \right)\right) \\
&= -\frac{2}{N} \sum_{n=1}^N \left(y_n - \sin\left(w_0 + w_1 x_n \right)\right) \cos\left(w_0 + w_  1 x_n \right)
\end{align*}

\begin{align*}
\frac{\partial E}{\partial w_1}
&= \frac{1}{N} \sum_{n=1}^N 2\left(y_n - \sin\left(w_0 + w_1 x_n \right)\right) \left(-\cos\left(w_0 + w_1 x_n \right)\right) x_n \\
&= -\frac{2}{N} \sum_{n=1}^N \left(y_n - \sin\left(w_0 + w_1 x_n \right)\right) \cos\left(w_0 + w_1 x_n \right) x_n
\end{align*}

```python
def grad_sine_MSE(w):
    """Compute the gradient of the negative log-likelihood."""
    x = sine['x']
    y = sine['y']
    y_hat = sine_model(w, x)
    grad_w0 = -2 * np.mean((y - y_hat) * np.cos(w[0] + w[1] * x))
    grad_w1 = -2 * np.mean((y - y_hat) * x * np.cos(w[0] + w[1] * x))
    return np.array([grad_w0, grad_w1])

grad_sine_MSE(np.array([0, 2]))
```

```python
(sine['dw0'], sine['dw1']) = np.array(
    [grad_sine_MSE(w) for w in sine['ws']]).T
sine['dw0'] = sine['dw0'].reshape(sine['w1'].shape)
sine['dw1'] = sine['dw1'].reshape(sine['w1'].shape)
fig = plot_gradient(sine['w0'], sine['w1'], sine['error'], sine['dw0'], sine['dw1'], scale=0.1)
fig
```

<link rel="stylesheet" href="berkeley.css">

<h2 class="cal cal-h2">The Gradient Descent Algorithm</h2>

Here we implement the most basic version of gradient descent.  This version uses a fixed step size and does not use any fancy tricks like momentum or adaptive step sizes (which we will see soon).  


```python
def gradient_descent(w_0, gradient,
    learning_rate=1, nepochs=10, epsilon=1e-6):
    """Basic gradient descent algorithm.
    Args:
        w_0: Initial weights (numpy array).
        gradient: Function to compute the gradient.
        learning_rate: Step size for each iteration.
        nepochs: Maximum number of iterations.
        epsilon: Convergence threshold.
    Returns:
        path: Array of weights at each iteration."""
    w_old = w_0
    path = [w_old]
    for e in range(nepochs):
        w = w_old - learning_rate * gradient(w_old)
        path.append(w)
        if np.linalg.norm(w - w_old) < epsilon: break
        w_old = w
    return np.array(path)
```

<link rel="stylesheet" href="berkeley.css">

<h3 class="cal cal-h3">Gradient Descent Applied to Logistic Regression</h3>


```python
# w0 = np.array([-10., -5.])
# w0 = np.array([-1., 2.])
w0 = np.array([0., 0])
path = gradient_descent(w0,
                     grad_NLL,
                     learning_rate=10,
                     nepochs=100)
errors = [neg_log_likelihood(w) for w in path]
fig = plot_gradient(
    cancer['w1'], cancer['w2'], cancer['error'],
    cancer['dw1'], cancer['dw2'], scale=2)
add_solution_path(fig, errors, path)
```

<link rel="stylesheet" href="berkeley.css">

<h3 class="cal cal-h3">Gradient Descent Applied to Sine Model</h3>


```python
w0 = np.array([1.2, 2.])
w0 = np.array([.5, 2.5])
w0 = np.array([1, 3])
path = gradient_descent(w0,
                        grad_sine_MSE,
                        learning_rate=.2,
                        nepochs=20)
errors = [sine_MSE(w) for w in path]
fig = plot_gradient(
    sine['w0'], sine['w1'], sine['error'],
    sine['dw0'], sine['dw1'], scale=.1)
add_solution_path(fig, errors, path)
```

<link rel="stylesheet" href="berkeley.css">

<h3 class="cal cal-h3">Loss Curves</h3>

In practice, we are typically unable to visualize the loss surface.  Instead, we can plot the loss value as a function of iteration number.  This is called a loss curve.  Here we plot the loss curve for gradient descent on our sine model.


```python
def make_loss_curve(path, error_func):
    """Make a loss curve from a path of weights."""
    errors = [error_func(w) for w in path]
    fig = px.line(x=np.arange(len(errors)), y=errors,
                  labels={'x': 'Iteration (Gradient Steps)', 'y': 'Loss (Error)'})
    fig.update_traces(line_width=4)
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20))
    return fig
```

```python
w0 = np.array([2.5, 1.9])
path = gradient_descent(w0,
                        grad_sine_MSE,
                        learning_rate=.1,
                        nepochs=50)

fig_loss = make_loss_curve(path, sine_MSE)
fig_loss.show()

errors = [sine_MSE(w) for w in path]
fig = plot_gradient(sine['w0'], sine['w1'], sine['error'],
    sine['dw0'], sine['dw1'], scale=.1)
add_solution_path(fig, errors, path)
fig.show()
```

```python
w0 = np.array([-10., -5.])
# w0 = np.array([-1., 2.])
# w0 = np.array([0., 0])
path = gradient_descent(w0,
                     grad_NLL,
                     learning_rate=10,
                     nepochs=100)

fig_loss = make_loss_curve(path, neg_log_likelihood)
fig_loss.show()

errors = [neg_log_likelihood(w) for w in path]
fig = plot_gradient(
    cancer['w1'], cancer['w2'], cancer['error'],
    cancer['dw1'], cancer['dw2'], scale=2)
add_solution_path(fig, errors, path)
```

<link rel="stylesheet" href="berkeley.css">

<h3 class="cal cal-h3">Slido Visualization</h3>


```python
x = np.linspace(-10, 10, 100)
y = x**2 + 1
fig = px.line(x=x, y=y, labels={'x': 'x', 'y': 'f(x)'})
fig.update_traces(line_width=5)

def grad_f(x):
    return 2 * x
w0 = 2
path = gradient_descent(w0, grad_f, learning_rate=1.08, nepochs=10)
fig.add_scatter(x=path, y=path**2 + 1, mode='markers+lines',
                marker=dict(size=10, color='red'),
                line=dict(color='black', width=2, dash="dash"),
                name='Gradient Descent Path')
```

<link rel="stylesheet" href="berkeley.css">

<h2 class="cal cal-h2">The Second Order Structure</h2>

Here we examine the second order structure of the loss function through a Taylor expansion.  

<link rel="stylesheet" href="berkeley.css">

<h3 class="cal cal-h3">The Hessian of the Logistic Regression Model</h3>

\begin{align}
\text{H} &= \begin{bmatrix}
\frac{\partial^2 E}{\partial w_0^2} & \frac{\partial^2 E}{\partial w_0 \partial w_1} \\
\frac{\partial^2 E}{\partial w_1 \partial w_0} & \frac{\partial^2 E}{\partial w_1^2}
\end{bmatrix}\\
\end{align}

We can start with the gradient:

\begin{align*}
\frac{\partial L}{\partial w_j}
&= \frac{1}{N} \sum_{i=1}^N \left( p(t_i|x_i) - t_i  \right) x_{ij}
= \frac{1}{N} \sum_{i=1}^N \left(  \sigma(w^T x_i) - t_i \right) x_{ij}
\end{align*}

Using the product rule and chain rule, we can compute the second derivatives of the loss function with respect to the parameters $w_k$:
\begin{align*}
\frac{\partial^2 }{\partial w_k} \frac{\partial L}{\partial w_j}   
&= \frac{1}{N} \sum_{i=1}^N \frac{\partial^2 }{\partial w_k}\left(  \sigma(w^T x_i) - t_i \right) x_{ij}\\
&= \frac{1}{N} \sum_{i=1}^N  x_{ij} \frac{\partial^2 }{\partial w_k}\sigma(w^T x_i) \\
&= \frac{1}{N} \sum_{i=1}^N  x_{ij} \sigma(w^T x_i)(1 - \sigma(w^T x_i)) x_{ik}
\end{align*}


```python
def hessian_NLL(w):
    """Compute the Hessian of the negative log-likelihood."""
    x = cancer['x']
    p = logistic_model(w, x)
    hessian = (x.T @ np.diag(p * (1 - p)) @ x ) / len(p)
    return hessian
```

<link rel="stylesheet" href="berkeley.css">

<h3 class="cal cal-h3">The Hessian of the Sine Regression Model</h3>

\begin{align}
\text{H} &= \begin{bmatrix}
\frac{\partial^2 E}{\partial w_0^2} & \frac{\partial^2 E}{\partial w_0 \partial w_1} \\
\frac{\partial^2 E}{\partial w_1 \partial w_0} & \frac{\partial^2 E}{\partial w_1^2}
\end{bmatrix}\\
\end{align}

We can start with the gradient:
Calculating each term we get:

\begin{align*}
\frac{\partial E}{\partial w_0}
&= -\frac{2}{N} \sum_{n=1}^N \left(y_n - \sin\left(w_0 + w_1 x_n \right)\right) \cos\left(w_0 + w_  1 x_n \right)
\end{align*}

\begin{align*}
\frac{\partial E}{\partial w_1}
&= -\frac{2}{N} \sum_{n=1}^N \left(y_n - \sin\left(w_0 + w_1 x_n \right)\right) \cos\left(w_0 + w_1 x_n \right) x_n
\end{align*}


Using the product rule and chain rule, we can compute the second derivatives of the loss function with respect to the parameters $w_0$ and $w_1$.
\begin{align}
\frac{\partial^2 E}{\partial w_0^2} &=
\frac{2}{N} \sum_{n=1}^N \left[ \cos^2(w_0 + w_1 x_n) + (y_n - \sin(w_0 + w_1 x_n)) \sin(w_0 + w_1 x_n) \right]\\
\frac{\partial^2 E}{\partial w_1^2} &=
\frac{2}{N} \sum_{n=1}^N \left[ \cos^2(w_0 + w_1 x_n) x_n^2 + (y_n - \sin(w_0 + w_1 x_n)) \sin(w_0 + w_1 x_n) x_n^2 \right]\\
\frac{\partial^2 E}{\partial w_0 \partial w_1} &=
\frac{2}{N} \sum_{n=1}^N \left[ \cos^2(w_0 + w_1 x_n) x_n + (y_n - \sin(w_0 + w_1 x_n)) \sin(w_0 + w_1 x_n) x_n \right]
\end{align}


```python
def hessian_sine_MSE(w):
    """Compute the Hessian of the negative log-likelihood."""
    x = sine['x']
    y = sine['y']
    dw0dw0 = 2 * np.mean(np.cos(w[0] + w[1] * x)**2 +
                (y - np.sin(w[0] + w[1] * x)) * np.sin(w[0] + w[1] * x))
    dw0dw1 = 2 * np.mean(x * np.cos(w[0] + w[1] * x)**2 +
                x * (y - np.sin(w[0] + w[1] * x)) * np.sin(w[0] + w[1] * x))
    dw1dw1 = 2 * np.mean(x**2 * np.cos(w[0] + w[1] * x)**2 +
                (x**2) * (y - np.sin(w[0] + w[1] * x)) * np.sin(w[0] + w[1] * x))
    hessian = np.array([[dw0dw0, dw0dw1], [dw0dw1, dw1dw1]])
    return hessian
```

We can also derive the gradient using a symbolic algebra library.  This is a powerful technique that allows us to compute the gradient of a function without having to derive it by hand.  We will use the `sympy` library to do this.


```python
# unfortunately on colab we need to upgrade sympy to get the needed functionality
# !pip install --upgrade sympy
```

```python
# import sympy as sp
# # define our symbols
# w0, w1, x, y = sp.symbols('w0 w1 x y')
# # Define a symbolic expression for the error
# E = (y - sp.sin(w0 + w1 * x))**2
# # Compute the gradient of E with respect to w0 and w1
# gE = [sp.diff(E, var) for var in (w0, w1)]
# gE
```

```python
# hessian_E = sp.hessian(E, (w0, w1))
# hessian_E
```

Using sympy to do everything algebraically we get the same result.

```python
# n = sp.symbols('n', integer=True)    # number of data points (symbolic length)
# i = sp.Idx('i', n)                   # index variable for summation
# w0, w1 = sp.symbols('w0 w1')         # weights
# x = sp.IndexedBase('x', shape=(n,))  # indexed variable for x data
# y = sp.IndexedBase('y', shape=(n,))  # indexed variable for y data

# E_i = (y[i] - sp.sin(w0 + w1 * x[i]))**2
# E = sp.summation(E_i, (i, 0, n - 1)) / n
# H = sp.hessian(E, (w0, w1))

# Hfun = sp.lambdify((w0, w1, x, y, n), H)
# def hessian_sine_MSE2(w):
#     return np.array(Hfun(w[0], w[1], sine['x'], sine['y'], len(sine['x'])))
```

```python
# hessian_sine_MSE(np.array([1.1, 2.5]))
```

```python
# hessian_sine_MSE2(np.array([1.1, 2.5]))
```

<link rel="stylesheet" href="berkeley.css">

<h3 class="cal cal-h3">Taylor Expansion of the Loss</h3>

The Taylor expansion of a function $f(x)$ around a point $x=a$ is given by:

$$
f(x) = f(a) + \nabla f(a)^T (x - a) + \frac{1}{2} (x - a)^T H(a) (x - a) + \ldots
$$

```python
def taylor_loss(w, w_star, L_star, g_star, H_star):
    delta = w - w_star
    return (L_star + delta @ g_star + 1/2 * delta.T @ H_star @ delta)
```

<link rel="stylesheet" href="berkeley.css">

<h4 class="cal cal-h4">Applied to Logistic Regression</h4>


```python
# w_star = np.array([-2., -2.])
w_star = cancer['grid_best']
L_star = neg_log_likelihood(w_star)
g_star = grad_NLL(w_star)
H_star = hessian_NLL(w_star)
s,u = np.linalg.eigh(H_star)
print("w_star:", w_star)
print("L_star:", L_star)
print("g_star:", g_star)
print("H_star:\n", H_star)
print("Eigenvalues of H_star:", s)
print("Eigegenvectors of H_star:\n", u)
```

Visualizing the Hessian at the optimal point we see that it is positive definite (both eigenvalues are positive).  This is consistent with the fact that the loss function is convex.

```python
fig = plot_gradient(cancer['w1'], cancer['w2'],
                    cancer['error'], cancer['dw1'], cancer['dw2'])

cancer['taylor_loss'] = np.array([
    taylor_loss(w, w_star, L_star, g_star, H_star)
for w in cancer['ws']]).reshape(cancer['w1'].shape)

fig.add_surface(z=cancer['taylor_loss'], x=cancer['w1'], y=cancer['w2'],
                 colorscale='plasma_r', opacity=0.5, showscale=False,
                 contours=dict(z=dict(show=True, highlightcolor="white",
                                      start=cancer['taylor_loss'].min(), end=cancer['taylor_loss'].max(),
                                      size=(cancer['taylor_loss'].max()-cancer['taylor_loss'].min())/50)),
                                      row=1, col=1)
fig.update_layout(scene=dict(zaxis=dict(range=[0, 3])))
# fig.write_image("images/loss_surface_3d_with_gradients.pdf", width=800, height=800)
display(fig)
```

We can visualize the eigenvectors of the Hessian at the optimal point on the quadratic approximation.  The eigenvectors give us the directions of curvature of the loss function.  The eigenvalues give us the amount of curvature in each direction.

The contours of the ellipse of the quadratic approximation are given by the equation:
$$
\sum_i \lambda_i \alpha_i^2 = \text{const}.
$$
If we solve for the direction of the $i^{th}$ eigenvector we get:
$$
\alpha_i = \pm \sqrt{\frac{\text{const}}{\lambda_i}}
$$
We can visualize the eigenvectors of the Hessian at the optimal point on the quadratic approximation. The eigenvectors give us the directions of curvature of the loss function. The eigenvalues give us the amount of curvature in each direction.

```python
fig = go.Figure()
fig.add_contour(z=cancer['taylor_loss'].flatten(), x=cancer['w1'].flatten(),
                y=cancer['w2'].flatten(),
                colorscale='viridis_r', opacity=0.5,
                contours=dict(start=cancer['taylor_loss'].min(), end=cancer['taylor_loss'].max(),
                            size=(cancer['taylor_loss'].max()-cancer['taylor_loss'].min())/50),
                colorbar=dict(x=1.05, y=0.3, len=0.75))
scaling = 0.5
lam, U = np.linalg.eigh(H_star)
scale = scaling / np.sqrt(np.abs(lam))

cx, cy = cancer['grid_best']
for i, (lami, ui, si) in enumerate(zip(lam, U.T, scale), start=1):
    fig.add_scatter(
        x=[cx, cx + si*ui[0]],
        y=[cy, cy + si*ui[1]],
        mode='lines+markers',
        line=dict(width=2),
        name=f'u{i} (λ={lami:.3g})'
    )
    fig.add_scatter(
        x=[cx, cx - si*ui[0]],
        y=[cy, cy - si*ui[1]],
        mode='lines',
        line=dict(width=2, dash='dot'),
        showlegend=False
    )

fig.update_layout(title="Taylor Approximation of Loss",
                  xaxis_title='w1', yaxis_title='w2',
                  height=600, width=1200)
# fig.write_image("images/taylor_approx_loss_contours.pdf", width=1200, height=600)
fig
```

<link rel="stylesheet" href="berkeley.css">

<h4 class="cal cal-h4">Applied to Sine Regression</h4>


```python
# w_star = np.array([-2., -2.])
sine['w_star'] = sine['grid_best']
sine['L_star'] = sine_MSE(sine['w_star'])
sine['g_star'] = grad_sine_MSE(sine['w_star'])
sine['H_star'] = hessian_sine_MSE(sine['w_star'])
sine['lam'], sine['U'] = np.linalg.eigh(sine['H_star'])
print("w_star:", sine['w_star'])
print("L_star:", sine['L_star'])
print("g_star:", sine['g_star'])
print("H_star:\n", sine['H_star'])
print("Eigenvalues of H_star:", sine['lam'])
print("Eigenvectors of H_star:\n", sine['U'])
```

Visualizing the Hessian at the optimal point we see that it is positive definite (both eigenvalues are positive).  This is consistent with the fact that the loss function is convex.

```python
fig = plot_gradient(sine['w0'], sine['w1'],
                    sine['error'], sine['dw0'], sine['dw1'], scale=0.1)

sine['taylor_loss'] = np.array([
    taylor_loss(w, sine['w_star'], sine['L_star'], sine['g_star'], sine['H_star'])
for w in sine['ws']]).reshape(sine['w1'].shape)

fig.add_surface(z=sine['taylor_loss'], x=sine['w0'], y=sine['w1'],
                 colorscale='plasma_r', opacity=0.5, showscale=False,
                 contours=dict(z=dict(show=True, highlightcolor="white",
                                      start=sine['taylor_loss'].min(), end=sine['taylor_loss'].max(),
                                      size=(sine['taylor_loss'].max()-sine['taylor_loss'].min())/50)),
                                      row=1, col=1)
fig.update_layout(scene=dict(zaxis=dict(range=[0, 3])))
# fig.write_image("images/loss_surface_3d_with_gradients.pdf", width=800, height=800)
display(fig)
```

```python
fig = go.Figure()
fig.add_contour(z=sine['taylor_loss'].flatten(), x=sine['w0'].flatten(),
                y=sine['w1'].flatten(),
                colorscale='viridis_r', opacity=0.5,
                contours=dict(start=sine['taylor_loss'].min(), end=sine['taylor_loss'].max(),
                            size=(sine['taylor_loss'].max()-sine['taylor_loss'].min())/50),
                colorbar=dict(x=1.05, y=0.3, len=0.75))
scaling = 0.5
scale = scaling / np.sqrt(np.abs(sine['lam']))

cx, cy = sine['grid_best']
for i, (lami, ui, si) in enumerate(zip(sine['lam'], sine['U'].T, scale), start=1):
    fig.add_scatter(
        x=[cx, cx + si*ui[0]],
        y=[cy, cy + si*ui[1]],
        mode='lines+markers',
        line=dict(width=2),
        name=f'u{i} (λ={lami:.3g})'
    )
    fig.add_scatter(
        x=[cx, cx - si*ui[0]],
        y=[cy, cy - si*ui[1]],
        mode='lines',
        line=dict(width=2, dash='dot'),
        showlegend=False
    )

fig.update_layout(title="Taylor Approximation of Loss",
                  xaxis_title='w1', yaxis_title='w2',
                  height=600, width=1200)
# fig.write_image("images/taylor_approx_loss_contours.pdf", width=1200, height=600)
fig
```

<link rel="stylesheet" href="berkeley.css">

<h2 class="cal cal-h2">Oscillating Loss Functions</h2>

Here we visualize what happens when we have a poorly conditioned quadratic loss function.  This can happen when the Hessian has very different eigenvalues.  In this case, gradient descent can oscillate and take a long time to converge.

```python
H_bad = np.array([[.1, 0], [0, 1]])
def quad(w):
    """A poorly conditioned quadratic function."""
    return np.sum((w @ H_bad) * w,1)

def grad_quad(w):
    """Gradient of a poorly conditioned quadratic function."""
    return np.array(w @ H_bad * 2)
```

```python
w1,w2 = np.meshgrid(np.linspace(-5, 5, 30), np.linspace(-5, 5, 30))
ws = np.hstack([w1.reshape(-1, 1), w2.reshape(-1, 1)])
error =  quad(ws).reshape(w1.shape)
contour = go.Contour(x=w1.flatten(), y=w2.flatten(), z=error.flatten(), colorscale='Viridis_r',
                contours=dict(start=0, end=20, size=.5))
go.Figure(data=contour)
```

Suppose we start at the point (-4,0) and use a learning rate of 1.  This is an ideal point and we can visualize the path taken by gradient descent on the loss surface.

```python
w0 = np.array([-4., 0])
path = gradient_descent(w0, grad_quad, learning_rate=1, nepochs=50)

fig = go.Figure()
fig.add_trace(contour)
#add arrows to lines
fig.add_scatter(x=path[:, 0], y=path[:, 1],
                mode='lines+markers', line=dict(color='black', width=2),
                marker=dict(size=10, color='black', symbol= "arrow-bar-up", angleref="previous"),
                name='Gradient Descent Path', legendgroup='Gradient Descent',
                showlegend=False)
fig.update_layout(margin=dict(l=5, r=5, t=5, b=5))
# fig.write_image("images/flat_quadratic.pdf", width=1000, height=500)
display(fig)
```

What happens if we start a different point and a learning rate of 1.

```python
w0 = np.array([-4.,-2.])
path = gradient_descent(w0, grad_quad, learning_rate=1, nepochs=50)

fig = go.Figure()
fig.add_trace(contour)
#add arrows to lines
fig.add_scatter(x=path[:, 0], y=path[:, 1],
                mode='lines+markers', line=dict(color='black', width=2),
                marker=dict(size=10, color='black', symbol= "arrow-bar-up", angleref="previous"),
                name='Gradient Descent Path', legendgroup='Gradient Descent',
                showlegend=False)
fig.update_layout(margin=dict(l=5, r=5, t=5, b=5))
# fig.write_image("images/flat_quadratic.pdf", width=1000, height=500)
display(fig)
```

Or if we decrease the learning rate slightly to .9.

```python
w0 = np.array([-4.,-2.])
path = gradient_descent(w0, grad_quad, learning_rate=.9, nepochs=50)

fig = go.Figure()
fig.add_trace(contour)
#add arrows to lines
fig.add_scatter(x=path[:, 0], y=path[:, 1],
                mode='lines+markers', line=dict(color='black', width=2),
                marker=dict(size=10, color='black', symbol= "arrow-bar-up", angleref="previous"),
                name='Gradient Descent Path', legendgroup='Gradient Descent',
                showlegend=False)
fig.update_layout(margin=dict(l=5, r=5, t=5, b=5))
# fig.write_image("images/flat_quadratic.pdf", width=1000, height=500)
display(fig)
```

<link rel="stylesheet" href="berkeley.css">

<h2 class="cal cal-h2">Gradient Descent with Momentum</h2>


Here we implement gradient descent with momentum.  This is a simple modification to the basic gradient descent algorithm that can help with oscillations and speed up convergence.

```python
def gd_momentum(w_0, gradient,
    learning_rate=1, nepochs=10, epsilon=1e-6, momentum=0.9):
    """Gradient descent with momentum.
    Args:
        w_0: Initial weights (numpy array).
        gradient: Function to compute the gradient.
        learning_rate: Step size for each iteration.
        nepochs: Maximum number of iterations.
        epsilon: Convergence threshold.
        momentum: Momentum factor.
    Returns:
        path: Array of weights at each iteration."""
    w_old = w_0
    path = [w_old]
    v = np.zeros_like(w_old)
    for e in range(nepochs):
        g = gradient(w_old)
        v = momentum * v - learning_rate * g
        w = w_old + v
        path.append(w)
        if np.linalg.norm(w - w_old) < epsilon: break
        w_old = w
    return np.array(path)
```

Let's apply gradient descent with momentum to the poorly conditioned quadratic loss function from before.  We will start at the point (-4,-2) and use a learning rate of .9 and a momentum factor of .3.  We can visualize the path taken by gradient descent with momentum on the loss surface.

```python
w0 = np.array([-4.,-2.])
path = gd_momentum(w0, grad_quad,
                   learning_rate=.9,
                   momentum=0.3,
                   nepochs=50)

fig = go.Figure()
fig.add_trace(contour)
#add arrows to lines
fig.add_scatter(x=path[:, 0], y=path[:, 1],
                mode='lines+markers', line=dict(color='black', width=2),
                marker=dict(size=10, color='black', symbol= "arrow-bar-up", angleref="previous"),
                name='Gradient Descent Path', legendgroup='Gradient Descent',
                showlegend=False)
fig.update_layout(margin=dict(l=5, r=5, t=5, b=5))
# fig.write_image("images/flat_quadratic.pdf", width=1000, height=500)
display(fig)
```

Try setting momentum to 0 and see what happens.

```python
w0 = np.array([2.2, 2.2])
#w0 = np.array([.5, 2.5])
# w0 = np.array([1, 3])
path = gd_momentum(w0,
                   grad_sine_MSE,
                   learning_rate=.1,
                   momentum=0.9,
                   nepochs=50)
errors = [sine_MSE(w) for w in path]
fig = plot_gradient(
    sine['w0'], sine['w1'], sine['error'],
    sine['dw0'], sine['dw1'], scale=.1)
add_solution_path(fig, errors, path)
```

