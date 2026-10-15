"""CSE 242 Assignment 1 — submit as Assignment1.py.

Fill # YOUR ANSWER HERE blocks with commented reasoning; fill function bodies
at # YOUR CODE HERE. Keep signatures. See README.md and INTERFACE.md.
Functions must not mutate inputs, prompt for input, or perform network I/O.
Only code inside the main guard should save/show figures.
"""
import numpy as np


# PART 1 (20 points), Q1: Show computations for the given A and B.
# YOUR ANSWER HERE
def matrix_operations(A=None, B=None):
    """Return AB, BA, (A+B).T, A@B.T, trace(A), trace(AB).

    Defaults: A=[[3,4],[2,1]], B=[[1,2],[0,-1]]. Other inputs are square
    real matrices of matching size. Return a tuple in the specified order.
    """
    # YOUR CODE HERE
    A = np.array([[3, 4], [2, 1]] if A is None else A, dtype=float)
    B = np.array([[1, 2], [0, -1]] if B is None else B, dtype=float)
    AB = A @ B
    return AB, B @ A, (A + B).T, A @ B.T, float(np.trace(A)), float(np.trace(AB))


# P1 Q2: Derive the given M eigenpairs and verify Mv=lambda*v.
# YOUR ANSWER HERE
def eigensystem(M=None):
    """Return (values, vectors) for real symmetric M (default [[2,-1],[-1,2]]).

    Vectors are columns paired with values; order, sign, and nonzero scale
    are free. Vectors must span each eigenspace, including repeated roots.
    """
    # YOUR CODE HERE
    M = np.array([[2, -1], [-1, 2]] if M is None else M, dtype=float)
    values, vectors = np.linalg.eigh(M)        # ascending values, orthonormal columns
    order = np.argsort(values)[::-1]           # report largest first, like pca()
    return values[order], vectors[:, order]


# P1 Q3: Prove trace(AB)=trace(BA) for compatible matrices.
# YOUR ANSWER HERE

# P1 Q4: Show the given C is invertible, and compute its inverse step by step.
# YOUR ANSWER HERE
def matrix_inverse(C=None):
    """Return inverse of nonsingular square C; default C=[[4,7],[2,6]]."""
    # YOUR CODE HERE
    C = np.array([[4, 7], [2, 6]] if C is None else C, dtype=float)
    return np.linalg.solve(C, np.eye(C.shape[0]))


# PART 2 (50 points), Q1: Normalize c*3**k/k!, derive expectation and variance.
# YOUR ANSWER HERE
def distribution_moments(rate=3.):
    """Return (c, mean, variance) for P(X=k)=c*rate**k/k!, k>=0, rate>0."""
    # YOUR CODE HERE
    rate = float(rate)
    return float(np.exp(-rate)), rate, rate


# P2 Q2: Show reasoning for intersection, conditional, and union probabilities.
# YOUR ANSWER HERE
def event_probabilities(p_a=.5, p_b=.3, p_a_given_b=.6):
    """Return (P(A intersect B), P(B|A), P(A union B)).

    Inputs describe a valid distribution with p_a,p_b>0.
    """
    # YOUR CODE HERE
    p_ab = p_a_given_b * p_b
    return p_ab, p_ab / p_a, p_a + p_b - p_ab


# P2 Q3: Derive MLE/MAP for H,H,T,H,T, including the prior and maximization.
# YOUR ANSWER HERE
def coin_estimates(heads=3, tosses=5):
    """Return (MLE, MAP) using prior density 2*theta on [0,1].

    0<=heads<=tosses, positive integer tosses. Include boundary maxima.
    """
    # YOUR CODE HERE
    return heads / tosses, (heads + 1) / (tosses + 1)


# P2 Q4: Derive both Gaussian MLEs for [1,3,5,7] step by step.
# YOUR ANSWER HERE
def gaussian_mle(data=None):
    """Return (mean MLE, variance MLE); default data=[1,3,5,7].

    Other inputs are finite 1D samples of length>=2 with nonzero variance.
    """
    # YOUR CODE HERE
    x = np.array([1, 3, 5, 7] if data is None else data, dtype=float).ravel()
    mean = x.mean()
    return float(mean), float(np.mean((x - mean) ** 2))


# P2 Q5: Derive the Gaussian posterior mode for the given observation/prior.
# YOUR ANSWER HERE
def gaussian_map(x=5., observation_variance=4., prior_mean=0., prior_variance=1.):
    """Return posterior mode for one observation and a Gaussian mean prior.

    Both variances are positive; inputs are variances, not standard deviations.
    """
    # YOUR CODE HERE
    obs_precision = 1.0 / observation_variance
    prior_precision = 1.0 / prior_variance
    return (obs_precision * x + prior_precision * prior_mean) / (obs_precision + prior_precision)



# P2 Q6: Prove Var(X)=E[X**2]-E[X]**2.
# YOUR ANSWER HERE

# PART 3 (30 points), Q1: Define covariance and derive Var(u.T@X)=u.T@Sigma@u.
# YOUR ANSWER HERE

# P3 Q2: Describe the Gaussian generator and observed correlation.
# YOUR ANSWER HERE
def generate_data(n_samples=500, seed=0, mean=None, covariance=None):
    """Return reproducible Gaussian samples (n_samples,2) using NumPy and seed.

    Defaults: mean=[2,-1], covariance=[[3,1.2],[1.2,1]]. Custom mean has shape
    (2,), covariance (2,2) is positive definite. Do not mutate input arrays.
    """
    # YOUR CODE HERE
    mean = np.array([2, -1] if mean is None else mean, dtype=float)
    covariance = np.array([[3, 1.2], [1.2, 1]] if covariance is None else covariance, dtype=float)
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(mean, covariance, size=n_samples)


# P3 Q3: Link centering/covariance/eigendecomposition to the derivation and explain the plot.
# YOUR ANSWER HERE
def pca(X):
    """Fit NumPy PCA without mutating finite X of shape (n,2), n>=2.

    Return dict: mean (2,), centered (n,2), covariance (2,2), eigenvalues (2,),
    components (2,2), scores (n,2). Use covariance divisor n-1, descending
    eigenvalues, orthonormal eigenvectors as columns, scores=centered@components.
    Signs/tied-eigenspace bases are free. Include rank-deficient and tied cases.
    NumPy eigh/SVD are allowed; fitted library PCA is not.
    """
    # YOUR CODE HERE
    X = np.array(X, dtype=float)
    n = X.shape[0]
    mean = X.mean(axis=0)
    centered = X - mean
    covariance = centered.T @ centered / (n - 1)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    eigenvalues = np.maximum(values[order], 0.0)
    components = vectors[:, order]
    return {
        "mean": mean,
        "centered": centered,
        "covariance": covariance,
        "eigenvalues": eigenvalues,
        "components": components,
        "scores": centered @ components,
    }


# P3 Q4: Interpret the variance fractions and dimensionality reduction tradeoff.
# YOUR ANSWER HERE
def explained_variance(eigenvalues):
    """Return same-shape fractions for nonnegative eigenvalues with positive sum."""
    # YOUR CODE HERE
    values = np.array(eigenvalues, dtype=float)
    return values / values.sum()


# P3 Q2/Q3/Q4: Provide labeled figures and discuss their meaning.
# YOUR ANSWER HERE
def make_plots(X, result):
    """Return a Matplotlib Figure (or sequence of Figures), without show/save.

    Plot samples and principal directions through their mean, and explained
    variance fractions. Label axes. Plot quality is manually reviewed.
    """
    # YOUR CODE HERE
    from matplotlib.figure import Figure   # Figure object: no pyplot global state

    X = np.asarray(X, dtype=float)
    mean = result["mean"]
    values = result["eigenvalues"]
    comps = result["components"]
    scores = result["scores"]
    total = values.sum()
    fractions = explained_variance(values) if total > 0 else np.zeros_like(values)
    pc_colors = ["#c8503c", "#2b6f9c"]
    point_style = dict(s=11, alpha=0.35, color="#4b5e5a", linewidths=0)

    fig = Figure(figsize=(15, 4.8), layout="constrained")
    ax_data, ax_pc, ax_var = fig.subplots(1, 3)

    # (a) samples and principal axes
    ax_data.scatter(X[:, 0], X[:, 1], label=f"samples (n={len(X)})", **point_style)
    for j in range(2):
        half = 2 * np.sqrt(values[j]) * comps[:, j]
        ax_data.annotate("", xy=mean + half, xytext=mean - half,
                         arrowprops=dict(arrowstyle="<->", color=pc_colors[j], lw=2.2))
        ax_data.plot([], [], color=pc_colors[j], lw=2.2,
                     label=f"PC{j + 1}  λ={values[j]:.2f}  ({fractions[j]:.1%})")
    ax_data.plot(*mean, "k+", ms=12, mew=1.8, label="sample mean")
    ax_data.set(xlabel="feature 1", ylabel="feature 2",
                title="Samples and principal axes (±2 sd)")
    ax_data.set_aspect("equal", adjustable="datalim")
    ax_data.legend(fontsize=8, loc="upper left")

    # (b) the same points in principal-component coordinates
    ax_pc.scatter(scores[:, 0], scores[:, 1], **point_style)
    ax_pc.axhline(0, color=pc_colors[0], lw=1)
    ax_pc.axvline(0, color=pc_colors[1], lw=1)
    ax_pc.set(xlabel="PC1 score", ylabel="PC2 score", title="Scores: data rotated onto the PCs")
    ax_pc.set_aspect("equal", adjustable="datalim")

    # (c) explained variance
    labels = ["PC1", "PC2"]
    bars = ax_var.bar(labels, fractions, color=pc_colors, width=0.55)
    ax_var.bar_label(bars, labels=[f"{f:.1%}" for f in fractions], padding=3, fontsize=9)
    ax_var.plot(labels, np.cumsum(fractions), "o--", color="#1b2422", ms=5, lw=1.2, label="cumulative")
    ax_var.set(ylim=(0, 1.12), xlabel="principal component", ylabel="fraction of total variance",
               title="Explained variance")
    ax_var.legend(fontsize=8, loc="center right")

    for ax in (ax_data, ax_pc, ax_var):
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    return fig


# REFLECTION: contributions, tasks completed, and external resources used.
# YOUR ANSWER HERE
'''
AI Generated guide w/ explanations. 
Interpreted and rewritten for individual solutions.
Course support material generated in an agent loop with this steering prompt:

```
hello your job is to create a study guide for all lectures for cse242. each 
study guide needs to be separate, per lecture, and i need you to create an 
agents file so that i can have a reproducible structure to create additional 
guides as more lectures are released. I need you to create an html guide for 
each of the lectures. each html file is independent and for each lecture, and 
needs to expertly cover all concepts covered in each lecture. building up the 
topics, trying to avoid claudisms, with a focus on developing an intuitive 
understanding of the subject matter. lean on professor quotes and examples as 
much as possible. create formatted visuals and graphics often. Aim for these 
notes to be textbook style, beginning from first princples and allowing the 
student to become an expert in the subject matter for each lecture. Do not just 
give solutions, give out reasoning, thinking and connections to the lecture as 
you work up to the solutions for the homework. Please be careful and put effort 
into optimal results on this task. When you are done. I need you to add a 
service for this with it's own html server, and a main index page. Please 
attach this server to the services gateway on this box. Do not ask me any 
additional questions. I am commuting back home and cannot answer any more 
questions. Good luck. You got this.
```

`make_plots()` and `main()` are pretty much verbatim Claude.

Models used: Opus 5.5 & Fable 5.1
'''

if __name__ == '__main__':
    # YOUR CODE HERE: call your functions and save/display the requested figures.
    # This block does not run when the grader imports your functions.
    from pathlib import Path

    np.set_printoptions(precision=4, suppress=True)

    print("Part 1")
    for label, value in zip(["AB", "BA", "(A+B)^T", "A B^T", "trace(A)", "trace(AB)"], matrix_operations()):
        print(f"  {label} =\n{np.array(value)}" if np.ndim(value) else f"  {label} = {value}")
    values, vectors = eigensystem()
    M = np.array([[2, -1], [-1, 2]], float)
    print(f"  eigenvalues = {values}; M v = λ v holds: {np.allclose(M @ vectors, vectors * values)}")
    print(f"  C^-1 =\n{matrix_inverse()}")

    print("Part 2")
    print("  (c, E[X], Var X)      =", distribution_moments())
    print("  (P(A∩B), P(B|A), P(A∪B)) =", event_probabilities())
    print("  (θ_MLE, θ_MAP)        =", coin_estimates())
    print("  (μ_MLE, σ²_MLE)       =", gaussian_mle())
    print("  μ_MAP                 =", gaussian_map())

    print("Part 3")
    X = generate_data()
    result = pca(X)
    fractions = explained_variance(result["eigenvalues"])
    print("  sample correlation    =", round(float(np.corrcoef(X, rowvar=False)[0, 1]), 4))
    print("  eigenvalues           =", result["eigenvalues"])
    print("  components (columns)  =\n", result["components"])
    print("  explained variance    =", fractions)
    reconstructed = result["mean"] + result["scores"][:, :1] @ result["components"][:, :1].T
    print("  1-PC reconstruction MSE =", round(float(np.mean(np.sum((X - reconstructed) ** 2, axis=1))), 4),
          " (≈ λ₂·(n−1)/n =", round(float(result["eigenvalues"][1] * (len(X) - 1) / len(X)), 4), ")")

    out = Path(__file__).with_name("assignment1_ex_figures.png")
    make_plots(X, result).savefig(out, dpi=160)
    print(f"  saved {out.name}")

