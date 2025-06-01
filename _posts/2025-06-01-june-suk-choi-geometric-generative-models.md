---
layout: post
title: "AI810 Blog Post (20247059): An Overview of Geometric Generative Models"
author: "June Suk Choi"
date: 2025-06-01 09:00:00 +0000
categories: 
tags: []
mathjax: true
---

Generative models have revolutionized our ability to synthesize realistic data, from images to text and beyond. While diffusion models have seen immense success, Flow Matching (FM) has emerged as a powerful and often more efficient alternative, relying on simulating Ordinary Differential Equations (ODEs). But what happens when our data isn't flat and Euclidean, or when it's inherently discrete? This post, goes through how Flow Matching can be elegantly extended to complex of data on manifolds, discrete structures, and even unified under a general process framework.

## 1. Flow Matching in Euclidean Space

Generative modeling aims to create new data samples indistinguishable from a given dataset. 
While diffusion models have been a prominent tool to generatively model a dataset, Flow Matching (FM) has emerged as an alternative tool.

Instead of gradually adding and then removing noise through a stochastic differential equation (SDE), Flow Matching learns to **deterministically** transport samples from a simple, known prior distribution (*e.g.*, a Gaussian, dubbed $$p_0$$) to a more complex, target data distribution ($$p_1$$) by simulating an Ordinary Differential Equation (ODE).

The primary goal of FM is to learn a time-dependent **velocity field**, denoted as: 

$$u_\theta(X_t, t),$$

which is a vector field parameterized by a neural network $$\theta$$. This term dictates the direction and the speed at which a sample $$X_t$$ should move at any given time $$t$$ ($$t \in [0,1]$$). Predictably, the transformation from the prior ($$p_0$$) to the data distribution ($$p_1$$) is governed by the following ODE:

$$dX_t = u_\theta(X_t, t)dt$$

$$X_0$$ could be a point sampled from our prior $$p_0$$ (*e.g.*, Gaussian $$\mathcal{N}(0,I)$$) at $$t=0$$. 

As we integrate this ODE forward with respect to time, $$X_t$$ traces a deterministic path. At $$t=1$$, we want $$X_1$$ to be a sample that looks just like it came from our target data distribution $$p_1$$. 

As such, the solution to this ODE ($$X_t = \psi_t(X_0)$$) defines a **flow** $$\psi_t$$. This **flow** acts as a deterministic transport map that "pushes" the entire prior distribution $$p_0$$ forward as time progresses. If $$p_t$$ is the distribution of samples at time $$t$$, then denote this relationship as $$\psi_{t\#}p_0 = p_t$$. Here, our goal is to learn $$\theta$$ such that $$\psi_{1\#}p_0 \approx p_1$$.

**Challenge: intractable marginal quantities.**

**If** we knew the ground truth, ideal velocity field $$u_t(x)$$ that transforms $$p_0$$ to $$p_1$$ along the marginal probability path $$p_t(x)$$, we could simply train our neural network $$u_\theta(x, t)$$ to mimic it. This ideal field is called the **marginal velocity field**. 

However, $$u_t(x)$$ and the intermediate marginal distributions $$p_t(x)$$ for $$0 < t < 1$$ are generally intractable to be computed directly. This renders a naive loss unusable, something like: 

$$\mathbb{E}_{t, x \sim p_t} \|\|u_\theta(x, t) - u_t(x)\|\|^2$$

**Going around the issue: conditional paths and velocities.**

Flow Matching goes around this issue with a simple trick.
Instead of focusing on the marginal path, we instead focus on the **conditional** probability path $$p_t(x\|z)$$ and its associated conditional velocity field $$u_t(x\|z)$$. Here, $$z$$ is a conditioning variable (or set of variables) that makes the path easy to define and sample from.

A simple and effective choice for the conditioning variable is the pair $$z = (x_0, x_1)$$, where $$x_0$$ is a sample from the prior $$p_0$$ and $$x_1$$ is a sample from the target $$p_1$$ (in practice, from our empirical dataset $$q_{data}$$).

Given $$x_0$$ and $$x_1$$, now we can define a simple, deterministic path connecting them. The most straightforward in the Euclidean space is a linear interpolation:

$$x_t(x_0, x_1) = (1-t)x_0 + tx_1$$

For this path, the conditional probability $$p_t(x\|x_0, x_1)$$ is a Dirac delta function centered on this interpolant:

$$p_t(x\|x_0, x_1) = \delta(x - ((1-t)x_0 + tx_1))$$

The corresponding conditional velocity field $$u_t(x_t\|x_0, x_1)$$, which is the rate of change of $$x_t$$ along this specific path, is thus simply:

$$u_t(x_t\|x_0, x_1) = \frac{d}{dt}((1-t)x_0 + tx_1) = x_1 - x_0$$

Notice that for this linear path, the target conditional velocity is **constant** with respect to time and the current position $$x_t$$.

**Why is conditional flow matching useful?**

Now that we have such tractable conditional quantities, we can define the **Conditional Flow Matching (CFM) loss**:

$$\mathcal{L}_{CFM}(\theta) = \mathbb{E}_{t \sim U[0,1],\, x_0 \sim p_0,\, x_1 \sim q_{data},\, x_t = (1-t)x_0+tx_1} \|\|u_\theta(x_t, t) - u_t(x_t\|x_0, x_1)\|\|^2$$

In practice, we sample a time $$t$$ uniformly from $$[0,1]$$, a starting point $$x_0$$ from our simple prior $$p_0$$ (e.g., $$\mathcal{N}(0,I)$$), and an endpoint $$x_1$$ from our dataset $$q_{data}$$, then compute $$x_t$$ using our chosen conditional path (e.g., linear interpolation). 
Then, we can train our neural network $$u_\theta(x_t, t)$$ to match the pre-defined conditional velocity $$u_t(x_t\|x_0, x_1)$$ (*e.g.*, $$x_1 - x_0$$).

What is very intriguing is that, this tractable CFM loss has the **same gradient** with respect to $$\theta$$ as the **intractable marginal Flow Matching loss**. 
This means that by minimizing $$\mathcal{L}_{CFM}$$, we are implicitly learning the correct marginal velocity field $$u_t(x)$$. 
This relies on what we call a **"marginalization trick"** that connects the conditional velocities to the marginal velocity:

$$u_t(x) p_t(x) = \mathbb{E}_{z \sim p(z)} [u_t(x\|z) p_t(x\|z)],$$

which ensures that the gradients are aligned. 
The requirement, though, is that the chosen conditional paths $$p_t(x\|z)$$ must satisfy certain boundary conditions to ensure that they correctly interpolate between $$p_0$$ and $$p_1$$.

## 2. Beyond Euclidean Spaces: Geometric Flow Matching

In Sec. 1, we cover flow matching in Euclidean spaces.
But what happens when our data naturally resides on **curved surfaces** or in spaces with more **complex structures**? 
Many real-world datasets don't conform to the "flatness" of Euclidean space, for example:
*   **Protein structures:** The backbone and sidechain orientations involve rotations, which live in spaces like SO(3) (the group of 3D rotations).
*   **Robotics:** Joint angles often represent points on tori or other non-Euclidean manifolds.
*   **Climate-related data:** Data like wind patterns or temperature are naturally defined on the surface of a sphere ($$S^2$$).
*   **3D shapes and surfaces:** Meshes representing objects are inherently geometric.

In these scenarios, blindly applying Euclidean operations like $$X_t = (1-t)X_0 + tX_1$$ (linear interpolation) or $$X_1 - X_0$$ (vector subtraction for velocity) is sadly **problematic**. 
Such operations might take us "off" of the manifold, or fail to respect the intrinsic distances and curvatures of the space. 
On a general manifold, there's often no inherent "addition" or "vector space structure" for the points themselves. 
This is why we cover **Geometric Flow Matching (GFM)** in this section; we want to extend Flow Matching to these curved domains.

**Differential geometry essentials.**

To navigate and perform calculations on manifolds, we need a few key concepts from differential geometry. Think of these as the specialized tools that replace our standard Euclidean rulers and protractors:

*   **Smooth manifolds ($$\mathcal{M}$$):** These are the fundamental "stages" where our non-Euclidean data lives. 
Informally, a smooth manifold is a space that, if you "zoom in" sufficiently at any point, looks like a *patch* of Euclidean space ($$\mathbb{R}^d$$). 
However, when zoomed out, it can be curved and have a complex topology, like surfaces of a sphere or a torus.

*   **Tangent spaces ($$T_x\mathcal{M}$$):** At every point $$x$$ on our manifold $$\mathcal{M}$$, we can define a **tangent space** $$T_x\mathcal{M}$$. 
This is a standard vector space (flat, like $$\mathbb{R}^d$$) that captures all possible "directions" or instantaneous velocities of curves passing through $$x$$. 
If we want to define a flow $$dX_t = u_t(X_t)dt$$ on a given manifold, the velocity vector $$u_t(X_t)$$ **must** reside in the tangent space $$T_{X_t}\mathcal{M}$$ at the current point $$X_t$$. It can be thought of as a local "flat land" where we can consider velocity vectors.

*   **Riemannian metric ($$g_x$$):** While tangent spaces are vector spaces, we need a way to measure **lengths** of these velocity vectors and angles between them.
To do so consistently across the manifold, we take into account **Riemannian metric** $$g_x$$. 
It's an inner product defined on each tangent space $$T_x\mathcal{M}$$. This metric allows us to **define the norm** of a tangent vector $$\|v\|_g = \sqrt{g_x(v,v)}$$, which is going to be crucial for our loss function. 
Simply put, this is a way of defining local distances and angles.

*   **Geodesics:** In Euclidean space, the shortest path between two points is a straight line. 
On a manifold, the equivalent concept is a **geodesic**, which are curves that locally minimize distance. 
For example, on a sphere, geodesics are great circles. 
GFM uses geodesics as the **natural generalization** of the straight-line conditional paths we used in Euclidean Flow Matching (Section 1).

*   **Exponential map ($$\exp_x(v)$$)**: This map allows us to "travel" from a point $$x$$ on the manifold along a geodesic in a direction specified by a tangent vector $$v \in T_x\mathcal{M}$$. 
Specifically, $$\exp_x(v)$$ maps to the point $$y$$ reached by following the geodesic starting at $$x$$ with initial velocity $$v$$ for a unit time.
The distance traveled along the geodesic is then denoted as $$\|v\|_g$$. 
It essentially maps a direction from the tangent space back onto the manifold: $$\exp_x: T_x\mathcal{M} \to \mathcal{M}$$. 
This is basically the generalization of Euclidean "addition" ($$x + v$$).

*   **Logarithmic Map ($$\log_x(y)$$)**: This is the (local) inverse of the exponential map. Given two points $$x$$ and $$y$$ on the manifold (not "too far" apart), $$\log_x(y)$$ gives you the unique tangent vector $$v \in T_x\mathcal{M}$$ such that $$\exp_x(v) = y$$.
This vector $$v$$ represents the initial direction and "length" of the geodesic path from $$x$$ to $$y$$. 
It maps a pair of points on the manifold to a direction vector in a tangent space: $$\log_x: \mathcal{M} \to T_x\mathcal{M}$$. 
This is basically the generalization of Euclidean "subtraction" ($$y - x$$).

Now we discuss how we can generalize Euclidean Flow Matching to Geometric Flow Matching.

**ODE on manifolds.** The fundamental ODE remains $$dX_t = u_t(X_t)dt$$, but now $$X_t \in \mathcal{M}$$ and the velocity field $$u_t(X_t)$$ must be a vector in the tangent space at $$X_t$$, i.e., $$u_t(X_t) \in T_{X_t}\mathcal{M}$$.

To simulate a small step $$\Delta t$$ from $$X_t$$, we use the exponential map:

$$X_{t+\Delta t} \approx \exp_{X_t}(u_\theta(X_t, t)\Delta t)$$

This is directly analogous to $$X_{t+\Delta t} \approx X_t + u_\theta(X_t, t)\Delta t$$ in Euclidean space, with $$\exp_{X_t}$$ replacing addition.

**Conditional paths and velocities on geodesics.** Just as we used **straight lines** in Euclidean space for our conditional paths $$p_t(x\|x_0, x_1)$$, in GFM we use **geodesics**, the "straightest" paths on the manifold.
Given a starting point $$x_0 \sim p_0$$ (where $$p_0$$ is now a prior distribution on the manifold) and an endpoint $$x_1 \sim q_{data}$$, the point $$x_t$$ on the geodesic connecting them at "time" or fraction $$t \in [0,1]$$ is:

$$x_t(x_0, x_1) = \exp_{x_0}(t \cdot \log_{x_0}(x_1))$$

To explain this equation in detail, first, we find the tangent vector at $$x_0$$ that points towards $$x_1$$ using $$\log_{x_0}(x_1)$$. 
Then, we scale this vector by $$t$$. 
And then finally, we "travel" along the geodesic from $$x_0$$ in this scaled direction using $$\exp_{x_0}$$.

The target conditional velocity $$u_t(x_t\|x_0, x_1)$$ is the vector in $$T_{x_t}\mathcal{M}$$ that would take $$x_t$$ to $$x_1$$ along the geodesic in the remaining $$1-t$$ time. This is given by:

$$u_t(x_t\|x_0, x_1) = \frac{\log_{x_t}(x_1)}{1-t}$$

This is perfectly analogous to the Euclidean conditional velocity $$u_t(x_t\|x_0, x_1) = (x_1 - x_t)/(1-t) = x_1 - x_0$$. Note that here, $$\log_{x_t}(x_1)$$ plays the role of "$$x_1 - x_t$$".

**Riemannian Conditional Flow Matching (RCFM) loss.** The loss function takes the same form as the Euclidean CFM loss, but the norm used to compare the predicted velocity $$u_\theta(x_t, t)$$ and the target conditional velocity $$u_t(x_t\|x_0, x_1)$$ is now computed using the **Riemannian metric** $$g$$ at point $$x_t$$:

$$\mathcal{L}_{RCFM}(\theta) = \mathbb{E}_{t, x_0 \sim p_0, x_1 \sim q_{data}, x_t} \|u_\theta(x_t, t) - u_t(x_t\|x_0, x_1)\|_g^2$$

The expectation is over time $$t$$, samples $$x_0$$ from the prior on $$\mathcal{M}$$, $$x_1$$ from the data on $$\mathcal{M}$$, and $$x_t$$ constructed via the geodesic path. 
The key idea here is that both $$u_\theta(x_t, t)$$ and $$u_t(x_t\|x_0, x_1)$$ are vectors in the same tangent space $$T_{x_t}\mathcal{M}$$, so their difference is well-defined, and its squared norm $$\|\cdot\|_g^2$$ is computed using the metric $$g_{x_t}$$.

## 3. Generating Discrete Data: Discrete Flow Matching

So far, we've explored how Flow Matching can generate data in continuous Euclidean spaces (Sec. 1) and on curved, continuous manifolds (Sec. 2). 
But what if our data isn't continuous in the first place? 
Many important datasets are inherently **discrete**:
*   **Natural language text:** Sequences of characters or words (tokens).
*   **DNA/RNA sequences:** Chains of specific nucleotides.
*   **Categorical data:** Product categories, user labels, or any data drawn from a finite set of options.

For such data, the concepts of ODEs, continuous velocity fields, and geodesics don't directly apply. 
We can't simply "flow" smoothly between discrete states.
Instead, we need a process that allows for **"jumps"** between these states. 
In this section, we discuss **Discrete Flow Matching (DFM)**, leveraging the mathematics of **Continuous-Time Markov Chains (CTMCs)**.

**Continuous-time markov chains.**

A CTMC is a stochastic process that describes how a system jumps between states in a discrete state space $$\mathcal{S}$$ over **continuous** time.

*   **Discrete State Space ($$\mathcal{S}$$):** This is the set of all possible values our data can take. 
For a single token from a vocabulary $$V$$, $$\mathcal{S}=V$$. 
For a sequence of $$L$$ tokens, $$\mathcal{S}=V^L$$, which can be huge.

*   **Process ($$X_t$$):** At any time $$t \ge 0$$, the system is in some state $$X_t \in \mathcal{S}$$.

*   **Infinitesimal transitions:** The behavior of a CTMC is defined by its instantaneous *transition rates*. The probability of transitioning from state $$x$$ to a different state $$y$$ in a small time interval $$h$$ is approximately proportional to $$h$$:

    $$\mathbb{P}(X_{t+h} = y \| X_t = x) = u_t(y,x)h + o(h) \quad \text{for } y \neq x.$$

    Here, $$u_t(y,x) \ge 0$$ is the **rate** of transitioning from $$x$$ to $$y$$ at time $$t$$. The probability of staying in state $$x$$ is:

    $$\mathbb{P}(X_{t+h} = x \| X_t = x) = 1 + u_t(x,x)h + o(h),$$
    
    where $$u_t(x,x) = -\sum_{y \neq x} u_t(y,x)$$ is the negative sum of rates of leaving state $$x$$. This is to make sure that probabilities sum into 1, leading to the condition $$\sum_{y \in \mathcal{S}} u_t(y,x) = 0$$ for all $$x, t$$.
      
*   **Evolution of Probabilities (Kolmogorov Forward Equation):** The probability $$p_t(x)$$ of being in state $$x$$ at time $$t$$ changes according to the Kolmogorov forward equation (also known as the master equation):

    $$\frac{d p_t(y)}{dt} = \sum_{x \in \mathcal{S}} u_t(y,x) p_t(x).$$

    This equation states that the rate of change of probability of being in state $$y$$ is the sum of all probability flows **into** $$y$$ **from** other states $$x$$, subtracted by the flow **out of** state $$y$$. 

    Here, $$u_t(y,x)p_t(x)$$ is the flux from $$x$$ to $$y$$.
    The flow out of state $$y$$ is captured by the $$u_t(y,y)p_t(y)$$ term; if we rewrite $$u_t(y,y) = -\sum_{x \neq y} u_t(x,y)$$, then the equation becomes: 

    $$\frac{d p_t(y)}{dt} = \sum_{x \neq y} [u_t(y,x)p_t(x) - u_t(x,y)p_t(y)].$$

**Discrete flow matching.**

The goal of DFM is to learn a parameterized rate function (analogously, "velocity field" in the discrete context) $$u_\theta(y,x,t)$$ that defines a CTMC. 
This CTMC should transport samples from a simple initial distribution $$p_0(x)$$ (e.g., random noise, or a uniform distribution over $$\mathcal{S}$$) at $$t=0$$ to the target data distribution $$p_1(x) \approx q_{data}$$ at $$t=1$$.

Just like in continuous Flow Matching, directly learning the marginal rates $$u_t(y,x)$$ is challenging because the marginal path $$p_t(x)$$ is intractable. 
In the context of DFM, we circumvent this using the following tricks:
1.  First, we define a **conditional CTMC** with tractable rates $$u_t(y,x\|z)$$, where $$z$$ is a conditioning variable, typically $$z=(x_0, x_1)$$ with $$x_0 \sim p_0$$ and $$x_1 \sim q_{data}$$.
2.  Now, this conditional CTMC defines a conditional probability path $$p_t(x\|x_0, x_1)$$ that interpolates between $$x_0$$ and $$x_1$$.
3.  Then, we can train a neural network $$u_\theta(y,x,t)$$ to match these target conditional rates.

**DFM loss and marginalization trick.**

The training objective for DFM is the **Conditional DFM (CDFM) loss**, which minimizes the difference between the model's predicted rates and the target conditional rates:

$$\mathcal{L}_{CDFM}(\theta) = \mathbb{E}_{t, x_0, x_1, x_t \sim p_t(\cdot\|x_0,x_1)} \left[ \sum_{y \in \mathcal{S}} (u_\theta(y, x_t, t) - u_t(y, x_t\|x_0,x_1))^2 \right]$$

The expectation is taken over uniformly sampled time $$t \in [0,1]$$, initial state $$x_0$$, target state $$x_1$$, and the intermediate state $$x_t$$ sampled from the conditional path. 
The sum is over all possible next states $$y$$, representing an $$\ell_2$$ norm for the vector of rates coming from $$x_t$$.

Similarly to its continuous counterpart, this **marginalization trick** lets us make sure that minimizing this tractable CDFM loss leads to learning the correct (but intractable) marginal rates $$u_t(y,x)$$. 

**Handling the high-dimensionality of discrete data.**

For many discrete data types, especially sequences (like text or DNA), the state space $$\mathcal{S}$$ is extraordinarily enormous. 
If $$x=(x^1, x^2, \ldots, x^L)$$ is a sequence of $$L$$ tokens, and each token $$x^j$$ can take $$K$$ values, then $$\|\mathcal{S}\| = K^L$$. 
Predicting a rate vector of size $$K^L$$ from each state $$x_t$$ is simply impossible.

DFM addresses this by using **factorization**.

**Factorized rates.** We just assume that transitions predominantly occur *one coordinate* (or token) at a time. 
The model $$u_\theta(y,x,t)$$ is structured such that it only predicts non-zero rates if $$y$$ differs from $$x$$ in exactly one coordinate, say $$j$$.

$$u_\theta(y,x,t) = \sum_{j=1}^L \delta(y^{\bar{j}}, x^{\bar{j}}) u_\theta^j(y^j, x, t)$$

Here, $$y^{\bar{j}}$$ denotes all coordinates of $$y$$ except the $$j$$-th, $$\delta$$ is the Kronecker delta ensuring other coordinates are unchanged, and $$u_\theta^j(y^j, x, t)$$ is the rate for the $$j$$-th coordinate to change from $$x^j$$ to $$y^j$$, given the full current state $$x$$. 
The model $$u_\theta^j$$ now only needs to output rates for **changing a single coordinate**, which is much more manageable for us (e.g., $$K$$ possible values for $$y^j$$).

**Factorized conditional paths.** The target conditional path $$p_t(x\|x_0,x_1)$$ is also typically factorized by coordinate:

$$p_t(x\|x_0, x_1) = \prod_{j=1}^L p_t^j(x^j\|x_0^j, x_1^j)$$

A common and simple choice for the per-coordinate conditional path $$p_t^j(x^j\|x_0^j, x_1^j)$$ is one where $$x^j$$ transitions from its initial state $$x_0^j$$ to its target state $$x_1^j$$ independently of other coordinates. For instance, let $$\kappa_t$$ be a monotonically increasing function from $$\kappa_0=0$$ to $$\kappa_1=1$$ (e.g., $$\kappa_t = t$$). Then, we can define:

$$p_t^j(x^j\|x_0^j, x_1^j) = \kappa_t \delta(x^j, x_1^j) + (1-\kappa_t)\delta(x^j, x_0^j)$$

This means at time $$t$$, the $$j$$-th coordinate $$X_t^j$$ (sampled from this path) will be $$x_1^j$$ with probability $$\kappa_t$$ and $$x_0^j$$ with probability $$1-\kappa_t$$.

**Target conditional rates for factorized paths.** From this per-coordinate path, we derive the target conditional rates $$u_t^j(y^j, X_t^j\|x_0^j, x_1^j)$$. 
If $$x_0^j \neq x_1^j$$, and the current state of the $$j$$-th coordinate $$X_t^j$$ is still $$x_0^j$$, the only allowed transition is to $$x_1^j$$. The rate for this transition is:

$$u_t^j(x_1^j, x_0^j \| x_0, x_1) = \frac{\dot{\kappa}_t}{1-\kappa_t}.$$

All other rates from $$x_0^j$$ are zero (i.e., $$u_t^j(y^j, x_0^j \| \ldots) = 0$$ for $$y^j \neq x_1^j, x_0^j$$). 
The rate of "staying" at $$x_0^j$$ is: 

$$u_t^j(x_0^j, x_0^j \| \ldots) = -\frac{\dot{\kappa}_t}{1-\kappa_t}.$$

If the $$j$$-th coordinate $$X_t^j$$ has already reached $$x_1^j$$ (or if $$x_0^j = x_1^j$$ to begin with), it stays there; all rates to transition out of $$x_1^j$$ are zero. This defines an "absorbing target" path for each coordinate.
The neural network's per-coordinate output $$u_\theta^j(y^j, x_t, t)$$ is then trained to match this simple and sparse target rate vector.

As we'll see in the next section, the principles of DFM can be combined with those for continuous data under a more general scope, allowing for the generation of more general data that might have both discrete and continuous traits (*e.g.*, molecules with specific atom types and continuous 3D coordinates).

## 4. A Unifying Perspective - Generator Matching (GM)

In the previous sections, we've seen how Flow Matching can be adapted for continuous data in Euclidean spaces (Sec. 1, CFM), on curved manifolds (Sec. 2, GFM), and for discrete data (Sec. 3, DFM). 
Here, we notice a pattern: we define some target process, find it hard to work with directly (due to intractable marginals), and then resort to simpler *conditional* paths and velocities/rates to make learning tractable.

**Generator Matching (GM)** takes this idea even further. 
It provides a powerful and unifying framework that can describe not just deterministic flows (ODEs) or simple jump processes (CTMCs), but a much broader class of **Continuous-Time Markov Processes (CTMPs)**. 
This includes processes that might simultaneously involve deterministic movement, random diffusion (like Brownian motion), and discrete jumps.

**What is Continuous-Time Markov Process (CTMP)?**

A CTMP $$X_t$$ is a stochastic process where the future state depends only on the current state, not on the sequence of events that preceded it (the Markov property).
Both the ODEs used in CFM/GFM and the CTMCs used in DFM are **special cases** of CTMPs.

**The Generator ($$\mathcal{L}_t$$)**

Every CTMP is uniquely characterized by its **generator**, denoted as $$\mathcal{L}_t$$. The generator is an operator that tells us how any reasonably smooth "test function" $$f(x)$$ is expected to change infinitesimally when applied to the process $$X_t$$ at state $$x$$.
More formally, for a state $$X_t = x$$, the generator is defined as:

$$\mathcal{L}_t f(x) = \lim_{h\to 0} \frac{\mathbb{E}[f(X_{t+h})\|X_t=x] - f(x)}{h}.$$

What this basically means is that, if you know the current value $$f(x)$$, then for a very small time step $$h$$, the expected value of $$f$$ at the next step $$X_{t+h}$$ is approximately $$f(x) + h \cdot \mathcal{L}_t f(x)$$. 
In other words, the generator captures the instantaneous "drift" or change tendency of $$f(X_t)$$.

Let's take a look at what the generators look like for the processes we've encountered:

**For a Flow process ($$dX_t = u_t(X_t)dt$$)**,
the generator is $$\mathcal{L}_t f(x) = \nabla f(x)^T u_t(x)$$.
This is simply the directional derivative of the function $$f$$ along the velocity vector $$u_t(x)$$. It tells us how fast $$f$$ changes if we move along the flow.

**For a diffusion process (e.g., $$dX_t = \sigma_t(X_t)dW_t$$, where $$W_t$$ is a Wiener process)**,
the generator can be described as the following equation:

$$\mathcal{L}_t f(x) = \frac{1}{2} \text{Tr}\left(\sigma_t(x)\sigma_t(x)^T \nabla^2 f(x)\right).$$

Here, $$\nabla^2 f(x)$$ is the Hessian (matrix of second derivatives) of $$f$$. The term $$\sigma_t(x)\sigma_t(x)^T$$ is the diffusion tensor. This generator involves second-order derivatives, reflecting the stochastic and diffusive nature of the process. For simple isotropic diffusion (where $$\sigma_t(x)$$ is a scalar), this simplifies to $$\frac{1}{2}\sigma_t(x)^2 \Delta f(x)$$, where $$\Delta$$ is the Laplacian.

**For a jump process (*i.e.*, CTMC with transition rates $$u_t(y,x)$$, or kernel $$Q_t(dy,x)$$ like in DFM)**,
the generator can be written as $$\mathcal{L}_t f(x) = \sum_{y \neq x} u_t(y,x) (f(y) - f(x))$$, or more generally using a jump kernel $$Q_t(dy\|x)$$:

$$\mathcal{L}_t f(x) = \int_{\mathcal{S}} (f(y)-f(x))Q_t(dy\|x).$$

This form calculates the expected change in $$f$$ due to a jump from $$x$$ to any other state $$y$$, weighted by the rate or probability of that jump.

**Benefits of using the Generator**

One of the most interesting results in the theory of CTMPs (on Euclidean space) is that any such process can essentially be decomposed. 
Its generator $$\mathcal{L}_t$$ can always be expressed as a sum of three parts: a **flow** part (first-order derivatives), a **diffusion** part (second-order derivatives), and a **jump** part (integral over state changes):

$$\mathcal{L}_t f(x) = \underbrace{\nabla f(x)^T u_t(x)}_{\text{Flow}} + \underbrace{\frac{1}{2} \text{Tr}(D_t(x) \nabla^2 f(x))}_{\text{Diffusion}} + \underbrace{\int (f(y)-f(x))J_t(dy\|x)}_{\text{Jump}},$$

where $$D_t(x) = \sigma_t(x)\sigma_t(x)^T$$ is the diffusion matrix and $$J_t$$ is the jump kernel.
This means a single framework built around learning the generator $$\mathcal{L}_t$$ can, in principle, learn models that combine deterministic trajectories with random diffusion and discrete jumps, offering a great flexibility.

**Generator Matching (GM)**

Directly parameterizing and learning the generator $$\mathcal{L}_t$$ (which is an operator acting on functions) is very difficult. 
As such, Generator Matching adapts the familiar Flow Matching strategy.

**Definition of Conditional Generators.** 
Just like CFM and DFM, we define a simpler **conditional generator** $$\mathcal{L}_{t\|z}$$ associated with a tractable conditional probability path $$p_t(x\|z)$$. Again, $$z=(x_0, x_1)$$ is a common choice, where $$x_0 \sim p_0$$ (prior) and $$x_1 \sim q_{data}$$ (target data). This conditional generator describes a simpler process that deterministically (or stochastically in a controlled way) transports $$x_0$$ to $$x_1$$.

**Marginalization trick for generators.** 
The marginal generator $$\mathcal{L}_t$$ (which we want to learn) can be related to the conditional generators $$\mathcal{L}_{t\|z}$$ via an expectation:

$$\mathcal{L}_t f(x) = \mathbb{E}_{z \sim p(z\|x,t)}[\mathcal{L}_{t\|z}f(x)].$$

Here, the exact form of the conditioning $$p(z\|x,t)$$ depends on the construction, but the core idea is that the average of conditional dynamics yields the marginal dynamics.

**Parameterizing the generator.**
Now, we need a way for our neural network $$\theta$$ to represent the generator. 
A common approach is a *linear parameterization*. 
Here, the idea is to represent the action of the learned generator $$\mathcal{L}_t^\theta$$ on a test function $$f$$ at point $$x$$ as an inner product:

$$(\mathcal{L}_t^\theta f)(x) = \langle \mathcal{K}(f,x), F_t^\theta(x) \rangle,$$

where:

*   $$\mathcal{K}(f,x)$$ is a "kernel" or feature extractor that takes the test function $$f$$ and point $$x$$ and outputs some representation. For example, if we are modeling a flow, $$\mathcal{K}(f,x)$$ could be $$\nabla f(x)$$. 
If modeling diffusion, it might involve $$\nabla^2 f(x)$$.

*   $$F_t^\theta(x)$$ is a vector-valued function (parameterized by our neural network $$\theta$$) that our model learns. For a flow, $$F_t^\theta(x)$$ would be the learned velocity field $$u_t^\theta(x)$$.

The target conditional generator $$\mathcal{L}_{t\|z}$$ would similarly have a target parameterization, which call it $$F_t^z(x)$$.

**Conditional Generator Matching (CGM) Loss.**
We train our model by minimizing the difference between our model's parameterization $$F_t^\theta(x_t)$$ and the target parameterization $$F_t^z(x_t)$$ derived from the known conditional generator. 
The expectation is taken over time $$t$$, conditioning variables $$z=(x_0, x_1)$$, and points $$x_t$$ sampled from the conditional path $$p_t(\cdot\|z)$$:

$$\mathcal{L}_{CGM}(\theta) = \mathbb{E}_{t, z, x_t \sim p_t(\cdot\|z)} [D(F_t^z(x_t), F_t^\theta(x_t))]$$

Here, $$D$$ is some distance or divergence measure (e.g., squared $$\ell_2$$ norm).

Training with the CGM loss lets us learn a neural network $$F_t^\theta(x)$$ that implicitly defines the marginal generator $$\mathcal{L}_t$$.
This generator is able to simulate complex processes that might combine flow-like, diffusion-like, and jump-like behaviors. 
This allows us to generate data with mixed characteristics, such as molecules with discrete atom types (jumps) and continuous 3D coordinates (flows/diffusion on manifolds).

The Generator Matching framework thus elegantly generalizes the ideas from CFM and DFM, providing a highly versatile tool for generative modeling across diverse data types and process dynamics. In the next section, we'll discuss some practical techniques that make these Flow Matching approaches even more effective.

## 5. Advanced Techniques & Making it Practical

So far, we've built a foundation for Flow Matching across various data domains. 
We understand how to define conditional paths and learn the corresponding velocity fields (or rates/generators). 
Now we will take a look at some advanced techniques that refine the training process, making Flow Matching more robust, efficient, and tailored to specific data characteristics when using the common $$z=(x_0, x_1)$$ conditioning.

#### 5.1. Multisample Couplings

Recall that in Conditional Flow Matching (CFM, Sec. 1), when we use $$z=(x_0, x_1)$$ as our conditioning variable, we sample $$x_0 \sim p_0$$ (our prior) and $$x_1 \sim q_{data}$$ (target data). 
We then define a conditional path, often a straight line in Euclidean space like $$x_t = (1-t)x_0 + tx_1$$, and train our model $$u_\theta(x_t,t)$$ to match the target velocity $$x_1-x_0$$.

The seemingly simple step of "sampling $$x_0$$ and $$x_1$$" actually occludes an important detail: how do we *"pair"* them? 
This pairing is formally known as a **coupling**, $$q(x_0, x_1)$$, which is a joint distribution whose marginals are $$p_0$$ and $$q_{data}$$. 
The choice of this coupling can actually impact the learning process.

**Why does the coupling matter?**
The principle is that we want to induce conditional paths that are as "straight" or "short" as possible. Straighter paths generally mean that the velocity field $$u_t(x_t|x_0,x_1)$$ doesn't change too erratically. This is beneficial because a simpler, less convoluted velocity field typically easier for the neural network to learn.
Additionally, when we simulate the learned ODE $$dX_t = u_\theta(X_t, t)dt$$ at inference time using discrete steps (e.g., with an Euler solver), straighter underlying paths typically lead to smaller accumulated errors.

**How do we find a "good" coupling?**
One way to encourage straight paths is to minimize the expected squared length of the true velocity vectors along these paths. 
For the simple linear path $$x_t = (1-t)x_0 + tx_1$$, the velocity is $$x_1-x_0$$. 
The "cost" or "length" of this path can be related to the distance between its endpoints. 
We can try to minimize an objective like:

$$\min_{q(x_0,x_1)} \int_0^1 \mathbb{E}_{x_t \sim p_t( \cdot | x_0,x_1)} [\|u_t(x_t|x_0,x_1)\|_2^2] dt.$$

For the linear path, this simplifies. 
More generally, one can show that this objective is upper-bounded by minimizing the expected squared Euclidean distance between the paired samples:

$$\min_{q(x_0,x_1)} \mathbb{E}_{(x_0,x_1)\sim q} [\|x_0 - x_1\|_2^2].$$

This is a classic problem in **Optimal Transport**.

**Practical implementation: Minibatch perfect matching.**
While solving the full optimal transport problem over entire distributions $$p_0$$ and $$q_{data}$$ can be complex, a practical approximation is to perform matching *within minibatches* during training:
1.  Sample a minibatch of $$N$$ prior samples: $$\{x_0^{(1)}, \dots, x_0^{(N)}\}$$.
2.  Sample a minibatch of $$N$$ data samples: $$\{x_1^{(1)}, \dots, x_1^{(N)}\}$$.
3.  Find a **permutation** $$\pi^*$$ of the indices $$\{1, \dots, N\}$$ that minimizes the sum of squared distances between paired samples:

    $$\pi^* = \underset{\pi \in S_N}{\text{argmin}} \sum_{i=1}^N \|x_0^{(i)} - x_1^{(\pi(i))}\|_2^2.$$

    This is known as the **perfect matching** problem (or linear assignment problem) and can be solved efficiently using algorithms like the Hungarian algorithm.
4.  Use these optimally paired $$(x_0^{(i)}, x_1^{(\pi^*(i))})$$ samples to construct the conditional paths $$x_t^{(i)}$$ and compute the CFM loss for that minibatch.

By using such multisample couplings, we can encourage the model to learn flows that are more direct (on average). 

#### 5.2. Equivariant Flow Matching

Many real-world datasets possess inherent **symmetries**, such as molecules (physical properties) or images (containing rotations or reflections).

If our generative model doesn't respect these symmetries, it might learn **spurious correlations**, or require more data to generalize. **Equivariant Flow Matching** aims to design the flow in a way that naturally incorporates these symmetries.

The core idea is to leverage the group actions associated with these symmetries. 
Let $$G$$ be the symmetry group (e.g., SE(3) for 3D rotations and translations, or the permutation group $$S_N$$ for $$N$$ identical particles). 
Let $$\rho(g)x$$ denote the action of a group element $$g \in G$$ on a data point $$x$$.

Instead of directly defining a path between a sampled $$x_0 \sim p_0$$ and $$x_1 \sim q_{data}$$, we first try to "align" $$x_1$$ to $$x_0$$ using an optimal group transformation.
1.  Sample $$x_0$$ from the prior and $$x_1$$ from the data.
2.  Find the group element $$g^* \in G$$ that minimizes the distance between $$x_0$$ and the transformed $$x_1$$:
    $$g^* = \underset{g \in G}{\text{argmin}} \|x_0 - \rho(g)x_1\|_2^2$$
    *   For **rotational alignment** (e.g., aligning two point clouds or molecular structures, where $$G = \text{SO}(3)$$), this can be solved using the **Kabsch algorithm**.
    *   For **permutational alignment** (e.g., matching sets of identical atoms where $$G = S_N$$), if we are matching two sets of points, this again becomes an assignment problem solvable by algorithms like the Hungarian algorithm (often by finding the permutation that minimizes the sum of distances between corresponding points after a potential global alignment).

3.  Once $$g^*$$ is found, define the conditional path (e.g., linear interpolation) between $$x_0$$ and the *aligned data sample* $$\tilde{x}_1 = \rho(g^*)x_1$$:

    $$x_t(x_0, \tilde{x}_1) = (1-t)x_0 + t\tilde{x}_1.$$

4.  The target conditional velocity is then $$u_t(x_t\|x_0, \tilde{x}_1) = \tilde{x}_1 - x_0$$.
5.  The Conditional Flow Matching loss is computed using these equivariantly constructed pairs:

    $$\mathcal{L}_{ECFM}(\theta) = \mathbb{E}_{t, x_0, x_1, g^*, x_t} \|u_\theta(x_t, t) - (\rho(g^*)x_1 - x_0)\|_2^2.$$

This way, the learned flow naturally respects the data's symmetries.
If the prior $$p_0$$ is itself invariant under $$G$$ (e.g., a rotationally invariant Gaussian for SE(3) data), then the entire generative process can be made equivariant.
Obviously the model doesn't need to learn the same pattern in multiple transformed orientations/permutations; it learns a canonical representation.

## 6. Additional Discussion: A Unified Perspective of Diffusion and Flow Matching

Throughout this overview, we've distinguished Flow Matching (FM) by its reliance on Ordinary Differential Equations (ODEs) to learn deterministic transformations, contrasting it with traditional Diffusion Models (DMs) that typically employ Stochastic Differential Equations (SDEs) for a gradual noising and de-noising process. However, these two paradigms are more deeply connected than they might first appear, particularly when we consider the deterministic "Probability Flow" ODE formulation inherent in score-based diffusion models.

#### 6.1. The Deterministic Core of Diffusion Models

Standard score-based diffusion models involve:
1.  A **forward process** (an SDE) that progressively adds noise to data $$x_0 \sim p_{data}$$ to obtain noisy samples $$X_t$$:

    $$dX_t = \mathbf{f}(X_t, t)dt + g(t)dW_t,$$

    where $$dW_t$$ is a standard Wiener process, $$\mathbf{f}(X_t, t)$$ is a drift coefficient, and $$g(t)$$ is a diffusion coefficient. 
    As $$t$$ goes from 0 to 1 (or $$T$$), $$X_t$$ approaches a simple prior distribution (e.g., Gaussian).

2.  A **reverse process** (also an SDE) learned to reverse this noising. This typically involves training a neural network $$s_\theta(X_t, t)$$ to approximate the score function $$\nabla_{X_t} \log p_t(X_t)$$, where $$p_t(X_t)$$ is the marginal probability density of $$X_t$$ at time $$t$$. The reverse SDE for generation is then:

    $$dX_t = [\mathbf{f}(X_t, t) - g(t)^2 s_\theta(X_t, t)]dt + g(t)d\bar{W}_t,$$

    where $$d\bar{W}_t$$ is a reverse-time Wiener process.

A crucial insight (Song et al., 2020 -> , "Score-Based Generative Modeling through Stochastic Differential Equations") is that for any such diffusion process, there exists a corresponding **deterministic ODE**, known as the **Probability Flow (PF) ODE**, whose trajectories share the exact same marginal probability densities $$p_t(X_t)$$ over time as the SDE:

$$\frac{dX_t}{dt} = \left[ \mathbf{f}(X_t, t) - \frac{1}{2} g(t)^2 \nabla_{X_t} \log p_t(X_t) \right].$$

If we substitute our learned score $$s_\theta(X_t, t)$$ for the true score, we get an ODE that can be used for deterministic sampling:

$$\frac{dX_t}{dt} = \underbrace{\left[ \mathbf{f}(X_t, t) - \frac{1}{2} g(t)^2 s_\theta(X_t, t) \right]}_{\text{deterministic velocity field } v_t^{\text{PF}}(X_t)}.$$

The term in the brackets is a deterministic velocity field that transforms samples from the prior at $$t=1$$ back to data samples at $$t=0$$ (if integrating backward in time, or vice-versa if defining forward).

#### 6.2. Flow Matching and the PF-ODE Velocity Field

Recall that Flow Matching's primary goal is to directly learn a velocity field $$u_\theta(X_t, t)$$ for an ODE $$dX_t = u_\theta(X_t, t)dt$$ such that this ODE transports a prior distribution $$p_0$$ to the data distribution $$p_1$$.

The connection here is that the velocity field $$v_t^{\text{PF}}(X_t)$$ of the Probability Flow ODE derived from a diffusion model is **exactly the type of velocity field that Flow Matching aims to learn**.
Thus, a trained score-based diffusion model, through its learned score function $$s_\theta$$, *implicitly defines* a velocity field for a deterministic flow capable of generation.

#### 6.3. Differences and Similarities of DM and FM

This point of view unifies the two approaches at a fundamental level.
In their deterministic sampling forms (FM directly, and DM via its PF-ODE), both aim to find an ODE that transforms an easy-to-sample prior distribution into the complex data distribution.

*   **Score-based DMs** first define an SDE (via choices of $$\mathbf{f}$$ and $$g$$), then learn the score $$s_\theta$$ (often via denoising score matching), and finally construct the PF-ODE velocity field $$v_t^{\text{PF}}$$ using $$s_\theta$$, $$\mathbf{f}$$, and $$g$$.
*   **Flow Matching** directly targets a velocity field $$u_\theta$$. It achieves this by regressing against a *target* velocity field $$u_t(x_t\|z)$$ derived from user-chosen conditional probability paths $$p_t(x\|z)$$. For example, for a linear conditional path $$x_t=(1-t)x_0+tx_1$$, the target velocity is simply $$x_1-x_0$$.

The key difference and a potential advantage of FM is the ability to *design* the conditional paths $$p_t(x\|z)$$. 
This design directly dictates the simplicity of the target velocity field $$u_t(x_t\|z)$$ that the network $$u_\theta$$ must learn.

For instance, **Rectified Flow** (a specific instance of FM) uses **linear** conditional paths. 
The resulting target velocity $$x_1-x_0$$ is constant along each path and very simple to compute. 
This can lead to highly stable and efficient training, as the model directly learns to "straighten" the paths between prior samples and data samples. 
This target is generally much simpler than the score function $$\nabla_{X_t} \log p_t(X_t)$$ which DMs learn.

**The Generator Perspective (Revisiting Sec. 4).** The Generator Matching framework provides the most abstract unification.
  *   The generator $$\mathcal{L}_t$$ of a diffusion SDE involves both a drift term (related to $$\mathbf{f}$$ and first-order derivatives of a test function) and a diffusion term (related to $$g$$ and second-order derivatives).
  *   The generator of an ODE (as in FM, or the PF-ODE) only contains a drift term.
  *   The PF-ODE effectively isolates the deterministic "mean" evolution of the probability densities from the stochastic SDE. Flow Matching directly aims to model such a deterministic evolution.

#### 6.4. Practical Implications

This point of view has a few practical implications, especially interms of training and design choices.
Flow Matching's direct regression onto a (potentially very simple) target velocity (e.g., in $$\mathcal{L}_{CFM} = \mathbb{E} \|u_\theta - u_t(\cdot\|z)\|^2$$) can be more direct than score matching objectives, which might involve estimating gradients of log-densities or relying on specific noise properties (like in denoising score matching). 
This can translate to benefits in training stability or computational efficiency.

Additionally, FM offers explicit control over the "straightness" or nature of the learned transport through the choice of conditional paths. 
This is particularly powerful when extending to manifolds (GFM), where **geodesics** provide natural "straight" paths.

Finally, while a diffusion model can be converted to an ODE sampler post-training, FM is designed around ODEs from the outset (*i.e.*, ODE-native). 
While score-based diffusion models arrive at a deterministic generative ODE in the viewpoint of SDEs and score functions, Flow Matching provides a framework to directly learn such an ODE, often with more explicit control over the properties of the learned flow. 
This shared deterministic foundation underscores a deep theoretical connection between these powerful generative modeling paradigms.


## Conclusion & Outlook

Based on the overview of flow matching we have followed through, we can say that generative modeling is no longer confined to the flat, grid-like structures of Euclidean space. 
We've seen how the foundational principles of Flow Matching—learning deterministic transformations via ODEs—can be adapted to far more complex data domains.
*   **Geometric Flow Matching (GFM)** lets us generate data on curved manifolds, respecting their intrinsic distances and structures.
*   **Discrete Flow Matching (DFM)** allows us to model discrete data using Continuous-Time Markov Chains.
*   **Generator Matching (GM)** provides an holistic theoretical viewpoint, using the concept of CTMP generators to unify flow, diffusion, and jump processes.

Across all these diverse settings, the core strategy remains consistent: the **"conditional path" trick**. 
By defining a simple, tractable circumvention between known start (prior) and end (data) points, we can effectively learn the complex, intractable marginal dynamics of the true data distribution (using geodesics, factorized discrete transitions, or paths governed by conditional generators). 
Practical enhancements like multisample couplings and equivariant flow matching allows us practical techniques to apply them in real-life data using these methods.

By embracing the inherent structure of data, whether geometric or discrete, this advanced suite of Flow Matching techniques is enabling us to model the world with greater fidelity than ever before.

