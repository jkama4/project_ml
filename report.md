# Predicting Heart Disease Risk with A NN and Traditional ML Models

## Abstract
WRITE AFTER WRITING THE COMPLETE REPORT!

## Introduction
Unfortunately, many people experience a heart disease at some point in their lives. They are so deadly, due to their randomness. It can overcome anyone. Cardiovascular disease, or heart disease, has different forms: coronary heart disease (CHD), cerebrovascular disease, peripheral artery disease (PAD), and aortic atherosclerosis. All of these have different main causes. However, the general causes of any heart disease, mostly depend on other factors, like 

## Mathematical Analysis of the Feedforward Neural Network

A feedforward **Neural Network** is conceptually not that complex as the name might indicate. The first time I read the term "neural network", I immediately linked it to the human brain. However, this is obviously not the case. You could say that a Neural Network is just a fancy squiggle fitting machine, just like a linear function, but instead of a line, you get a weird shape that fits to the data.

### Forward Pass
A Neural Network starts with a **forward pass**. The model first processes the input. At each layer, the model computes a weighted sum of the inputs, adds a bias, and then applies an  **activation function**. The activation function will produce an output. For our purpose, we used a basic ReLU activation function.

**ReLU: f(x) = max(0,x)**

Then, the Neural Network may or may not add another bias and apply some weights to these outputs, and even add a function to combine all outcomes from all inputs in some way. Finally a result is returned by the Neural Network.

### Loss
After the forward pass, the network calculates the loss. Chances are very small that the model got the weights optimised instantly. To calculate the loss, it calculates the difference between the predicted output and the actual target value. Since we have a binary classifier, we use **binary cross-entropy loss**, or simply log loss.

$$
\text{loss}(q) = - \sum_{x \in X_Pos} \log\,q_x(Pos) \;-\; \sum_{x \in X_Neg} \log\,q_x(Neg)
$$

### Backpropagation
Since the network can now compute the loss for each output for some given input, it can start backpropagating. Using backpropagation, the network computes the gradient (the derivative) of the loss function with respect to each of the network's parameters (weights and biases). This is done using the chain rule of calculus, which propagates the error backword through the network.

$$
\frac{\partial \ell}{\partial \text{input}}
\;=\;
\frac{\partial \ell}{\partial \text{output}}
\;\cdot\;
\frac{\partial \text{output}}{\partial \text{input}}
$$

### Updating Weights and Biases
Once the gradients are computed, an optimisation algorithm updates the weights and biases in the opposite direction of the gradient. In our case, we use Adam. The update rule typically looks like the following:

$$
\theta_{\text{new}} = \theta_{\text{old}} - \eta \cdot \nabla_{\theta} L
$$

**Where:**

$$
\begin{aligned}
\theta &:\; \text{Model parameters (e.g., weights, biases)} \\
\theta_{\text{old}} &:\; \text{Parameter values before the update} \\
\theta_{\text{new}} &:\; \text{Parameter values after the update} \\
\eta &:\; \text{Learning rate, controlling the step size} \\
\nabla_{\theta} L &:\; \text{Gradient of the loss function with respect to the parameters}
\end{aligned}
$$

Imagine the loss surface like a mountain upside down. Therefore, we need to go in the opposite direction of the gradient.

### Iterate the Process
This whole process (forward pass, loss calculations, backpropagation, and parameter updates) is repeated for many iterations until the loss converges or the limit of iterations has been reached. Luckily for us, these steps are implemented internall in scikit-learn's MLPClassifier. We only call the .fit() function as a user. Sci-Kit learn does the computing and parameter tuning.

# How does the NN work?
Like mentioned earlier, the Neural Network is a squiggle fitting machine, rather than a regular function. It is non-linear, and it can take on very complex shapes. It can capture features that other machine learning models might miss. Also, since the Neural Network has the ability to calculate the loss across multiple layers, the Neural Network could potentially converge to a more optimal solution.
