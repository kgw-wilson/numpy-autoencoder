# More Info on Weight Initialization

This document is meant to serve as a reference for the AuotEncoder.init_weights function.

For more info, see the Wikipedia links throughout the document.

## Calculations

We want the output variance of each layer to be similar to the input variance so that signals don't vanish or explode. For a layer:

    z = x @ W + b

Here:

• x has shape [batch_size, n_in]  

• W has shape [n_in, n_out]  

• b has shape [1, n_out] (broadcasted across the batch)  

• z has shape [batch_size, n_out]

The variance of the layer output needs to be similar to the variance of the input so that signals don't explode or vanish as they pass through the network.

## Weight Initialization from Standard Normal Distribution

Taking the variance of both sides of the above equation yields:

    var(z) = var(x @ W + b)

The usual convention is to set var(z) = 1 so that variance does not grow or shrink on average as signal passes through the network. Then:

    1 = var(x @ W + b)

b is initialized to zero to simpmlify the math, so it doesn't contribute to the sum. So we can simplify as:

    1 = var(x @ W)

If we assume all output neurons behave the same statistically, then matrix multiplication is the sum of products:

    1 = var(sum_i {x_i * W_i})

We assume all weights and inputs are uncorrelated, so that means the variance of their sum is equal to the sum of their variances (see [Wikipedia](https://en.wikipedia.org/wiki/Variance#Linear_combinations) for more info). Thus:

    1 = sum_i{var(x_i) * var(W_i)}

x_i is just a scalar value (the value of the input layer's neuron) and the variance of a constant is 1. Then:

    1 = sum_i{var(W_i)}

Each weight will be sampled from a normal distribution with constant variance sigma_w^2, so:

    1 = sum_i{sigma_w^2}

i represents the indices of input neurons, so if we say that i ranges from i to neurons_in, then the sum from i to neurons_in of the unknown constant simplifies to:

    1 = neurons_in * sigma_w^2

Solving for the standard deviation of our weight matrix (sigma_w) yeilds:

    sigma_w = sqrt{1 / neurons_in}

## Using ReLU

Because the ReLU function zeros out all activations below 0, we multiply by 2 to account for half of the variance being zeroed out. This gives us a final value of:

    sigma_w = sqrt{2 / neurons_in}

when using the ReLU activation function for a network whose weights are initialized from a standard normal distribution.

## Weight Initialization from Uniform Distribution

The math is exactly the same as the first few steps above except weights will be sampled from a uniform distribution instead of a standard normal. The variance of a uniform distribution over an interval [a,b] is:

    (b - a)^2 / 12

We want the interval to be centered around 0 to give the weights a mean value of 0. This helps prevent their values from growing or shrinking on average throughout the network. Thus, the starting point a will be the end point b reflected across the origin giving us a = -b and the interval [-b, b]. Substituting with a = -b gives us:

    (b - (-b))^2 / 12 = (2b)^2 / 12 = 4b^2 / 12 = b^2 / 3

Thus, we use this variance instead of the constant sigma_w we used with the standard normal in the steps above:

    var(z) = 1 = sum_i{var(W_i)} = sum_i{b^2 / 3}

Both var(W_i) and b do not depend on i, so we can simplify the sums with:

    nuerons_in * var(W_i) = (neurons_in * b^2) / 3

During backprop (the process of iteratively adjusting the weights going backward from a final output), weights change in the backward direction. Thus, to avoid those differences in weights (gradients) from vanishing or exploding too, we also have to consider the number of neurons in the output layer. Derivations by Glorot and Bengio to find a compromise between the forward and backward passes result in the following:

    var(W_i) = 2 / (neurons_in + neurons_out)

So we can combine the two equations:

    (2 * neurons_in) / (neurons_in + neurons_out) = (neurons_in * b^2) / 3

Dividing both sides by neurons_in and multiplying both sides by 3 we find:

    6 / (neurons_in + neurons_out) = b^2

So:

    b = sqrt{6 / (neurons_in + neurons_out)}
