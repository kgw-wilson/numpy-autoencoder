# Backprop Calculations

This document is meant to serve as a reference for the AuotEncoder.backprop function. Thus, it goes through the backprop calculations using a format very similar to the code.

For more info, see [Wikipedia](https://en.wikipedia.org/wiki/Backpropagation#Finding_the_derivative_of_the_error).
        
## Calculations

At the last layer:

    1. z_i = x_{i-1} @ W_i + b_i
    2. act(z_i) = out
    3. loss = 1/N sum_i{(output - input)^2} # <- Using Mean Squared Error

Where N = batch_size * input length. In MNIST, input images are 28x28, so N = batch_size * 784. Here, act just represents the activation function which can be ReLU or Sigmoid in the code.

Then, to compute the deltas to apply to W and b, use the chain rule:

    4. dl/dz_i = dl/dout * dout/dz_i
    5. dl/dout = 2 * (output - input) / N # <-  Take derivative w.r.t. out, summation disappears because the derivative is element-wise
    6. dout/dz_i = dact/dz_i
    7. dl/dz_i = 2/N * (output - input) * dact/dz_i

    8. dl/dW_i = dl/dz_i * dz_i/dW_i
    9. dz_i/dW_i = x_{i-1}
    10. dl/dW_i = x_{i-1}.T @ dl/dz_i # <- transpose to make shape match

    11. dl/db_i = dl/dz_i * dz_i/db_i
    12. dz_i/db_i = 1
    13. dl/db_i = mean(dl/dz_i, axis=0) # <- Take mean over batch

We can also solve for dl/dx_{i-1}, the error signal for the previous layer:

    14. dl/dx_{i-1} = dl/dz_i * dz_i/dx_{i-1}
    15. dz_i/dx_{i-1} = W_i
    16. dl/dx_{i-1} = dl/dz_i @ W_i.T # <- transpose to make shape match

Where x_{i-1} is the output of the previous layer (or input image for i=1). We have already computed the loss for the last layer above. Then, for the second to last layer i-1, we want to find dl/dz_{i-1} so we can calculate dl/dW_{i-1} and dl/db_{i-1}. Now, x_{i-1} is the output of our current layer.

    17. z_{i-1} = x_{i-2} @ W_{i-1} + b_{i-1}
    18. x_{i-1} = act(z_{i-1})

To find the deltas at the current layer, we again start with the derivative of loss with respect to the pre-activations z, as we did on line 4 except x_{i-1} is the output. This time it's for layer i-1:
    
    19. dl/dz_{i-1} = dl/dx_{i-1} * dx_{i-1}/dz_{i-1}
    20. dl/dz_{i-1} = dl/dx_{i-1} * dact/dz_{i-1}

Then we can compute deltas for W_{i-1} the same way we did on line 8:

    21. dl/dW_{i-1} = dl/dz_{i-1} * dz_{i-1}/dW_{i-1}

Referencing line 17, we take the derivative of z_{i-1} with respect to W_{i-1} and get:
    
    22. dz_{i-1}/dW_{i-1} = x_{i-2}

So if we substitute the value for dl/dz_{i-1} we found on line 20 and the value for dz_{i-1}/dW_{i-1} into the equation on line 21:

    23. dl/dW_{i-1} = dl/dx_{i-1} * dact/dz_{i-1} @ x_{i-2}

And finally we substitute in for dl/dx_{i-1} from line 16 and get
    
    24. dl/dW_{i-1} = (dl/dz_i @ W_i.T) * dact/dz_{i-1} @ x_{i-2}

Which reveals that the changes in the second to last layer depend on the dl/dz_i values and the weights of the last layer. This means that if we record the dl/dz values for each layer i, we can compute the updates to the weights at layer i-i using that value and the weights at i.

Similarly, for the biases we do the same thing we did on line 11:

    25. dl/db_{i-1} = dl/dz_{i-1} * dz_{i-1}/db_{i-1}

Where looking at line 17, dz_{i-1}/db_{i-1} is just 1.

Then, substitute in our value for dl/dz_{i-1} from line 20:

    26. dl/db_{i-1} = dl/dx_{i-1} * dact/dz_{i-1}

And finally we substitute in for dl/dx_{i-1} and get
    
    27. dl/db_{i-1} = (dl/dz_i @ W_i.T) * dact/dz_{i-1}

Which is the same equation as on line 24 except without the x term.
