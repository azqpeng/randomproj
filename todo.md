

TO DO

- Select regression dataset that you will be working on 
- Implement unpruned model
- Implement mask pruning (randomly sselect some )
- Implement random gaussian projections



Suppose we have a hidden layer of width n. Then, between two equal hidden layers, their is a n-by-n weight matrix, for a total of n^2 training weights. 

In model 1, these weights are trained as given. 
In models 2 and 3, there is some subspace that is randomly projected to "cover" the n-by-n weight matrix. 
    TO DO: Is this subspace compared to n, or compared to the parameter space n^2? 
    
    In model 2, this projection is sort of cool - just create an n-by-n mask, where d random nodes are selected in n^2 as 1's and the others are 0's. 
    
    In model 3, take a w-length vector, and use a random projection to transform it into an n^2 matrix. Be sure to use Kaiming initialization.


FORWARD PROPOGATION:
- 

BACKWARDS PROPOGATION:
- In backprop, the NN adjusts its parameters proportionate to the error in its guess. It does this by traversing backwards from the output, collecting the derivatives of the error with respect to the parameters of the functions (gradients), and optimizing the parameters using gradient descent. For a more detailed walkthrough of backprop, check out this video from 3Blue1Brown.

- Weight Initialization
  - Kaiming
  - Automatic Diferentiaion


RUNNING THE EXPERIMENTS:
- Iterations required to converge
- Loss achieved
