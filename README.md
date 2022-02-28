# gaussian-processes
This project worked with Gaussian processes using an RBF kernel and a polynomial kernel with degree 1 (i.e., linear).
This was an assignment from my graduate level machine learning class.

## Results
### Visualizing performance on the 1D dataset
For this part, I had a given function to be learned, and I used both linear and RBF kernels to do so. Below are the plots of the learned function vs the real function.
<img src="https://github.com/bjmcshane/gaussian-processes/blob/main/images/1D_linear.png?raw=true" alt="drawing" width="350"/>
<img src="https://github.com/bjmcshane/gaussian-processes/blob/main/images/1D_rbf.png?raw=true" alt="drawing" width="350"/>



### Performance as a function of iterations
I had to run both linear and rbf kernels on all 4 given datasets, and record/plot the MNLL (mean negative log likelihood, think of it as the error) as a function of training iterations. Below are the results.
<img src="https://github.com/bjmcshane/gaussian-processes/blob/main/images/1D_linear_mnll.png?raw=true" alt="drawing" width="350"/>
<img src="https://github.com/bjmcshane/gaussian-processes/blob/main/images/1D_rbf_mnll.png?raw=true" alt="drawing" width="350"/>
<img src="https://github.com/bjmcshane/gaussian-processes/blob/main/images/CRIME_LINEAR.png?raw=true" alt="drawing" width="350"/>
<img src="https://github.com/bjmcshane/gaussian-processes/blob/main/images/crime_rbf.png?raw=true" alt="drawing" width="350"/>
<img src="https://github.com/bjmcshane/gaussian-processes/blob/main/images/artsmall_linear.png?raw=true" alt="drawing" width="350"/>
<img src="https://github.com/bjmcshane/gaussian-processes/blob/main/images/artsmall_rbf.png?raw=true" alt="drawing" width="350"/>
<img src="https://github.com/bjmcshane/gaussian-processes/blob/main/images/housing_linear.png?raw=true" alt="drawing" width="350"/>
<img src="https://github.com/bjmcshane/gaussian-processes/blob/main/images/housing_rbf.png?raw=true" alt="drawing" width="350"/>



### Comparison to Bayesian Linear Regression
Below compares the resulting error and the optimized parameters of our models to those of the bayesian linear regression from earlier in the semester.
<img src="https://github.com/bjmcshane/gaussian-processes/blob/main/images/gp_mse.png?raw=true" alt="drawing" width="350"/>
<img src="https://github.com/bjmcshane/gaussian-processes/blob/main/images/blr_mse.png?raw=true" alt="drawing" width="350"/>
