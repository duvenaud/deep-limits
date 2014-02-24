deep-limits
===========

Source code for the [paper](https://github.com/duvenaud/deep-limits/blob/master/latex/verydeep.pdf)

*Avoiding Pathologies in Very Deep Networks*
By
[David Duvenaud](http://mlg.eng.cam.ac.uk/duvenaud/),
[Oren Rippel](http://math.mit.edu/~rippel/),
[Ryan P. Adams](http://people.seas.harvard.edu/~rpa/),
and
[Zoubin Ghahramani](http://mlg.eng.cam.ac.uk/zoubin/)

To appear in AISTSATS 2014.

Abstract:
Choosing appropriate architectures and regularization strategies for deep networks is crucial to good predictive performance.  To shed light on this problem, we analyze the analogous problem of constructing useful priors on compositions of functions.  Specifically, we study the deep Gaussian process, a type of infinitely-wide, deep neural network.  We show that in standard architectures, the representational capacity of the network tends to capture fewer degrees of freedom as the number of layers increases, retaining only a single degree of freedom in the limit.  We propose an alternate network architecture which does not suffer from this pathology.  We also examine deep covariance functions, obtained by composing infinitely many feature transforms.  Lastly, we characterize the class of models obtained by performing dropout on Gaussian processes.


The source directory contains code to generate all the figures.

Feel free to email me with any questions at (dkd23@cam.ac.uk).

