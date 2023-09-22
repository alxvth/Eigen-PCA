# Principal components analysis with Eigen

Performs a [principal components analysis](https://en.wikipedia.org/wiki/Principal_component_analysis) with [Eigen](https://gitlab.com/libeigen/eigen).

By default, the plugin internally centers the data so that each dimension/channel has zero mean.
Other normalization steps before this centering are possible as well: [Mean normalization](https://en.wikipedia.org/wiki/Feature_scaling#Mean_normalization) and [Rescaling (min-max normalization)](https://en.wikipedia.org/wiki/Feature_scaling#Rescaling_(min-max_normalization)).
This plugin implements two PCA computation algorithms: Explicitly computing the eigenvectors of the covariance matrix and singular value decomposition.

To run the test, make sure to fetch all submodules (Eigen) when cloning the repo (`--recurse-submodule`) and prepare the test data as described in [test/README.md](test/README.md).

