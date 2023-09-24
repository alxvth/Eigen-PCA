# Principal components analysis with Eigen

Header-only C++ [principal components analysis](https://en.wikipedia.org/wiki/Principal_component_analysis) with [Eigen](https://gitlab.com/libeigen/eigen) (3.4.1 and newer).


## Example code

```cpp
#include "pca/eigen-pca.hpp"

// Define your data in the format
// [p0d0, p0d1, ..., p1d0, p1d1, ..., pNd0, pNd1, ..., pNdM]
// with p0 being point 0 and d0 dimension 0, up to point N and dimension M
std::vector<float> data_in;
size_t num_dims = 50;

// Compute first two principle components
std::vector<float> pca_out;
size_t num_comp = 2;

math::pca(data_in, num_dims, pca_out, num_comp);
```

## Settings
Two normalization procedures are implemented:
Other normalization steps before this centering are possible as well: 
- [Mean normalization](https://en.wikipedia.org/wiki/Feature_scaling#Mean_normalization) (`math::DATA_NORM::MEAN`) 
- [Rescaling (min-max normalization)](https://en.wikipedia.org/wiki/Feature_scaling#Rescaling_(min-max_normalization)) (`math::DATA_NORM::MINMAX`)
- No normalization (`math::DATA_NORM::NONE`)

Each normalization centers the data such that each dimension/channel has zero mean. 

This project implements two PCA computation algorithms: 
- Explicitly computing the eigenvectors of the covariance matrix (`math::PCA_ALG::COV`)
- Singular value decomposition (`math::PCA_ALG::SVD`).

The default settings are:
```cpp
math::pca(data_in, num_dims, pca_out, num_comp, math:PCA_ALG::SVD, math:DATA_NORM::MINMAX);
```

## Tests
To run some test, follow the [setup instructions](test/README.md), which include creating reference data and ensuring that Eigen is available.
