# Principal components analysis with Eigen

Header-only C++ [principal components analysis](https://en.wikipedia.org/wiki/Principal_component_analysis) with [Eigen](https://gitlab.com/libeigen/eigen) (3.4.1 and newer).

To run the [tests](test/README.md), make sure to Eigen as a submodules when cloning the repo and prepare the test data:
```
git clone --recurse-submodule https://github.com/alxvth/Eigen-PCA.git

cd Eigen-PCA/test
pip install -r requirements.txt
python create_reference_data.py
```

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
By default, the the data is centered such that each dimension/channel has zero mean.
Other normalization steps before this centering are possible as well: [Mean normalization](https://en.wikipedia.org/wiki/Feature_scaling#Mean_normalization) and [Rescaling (min-max normalization)](https://en.wikipedia.org/wiki/Feature_scaling#Rescaling_(min-max_normalization)).
This plugin implements two PCA computation algorithms: Explicitly computing the eigenvectors of the covariance matrix and singular value decomposition.
