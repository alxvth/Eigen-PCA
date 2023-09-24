# PCA Plugin Test

Before setting up the test project with cmake, make sure that Eigen is availbale (as a submodules of this repo) and prepare the test data:
```
git clone --recurse-submodule https://github.com/alxvth/Eigen-PCA.git

cd Eigen-PCA/test
pip install -r requirements.txt
python create_reference_data.py

mkdir build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

Running `./create_reference_data.py` creates reference data sets and reference PCA transformations in `.data`  using [scikit-learn](https://scikit-learn.org) and [numpy](https://numpy.org/).

## Dependencies
Here we use Eigen as a submodule. In your project you can also setup [Eigen](https://gitlab.com/libeigen/eigen) as described in their [documentation](https://eigen.tuxfamily.org/dox/TopicCMakeGuide.html).

Other dependencies, as specified in `./vcpkg.json` are
- [Catch2](https://github.com/catchorg/Catch2) for unit testing
- [nlohmann_json](https://github.com/nlohmann/json) for reading meta data about the test data from json files in cpp
