
This page describes how to reproduce the experimental results reported in our paper.

Note that this page (and the clean, easy-to-use version of our code) are still under construction and we refer the reader to https://smarturl.it/Maddness for the latest version.
<!-- Note that we built Maddness as a fork of [Bolt](https://github.com/dblalock/bolt); while this gives us access to highly optimized implementations of rival methods, it does mean that this "standalone" version of Maddness is not thoroughly tested and is a work in progress. -->

<!-- Also note that this code is built on the [Bolt repository](https://github.com/dblalock/bolt), so the implementations for competing methods are highly optimized. -->

## Install Dependencies

To run the experiments, you will first need to obtain the following tools / libraries and datasets.

### C++ Code

- [Xcode](https://developer.apple.com/xcode/), to run the C++ timing benchmarks using Xcode (this is the "official" version that is much better tested and actually works at the moment)
- [Bazel](http://bazel.build), Google's open-source build system (support coming soon...)

### Python Code:
- [Joblib](https://github.com/joblib/joblib) - for caching function output
- [Scikit-learn](https://github.com/scikit-learn/scikit-learn) - for k-means
- [Kmc2](https://github.com/obachem/kmc2) - for k-means seeding
- [Pandas](http://pandas.pydata.org) - for storing results and reading in data
- [Seaborn](https://github.com/mwaskom/seaborn) - for plotting, if you want to reproduce our figures

### Datasets

- Download [Caltech 101](http://www.vision.caltech.edu/Image_Datasets/Caltech101/) and the [UCR Time Series Archive](https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/).
- Edit `python/datasets/paths.py` to point to where you're storing them.

The activations and weights from the CIFAR-10 and CIFAR-100 datasets are already included under `python/assets`.

## View Existing Results

All results are in `python/results/amm`. The timing results are in the subdirectory `timing`.

## Reproduce Timing / Throughput results

The C++ code is driven by [Catch](https://github.com/catchorg/Catch2) run via Xcode. You can just open Bolt.xcodeproj (Maddness was built as a fork of [Bolt](https://github.com/dblalock/bolt)) and press run with the appropriate arguments. For different experiments, the arguments are:

- f() speed for various methods: `[scan][amm]~[old]`
- g() speed for various methods: `[encode][amm]~[old]`
- h() speed for various (not reported in the paper, but interesting): `[lut][amm]\~[old]`
- Overall AMM speed: `[matmul][amm]~[old]`.

We highly recommend running this when the machine is otherwise idle. Also note that we haven't yet automated having the C++ code dump results into the appropriate files, so you'll have to manually paste the output into the corresponding file in `python/results/amm/timing`.

Coming soon: Working Bazel build for all the code and wrapper shell scripts to run and store the output of each experiment.

## Reproduce Accuracy Results

From the `python` directory, run `python -m python.amm_main`. This will run all the methods we showed in the body of the paper (and some others that run quickly) on CIFAR-10, CIFAR-100, Caltech 101 using both the Sobel and Gaussian filters, and the datasets from the UCR Time Series Archive.

## Reproduce Plots

From the `python` directory, run `python -m python.amm_figs2`. You can uncomment different lines in `main` to only produce subsets of the plots.

## Other notes

Our method is called Mithral in the source code, not Maddness.
