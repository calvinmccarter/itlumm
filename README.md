### Paper

"Look-ups are not (yet) all you need for deep learning inference" was presented at the [Sparsity in Neural Networks Workshop 2022](https://www.sparseneural.net/accepted-papers). See the [arXiv preprint](https://arxiv.org/abs/2207.05808) and please cite as follows:

```bibtex
@misc{https://doi.org/10.48550/arxiv.2207.05808,
  doi = {10.48550/ARXIV.2207.05808},
  url = {https://arxiv.org/abs/2207.05808},
  author = {McCarter, Calvin and Dronen, Nicholas},
  title = {Look-ups are not (yet) all you need for deep learning inference},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
```

### Code

This repo is mostly copypasta from the [Bolt repo](https://github.com/dblalock/bolt) which contains code for [MADDNESS](http://proceedings.mlr.press/v139/blalock21a.html) -- see our `bolt` directory. The `maddness-old` directory is deprecated and can be safely ignored. In the code, `ITLUMM` and `Pluto` refer to the same method; `Vingilote` refers to an early, inferior version of `ITLUMM`. The `driveit` directory contains code related to acceleration of full NNs with replacement of linear layers and fine-tuning. The `snn2022` directory contains materials for the SNN2022 paper.
