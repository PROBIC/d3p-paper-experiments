# d3p-paper-experiments
Experiments for d3p paper

See the subdirectory README files for information for the individual experiments in the paper.

## Overall setup
This repo relies on [nivida-docker](https://github.com/NVIDIA/nvidia-docker) for providing clean working environment with GPU support.

Build and run the docker container:
```
docker build . -t d3p-experiments
nvidia-docker run -it --rm d3p-experiments
```
Follow potentially shown instructions to adjust available memory for the container.

By default, only the CPU backend for JAX is installed. To enable GPU support, inside the container run
```
./reinstall_jaxlib_cuda.sh
```

## License
Code in this repository is licensed under the same CC-BY-NC-ND license as the associated paper.