#!/bin/bash

cuda_ver=$1
if [ -z "${cuda_ver}" ];
then
    echo "must give a cuda version, e.g., 100, 101, 110, ..."
    exit 1
fi

jaxlib=`pip freeze | grep jaxlib`
if [ -z "${jaxlib}" ];
then
    echo "could not detect current jaxlib version"
    exit 1
fi

pip install ${jaxlib}+cuda${cuda_ver} -f https://storage.googleapis.com/jax-releases/jax_releases.html
