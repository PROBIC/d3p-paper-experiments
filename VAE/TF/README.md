Scripts for experiments regarding performance of TF on VAE in Sec 5.1.

- `vae_tf.py`: Implementation of the VAE model and inference on MNIST for TensorFlow.
- `vae_tf_fashion.py`: Implementation of the VAE model and inference on Fashion-MNIST for TensorFlow.
- `vae_tf_cifar.py`: Implementation of the VAE model and inference on CIFAR10 for TensorFlow.
- `vae_tf_sgd.py`: Same as `vae_tf.py` but using SGD instead of Adam optimizer; not shown in paper.
- `runmultiple.sh`: Runs all required evaluation runs of `vae_tf*.py` to create measurements of TensorFlow for Tables 2 and 3, Sec. 5.1.
- `process.py`: Scripts for processing results obtained from runs and creating the tables.
