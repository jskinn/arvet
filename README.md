# John's Benchmark Framework
A framework for doing robot vision experiments

## Dependencies
The code is in python 3, tested with both 3.4 and 3.5. Does not support python 2.
The core structure depends on the following python libraries, which can be installed with pip:
- pickle
- numpy
- transforms3d
- pymongo
- unittest

Image dataset importing and image feature systems (module systems.features) depend on:
- opencv 3.0

Unreal Engine simulators depend on:
- unrealcv (use my fork from my github for additional camera features and python3 support)

Deep-learning depends on:
- keras (requires scikit-learn, pillow, h5py)
- tensorflow (see [https://www.tensorflow.org/install/install_linux], best with GPU and CUDA)
