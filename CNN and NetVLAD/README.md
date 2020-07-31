# Convolutional Neural Networks with a NetVLAD layer.
- `process_dataset.py` and `load_dataset.py` for processing and loading the [Caltech-UCSD Birds dataset](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)
- `pretrained.py` implements MLFFNN based classifiers using features from VGG16 and GoogLeNet.
- `cnn.py` implements a simple 2 layer cnn with max-pool and fully-connected classifier layers.
- `vlad.py` augments the above model with NetVLAD layer before the fully connected layers.
- NetVLAD implementation borrows inspiration from [this repo](https://github.com/lyakaap/NetVLAD-pytorch).