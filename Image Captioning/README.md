# Image Captioning.
- endoder is a CNN followed by NetVLAD, decoder is RNN or LSTM.
- `glove.py` implements the GloVe model over the available corpus.
- `caption_glove.py` extracts the captions and images from the dataset and trains a glove model over the same.
- `image_caption_dataset.py` implements PyTorch Data Loaders.
- `rnn.py` is the model and train code for CNN-RNN based encoder-decoder archiecture.
- GloVe model implemenation is borrowed from [this repo](https://github.com/kefirski/pytorch_GloVe).