Multi-Layer Perceptron From Scratch

A simple multi-layer neural network (MLP) built from scratch in Python using only NumPy — no PyTorch, no TensorFlow. It classifies points on a 2D plot into multiple classes, and plots the loss and accuracy as it trains. there is also a graph named loss_acc.png what will show you what the loss and accuracy look like through out the training 

Uses SGD with momentum. Trained on the nnfs spiral dataset.

Results
Training accuracy: ~92–93%
Test accuracy: ~86–91%
Known limitations
The spiral dataset has overlapping points near the center, which makes it harder for the network to learn a clean boundary.
Currently trained on CPU only, so it's hard to scale up to bigger datasets for now.

Usage
pip install numpy nnfs matplotlib
python train.py
python predict.py
