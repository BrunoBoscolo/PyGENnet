import numpy as np
import struct

def load_mnist(images_filepath, labels_filepath):
    """
    Loads MNIST data from IDX files.
    """
    with open(labels_filepath, 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)

    with open(images_filepath, 'rb') as f:
        magic, size, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(len(labels), rows, cols)

    return images, labels
