In this experiment I took a single layer RBM, and trained it with 2 sets of images. Black Background (all white converted to black) and White Background (original)
The results were that the black bg models all trained faster and more clearly. It seems that having too much white (translates to a binary 11111111_11111111_11111111)
causes too much noise and affects learning.