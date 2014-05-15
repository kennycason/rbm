Restricted Boltzmann Machine
===========
This is a simple implementation modeled from http://blog.echen.me/2011/07/18/introduction-to-restricted-boltzmann-machines/
Currently has no bias

Results RBM(visual=6,hidden=4)
```
Training Data:
[[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
 [1.0, 0.0, 1.0, 0.0, 0.0, 0.0]
 [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
 [0.0, 0.0, 1.0, 1.0, 1.0, 0.0]
 [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
 [0.0, 0.0, 1.0, 1.0, 1.0, 0.0]]

Input:  [[0.0, 0.0, 0.0, 1.0, 1.0, 0.0]]
Output: [[0.0, 0.0, 1.0, 1.0, 1.0, 0.0]]

Inputs: [[0.0, 0.0, 0.0, 1.0, 1.0, 0.0] [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]]
Outputs: [0.0, 0.0, 1.0, 1.0, 1.0, 0.0] [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]]
```