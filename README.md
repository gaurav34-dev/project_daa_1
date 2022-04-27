# Image Quilting for Texture Synthesis 

Image quilting is a technique for stitching together portions of existing images to create new images.
It has applications of 

1) Texture synthesis is the process of producing big textures from small real-world data.


>The approach works with photos directly and does not require 3D data.

We'll use Python and NumPy to implement texture synthesis in this repository.

## Texture Synthesis

An input image and a block size are used to start the process:

![input block](input.png)

Then, between the overlap of two blocks, we design a minimum cost path:

<img src="slide.png" width=500 />

Then, by tiling small chunks of the input image, we create a synthesized image.

![build](build.png)

(a) We basically pick blocks at random here.

(b) Here, we select blocks with the least amount of overlap error.

(c) We follow the same steps as in (c), but we also cut along the minimum error border.

reference-https://www.youtube.com/watch?v=QMiCNJofJUk
          https://people.eecs.berkeley.edu/~efros/research/quilting/quilting.pdf
