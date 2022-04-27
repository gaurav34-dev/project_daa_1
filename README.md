# Image Quilting for Texture Synthesis 

Image quilting is a technique for stitching together portions of existing images to create new images.
We describe a simple image-based method for creating unique visual appearances that involves sewing together small pieces of existing images to create a new image. This is referred to as image quilting.
It has applications of 

1) Texture synthesis is the process of producing big textures from small real-world data.

>The approach works with photos directly and does not require 3D data.

We'll use Python and NumPy to implement texture synthesis in this repository.

## Texture Synthesis
Texture synthesis is a different approach of making textures. Visual recurrence is minimized since synthetic textures can be generated in any scale. By appropriately addressing the boundary requirements, texture synthesis can also yield tileable images. Image de-noising, occlusion fill-in, and compression are only a few examples of texture synthesis's potential applications.

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
