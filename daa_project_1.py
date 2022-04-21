#!/usr/bin/env python
# coding: utf-8

# In[10]:


from PIL import Image, ImageFilter

# Open image file
im = Image.open('/Users/gauravbaser/Downloads/lily.gif')

print("\n** Analysing image **\n")

# Display image format, size, colour mode
print("Format:", im.format, "\nWidth:", im.width, "\nHeight:", im.height, "\nMode:", im.mode)

# Check if GIF is animated
frames = im.n_frames
print("Number of frames: " + str(frames))

# Convert to RGB
rgb_im = im.convert('RGB')

print("\n** Converting image **\n")


# Iterate through frames and pixels, top row first
for z in range(frames):

    # Go to frame
    im.seek(z)
    print("Frame: ", im.tell())

    for y in range(im.width):
        for x in range(im.height):

            # Get RGB values of each pixel
            r, g, b = rgb_im.getpixel((x, y))

            print(r, g, b)


# In[ ]:




