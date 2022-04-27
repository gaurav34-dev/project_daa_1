#libraries install 
%pip install scikit-image
%pip install skimage
%pip install ssl
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import numpy as np
import math
from skimage import io, util
import heapq

             #Finding out the random patch
def randomPatch_1(texture_portion, patchLengthsegment):
    h1, w1, _ = texture_portion.shape
    i = np.random.randint(h1 - patchLengthsegment)
    j = np.random.randint(w1 - patchLengthsegment)

    return texture_portion[i:i+patchLengthsegment, j:j+patchLengthsegment]

          #Finding out the overlapping difference
def L2OverlapDiff_1(patch, patchLengthsegment, overlap, res, y1, x1):
    error = 0

    if x1 > 0:
        leftpart = patch[:, :overlap] - res[y1:y1+patchLengthsegment, x1:x1+overlap]
        error = error + np.sum(leftpart**2)
    if y1 > 0:
        up   = patch[:overlap, :] - res[y1:y1+overlap, x1:x1+patchLengthsegment]
        error = error + np.sum(up**2)

    if x1 > 0 and y1 > 0:
        corner = patch[:overlap, :overlap] - res[y1:y1+overlap, x1:x1+overlap]
        error = error - np.sum(corner**2)

    return error
 
                  #Finding out the best random patch in all
def randomBestPatch_1(texture_portion, patchLengthsegment, overlap, res, y1, x1):
    h1, w1, _ = texture_portion.shape
    errors = np.zeros((h1 - patchLengthsegment, w1 - patchLengthsegment))
    i=0
    while i<(h1 - patchLengthsegment):
        j=0
        while j<(w1 - patchLengthsegment):                                    
            patch = texture_portion[i:i+patchLengthsegment, j:j+patchLengthsegment]
            e = L2OverlapDiff_1(patch, patchLengthsegment, overlap, res, y1, x1)
            errors[i, j] = e
            j=j+1
        i=i+1
    i, j = np.unravel_index(np.argmin(errors), errors.shape)
    return texture_portion[i:i+patchLengthsegment, j:j+patchLengthsegment]


      #Applying the min-cut
def minCutPath_1(errors):
    # Dijkstra's algorithm vertical
    pq = [(error, [i]) for i, error in enumerate(errors[0])]
    heapq.heapify(pq)

    h1, w1 = errors.shape
    seen = set()

    while pq:
        error, path = heapq.heappop(pq)
        curDepth = len(path)
        curIndex = path[-1]

        if curDepth == h1:
            return path
    
        for delta in -1, 0, 1:
            nextIndex = curIndex + delta

            if 0 <= nextIndex and nextIndex < w1:
                if (curDepth, nextIndex) not in seen:
                    cumError = error + errors[curDepth, nextIndex]
                    heapq.heappush(pq, (cumError, path + [nextIndex]))
                    seen.add((curDepth, nextIndex))
        

def minCutPath2(errors):
    # Dynamic programming, used
    errors = np.pad(errors, [(0, 0), (1, 1)], 
                    mode='constant', 
                    constant_values=np.inf)

    cumError = errors[0].copy()
    paths = np.zeros_like(errors, dtype=int)    
    
    i=1
    while i!=len(errors):
        M = cumError
        L = np.roll(M, 1)
        R = np.roll(M, -1)

        # Optimize with np.choose?
        cumError = np.min((L, M, R), axis=0) + errors[i]
        paths[i] = np.argmin((L, M, R), axis=0)
        i=i+1
    paths -= 1
    
    minCutPath_1 = [np.argmin(cumError)]
    i=1
    while i!=(reversed(len(errors))):
        minCutPath_1.append(minCutPath_1[-1] + paths[i][minCutPath_1[-1]])
        i=i+1
    return map(lambda x1: x1 - 1, reversed(minCutPath_1))

          #Getting the minimum cut patch 
def minCutPatch_1(patch, patchLengthsegment, overlap, res, y1, x1):
    patch = patch.copy()
    dy, dx, _ = patch.shape
    minCut = np.zeros_like(patch, dtype=bool)

    if x1 > 0:
        leftpart = patch[:, :overlap] - res[y1:y1+dy, x1:x1+overlap]
        leftL2 = np.sum(leftpart**2, axis=2)
        for i, j in enumerate(minCutPath_1(leftL2)):
            minCut[i, :j] = True

    if y1 > 0:
        up = patch[:overlap, :] - res[y1:y1+overlap, x1:x1+dx]
        upL2 = np.sum(up**2, axis=2)
        for j, i in enumerate(minCutPath_1(upL2.T)):
            minCut[:i, j] = True

    np.copyto(patch, res[y1:y1+dy, x1:x1+dx], where=minCut)

    return patch

          #Quilting
def quilt(texture_portion, patchLengthsegment, numPatches, mode="cut", sequence=False):
    texture_portion = util.img_as_float(texture_portion)

    overlap = patchLengthsegment // 6
    numPatchesHigh_1, numPatchesWide_1 = numPatches

    h1 = (numPatchesHigh_1 * patchLengthsegment) - (numPatchesHigh_1 - 1) * overlap
    w1 = (numPatchesWide_1 * patchLengthsegment) - (numPatchesWide_1 - 1) * overlap

    res = np.zeros((h1, w1, texture_portion.shape[2]))
    i=0
    while i!=(numPatchesHigh_1):
        j=0
        while j!=(numPatchesWide_1):
            y1 = i * (patchLengthsegment - overlap)
            x1 = j * (patchLengthsegment - overlap)

            if i == 0 and j == 0 or mode == "random":
                patch = randomPatch_1(texture_portion, patchLengthsegment)
            elif mode == "best":
                patch = randomBestPatch_1(texture_portion, patchLengthsegment, overlap, res, y1, x1)
            elif mode == "cut":
                patch = randomBestPatch_1(texture_portion, patchLengthsegment, overlap, res, y1, x1)
                patch = minCutPatch_1(patch, patchLengthsegment, overlap, res, y1, x1)
            
            res[y1:y1+patchLengthsegment, x1:x1+patchLengthsegment] = patch

            if sequence:
                io.imshow(res)
                io.show()
            j=j+1
        i=i+1
    return res

           #Find out Quilt Size
def quiltSize(texture_portion, patchLengthsegment, shape, mode="cut"):
    overlap = patchLengthsegment // 6
    h1, w1 = shape

    numPatchesHigh_1 = math.ceil((h1 - patchLengthsegment) / (patchLengthsegment - overlap)) + 1 or 1
    numPatchesWide_1 = math.ceil((w1 - patchLengthsegment) / (patchLengthsegment - overlap)) + 1 or 1
    res = quilt(texture_portion, patchLengthsegment, (numPatchesHigh_1, numPatchesWide_1), mode)

    return res[:h1, :w1]


# In[4]:


s = "https://raw.githubusercontent.com/gbaser54/project_daa_1/master/"

texture = io.imread(s + "test.png")
io.imshow(texture)
io.show()

io.imshow(quilt(texture, 25, (6, 6), "random"))
io.show()

io.imshow(quilt(texture, 25, (6, 6), "best"))
io.show()

io.imshow(quilt(texture, 20, (6, 6), "cut"))
io.show()


# In[ ]:




