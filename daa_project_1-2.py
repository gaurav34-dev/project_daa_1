%pip install scikit-image
%pip install skimage
%pip install ssl
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import numpy as np
import math
from skimage import io, util
import heapq


def randomPatch_1(texture, patchLength):
    h, w, _ = texture.shape
    i = np.random.randint(h - patchLength)
    j = np.random.randint(w - patchLength)

    return texture[i:i+patchLength, j:j+patchLength]


def L2OverlapDiff_1(patch, patchLength, overlap, res, y, x):
    error = 0

    if x > 0:
        left = patch[:, :overlap] - res[y:y+patchLength, x:x+overlap]
        error = error + np.sum(left**2)
    if y > 0:
        up   = patch[:overlap, :] - res[y:y+overlap, x:x+patchLength]
        error = error + np.sum(up**2)

    if x > 0 and y > 0:
        corner = patch[:overlap, :overlap] - res[y:y+overlap, x:x+overlap]
        error = error - np.sum(corner**2)

    return error
 

def randomBestPatch_1(texture, patchLength, overlap, res, y, x):
    h, w, _ = texture.shape
    errors = np.zeros((h - patchLength, w - patchLength))
    i=0
    while i<(h - patchLength):
        j=0
        while j<(w - patchLength):                                    
            patch = texture[i:i+patchLength, j:j+patchLength]
            e = L2OverlapDiff_1(patch, patchLength, overlap, res, y, x)
            errors[i, j] = e
            j=j+1
        i=i+1
    i, j = np.unravel_index(np.argmin(errors), errors.shape)
    return texture[i:i+patchLength, j:j+patchLength]



def minCutPath_1(errors):
    # dijkstra's algorithm vertical
    pq = [(error, [i]) for i, error in enumerate(errors[0])]
    heapq.heapify(pq)

    h, w = errors.shape
    seen = set()

    while pq:
        error, path = heapq.heappop(pq)
        curDepth = len(path)
        curIndex = path[-1]

        if curDepth == h:
            return path
    
        for delta in -1, 0, 1:
            nextIndex = curIndex + delta

            if 0 <= nextIndex and nextIndex < w:
                if (curDepth, nextIndex) not in seen:
                    cumError = error + errors[curDepth, nextIndex]
                    heapq.heappush(pq, (cumError, path + [nextIndex]))
                    seen.add((curDepth, nextIndex))
        

def minCutPath2(errors):
    # dynamic programming, used
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

        # optimize with np.choose?
        cumError = np.min((L, M, R), axis=0) + errors[i]
        paths[i] = np.argmin((L, M, R), axis=0)
        i=i+1
    paths -= 1
    
    minCutPath_1 = [np.argmin(cumError)]
    i=1
    while i!=(reversed(len(errors))):
        minCutPath_1.append(minCutPath_1[-1] + paths[i][minCutPath_1[-1]])
        i=i+1
    return map(lambda x: x - 1, reversed(minCutPath_1))


def minCutPatch_1(patch, patchLength, overlap, res, y, x):
    patch = patch.copy()
    dy, dx, _ = patch.shape
    minCut = np.zeros_like(patch, dtype=bool)

    if x > 0:
        left = patch[:, :overlap] - res[y:y+dy, x:x+overlap]
        leftL2 = np.sum(left**2, axis=2)
        for i, j in enumerate(minCutPath_1(leftL2)):
            minCut[i, :j] = True

    if y > 0:
        up = patch[:overlap, :] - res[y:y+overlap, x:x+dx]
        upL2 = np.sum(up**2, axis=2)
        for j, i in enumerate(minCutPath_1(upL2.T)):
            minCut[:i, j] = True

    np.copyto(patch, res[y:y+dy, x:x+dx], where=minCut)

    return patch


def quilt(texture, patchLength, numPatches, mode="cut", sequence=False):
    texture = util.img_as_float(texture)

    overlap = patchLength // 6
    numPatchesHigh_1, numPatchesWide_1 = numPatches

    h = (numPatchesHigh_1 * patchLength) - (numPatchesHigh_1 - 1) * overlap
    w = (numPatchesWide_1 * patchLength) - (numPatchesWide_1 - 1) * overlap

    res = np.zeros((h, w, texture.shape[2]))
    i=0
    while i!=(numPatchesHigh_1):
        j=0
        while j!=(numPatchesWide_1):
            y = i * (patchLength - overlap)
            x = j * (patchLength - overlap)

            if i == 0 and j == 0 or mode == "random":
                patch = randomPatch_1(texture, patchLength)
            elif mode == "best":
                patch = randomBestPatch_1(texture, patchLength, overlap, res, y, x)
            elif mode == "cut":
                patch = randomBestPatch_1(texture, patchLength, overlap, res, y, x)
                patch = minCutPatch_1(patch, patchLength, overlap, res, y, x)
            
            res[y:y+patchLength, x:x+patchLength] = patch

            if sequence:
                io.imshow(res)
                io.show()
            j=j+1
        i=i+1
    return res


def quiltSize(texture, patchLength, shape, mode="cut"):
    overlap = patchLength // 6
    h, w = shape

    numPatchesHigh_1 = math.ceil((h - patchLength) / (patchLength - overlap)) + 1 or 1
    numPatchesWide_1 = math.ceil((w - patchLength) / (patchLength - overlap)) + 1 or 1
    res = quilt(texture, patchLength, (numPatchesHigh_1, numPatchesWide_1), mode)

    return res[:h, :w]


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




