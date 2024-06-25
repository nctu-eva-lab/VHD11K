''' 
    (Multithread ver) Check if there is any duplicate images under the given directories. 
    Grouping the images by size before comparing can significantly decrease pairs to be compared and then
    reduce required time.
'''
# NOTE: If handling large amount of images, it might take some time to create the remove script.

import os
import numpy as np
from PIL import Image
from collections import defaultdict
from tqdm.auto import tqdm
from typing import List
import threading
import queue

class Worker(threading.Thread):
    def __init__(self, q, imageBySize, results, keyLock, resultLock):
        super().__init__()
        self.q: queue.Queue = q
        self.results: List = results
        self.totalJobNum: int = self.q.qsize()
        self.keyLock: threading.Lock = keyLock
        self.resultLock: threading.Lock = resultLock
        self.imageBySize = imageBySize
    
    def run(self):
        while self.q.qsize() > 0:
            self.keyLock.acquire()
            try:
                currentSize = self.q.get(block=False)
            except queue.Empty:
                # explicit terminate the worker when there is no job left in the queue
                self.keyLock.release()
                self.q.task_done()
            else:
                print(f"Duplicates checking job remains: {self.q.qsize()}/{self.totalJobNum}")
                self.keyLock.release()

            path2DuplicateImages = []
            for sourceInd, (path2SrcImg, sourceImage) in enumerate(imageBySize[currentSize]):
                removeBuffer = []
                for targetInd, (path2TgtImg, targetImage) in enumerate(imageBySize[currentSize]):
                    if targetInd == sourceInd: continue
                    
                    if np.sum(np.bitwise_xor(sourceImage, targetImage)) == 0:
                        path2DuplicateImages.append(path2TgtImg)
                        removeBuffer.append((path2TgtImg, targetImage))

                # remove all duplicate images from source list at the end of each round
                # to assure the remaining source image is as expected
                for imageToBeRemoved in removeBuffer:
                    imageBySize[currentSize].remove(imageToBeRemoved)

            self.resultLock.acquire()
            self.results.extend(path2DuplicateImages)
            self.resultLock.release()


if __name__ == "__main__":
    imageRoot1 = "/eva_data6/iammingggg/harmful_contents/crawled_from_google/google_unharmful_100_max300/image"
    imageRoot2 = "/eva_data3/iammingggg/harmful_images_labels/google_harmful_images_300"
    # imageRoot = "/eva_data6/iammingggg/google_unharmful_100_max300"
    outScriptDir = "./scripts"
    scriptName = "remove_dup_google_harmful_old"
    threadNum = 16

    # read images and group them by their sizes.
    # if two images are of different size, they must not be the same.
    imageBySize = defaultdict(list)
    for imageName in tqdm(sorted(os.listdir(imageRoot1)), desc="Reading images"):
        path2Image = os.path.join(imageRoot1, imageName)
        img = np.array(Image.open(path2Image))
        imageBySize[img.shape].append((path2Image, img))
    for imageName in tqdm(sorted(os.listdir(imageRoot2)), desc="Reading images"):
        path2Image = os.path.join(imageRoot2, imageName)
        img = np.array(Image.open(path2Image))
        imageBySize[img.shape].append((path2Image, img))

    # create job queue for multithread workers
    # (treat each of the image group as one job)
    keyQueue = queue.Queue()
    for key in imageBySize.keys():
        keyQueue.put(key)
    
    # two locks to avoid data hazard
    keyLock = threading.Lock()
    resultLock = threading.Lock()
    path2DuplicateImages = []
    # each of the workers handle one image group
    workers = [Worker(keyQueue, imageBySize, path2DuplicateImages, keyLock, resultLock) for _ in range(threadNum)]

    for ind in range(threadNum):
        workers[ind].start()
    
    # for ind in range(tqdm(threadNum, desc="Waiting for threads to end")):
    for ind in range(threadNum):
        workers[ind].join()

    path2DuplicateImages.sort()

    # print(f'\nFound {len(path2DuplicateImages)} duplicate images in \"{imageRoot}\".')
    print(f'\nFound {len(path2DuplicateImages)} duplicate images in \"{imageRoot1}\" and \"{imageRoot2}\".')
    print(f'Create remove script at \"{os.path.join(outScriptDir, scriptName)}.sh\".')

    # create script to remove duplicate images
    with open(os.path.join(outScriptDir, f"{scriptName}.sh"), 'a') as file:
        for path2DuplicateImage in path2DuplicateImages:
            print(f'rm {path2DuplicateImage}', file=file)