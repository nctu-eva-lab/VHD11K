'''
    Extract frames from videos under the given directory.
    The extraction method can be either random selection or selection at the fixed interval.
'''

from torchvision.io import read_video
from random import sample, shuffle
import os
from tqdm.auto import tqdm
from torchvision.transforms import ToPILImage
from PIL import Image

isRandomSelection = False
extractionInterval = 1  # set isRandomSelection=False; extract frames with fixed interval
extractedFrameNum = 8   # set isRandomSelection=True; extract frames with fixed total frames per video

root = "/eva_data3/iammingggg/DFC_temp/test_videos"  # video root
outputDir = "/eva_data3/iammingggg/DFC_temp/extracted_frames_all"  # extracted images output dir
outputPrefix = "DFDC"
os.makedirs(outputDir, exist_ok=True)

extractedFrameCnt = 0
videoNames = sorted([videoName for videoName in os.listdir(root) if os.path.splitext(videoName)[1] == '.mp4'])
for videoInd, videoName in tqdm(enumerate(videoNames), total=len(videoNames), desc='Extracting frames'):
    frames, _, _ = read_video(os.path.join(root, videoName), output_format='TCHW')

    if isRandomSelection:
        extractedInds = sample(range(frames.shape[0]), k=extractedFrameNum)
        extractedFrames = [ToPILImage()(frame) for frame in frames[extractedInds]]
    else:  # fix extraction interval
        extractedFrames = [ToPILImage()(frame) for frame in frames[::extractionInterval]]

    extractedFrameCnt += len(extractedFrames)

    for extractedInd, extractedFrame in enumerate(extractedFrames):
        extractedFrame.save(
            os.path.join(outputDir, f"{outputPrefix}_{os.path.splitext(videoName)[0]}_{extractedInd:04d}.png"))

print(f'\nRead {len(videoNames)} images from "{root}".')
print(f'{extractedFrameCnt} extracted images are stored in "{outputDir}".')
