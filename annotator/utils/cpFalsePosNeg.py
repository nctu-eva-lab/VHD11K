'''
    Create a script to copy false positive & false negative contents to the given directories.
    This is useful when choosing in-context learning samples.
'''

import os
import csv
from collections import defaultdict

# image roots
sourceRoot = "/eva_data6/iammingggg/harmful_contents/image_val"
falsePosDestRoot = "/eva_data6/iammingggg/harmful_contents/image_val_inCtxLearning_exp/image_val_true_pos_ver12"
falseNegDestRoot = "/eva_data6/iammingggg/harmful_contents/image_val_inCtxLearning_exp/image_val_true_neg_ver12"

# extract annotation info
with open("annotation/experiments/image_val_inCtxLearnVer12_annotation.csv") as csvFile:
    reader = csv.DictReader(csvFile)

    imageAndDecisions = defaultdict(list)
    for row in reader:
        imageAndDecisions[os.path.basename(row['imagePath'])].append(row['decision'])

# [HOD; Images]
# imageAndDecisions = defaultdict(list)
# predictHarmfulImages = [os.path.splitext(f)[0] for f in os.listdir("/eva_data6/iammingggg/HOD_data/detection/sample_real_fake_everyPrompt/labels")]
# for file in sorted(os.listdir(sourceRoot)):
#     if os.path.isfile(os.path.join(sourceRoot, file)):
#         imageName = os.path.splitext(file)[0]
#         if imageName in predictHarmfulImages:
#             imageAndDecisions[file].append("yes")
#         else:
#             imageAndDecisions[file].append("no")

# [HOD; Video]
videoRoot = "/eva_data6/denny/Autogen/annotation_all/0_all_videos"
# imageAndDecisions = defaultdict(list)
# aggregateResults = defaultdict(list)
# predictHarmfulFrames = [os.path.splitext(f)[0] for f in os.listdir("/eva_data6/iammingggg/HOD_data/detection/video_all_frames/labels")]
# for file in sorted(os.listdir(sourceRoot)):
#     if os.path.isfile(os.path.join(sourceRoot, file)):
#         videoName = "_".join(file.split('_')[:-1])+'.mp4'
#         if os.path.splitext(file)[0] in predictHarmfulFrames:
#             aggregateResults[videoName].append(True)
#         else:
#             aggregateResults[videoName].append(False)
# for key, results in aggregateResults.items():
#     imageAndDecisions[key].append("yes" if any(results) else "no")


# ground truth: additional csv file to record such info
with open("/eva_data6/iammingggg/harmful_contents/image_val_groundTruth_denny.csv") as csvFile:
    reader = csv.DictReader(csvFile)

    for row in reader:
        imageAndDecisions[os.path.basename(row['image'])].append(row['groundTruth'])
        
        # [HOD; Images]
        # imageAndDecisions[os.path.basename(row['imagePath'])].append(row['decision'])
        
        # [HOD; Video]
        # imageAndDecisions[os.path.basename(row['videoPath'])].append(row['decision'])

FPImages = []
FNImages = []
# false positive
FPCnt = 0
FPTotal = 0
for key, val in imageAndDecisions.items():
    # if val[1] == 'no':  # ground truth
    if val[1] == 'yes':  # ground truth
        FPTotal += 1
        if val[0] == 'yes':  # annotation
            FPCnt += 1
            FPImages.append(key)

# false negative
FNCnt = 0
FNTotal = 0
for key, val in imageAndDecisions.items():
    # if val[1] == 'yes':
    if val[1] == 'no':
        FNTotal += 1
        if val[0] == 'no':
            FNCnt += 1
            FNImages.append(key)

print(f'False Pos.: {FPCnt/FPTotal: >5.2%}')
print(f'False Neg.: {FNCnt/FNTotal: >5.2%}')

# from pprint import pprint
# pprint(FPImages)
# pprint(FNImages)

FPImages.sort()
FNImages.sort()
# with open('utils/video_all_frames_cpFalsePosNeg.sh', 'w') as cpFile:
with open('utils/image_val_cpTruePosNeg.sh', 'w') as cpFile:
    print(f'mkdir -p {falsePosDestRoot}', file=cpFile)
    print(f'mkdir -p {falseNegDestRoot}', file=cpFile)
    for FPImage in FPImages:
        if '.mp4' in FPImage:
            print(f'cp {os.path.join(videoRoot, FPImage)} {falsePosDestRoot}', file=cpFile)
        else:
            print(f'cp {os.path.join(sourceRoot, FPImage)} {falsePosDestRoot}', file=cpFile)

    for FNImage in FNImages:
        if '.mp4' in FPImage:
            print(f'cp {os.path.join(videoRoot, FNImage)} {falseNegDestRoot}', file=cpFile)
        else:
            print(f'cp {os.path.join(sourceRoot, FNImage)} {falseNegDestRoot}', file=cpFile)
