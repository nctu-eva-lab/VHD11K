'''
    Extract require information from annotation file. (e.g. decision, harmfulTypes, ...)
'''

import os
import csv
from collections import defaultdict

# path2GT = "/eva_data6/iammingggg/harmful_contents/image_val_groundTruth_denny.csv"
path2GT = None
# path2Anns = ["annotation/sampled_real_5000_image_annotation.csv", "annotation/sampled_fake_5000_image_annotation.csv"]
path2Anns = [
    "/home/denny/autogen/annotator/annotation/video_all_0530_p1p2p3_downsampled.csv",
    "/home/iammingggg/detection/autogen/annotator/annotation/sample_real_fake_10000_evenHarmfulUnharmful_annotation.csv",
]
outputPath = "extracted/image_10000_video_1000_arguments.txt"
imageMode = False   # True: image mode; False: video mode


if path2GT is not None:
    contentAndGroundTruth = {}
    # ground truth: additional csv file to record such info
    with open(path2GT) as csvFile:
        reader = csv.DictReader(csvFile)

        for row in reader:
            contentAndGroundTruth[os.path.basename(row['image'])] = row['groundTruth']

contentAndDecision = []
contentAndDecisionFP = []
contentAndDecisionFN = []
for path2Ann in path2Anns:
    with open(path2Ann) as csvFile:
        reader = csv.DictReader(csvFile)

        for row in reader:
            contentPath = os.path.basename(row['imagePath']) if 'video' not in path2Ann else os.path.basename(
                row['videoPath'])
            if row['decision'] == 'yes':
                if imageMode:
                    contentAndDecision.append({
                        "imagePath": contentPath,
                        "decision": row['decision'],
                        "harmfulType": row['harmfulType']
                    })
                else:
                    contentAndDecision.append({
                        "videoPath": contentPath,
                        "decision": row['decision'],
                        "harmfulType": row['harmfulType']
                    })
            # contentAndDecision.append(row['harmfulType'])
            # for key in ["affirmativeDebater_argument_0", "affirmativeDebater_argument_1", "negativeDebater_argument_0", "negativeDebater_argument_1"]:
            #     if row[key].lower().strip() != "none":
            #         contentAndDecision.append(row[key])

            if path2GT is not None:
                if contentAndGroundTruth[contentPath] == 'no' and row['decision'] == 'yes':  # annotation
                    buffer = {"decision": row['decision'], "harmfulType": row['harmfulType']}
                    if imageMode:
                        buffer['imagePath'] = contentPath
                    else:
                        buffer['videoPath'] = contentPath

                    contentAndDecisionFP.append(buffer)
                if contentAndGroundTruth[contentPath] == 'yes' and row['decision'] == 'no':  # annotation
                    buffer = {"decision": row['decision'], "harmfulType": row['harmfulType']}
                    if imageMode:
                        buffer['imagePath'] = contentPath
                    else:
                        buffer['videoPath'] = contentPath

                    contentAndDecisionFN.append(buffer)

# contentAndDecision.sort(key=lambda x: x['imagePath'])

with open(outputPath, 'w') as file:
    if imageMode:
        writer = csv.DictWriter(file, fieldnames=['imagePath', 'decision', 'harmfulType'])
    else:
        writer = csv.DictWriter(file, fieldnames=['videoPath', 'decision', 'harmfulType'])
    # writer = csv.DictWriter(file, fieldnames=['harmfulType'])

    writer.writeheader()
    writer.writerows(contentAndDecision)
    # writer.writerows([{'harmfulType': harmfulType} for harmfulType in sorted(list(set(contentAndDecision)))])
    # file.writelines(contentAndDecision)

if path2GT is not None:
    if imageMode:
        contentAndDecisionFP.sort(key=lambda x: x['imagePath'])
        contentAndDecisionFN.sort(key=lambda x: x['imagePath'])
        fieldnames = ['imagePath', 'decision', 'harmfulType']
    else:
        contentAndDecisionFP.sort(key=lambda x: x['videoPath'])
        contentAndDecisionFN.sort(key=lambda x: x['videoPath'])
        fieldnames = ['videoPath', 'decision', 'harmfulType']

    with open(f"{outputPath.split('_annotationOnly')[0]}_FP_annotationOnly.csv", 'w') as csvFile:
        writer = csv.DictWriter(csvFile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(contentAndDecisionFP)

    with open(f"{outputPath.split('_annotationOnly')[0]}_FN_annotationOnly.csv", 'w') as csvFile:
        writer = csv.DictWriter(csvFile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(contentAndDecisionFN)
