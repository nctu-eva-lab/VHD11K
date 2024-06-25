''' Create and print confusion matrix out of the given ground truth and the corresponding annoations. '''

import os
import csv
from collections import defaultdict


def confusionMatrix(imageAndDecisions: defaultdict, lackOfMessageToggle: bool = False):
    # true positive
    TPCnt = 0
    TPTotal = 0
    for key, val in imageAndDecisions.items():
        if val[1] == 'yes':     # ground truth
            TPTotal += 1
            if val[0] == 'yes' or (lackOfMessageToggle and val[0] == "no" and "lack of direct message" in imageAndHarmfulType[key]): # annotation
                TPCnt += 1

    # true negative
    TNCnt = 0
    TNTotal = 0
    for key, val in imageAndDecisions.items():
        if val[1] == 'no':
            TNTotal += 1
            if val[0] == 'no' and not (lackOfMessageToggle and "lack of direct message" in imageAndHarmfulType[key]):
                TNCnt += 1

    # false positive
    FPCnt = 0
    FPTotal = 0
    for key, val in imageAndDecisions.items():
        if val[1] == 'no':
            FPTotal += 1
            if val[0] == 'yes' or (lackOfMessageToggle and val[0] == "no" and "lack of direct message" in imageAndHarmfulType[key]):
                FPCnt += 1

    # false negative
    FNCnt = 0
    FNTotal = 0
    for key, val in imageAndDecisions.items():
        if val[1] == 'yes':
            FNTotal += 1
            if val[0] == 'no' and not (lackOfMessageToggle and "lack of direct message" in imageAndHarmfulType[key]):
                FNCnt += 1

    return TPCnt, TPTotal, TNCnt, TNTotal, FPCnt, FPTotal, FNCnt, FNTotal

def printConfusionMatrix(TPCnt, TPTotal, TNCnt, TNTotal, FPCnt, FPTotal, FNCnt, FNTotal, outFileName, cmType: str):
    precision = TPCnt / (TPCnt+FPCnt)
    recall = TPCnt / (TPCnt+FNCnt)
    f1 = 2 * (precision*recall) / (precision+recall)
    print(f'[{cmType}]')
    print(f'              | GT Pos. | GT Neg. |')
    print(f'Annotate Pos. | {TPCnt:>7d} | {FPCnt:7d} |')
    print(f'Annotate Neg. | {FNCnt:>7d} | {TNCnt:7d} |')
    print(f'-'*20)
    print(f'True Pos. : {TPCnt/TPTotal*100: 6.2f}% ({TPCnt}/{TPTotal})')
    print(f'True Neg. : {TNCnt/TNTotal*100: 6.2f}% ({TNCnt}/{TNTotal})')
    print(f'False Pos.: {FPCnt/FPTotal*100: 6.2f}% ({FPCnt}/{FPTotal})')
    print(f'False Neg.: {FNCnt/FNTotal*100: 6.2f}% ({FNCnt}/{FNTotal})')
    print(f'Accuracy  : {(TPCnt+TNCnt)/(TPCnt+FPCnt+FNCnt+TNCnt)*100:6.2f}%')
    print(f'Precision : {precision*100:6.2f}%')
    print(f'Recall    : {recall*100:6.2f}%')
    print(f'F1-score  : {f1:>7.4f}')
    print(f'='*20)

    with open(outFileName, 'a') as outFile:
        print(f'[{cmType}]', file=outFile)
        print(f'              | GT Pos. | GT Neg. |', file=outFile)
        print(f'Annotate Pos. | {TPCnt:>7d} | {FPCnt:7d} |', file=outFile)
        print(f'Annotate Neg. | {FNCnt:>7d} | {TNCnt:7d} |', file=outFile)
        print(f'-'*20, file=outFile)
        print(f'True Pos. : {TPCnt/TPTotal*100: 6.2f}% ({TPCnt}/{TPTotal})', file=outFile)
        print(f'True Neg. : {TNCnt/TNTotal*100: 6.2f}% ({TNCnt}/{TNTotal})', file=outFile)
        print(f'False Pos.: {FPCnt/FPTotal*100: 6.2f}% ({FPCnt}/{FPTotal})', file=outFile)
        print(f'False Neg.: {FNCnt/FNTotal*100: 6.2f}% ({FNCnt}/{FNTotal})', file=outFile)
        print(f'Accuracy  : {(TPCnt+TNCnt)/(TPCnt+FPCnt+FNCnt+TNCnt)*100:6.2f}%', file=outFile)
        print(f'Precision : {precision*100:6.2f}%', file=outFile)
        print(f'Recall    : {recall*100:6.2f}%', file=outFile)
        print(f'F1-score  : {f1:>7.4f}', file=outFile)
        print(f'='*20, file=outFile)


if __name__ == '__main__':
    # whether to treat "lack of message" as harmful
    lackOfMessageToggle = False

    # extract annotation info
    with open("annotation/SMID_annotation.csv") as csvFile:
        reader = csv.DictReader(csvFile)

        imageAndDecisionsAll = defaultdict(list)
        imageAndDecisionsReal = defaultdict(list)
        imageAndDecisionsSyn = defaultdict(list)
        imageAndHarmfulType = defaultdict(list)
        for row in reader:
            imageAndDecisionsAll[os.path.basename(row['imagePath'])].append(row['decision'])
            if 'sdxl' in row['imagePath']:
                imageAndDecisionsSyn[os.path.basename(row['imagePath'])].append(row['decision'])
            else:
                imageAndDecisionsReal[os.path.basename(row['imagePath'])].append(row['decision'])

            imageAndHarmfulType[os.path.basename(row['imagePath'])].append(row['harmfulType'])
                            
    # ground truth ver. 1: images under "xxx_manualSelection" are considered as harmful (ground truth: harmful)
    # files = os.listdir("/eva_data3/iammingggg/harmful_images_labels/google_harmful_images_300_each3_manualSelection")
    # for f in files:
    #     imageAndDecisionsAll[f].append("yes")
    # for key, val in imageAndDecisionsAll.items():
    #     if len(val) < 2:
    #         imageAndDecisionsAll[key].append("no")

    # ground truth ver. 2: additional csv file to record such info
    with open("/eva_data6/SMID/img_400px/all_good_bad_groundTruth.csv") as csvFile:
        reader = csv.DictReader(csvFile)

        for row in reader:
            # imageAndDecisionsAll[os.path.basename(row['image'])].append(row['groundTruth'])
            imageAndDecisionsAll[os.path.basename(row['imagePath'])].append(row['groundTruth'])
            # if 'sdxl' in row['image']:
            #     imageAndDecisionsSyn[os.path.basename(row['image'])].append(row['groundTruth'])
            # else:
            #     imageAndDecisionsReal[os.path.basename(row['image'])].append(row['groundTruth'])

    os.makedirs('confusionMatrix', exist_ok=True)
    outFileName = 'confusionMatrix/SMID_confusionMatrix.txt'
    TPCntAll, TPTotalAll, TNCntALl, TNTotalAll, FPCntAll, FPTotalAll, FNCntAll, FNTotalAll = confusionMatrix(imageAndDecisionsAll, lackOfMessageToggle)
    printConfusionMatrix(TPCntAll, TPTotalAll, TNCntALl, TNTotalAll, FPCntAll, FPTotalAll, FNCntAll, FNTotalAll,
                        cmType="All", outFileName = outFileName)

    # TPCntReal, TPTotalReal, TNCntReal, TNTotalReal, FPCntReal, FPTotalReal, FNCntReal, FNTotalReal = confusionMatrix(imageAndDecisionsReal, lackOfMessageToggle)
    # printConfusionMatrix(TPCntReal, TPTotalReal, TNCntReal, TNTotalReal, FPCntReal, FPTotalReal, FNCntReal, FNTotalReal,
    #                     cmType="Real", outFileName = outFileName)

    # TPCntSyn, TPTotalSyn, TNCntSyn, TNTotalSyn, FPCntSyn, FPTotalSyn, FNCntSyn, FNTotalSyn = confusionMatrix(imageAndDecisionsSyn, lackOfMessageToggle)
    # printConfusionMatrix(TPCntSyn, TPTotalSyn, TNCntSyn, TNTotalSyn, FPCntSyn, FPTotalSyn, FNCntSyn, FNTotalSyn,
    #                     cmType="Syn", outFileName = outFileName)
