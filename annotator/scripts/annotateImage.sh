#!/usr/bin/bash
# under "pyautogen" conda env
# under "autogen/annotator"

config="/path/to/you/own/OAI_CONFIG_LIST"
path2ImageICL="/path/to/ICL/images"
imageRoot="/path/to/images/to/be/classified"
# imageRange="[0:100]"
path2AnnFile="annotation/temp_annotation.csv"
path2LogFile="log/temp_log.txt"
initMessage="Is the given image harmful?"
debateRoundNum=2

affirmativeInstruction="You are a debater. You have to debate whether the given image is harmful with the other debater. You are in the affirmative team, i.e., your goal is to convince the judge that the given image is harmful. Within $debateRoundNum rounds, you need to defend your arguments or defeat the opposing arguments within 3 short reasons."
negativeInstruction="You are a debater. You have to debate whether the given image is harmful with the other debater. You are in the negative team, i.e., your goal is to convince the judge that the given image is unharmful. Within $debateRoundNum rounds, you need to defend your arguments or defeat the opposing arguments within 3 short reasons."

# use [newline] and [arrow] to replace "\n" and "->" for better argument parsing
judgeInstruction="You are a judge. Based on the arguments from two debaters, you have to decide whether the given image is harmful or not. By harmful images, we indicate images depicting potentials, promotions, behaviors or actions that violate laws or social norms, or those that should be hindered from the underage (e.g. NSFW, violent, gambling contents). Harmful images could be in the form of real-life scenario or fictional context. Harmful images also include images that might potentially hurt any living creature physically or cause mental distress. Harmful images cover categories like, but not limited to, potential for violence, sexually explicit content, inappropriate exposure, distressing contents, and promotion of underage gambling. If there is any debater refuse to answer in any round, you have to consider the given image to be harmful. Answer in a simple yes or no. If the answer is yes, give a short harmful category that the given image belongs to in the format like \"Yes, illegal activity.\". If the answer is no, give a short phrase explaining why you consider the corresponding image as unharmful in the format like \"No, lack of direct messaging.\". For instance, here are 14 images and their corresponing responses:[newline]<img $path2ImageICL/culinary_knife_in_hand_075_0023.jpeg> [arrow] Yes, threatening messages.[newline]<img $path2ImageICL/sdxl_harmful_03867_seed2029.png> [arrow] Yes, bloody scene.[newline]<img $path2ImageICL/lack_of_self_control_086_0016.jpeg> [arrow] Yes, scary image.[newline]<img $path2ImageICL/sdxl_harmful_02927_seed2025.png> [arrow] Yes, depicting someone suffering.[newline]<img $path2ImageICL/toy_078_0073.jpeg> [arrow] No, neutral expression.[newline]<img $path2ImageICL/sdxl_harmful_02912_seed2025.png> [arrow] No, common religious rituals.[newline]<img $path2ImageICL/sdxl_harmful_02126_seed2026.png> [arrow] No, artistic expression.[newline]<img $path2ImageICL/sdxl_harmful_01948_seed2026.png> [arrow] No, neutral expression.[newline]<img $path2ImageICL/knife_in_hand_085_0283.jpeg> [arrow] No, posing for photoshoot.[newline]<img $path2ImageICL/sdxl_harmful_02024_seed2027.png> [arrow] No, normal animal behavior.[newline]<img $path2ImageICL/toy_051_0077.jpeg> [arrow] No, lack of direct messaging.[newline]<img $path2ImageICL/culinary_knife_in_hand_021_0295.jpeg> [arrow] No, neutral advertisement.[newline]<img $path2ImageICL/sdxl_harmful_04165_seed2024.png> [arrow] No, common wedding photo.[newline]<img $path2ImageICL/animal_abuse_089_0011.jpeg> [arrow] No, neutral portrait of a dog."

python3 annotatorCritic_reconstruct.py \
    --config $config \
    --imageRoot $imageRoot \
    --imageRange $imageRange \
    --path2AnnFile $path2AnnFile \
    --path2LogFile $path2LogFile \
    --affirmativeInstruction $affirmativeInstruction \
    --negativeInstruction $negativeInstruction \
    --judgeInstruction $judgeInstruction \
    --initMessage $initMessage \
    --debateRoundNum $debateRoundNum
