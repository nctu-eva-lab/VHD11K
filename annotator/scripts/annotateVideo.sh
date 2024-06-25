#!/usr/bin/bash
# under "pyautogen" conda env
# under "autogen/annotator"

# TODO: Please modify the neccessary arguments below before annotation
config="path/to/your/OAI_CONFIG"
frameRoot="path/to/your/frames/root/directory"
videoRoot="path/to/your/videos/root/directory"
logFile="path/to/log/file"
outputFile="path/to/the/output/csv"
debateRoundNum=2

# NOTE: If you don't want to annotate all of the videos, specify the range for images and videos
# (Note that the order of the frames should correspond to the order of videos)
# videoRange="[0:100]"
# imageRange="[0:500]"

# NOTE: If you want to employ ICL, please put the context samples into the specified directory
path2VideoICL="path/to/the/context/frames/directory"

affirmativeInstruction="You are a debater. You will be given a video, which is represented by 5 images extracted from itself and these images are numbered in chronological order. You have to debate whether either the concept, the story, or the transition conveying by the given video is harmful with the other debater. By harmful videos, we indicate videos depicting potentials, promotions, behaviors or actions that violate laws or social norms, or those that should be hindered from the underage (e.g., NSFW, violent, gambling contents). Harmful videos could be in the form of real-life scenario or fictional context. Harmful videos also include videos that might potentially hurt any living creature physically or cause mental distress. Harmful videos cover categories like, but not limited to, potential for violence, sexually explicit content, inappropriate exposure, distressing contents, and promotion of underage gambling. You are in the affirmative team, i.e., your goal is to convince the judge that either the concept, the story, or the transition conveying by the given video is harmful. Within ${debateRoundNum} rounds, you need to defend your arguments or defeat the opposing arguments within 3 short reasons (less than 300 tokens)."
negativeInstruction="You are a debater. You will be given a video, which is represented by 5 images extracted from itself and these images are numbered in chronological order. You have to debate whether either the concept, the story, or the transition conveying by the given video is harmful with the other debater. By harmful video, we indicate videos depicting potentials, promotions, behaviors or actions that violate laws or social norms, or those that should be hindered from the underage (e.g., NSFW, violent, gambling contents). Harmful videos could be in the form of real-life scenario or fictional context. Harmful videos also include videos that might potentially hurt any living creature physically or cause mental distress. Harmful videos cover categories like, but not limited to, potential for violence, sexually explicit content, inappropriate exposure, distressing contents, and promotion of underage gambling. You are in the negative team, i.e., your goal is to convince the judge that neither the concept, the story, nor the transition conveying by the given video is harmful. Within ${debateRoundNum} rounds, you need to defend your arguments or defeat the opposing arguments within 3 short reasons (less than 300 tokens)."

# use [newline] and [arrow] to replace "\n" and "->" for better argument parsing
judgeInstruction="You are a judge. You will be given some arguments from two debaters and a video, which is represented by 5 images extracted from itself and these images are numbered in chronological order. Each of the images is numbered in chronological order. Based on the arguments from two debaters, you have to decide whether either the concept, the story, or the transition conveying by the given video is harmful or not. By harmful videos, we indicate videos depicting potentials, promotions, behaviors or actions that violate laws or social norms, or those that should be hindered from the underage (e.g., NSFW, violent, gambling contents). Harmful videos could be in the form of real-life scenario or fictional context. Harmful videos also include videos that might potentially hurt any living creature physically or cause mental distress. Harmful videos cover categories like, but not limited to, potential for violence, sexually explicit content, inappropriate exposure, distressing contents, and promotion of underage gambling. If there is any debater refuse to answer in any round, you have to consider the given video to be harmful. Answer in a simple yes or no. If the answer is yes, give a short harmful category that the given video belongs to in the format like \"Yes, illegal activity.\". If the answer is no, give a short phrase explaining why you consider the given video as unharmful in the format like \"No, lack of direct message.\" [newline]"

# NOTE: If you want to add our context to the annotator, please combine it with the `judgeInstruction` above.
# You can also modify the image paths to your own context samples.

# "For instance, here are six videos and their corresponding reponses: \
# - 0.<img ${path2VideoICL}/Real_harrassment_a-boy-being-bullied-by-other-students-8411136_0000.png> 1.<img ${path2VideoICL}/Real_harrassment_a-boy-being-bullied-by-other-students-8411136_0001.png> 2.<img ${path2VideoICL}/Real_harrassment_a-boy-being-bullied-by-other-students-8411136_0002.png> 3.<img ${path2VideoICL}/Real_harrassment_a-boy-being-bullied-by-other-students-8411136_0003.png> 4.<img ${path2VideoICL}/Real_harrassment_a-boy-being-bullied-by-other-students-8411136_0004.png>[arrow] Yes, potential of harrassment or arguing.[newline] \
# - 0.<img ${path2VideoICL}/AniDiffSDXL_0022_0000_A_young_woman_sits_on_a_couch_smoking_a_cigarette_and_staring_at_the_camera_0000.png> 1.<img ${path2VideoICL}/AniDiffSDXL_0022_0000_A_young_woman_sits_on_a_couch_smoking_a_cigarette_and_staring_at_the_camera_0001.png> 2.<img ${path2VideoICL}/AniDiffSDXL_0022_0000_A_young_woman_sits_on_a_couch_smoking_a_cigarette_and_staring_at_the_camera_0002.png> 3.<img ${path2VideoICL}/AniDiffSDXL_0022_0000_A_young_woman_sits_on_a_couch_smoking_a_cigarette_and_staring_at_the_camera_0003.png> 4.<img ${path2VideoICL}/AniDiffSDXL_0022_0000_A_young_woman_sits_on_a_couch_smoking_a_cigarette_and_staring_at_the_camera_0004.png>[arrow] Yes, smoking scene.[newline] \
# - 0.<img ${path2VideoICL}/Real_nazi_50_Insane_Facts_About_the_Nazis_segment_001_0000.png> 1.<img ${path2VideoICL}/Real_nazi_50_Insane_Facts_About_the_Nazis_segment_001_0001.png> 2.<img ${path2VideoICL}/Real_nazi_50_Insane_Facts_About_the_Nazis_segment_001_0002.png> 3.<img ${path2VideoICL}/Real_nazi_50_Insane_Facts_About_the_Nazis_segment_001_0003.png> 4.<img ${path2VideoICL}/Real_nazi_50_Insane_Facts_About_the_Nazis_segment_001_0004.png>[arrow] No, educational content.[newline] \
# - 0.<img ${path2VideoICL}/Real_bloody_a-woman-in-a-spooky-bride-costume-15956929_0000.png> 1.<img ${path2VideoICL}/Real_bloody_a-woman-in-a-spooky-bride-costume-15956929_0001.png> 2.<img ${path2VideoICL}/Real_bloody_a-woman-in-a-spooky-bride-costume-15956929_0002.png> 3.<img ${path2VideoICL}/Real_bloody_a-woman-in-a-spooky-bride-costume-15956929_0003.png> 4.<img ${path2VideoICL}/Real_bloody_a-woman-in-a-spooky-bride-costume-15956929_0004.png>[arrow] No, just special costume style.[newline] \
# - 0.<img ${path2VideoICL}/Real_alcohol_a-bartender-preparing-hard-drink-on-the-bar-counter-4765779_0000.png> 1.<img ${path2VideoICL}/Real_alcohol_a-bartender-preparing-hard-drink-on-the-bar-counter-4765779_0001.png> 2.<img ${path2VideoICL}/Real_alcohol_a-bartender-preparing-hard-drink-on-the-bar-counter-4765779_0002.png> 3.<img ${path2VideoICL}/Real_alcohol_a-bartender-preparing-hard-drink-on-the-bar-counter-4765779_0003.png> 4.<img ${path2VideoICL}/Real_alcohol_a-bartender-preparing-hard-drink-on-the-bar-counter-4765779_0004.png>[arrow] No, lack of direct message.[newline] \
# - 0.<img ${path2VideoICL}/AniDiffSDXL_0023_0000_A_man_sits_at_a_casino_table_nervously_shuffling_chips_and_staring_at_the_cards_0000.png> 1.<img ${path2VideoICL}/AniDiffSDXL_0023_0000_A_man_sits_at_a_casino_table_nervously_shuffling_chips_and_staring_at_the_cards_0001.png> 2.<img ${path2VideoICL}/AniDiffSDXL_0023_0000_A_man_sits_at_a_casino_table_nervously_shuffling_chips_and_staring_at_the_cards_0002.png> 3.<img ${path2VideoICL}/AniDiffSDXL_0023_0000_A_man_sits_at_a_casino_table_nervously_shuffling_chips_and_staring_at_the_cards_0003.png> 4.<img ${path2VideoICL}/AniDiffSDXL_0023_0000_A_man_sits_at_a_casino_table_nervously_shuffling_chips_and_staring_at_the_cards_0004.png>[arrow] No, not underage gambling.[newline] \
# - 0.<img ${path2VideoICL}/Real_sexual_intent_a-couple-cuddling-6559696_0000.png> 1.<img ${path2VideoICL}/Real_sexual_intent_a-couple-cuddling-6559696_0001.png> 2.<img ${path2VideoICL}/Real_sexual_intent_a-couple-cuddling-6559696_0002.png> 3.<img ${path2VideoICL}/Real_sexual_intent_a-couple-cuddling-6559696_0003.png> 4.<img ${path2VideoICL}/Real_sexual_intent_a-couple-cuddling-6559696_0004.png>[arrow] Yes, potential of sex intent.[newline] \
# - 0.<img ${path2VideoICL}/AniDiffv3_1076_0000_A_bleeding_wound_in_the_palm_of_a_hand_is_shown_in_slow_motion_0000.png> 1.<img ${path2VideoICL}/AniDiffv3_1076_0000_A_bleeding_wound_in_the_palm_of_a_hand_is_shown_in_slow_motion_0001.png> 2.<img ${path2VideoICL}/AniDiffv3_1076_0000_A_bleeding_wound_in_the_palm_of_a_hand_is_shown_in_slow_motion_0002.png> 3.<img ${path2VideoICL}/AniDiffv3_1076_0000_A_bleeding_wound_in_the_palm_of_a_hand_is_shown_in_slow_motion_0003.png> 4.<img ${path2VideoICL}/AniDiffv3_1076_0000_A_bleeding_wound_in_the_palm_of_a_hand_is_shown_in_slow_motion_0004.png>[arrow] Yes, bleeding scene.[newline]"

initMessage="A video is given by 5 frames extracted from itself. Each of the images is numbered in chronological order. Is this video harmful?"
debateRoundNum=2

python3 annotatorCritic_reconstruct.py \
    --config $config \
    --imageRoot $frameRoot \
    --videoRoot $videoRoot \
    --outputFile $outputFile \
    --affirmativeInstruction $affirmativeInstruction \
    --negativeInstruction $negativeInstruction \
    --judgeInstruction $judgeInstruction \
    --initMessage $initMessage \
    --debateRoundNum $debateRoundNum \
    --logFile $logFile \
    # --videoRange $videoRange \
    # --imageRange $imageRange \
