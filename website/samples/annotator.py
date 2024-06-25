# NOTE: example from "Agent Chat with Multimodal Models: GPT-4V"
# NOTE: annotate 1 images cost ~= USD$0.166
# FIXME: debater_1 might refuse to answer by saying "I'm sorry, but I can't assist with that request."

from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import autogen
from autogen import Agent, AssistantAgent, ConversableAgent, UserProxyAgent, GroupChat
from autogen.agentchat.contrib.multimodal_conversable_agent import MultimodalConversableAgent

import os
import csv
from pprint import pprint
import warnings
from tqdm.auto import tqdm
from collections import defaultdict

def initAutogenConfig(path2OaiConfigList: str, configType="gpt4v") -> Dict:
    assert configType in ["gpt4", "gpt4v", "gpt3"], \
        f'Please choose a configType from ["gpt4", "gpt4v", "gpt3"]'

    # config settings
    if configType == "gpt4":
        config_list_gpt4 = autogen.config_list_from_json(
            path2OaiConfigList,
            filter_dict={
                "model": ["gpt-4", "gpt-4-0314", "gpt4", "gpt-4-32k", "gpt-4-32k-0314", "gpt-4-32k-v0314"],
            },
        )
        # gpt4_llm_config = {"config_list": config_list_gpt4, "cache_seed": 42}

        return config_list_gpt4
    elif configType == "gpt4v":
        config_list_4v = autogen.config_list_from_json(
            path2OaiConfigList,
            filter_dict={
                "model": ["gpt-4-vision-preview"],
            },
        )
        # gpt4v_llm_config = {"config_list": config_list_4v, "cache_seed": 42}
     
        return config_list_4v
    elif configType == "gpt3":
        config_list_gpt3 = autogen.config_list_from_json(
            path2OaiConfigList,
            filter_dict={
                "model": ["gpt-3.5-turbo"],
            },
        )
        # gpt3_llm_config = {"config_list": config_list_gpt3, "cache_seed": 42}

        return config_list_gpt3

def initDebateAgents(llmConfigList: Dict, 
                     debater1Instruction: str, debater2Instruction: str, judgeInstruction: str,
                     debater1MaxTokenNum: int = 500, debater2MaxTokenNum: int = 500, judgeMaxTokenNum: int = 100, 
                     cache_seed: int = 42) -> Tuple[UserProxyAgent, MultimodalConversableAgent, MultimodalConversableAgent, MultimodalConversableAgent]:
    
    # initiate agents
    user_proxy = autogen.UserProxyAgent(
        name="User_proxy",
        llm_config={"config_list": llmConfigList, "cache_seed": cache_seed},
        system_message="A human admin.",
        human_input_mode="NEVER",  # Try between ALWAYS or NEVER
        # human_input_mode="TERMINATE",
        max_consecutive_auto_reply=0,
        code_execution_config={
            "use_docker": False
        },  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
    )

    # FIXED: 
    #   [v] refuse to answer -> treat as harmful
    #   [v] judge: yes/no with a harmful category
    #   [v] 要求辯手針對對方的回應攻擊
    #   [v] `speaker_selection_method`: force them to debate 3 rounds before the final decision
    #   [v] explicitly ask for brief reasons
    debater_1 = MultimodalConversableAgent(
        name="debater_1",
        # system_message="You are a debater. You have to give several reasons until the judge is convinced that the given image is harmful, or a decision is made.",
        # system_message="You are a debater. You have to debate whether the given image is harmful with the other debater. You are in the affirmative team, i.e., your goal is to convince the judge that the given image is harmful. Within the 3 rounds, you need to briefly give several reasons to defend your arguments or defeat the opposing arguments.",
        system_message=debater1Instruction,
        max_consecutive_auto_reply=10,
        llm_config={"config_list": llmConfigList, "temperature": 0.5, "max_tokens": debater1MaxTokenNum},
    )
    debater_2 = MultimodalConversableAgent(
        name="debater_2",
        # system_message="You are a debater. You have to give several reasons until the judge is convinced the given image is unharmful, or a decision is made.",
        # system_message="You are a debater. You have to debate whether the given image is unharmful with the other debater. You are in the negative team, i.e., your goal is to convince the judge that the given image is unharmful. Within the 3 rounds, you need to briefly give several reasons to defend your arguments or defeat the opposing arguments.",
        system_message=debater2Instruction,
        max_consecutive_auto_reply=10,
        llm_config={"config_list": llmConfigList, "temperature": 0.5, "max_tokens": debater2MaxTokenNum},
    )
    judge = MultimodalConversableAgent(
        name="judge",
        # system_message="""You are a judge. Based on the argments from two debaters, you have to decide whether the given image is harmful or not. Answer in a simple yes or no. If the answer is yes or you refuse to answer, give a short harmful category that the given image belongs to or a short reason of why you refuse to answer in the format like "Yes, illegal activity." or "Refuse to answer, NSFW", respectively.""",
        # system_message="""You are a judge. Based on the argments from two debaters, you have to decide whether the given image is harmful or not. Answer in a simple yes or no. If the answer is yes, give a short harmful category that the given image belongs to in the format like "Yes, illegal activity.".""",
        # system_message="""You are a judge. Based on the argments from two debaters, you have to decide whether the given image is harmful or not. Answer in a simple yes or no with brief explanation of how you make this decision.""",
        system_message=judgeInstruction,
        max_consecutive_auto_reply=10,
        llm_config={"config_list": llmConfigList, "temperature": 0.5, "max_tokens": judgeMaxTokenNum},
    )

    # static var decorator
    # ref: https://stackoverflow.com/questions/279561/what-is-the-python-equivalent-of-static-variables-inside-a-function
    def static_vars(**kwargs):
        def decorate(func):
            for k in kwargs:
                setattr(func, k, kwargs[k])
            return func
        return decorate

    @static_vars(agentCnt=0)
    def custom_speaker_selection_func(last_speaker: Agent, groupchat: GroupChat) -> Union[Agent, str, None]:
        agentSequence = [
            'debater_1', 'debater_2',
            'debater_1', 'debater_2',
            'debater_1', 'debater_2',
            'judge', None
        ]
        nextAgentName = agentSequence[custom_speaker_selection_func.agentCnt]
        custom_speaker_selection_func.agentCnt += 1
        
        if nextAgentName is not None:
            return groupchat.agent_by_name(nextAgentName)
        else:
            # end the iteration (by returning None) and reset counter
            custom_speaker_selection_func.agentCnt = 0

            return None

    groupchat = autogen.GroupChat(agents=[user_proxy, debater_2, debater_1, judge],
                                  messages=[],
                                  max_round=12,
                                  speaker_selection_method=custom_speaker_selection_func)
    manager = autogen.GroupChatManager(groupchat=groupchat, 
                                       llm_config={"config_list": llmConfigList, "cache_seed": cache_seed})

    return user_proxy, debater_2, debater_1, judge, manager

def responseParsing(agent) -> Tuple[Dict, str]:
    response = defaultdict(list)
    for conversation in agent.chat_messages[manager]:
        if conversation['role'] == 'assistant':
            response['judge'].append(conversation['content'][0]['text'])
        else:   # conversation['role'] == 'user'
            # conversation['name'] can be either:
            # 1. User_proxy
            # 2. debater_1
            # 3. debater_2
            response[conversation['name']].append(conversation['content'][0]['text'])

    # FIXED: parsing
    if len(response["judge"]) == 0:
        response["judge"] = [f"Judge reply not found."]
        warnings.warn(f"Judge reply not found. Image: {path2Image}")
    elif "yes," in response["judge"][0].lower():
        harmfulType = ' '.join(response["judge"][0].split(', ')[1:]).strip('.')
        response["judge"][0] = "yes"
    elif "no." in response["judge"][0].lower() or "no," in response["judge"][0].lower():
        harmfulType = "not harmful"
        response["judge"][0] = "no"
    else:
        harmfulType = ' '.join(response["judge"][0].split(', ')[1:]).strip('.')
        response["judge"][0] = "yes"
        warnings.warn(f'Judge refuse to answer. Treat as harmful. Image: {path2Image}')

    return response, harmfulType

# Ask the question with an image
# user_proxy.initiate_chat(
#     debater_1,
#     # message="""What's the breed of this dog?
#     message="""Is this image harmful?
# <img https://th.bing.com/th/id/R.422068ce8af4e15b0634fe2540adea7a?rik=y4OcXBE%2fqutDOw&pid=ImgRaw&r=0>.""",
# )

# user_proxy.send(
#     message="""What is this breed?
# <img https://th.bing.com/th/id/OIP.29Mi2kJmcHHyQVGe_0NG7QHaEo?pid=ImgDet&rs=1>

# Among the breeds, which one barks less?""",
#     recipient=image_agent,
# )

if __name__ == '__main__':
    gpt4v_llm_config = initAutogenConfig("OAI_CONFIG_LIST_antifakePrompt006", configType='gpt4v')

    user_proxy, debater_2, debater_1, judge, manager = initDebateAgents(
        gpt4v_llm_config,
        debater1Instruction="You are a debater. You have to debate whether the given image is harmful with the other debater. You are in the affirmative team, i.e., your goal is to convince the judge that the given image is harmful. Within 3 rounds, you need to briefly give several reasons to defend your arguments or defeat the opposing arguments.",
        debater2Instruction="You are a debater. You have to debate whether the given image is harmful with the other debater. You are in the negative team, i.e., your goal is to convince the judge that the given image is unharmful. Within 3 rounds, you need to briefly give several reasons to defend your arguments or defeat the opposing arguments.",
        judgeInstruction="""You are a judge. Based on the arguments from two debaters, you have to decide whether the given image is harmful or not. Answer in a simple yes or no. If the answer is yes, give a short harmful category that the given image belongs to in the format like "Yes, illegal activity.".""")

    # path2Images = [
        # "/eva_data3/iammingggg/google_harmful_images_300/alcohol_00001.jpeg",
        # "/eva_data3/iammingggg/google_harmful_images_300/alcohol_00002.jpeg",
        # "/eva_data3/iammingggg/google_harmful_images_300/animal_abuse_00000.jpeg",
        # "/eva_data3/iammingggg/google_harmful_images_300/animal_abuse_00001.png",
        # "/eva_data3/iammingggg/google_harmful_images_300/animal_genitalia_and_human_00001.jpeg",
        # "/eva_data3/iammingggg/google_harmful_images_300/animal_genitalia_and_human_00002.jpeg",

        # "/eva_data3/iammingggg/google_hiveAI_300_each30/general_suggestive_00020.jpeg",
        # "/eva_data3/iammingggg/google_hiveAI_300_each30/illicit_injectables_00005.jpeg",
        # "/eva_data3/iammingggg/google_hiveAI_300_each30/kkk_00005.jpeg",
        # "/eva_data3/iammingggg/harmful_images_labels/google_hiveAI_300_each30/male_nudity_00020.jpeg",
        # "/eva_data3/iammingggg/google_hiveAI_300_each30/female_nudity_00010.png",
        # "/eva_data3/iammingggg/google_hiveAI_300_each30/fight_00002.jpeg",
        # "/eva_data3/iammingggg/google_hiveAI_300_each30/medical_injectables_00003.jpeg",
        # "/eva_data3/iammingggg/google_hiveAI_300_each30/self_harm_00005.jpeg",
        # "/eva_data3/iammingggg/google_hiveAI_300_each30/knife_in_hand_00022.jpeg",
    # ]

    imageRoot = "/eva_data3/iammingggg/harmful_images_labels/google_harmful_images_300_each3_undone"
    # path2Images = [os.path.join(imageRoot, f) for f in sorted(os.listdir(imageRoot))][73:]
    path2Images = [os.path.join(imageRoot, f) for f in sorted(os.listdir(imageRoot))]

    with open('/eva_data3/iammingggg/harmful_images_labels/google_harmful_images_300_each3_undone_annotation.csv', 
              'a', newline='') as csvFile:
        fieldnames = ['imagePath', 'decision', 'harmfulType', 
                      'debater_1_argument_0', 'debater_1_argument_1', 'debater_1_argument_2',
                      'debater_2_argument_0', 'debater_2_argument_1', 'debater_2_argument_2',]
        writer = csv.DictWriter(csvFile, fieldnames=fieldnames)
        writer.writeheader()

        for path2Image in tqdm(path2Images, desc="annotating images"):
            manager.reset()

            user_proxy.initiate_chat(
                manager,
                message=f"""Is the given image harmful? <img {path2Image}>""",
                silent=True,        # does not seem working...
                max_turns=5
            )

            response, harmfulType = responseParsing(judge)
        
            writer.writerow({
                'imagePath': path2Image,
                'decision': response["judge"][0],
                'harmfulType': harmfulType,
                'debater_1_argument_0': response['debater_1'][0],
                'debater_1_argument_1': response['debater_1'][1],
                'debater_1_argument_2': response['debater_1'][2],
                'debater_2_argument_0': response['debater_2'][0],
                'debater_2_argument_1': response['debater_2'][1],
                'debater_2_argument_2': response['debater_2'][2],
            })

            manager.reset()