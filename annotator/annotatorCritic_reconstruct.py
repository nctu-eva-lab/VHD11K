''' same annotator with the additional critic to analyze the reason why the judge gives different annotations to images. '''
# NOTE: example from "Agent Chat with Multimodal Models: GPT-4V"
# NOTE: (2-round debation) annotate 1 images ~= USD$0.14
# FIXED: affirmativeDebater might refuse to answer by saying "I'm sorry, but I can't assist with that request."
#       -> reckon as harmful when this happens

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
from PIL import Image
import argparse
from openai import BadRequestError


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
                # "model": ["gpt-4-turbo"],
                # "model": ["gpt-4o"],
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


def initDebateAgents(
    llmConfigList: Dict,
    debateRoundNum: int,
    affirmativeInstruction: str,
    negativeInstruction: str,
    judgeInstruction: str,
    criticInstruction: str,
    debater1MaxTokenNum: int = 500,
    debater2MaxTokenNum: int = 500,
    judgeMaxTokenNum: int = 100,
    criticMaxTokenNum: Union[int, None] = 500,
    cache_seed: int = 42
) -> Tuple[UserProxyAgent, MultimodalConversableAgent, MultimodalConversableAgent,
           MultimodalConversableAgent]:
    ''' Disable critic by passing `None` as `criticInstruction`. '''

    # initiate agents
    userProxy = autogen.UserProxyAgent(
        name="userProxy",
        llm_config={
            "config_list": llmConfigList,
            "cache_seed": cache_seed
        },
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
    # affirmative team
    affirmativeDebater = MultimodalConversableAgent(
        name="affirmativeDebater",
        system_message=affirmativeInstruction,
        max_consecutive_auto_reply=10,
        llm_config={
            "config_list": llmConfigList,
            "temperature": 0.5,
            "max_tokens": debater1MaxTokenNum
        },
    )
    # negative team
    negativeDebater = MultimodalConversableAgent(
        name="negativeDebater",
        system_message=negativeInstruction,
        max_consecutive_auto_reply=10,
        llm_config={
            "config_list": llmConfigList,
            "temperature": 0.5,
            "max_tokens": debater2MaxTokenNum
        },
    )
    judge = MultimodalConversableAgent(
        name="judge",
        system_message=judgeInstruction,
        max_consecutive_auto_reply=10,
        llm_config={
            "config_list": llmConfigList,
            "temperature": 0.5,
            "max_tokens": judgeMaxTokenNum
        },
    )
    if criticInstruction is not None:
        # disable `critic` by passing `None` as criticInstruction
        critic = MultimodalConversableAgent(
            name="critic",
            system_message=criticInstruction,
            max_consecutive_auto_reply=10,
            llm_config={
                "config_list": llmConfigList,
                "temperature": 0.5,
                "max_tokens": criticMaxTokenNum
            },
        )
    else:
        print("//// Critic disabled. ////")

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
        # `agentSequence` must end with `None` for elegant termination
        agentSequence = ['affirmativeDebater', 'negativeDebater'] * debateRoundNum
        agentSequence += ['judge', 'critic', None] if criticInstruction is not None else ['judge', None]
        nextAgentName = agentSequence[custom_speaker_selection_func.agentCnt]
        custom_speaker_selection_func.agentCnt += 1

        if nextAgentName is not None:
            return groupchat.agent_by_name(nextAgentName)
        else:
            # end the iteration (by returning None) and reset counter
            custom_speaker_selection_func.agentCnt = 0

            return None

    if criticInstruction is not None:
        groupchat = autogen.GroupChat(agents=[userProxy, negativeDebater, affirmativeDebater, critic, judge],
                                      messages=[],
                                      max_round=12,
                                      speaker_selection_method=custom_speaker_selection_func)
        manager = autogen.GroupChatManager(groupchat=groupchat,
                                           llm_config={
                                               "config_list": llmConfigList,
                                               "cache_seed": cache_seed
                                           })

        return userProxy, negativeDebater, affirmativeDebater, judge, critic, manager
    else:
        groupchat = autogen.GroupChat(agents=[userProxy, negativeDebater, affirmativeDebater, judge],
                                      messages=[],
                                      max_round=12,
                                      speaker_selection_method=custom_speaker_selection_func)
        manager = autogen.GroupChatManager(groupchat=groupchat,
                                           llm_config={
                                               "config_list": llmConfigList,
                                               "cache_seed": cache_seed
                                           })

        # return `None` for empty critic
        return userProxy, negativeDebater, affirmativeDebater, judge, None, manager


def responseParsing(agent, debug=False) -> Tuple[Dict, str]:
    response = defaultdict(list)
    for conversation in agent.chat_messages[manager]:
        if conversation['role'] == 'assistant':
            response['judge'].append(conversation['content'][0]['text'])
        else:  # conversation['role'] == 'user'
            # conversation['name'] can be either:
            # 1. userProxy
            # 2. affirmativeDebater
            # 3. negativeDebater
            response[conversation['name']].append(conversation['content'][0]['text'])

    # FIXED: parsing; both yes/no have their corresponing category
    if len(response["judge"]) == 0:
        response["judge"] = [f"Judge reply not found."]
        harmfulType = "none"
        warnings.warn(f"Judge reply not found.")
    elif "yes," in response["judge"][0].lower():
        harmfulType = ' '.join(response["judge"][0].split(', ')[1:]).strip('.')
        response["judge"][0] = "yes"
        response["judge"].append(harmfulType)
    elif "no." in response["judge"][0].lower() or "no," in response["judge"][0].lower():
        if len(response['judge'][0]) > 1:
            harmfulType = ' '.join(response["judge"][0].split(', ')[1:]).strip('.')
        else:
            harmfulType = "not harmful"
        response["judge"][0] = "no"
        response["judge"].append(harmfulType)
    else:
        harmfulType = ' '.join(response["judge"][0].split(', ')[1:]).strip('.')
        response["judge"][0] = "yes"
        response["judge"].append(harmfulType)
        warnings.warn(f'Judge refuse to answer. Treat as harmful.')

    if debug:
        from pprint import pprint

        pprint(response)

    return response, harmfulType


def parseArgs():
    parser = argparse.ArgumentParser()

    # basic configs: authentication, path to images/videos.
    parser.add_argument("--config", default="OAI_CONFIG_LIST", help="path to OAI config file")
    parser.add_argument("--imageRoot", required=True, help="path to images to be annotated.")
    parser.add_argument(
        "--videoRoot",
        default=None,
        help=
        "path to original video to be annotated. (need to specify extracted image path by imageRoot.) ignore to disable"
    )
    # TODO: image/video slice control through argparse
    parser.add_argument("--imageRange",
                        nargs='?',
                        default=None,
                        help="specify the images to read from imageRoot")
    parser.add_argument("--videoRange",
                        nargs='?',
                        default=None,
                        help="specify the video to read from videoRoot")
    parser.add_argument("--path2AnnFile", default="annotation/out.csv", help="path to output annotated csv file")
    parser.add_argument("--path2LogFile", default="log/out.csv", help="path to output log csv file")

    # agent configs
    parser.add_argument("--affirmativeInstruction",
                        nargs="+",
                        required=True,
                        type=str,
                        help="system message (instruction) for affirmative debater.")
    parser.add_argument("--negativeInstruction",
                        nargs="+",
                        required=True,
                        type=str,
                        help="system message (instruction) for negative debater.")
    parser.add_argument("--judgeInstruction",
                        nargs="+",
                        required=True,
                        help="system message (instruction) for judge.")
    parser.add_argument("--criticInstruction",
                        nargs="*",
                        default=None,
                        type=str,
                        help="system message (instruction) for critic. (ignore to disable)")
    parser.add_argument("--initMessage",
                        nargs="+",
                        required=True,
                        type=str,
                        help="initial message of every round.")
    parser.add_argument(
        "--debateRoundNum",
        required=True,
        type=int,
        help="number of rounds that two debater have to conduct before the judge make the decision.")
    parser.add_argument(
        "--debater1MaxTokenNum",
        default=500,
        type=int,
        help=
        "max token of the affirmative debater. (set < 500 might result in incomplete/truncated responses.)")
    parser.add_argument(
        "--debater2MaxTokenNum",
        default=500,
        type=int,
        help="max token of the negative debater. (set < 500 might result in incomplete/truncated responses.)")
    parser.add_argument(
        "--judgeMaxTokenNum",
        default=500,
        type=int,
        help="max token of the judge. (num < 500 might result in incomplete/truncated responses.)")
    parser.add_argument("--cacheSeed",
                        default=42,
                        type=int,
                        help="cache seed of autogen. this is to guarantee response reproducibility.")

    args = parser.parse_args()

    # postprocess
    args.affirmativeInstruction = ' '.join(args.affirmativeInstruction)
    args.negativeInstruction = ' '.join(args.negativeInstruction)
    args.judgeInstruction = ' '.join(args.judgeInstruction).replace('[newline]', '\n').replace('[arrow]', '->')
    args.initMessage = ' '.join(args.initMessage)
    if args.criticInstruction is not None:
        args.criticInstruction = ' '.join(args.criticInstruction)

    # args.imageRange = [lower bound, upper bound]
    # args.videoRange = [lower bound, upper bound]
    if args.imageRange is not None:
        assert len(args.imageRange.split(":")) == 2, f"Please assure the imageRange ({args.imageRange}) is in format like \"[0:32]\""

        args.imageRange = list(map(int, args.imageRange.lstrip("[").rstrip("]").split(":")))

        assert args.imageRange[0] <= args.imageRange[1], f"Please make sure that the lower bound ({args.imageRange[0]}) <= upper bound ({args.imageRange[1]})"
    if args.videoRange is not None:
        assert len(args.videoRange.split(":")) == 2, f"Please assure the videoRange ({args.videoRange}) is in format like \"[0:32]\""

        args.videoRange = list(map(int, args.videoRange.lstrip("[").rstrip("]").split(":")))

        assert args.videoRange[0] <= args.videoRange[1], f"Please make sure that the lower bound ({args.videoRange[0]}) <= upper bound ({args.videoRange[1]})"

    print(args)

    return args


if __name__ == '__main__':
    args: argparse.Namespace = parseArgs()

    gpt4v_llm_config = initAutogenConfig(args.config, configType='gpt4v')

    path2Images = [os.path.join(args.imageRoot, f) for f in sorted(os.listdir(args.imageRoot))]
    if args.imageRange is not None:
        assert args.imageRange[1] <= len(path2Images), (args.imageRange[1], len(path2Images))
        assert 0 <= args.imageRange[0], f"image lower bound ({args.imageRange[0]}) exceed the boundary"
        path2Images = path2Images[args.imageRange[0]:args.imageRange[1]]

    if args.videoRoot is not None:
        videoNames = [f for f in sorted(os.listdir(args.videoRoot)) if os.path.splitext(f)[1] == ".mp4"]
        if args.videoRange is not None:
            assert args.videoRange[1] <= len(
                videoNames), f"video upper bound ({args.videoRange[1]}) exceed the boundary"
            assert 0 <= args.videoRange[0], f"video lower bound ({args.videoRange[0]}) exceed the boundary"
            videoNames = videoNames[args.videoRange[0]:args.videoRange[1]]

    # write header only when the output file does not exist
    annFileExist = os.path.exists(args.path2AnnFile)
    logFileExist = os.path.exists(args.path2LogFile)

    with open(args.path2AnnFile, 'a', newline='') as annFile:
        annFieldnames = [
            'imagePath',
            'decision',
            'harmfulType',
            'affirmativeDebater_argument_0',
            'affirmativeDebater_argument_1',
            'negativeDebater_argument_0',
            'negativeDebater_argument_1',
        ]

        # critic is not disabled
        if args.criticInstruction is not None:
            annFieldnames.append('critic')

        annWriter = csv.DictWriter(annFile, fieldnames=annFieldnames)

        if not annFileExist:
            annWriter.writeheader()

        with open(args.path2LogFile, 'a', newline='') as logFile:
            logFieldnames = [
                'imagePath',
                'error',
            ]
            logWriter = csv.DictWriter(logFile, fieldnames=logFieldnames)

            if not logFileExist:
                logWriter.writeheader()

            if args.videoRoot is None:  # annotate images
                userProxy, negativeDebater, affirmativeDebater, judge, critic, manager = initDebateAgents(
                    gpt4v_llm_config,
                    debateRoundNum=args.debateRoundNum,
                    affirmativeInstruction=args.affirmativeInstruction,
                    negativeInstruction=args.negativeInstruction,
                    judgeInstruction=args.judgeInstruction,
                    criticInstruction=args.criticInstruction,  # disable critic by passing in `None`
                    cache_seed=args.cacheSeed,
                )

                # ignore those images are already annotated
                if annFileExist:
                    with open(args.path2AnnFile) as annFile:
                        annReader = csv.DictReader(annFile)
                        donePath2Images = [os.path.basename(row['imagePath']) for row in annReader]
                        path2Images = [p for p in path2Images if os.path.basename(p) not in donePath2Images]

                errorImagesAndMessages = []
                for path2Image in tqdm(path2Images, desc="annotating images"):
                    manager.reset()

                    try:
                        userProxy.initiate_chat(manager,
                                                message=f"""{args.initMessage} <img {path2Image}>""",
                                                silent=False,
                                                max_turns=5)
                    except BadRequestError as e:
                        warnings.warn(str(e))

                        if "Your input image may contain content that is not allowed by our safety system.'" in str(
                                e):
                            responseBuffer = {
                                'imagePath': path2Image,
                                'decision': "yes",
                                'harmfulType': "API refuse to answer",
                                'affirmativeDebater_argument_0': 'None',
                                'affirmativeDebater_argument_1': 'None',
                                'negativeDebater_argument_0': 'None',
                                'negativeDebater_argument_1': 'None',
                            }
                            if critic is not None:
                                responseBuffer['critic'] = 'None'
                        else:
                            responseBuffer = {
                                'imagePath': path2Image,
                                'decision': "error",
                                'harmfulType': "error",
                                'affirmativeDebater_argument_0': 'None',
                                'affirmativeDebater_argument_1': 'None',
                                'negativeDebater_argument_0': 'None',
                                'negativeDebater_argument_1': 'None',
                            }
                            if critic is not None:
                                responseBuffer['critic'] = 'None'

                        logWriter.writerow({
                            'imagePath': path2Image,
                            'error': str(e),
                        })
                    else:
                        response, harmfulType = responseParsing(judge)

                        if len(response['affirmativeDebater']) < args.debateRoundNum:
                            warnings.warn(
                                f"affirmativeDebater: less than {args.debateRoundNum} responses ({len(response['affirmativeDebater'])})"
                            )
                            logWriter.writerow({
                                'imagePath':
                                path2Image,
                                'error':
                                f"affirmativeDebater: less than {args.debateRoundNum} responses ({len(response['affirmativeDebater'])})",
                            })
                            response['affirmativeDebater'].extend(['None'] *
                                                                  (args.debateRoundNum - len(response['affirmativeDebater'])))
                        if len(response['negativeDebater']) < args.debateRoundNum:
                            warnings.warn(
                                f"negativeDebater: less than {args.debateRoundNum} responses ({len(response['negativeDebater'])})"
                            )
                            logWriter.writerow({
                                'imagePath':
                                path2Image,
                                'error':
                                f"negativeDebater: less than {args.debateRoundNum} responses ({len(response['negativeDebater'])})",
                            })
                            response['negativeDebater'].extend(['None'] *
                                                               (args.debateRoundNum - len(response['negativeDebater'])))

                        responseBuffer = {
                            'imagePath': path2Image,
                            'decision': response["judge"][0],
                            'harmfulType': harmfulType,
                            'affirmativeDebater_argument_0': response['affirmativeDebater'][0],
                            'affirmativeDebater_argument_1': response['affirmativeDebater'][1],
                            'negativeDebater_argument_0': response['negativeDebater'][0],
                            'negativeDebater_argument_1': response['negativeDebater'][1],
                        }

                        if critic is not None:
                            responseBuffer['critic'] = response['critic'][0]

                    annWriter.writerow(responseBuffer)
                    manager.reset()
            else:  # annotate videos
                userProxy, negativeDebater, affirmativeDebater, judge, critic, manager = initDebateAgents(
                    gpt4v_llm_config,
                    debateRoundNum=args.debateRoundNum,
                    affirmativeInstruction=args.affirmativeInstruction,
                    negativeInstruction=args.negativeInstruction,
                    judgeInstruction=args.judgeInstruction,
                    criticInstruction=args.criticInstruction,  # disable critic by passing None as criticInstruction
                    cache_seed=args.cacheSeed,
                )
                # NOTE: When annotating videos, `path2Images` contains extracted images with names
                # "<video>_<counter:04d>.png". Thus, if both `videoNames` and `path2Images` are sorted, we can
                # access the extracted images of corresponding videos in alphebatic order batch by batch without
                # explicitly searching them in every round.
                dummyCnt = 0
                for videoName in tqdm(videoNames, desc="annotating videos"):
                    # prepare prompt
                    videoRawName = os.path.splitext(videoName)[0]  # video name without file extension
                    initMessage = args.initMessage

                    imageNumberCnt = 0
                    while dummyCnt < len(path2Images) and videoRawName in path2Images[dummyCnt]:
                        initMessage += f" {imageNumberCnt}.<img {path2Images[dummyCnt]}>"
                        dummyCnt += 1
                        imageNumberCnt += 1

                    manager.reset()

                    try:
                        userProxy.initiate_chat(manager, message=initMessage, silent=False, max_turns=5)
                    except BadRequestError as e:
                        warnings.warn(str(e))
                        responseBuffer = {
                            'videoPath': videoName,
                            'decision': "yes",
                            'harmfulType': "API refuses request",
                            'affirmativeDebater_argument_0': 'None',
                            'affirmativeDebater_argument_1': 'None',
                            'negativeDebater_argument_0': 'None',
                            'negativeDebater_argument_1': 'None',
                        }
                        if critic is not None:
                            responseBuffer['critic'] = 'None'

                        logWriter.writerow({
                            'videoPath': videoName,
                            'error': str(e),
                        })
                    else:
                        response, harmfulType = responseParsing(judge)
                        if len(response['affirmativeDebater']) < args.debateRoundNum:
                            warnings.warn(
                                f"affirmativeDebater: less than {args.debateRoundNum} responses ({len(response['affirmativeDebater'])})"
                            )
                            logWriter.writerow({
                                'videoPath':
                                videoName,
                                'error':
                                f"affirmativeDebater: less than {args.debateRoundNum} responses ({len(response['affirmativeDebater'])})",
                            })
                            response['affirmativeDebater'].extend(['None'] *
                                                                  (args.debateRoundNum - len(response['affirmativeDebater'])))
                        if len(response['negativeDebater']) < args.debateRoundNum:
                            warnings.warn(
                                f"negativeDebater: less than {args.debateRoundNum} responses ({len(response['negativeDebater'])})"
                            )
                            logWriter.writerow({
                                'videoPath':
                                videoName,
                                'error':
                                f"negativeDebater: less than {args.debateRoundNum} responses ({len(response['negativeDebater'])})",
                            })
                            response['negativeDebater'].extend(['None'] *
                                                               (args.debateRoundNum - len(response['negativeDebater'])))
                        if harmfulType == "none":
                            logWriter.writerow({
                                'videoPath': videoName,
                                'error': f"judge response not found",
                            })

                        responseBuffer = {
                            'videoPath': videoName,
                            'decision': response["judge"][0],
                            'harmfulType': harmfulType,
                            'affirmativeDebater_argument_0': response['affirmativeDebater'][0],
                            'affirmativeDebater_argument_1': response['affirmativeDebater'][1],
                            'negativeDebater_argument_0': response['negativeDebater'][0],
                            'negativeDebater_argument_1': response['negativeDebater'][1],
                        }
                        if critic is not None:
                            responseBuffer['critic'] = response['critic'][0]

                    annWriter.writerow(responseBuffer)
                    manager.reset()
