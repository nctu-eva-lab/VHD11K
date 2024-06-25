# Optional Dependencies

## LLM Caching

To use LLM caching with Redis, you need to install the Python package with
the option `redis`:

```bash
pip install "pyautogen[redis]"
```

See [LLM Caching](Use-Cases/agent_chat.md#llm-caching) for details.

## IPython Code Executor

To use the IPython code executor, you need to install the `jupyter-client`
and `ipykernel` packages:

```bash
pip install "pyautogen[ipython]"
```

To use the IPython code executor:

```python
from autogen import UserProxyAgent

proxy = UserProxyAgent(name="proxy", code_execution_config={"executor": "ipython-embedded"})
```

## blendsearch

`pyautogen<0.2` offers a cost-effective hyperparameter optimization technique [EcoOptiGen](https://arxiv.org/abs/2303.04673) for tuning Large Language Models. Please install with the [blendsearch] option to use it.

```bash
pip install "pyautogen[blendsearch]<0.2"
```

Example notebooks:

[Optimize for Code Generation](https://github.com/microsoft/autogen/blob/main/notebook/oai_completion.ipynb)

[Optimize for Math](https://github.com/microsoft/autogen/blob/main/notebook/oai_chatgpt_gpt4.ipynb)

## retrievechat

`pyautogen` supports retrieval-augmented generation tasks such as question answering and code generation with RAG agents. Please install with the [retrievechat] option to use it.

```bash
pip install "pyautogen[retrievechat]"
```

RetrieveChat can handle various types of documents. By default, it can process
plain text and PDF files, including formats such as 'txt', 'json', 'csv', 'tsv',
'md', 'html', 'htm', 'rtf', 'rst', 'jsonl', 'log', 'xml', 'yaml', 'yml' and 'pdf'.
If you install [unstructured](https://unstructured-io.github.io/unstructured/installation/full_installation.html)
(`pip install "unstructured[all-docs]"`), additional document types such as 'docx',
'doc', 'odt', 'pptx', 'ppt', 'xlsx', 'eml', 'msg', 'epub' will also be supported.

You can find a list of all supported document types by using `autogen.retrieve_utils.TEXT_FORMATS`.

Example notebooks:

[Automated Code Generation and Question Answering with Retrieval Augmented Agents](https://github.com/microsoft/autogen/blob/main/notebook/agentchat_RetrieveChat.ipynb)

[Group Chat with Retrieval Augmented Generation (with 5 group member agents and 1 manager agent)](https://github.com/microsoft/autogen/blob/main/notebook/agentchat_groupchat_RAG.ipynb)

[Automated Code Generation and Question Answering with Qdrant based Retrieval Augmented Agents](https://github.com/microsoft/autogen/blob/main/notebook/agentchat_qdrant_RetrieveChat.ipynb)

## Teachability

To use Teachability, please install AutoGen with the [teachable] option.

```bash
pip install "pyautogen[teachable]"
```

Example notebook:  [Chatting with a teachable agent](https://github.com/microsoft/autogen/blob/main/notebook/agentchat_teachability.ipynb)

## Large Multimodal Model (LMM) Agents

We offered Multimodal Conversable Agent and LLaVA Agent. Please install with the [lmm] option to use it.

```bash
pip install "pyautogen[lmm]"
```

Example notebooks:

[LLaVA Agent](https://github.com/microsoft/autogen/blob/main/notebook/agentchat_lmm_llava.ipynb)

## mathchat

`pyautogen<0.2` offers an experimental agent for math problem solving. Please install with the [mathchat] option to use it.

```bash
pip install "pyautogen[mathchat]<0.2"
```

Example notebooks:

[Using MathChat to Solve Math Problems](https://github.com/microsoft/autogen/blob/main/notebook/agentchat_MathChat.ipynb)

## Graph

To use a graph in `GroupChat`, particularly for graph visualization, please install AutoGen with the [graph] option.


```bash
pip install "pyautogen[graph]"
```

Example notebook:  [Graph Modeling Language with using select_speaker](https://github.com/microsoft/autogen/blob/main/notebook/agentchat_graph_modelling_language_using_select_speaker.ipynb)
