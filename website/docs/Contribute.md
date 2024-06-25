# Contributing

This project welcomes and encourages all forms of contributions, including but not limited to:

- Pushing patches.
- Code review of pull requests.
- Documentation, examples and test cases.
- Readability improvement, e.g., improvement on docstr and comments.
- Community participation in [issues](https://github.com/microsoft/autogen/issues), [discussions](https://github.com/microsoft/autogen/discussions), [discord](https://discord.gg/pAbnFJrkgZ), and [twitter](https://twitter.com/pyautogen).
- Tutorials, blog posts, talks that promote the project.
- Sharing application scenarios and/or related research.

Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit <https://cla.opensource.microsoft.com>.

If you are new to GitHub [here](https://help.github.com/categories/collaborating-with-issues-and-pull-requests/) is a detailed help source on getting involved with development on GitHub.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Roadmaps

To see what we are working on and what we plan to work on, please check our
[Roadmap Issues](https://github.com/microsoft/autogen/issues?q=is%3Aopen+is%3Aissue+label%3Aroadmap).

## How to make a good bug report

When you submit an issue to [GitHub](https://github.com/microsoft/autogen/issues), please do your best to
follow these guidelines! This will make it a lot easier to provide you with good
feedback:

- The ideal bug report contains a short reproducible code snippet. This way
  anyone can try to reproduce the bug easily (see [this](https://stackoverflow.com/help/mcve) for more details). If your snippet is
  longer than around 50 lines, please link to a [gist](https://gist.github.com) or a GitHub repo.

- If an exception is raised, please **provide the full traceback**.

- Please include your **operating system type and version number**, as well as
  your **Python, autogen, scikit-learn versions**. The version of autogen
  can be found by running the following code snippet:

```python
import autogen
print(autogen.__version__)
```

- Please ensure all **code snippets and error messages are formatted in
  appropriate code blocks**.  See [Creating and highlighting code blocks](https://help.github.com/articles/creating-and-highlighting-code-blocks)
  for more details.

## Becoming a Reviewer

There is currently no formal reviewer solicitation process. Current reviewers identify reviewers from active contributors. If you are willing to become a reviewer, you are welcome to let us know on discord.

## Guidance for Maintainers

### General

- Be a member of the community and treat everyone as a member. Be inclusive.
- Help each other and encourage mutual help.
- Actively post and respond.
- Keep open communication.

### Pull Requests

- For new PR, decide whether to close without review. If not, find the right reviewers. The default reviewer is microsoft/autogen. Ask users who can benefit from the PR to review it.

- For old PR, check the blocker: reviewer or PR creator. Try to unblock. Get additional help when needed.
- When requesting changes, make sure you can check back in time because it blocks merging.
- Make sure all the checks are passed.
- For changes that require running OpenAI tests, make sure the OpenAI tests pass too. Running these tests requires approval.
- In general, suggest small PRs instead of a giant PR.
- For documentation change, request snapshot of the compiled website, or compile by yourself to verify the format.
- For new contributors who have not signed the contributing agreement, remind them to sign before reviewing.
- For multiple PRs which may have conflict, coordinate them to figure out the right order.
- Pay special attention to:
  - Breaking changes. Don’t make breaking changes unless necessary. Don’t merge to main until enough headsup is provided and a new release is ready.
  - Test coverage decrease.
  - Changes that may cause performance degradation. Do regression test when test suites are available.
  - Discourage **change to the core library** when there is an alternative.

### Issues and Discussions

- For new issues, write a reply, apply a label if relevant. Ask on discord when necessary. For roadmap issues, add to the roadmap project and encourage community discussion. Mention relevant experts when necessary.

- For old issues, provide an update or close. Ask on discord when necessary. Encourage PR creation when relevant.
- Use “good first issue” for easy fix suitable for first-time contributors.
- Use “task list” for issues that require multiple PRs.
- For discussions, create an issue when relevant. Discuss on discord when appropriate.

## Docker for Development

For developers contributing to the AutoGen project, we offer a specialized Docker environment. This setup is designed to streamline the development process, ensuring that all contributors work within a consistent and well-equipped environment.

### Autogen Developer Image (autogen_dev_img)

- **Purpose**: The `autogen_dev_img` is tailored for contributors to the AutoGen project. It includes a suite of tools and configurations that aid in the development and testing of new features or fixes.
- **Usage**: This image is recommended for developers who intend to contribute code or documentation to AutoGen.
- **Forking the Project**: It's advisable to fork the AutoGen GitHub project to your own repository. This allows you to make changes in a separate environment without affecting the main project.
- **Updating Dockerfile**: Modify your copy of `Dockerfile` in the `dev` folder as needed for your development work.
- **Submitting Pull Requests**: Once your changes are ready, submit a pull request from your branch to the upstream AutoGen GitHub project for review and integration. For more details on contributing, see the [AutoGen Contributing](https://microsoft.github.io/autogen/docs/Contribute) page.

### Building the Developer Docker Image

- To build the developer Docker image (`autogen_dev_img`), use the following commands:

  ```bash
  docker build -f .devcontainer/dev/Dockerfile -t autogen_dev_img https://github.com/microsoft/autogen.git#main
  ```

- For building the developer image built from a specific Dockerfile in a branch other than main/master

  ```bash
  # clone the branch you want to work out of
  git clone --branch {branch-name} https://github.com/microsoft/autogen.git

  # cd to your new directory
  cd autogen

  # build your Docker image
  docker build -f .devcontainer/dev/Dockerfile -t autogen_dev-srv_img .
  ```

### Using the Developer Docker Image

Once you have built the `autogen_dev_img`, you can run it using the standard Docker commands. This will place you inside the containerized development environment where you can run tests, develop code, and ensure everything is functioning as expected before submitting your contributions.

```bash
docker run -it -p 8081:3000 -v `pwd`/autogen-newcode:newstuff/ autogen_dev_img bash
```

- Note that the `pwd` is shorthand for present working directory. Thus, any path after the pwd is relative to that. If you want a more verbose method you could remove the "`pwd`/autogen-newcode" and replace it with the full path to your directory

```bash
docker run -it -p 8081:3000 -v /home/AutoGenDeveloper/autogen-newcode:newstuff/ autogen_dev_img bash
```

### Develop in Remote Container

If you use vscode, you can open the autogen folder in a [Container](https://code.visualstudio.com/docs/remote/containers).
We have provided the configuration in [devcontainer](https://github.com/microsoft/autogen/blob/main/.devcontainer). They can be used in GitHub codespace too. Developing AutoGen in dev containers is recommended.

### Pre-commit

Run `pre-commit install` to install pre-commit into your git hooks. Before you commit, run
`pre-commit run` to check if you meet the pre-commit requirements. If you use Windows (without WSL) and can't commit after installing pre-commit, you can run `pre-commit uninstall` to uninstall the hook. In WSL or Linux this is supposed to work.

### Write tests

Tests are automatically run via GitHub actions. There are two workflows:

1. [build.yml](https://github.com/microsoft/autogen/blob/main/.github/workflows/build.yml)
1. [openai.yml](https://github.com/microsoft/autogen/blob/main/.github/workflows/openai.yml)

The first workflow is required to pass for all PRs (and it doesn't do any OpenAI calls). The second workflow is required for changes that affect the OpenAI tests (and does actually call LLM). The second workflow requires approval to run. When writing tests that require OpenAI calls, please use [`pytest.mark.skipif`](https://github.com/microsoft/autogen/blob/b1adac515931bf236ac59224269eeec683a162ba/test/oai/test_client.py#L19) to make them run in only when `openai` package is installed. If additional dependency for this test is required, install the dependency in the corresponding python version in [openai.yml](https://github.com/microsoft/autogen/blob/main/.github/workflows/openai.yml).

Make sure all tests pass, this is required for [build.yml](https://github.com/microsoft/autogen/blob/main/.github/workflows/build.yml) checks to pass

#### Running tests locally

To run tests, install the [test] option:

```bash
pip install -e."[test]"
```

Then you can run the tests from the `test` folder using the following command:

```bash
pytest test
```

Tests for the `autogen.agentchat.contrib` module may be skipped automatically if the
required dependencies are not installed. Please consult the documentation for
each contrib module to see what dependencies are required.

See [here](https://github.com/microsoft/autogen/blob/main/notebook/contributing.md#testing) for how to run notebook tests.

#### Skip flags for tests

- `--skip-openai` for skipping tests that require access to OpenAI services.
- `--skip-docker` for skipping tests that explicitly use docker
- `--skip-redis` for skipping tests that require a Redis server

For example, the following command will skip tests that require access to
OpenAI and docker services:

```bash
pytest test --skip-openai --skip-docker
```

### Coverage

Any code you commit should not decrease coverage. To run all unit tests, install the [test] option:

```bash
pip install -e."[test]"
coverage run -m pytest test
```

Then you can see the coverage report by
`coverage report -m` or `coverage html`.

### Documentation
#### Build documentation locally
1\. To build and test documentation locally, first install [Node.js](https://nodejs.org/en/download/). For example,
```bash
nvm install --lts
```

Then, install `yarn` and other required packages:
```bash
npm install --global yarn
pip install pydoc-markdown pyyaml termcolor
```

2\. You also need to install quarto. Please click on the `Pre-release` tab from [this website](https://quarto.org/docs/download/) to download the latest version of `quarto` and install it. Ensure that the `quarto` version is `1.5.23` or higher.

3\. Finally, run the following commands to build:

```console
cd website
yarn install --frozen-lockfile --ignore-engines
pydoc-markdown
python process_notebooks.py render
yarn start
```

The last command starts a local development server and opens up a browser window.
Most changes are reflected live without having to restart the server.

#### Build with Docker
To build and test documentation within a docker container. Use the Dockerfile in the `dev` folder as described above to build your image:

```bash
docker build -f .devcontainer/dev/Dockerfile -t autogen_dev_img https://github.com/microsoft/autogen.git#main
```

Then start the container like so, this will log you in and ensure that Docker port 3000 is mapped to port 8081 on your local machine

```bash
docker run -it -p 8081:3000 -v `pwd`/autogen-newcode:newstuff/ autogen_dev_img bash
```

Once at the CLI in Docker run the following commands:

```bash
cd website
yarn install --frozen-lockfile --ignore-engines
pydoc-markdown
python process_notebooks.py render
yarn start --host 0.0.0.0 --port 3000
```

Once done you should be able to access the documentation at `http://127.0.0.1:8081/autogen`

Note:
some tips in this guide are based off the contributor guide from [flaml](https://microsoft.github.io/FLAML/docs/Contribute).
