# FROM python:3.8.13-slim-buster
FROM python:3.8.13-slim-buster
RUN apt-get update && \
    apt-get upgrade -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git \
    graphviz \
 && apt-get clean && \
  rm -rf /var/lib/apt

RUN pip install -U pip tox pipx
RUN pip install -U setuptools setuptools_scm wheel
RUN git config --global user.email "gunthergl@gmx.net"
RUN git config --global user.name "gunthergl"


# Create a working directory
RUN mkdir /nbnode
WORKDIR /nbnode

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash nbnode_user

### Add relevant code sources
#       Add directories to the container
ADD src /nbnode/src
ADD .git /nbnode/.git
# ADD tests /nbnode/tests
# ADD container /nbnode
# ADD .coveragerc /nbnode
# ADD dist /nbnode/dist
# ADD docs /nbnode/docs

#       Add files to the container
ADD AUTHORS.rst /nbnode
ADD CHANGELOG.rst /nbnode
ADD CONTRIBUTING.rst /nbnode
ADD .gitignore /nbnode
ADD LICENSE.txt /nbnode
ADD pyproject.toml /nbnode
ADD setup.cfg /nbnode
ADD setup.py /nbnode
ADD tox.ini /nbnode
# ADD .isort.cfg /nbnode
# ADD .gitlab-ci.yml /nbnode
# ADD .pre-commit-config.yaml /nbnode
# ADD README.rst /nbnode
# ADD .readthedocs.yml /nbnode
# ADD .tox /nbnode
# ADD removeme_.gitlab-ci.yml /nbnode
# ADD .vscode /nbnode

RUN chown -R nbnode_user:nbnode_user /nbnode

WORKDIR /nbnode
USER nbnode_user



RUN ls -lath
RUN tox -e build


# Run at root of the cloned repo
# https://stackoverflow.com/questions/64804749/why-is-docker-build-not-showing-any-output-from-commands
# DOCKER_BUILDKIT=1 BUILDKIT_PROGRESS=plain docker build --file ./container/nbnode_py38_cpu.dockerfile -t registry.gunthergl.com/nbnode-cpu:latest .
# DOCKER_BUILDKIT=1 BUILDKIT_PROGRESS=plain docker build --file ./container/nbnode_py38_cpu.dockerfile -t registry.gunthergl.com/nbnode-cpu:0.9 .
# docker push registry.gunthergl.com/nbnode-cpu:0.9
# docker push registry.gunthergl.com/nbnode-cpu:latest
# docker run --rm -it --entrypoint bash registry.gunthergl.com/nbnode-cpu:latest
