# FROM python:3.8.13-slim-buster
FROM python:3.8.13-slim-buster
RUN apt-get update && \
    apt-get upgrade -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git \
 && apt-get clean && \
  rm -rf /var/lib/apt

RUN pip install -U pip tox pipx
RUN pip install -U setuptools setuptools_scm wheel
RUN git config --global user.email "gunthergl@gmx.net"
RUN git config --global user.name "gunthergl"


# Create a working directory
RUN mkdir /nbnode_pyscaffold
WORKDIR /nbnode_pyscaffold

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash nbnode_user 

### Add relevant code sources
#       Add directories to the container
ADD src /nbnode_pyscaffold/src
ADD .git /nbnode_pyscaffold/.git
# ADD tests /nbnode_pyscaffold/tests
# ADD container /nbnode_pyscaffold
# ADD .coveragerc /nbnode_pyscaffold
# ADD dist /nbnode_pyscaffold/dist
# ADD docs /nbnode_pyscaffold/docs

#       Add files to the container
ADD AUTHORS.rst /nbnode_pyscaffold
ADD CHANGELOG.rst /nbnode_pyscaffold
ADD CONTRIBUTING.rst /nbnode_pyscaffold
ADD .gitignore /nbnode_pyscaffold
ADD LICENSE.txt /nbnode_pyscaffold
ADD pyproject.toml /nbnode_pyscaffold
ADD setup.cfg /nbnode_pyscaffold
ADD setup.py /nbnode_pyscaffold
ADD tox.ini /nbnode_pyscaffold
# ADD .isort.cfg /nbnode_pyscaffold
# ADD .gitlab-ci.yml /nbnode_pyscaffold
# ADD .pre-commit-config.yaml /nbnode_pyscaffold
# ADD README.rst /nbnode_pyscaffold
# ADD .readthedocs.yml /nbnode_pyscaffold
# ADD .tox /nbnode_pyscaffold
# ADD removeme_.gitlab-ci.yml /nbnode_pyscaffold
# ADD .vscode /nbnode_pyscaffold

RUN chown -R nbnode_user:nbnode_user /nbnode_pyscaffold

WORKDIR /nbnode_pyscaffold
USER nbnode_user



RUN ls -lath
RUN tox -e build


# Run at root of the cloned repo
# https://stackoverflow.com/questions/64804749/why-is-docker-build-not-showing-any-output-from-commands
# DOCKER_BUILDKIT=1 BUILDKIT_PROGRESS=plain docker build --file ./container/nbnode_py38_cpu.dockerfile -t registry.gunthergl.com/nbnode-cpu:latest .
# DOCKER_BUILDKIT=1 BUILDKIT_PROGRESS=plain docker build --file ./container/nbnode_py38_cpu.dockerfile -t registry.gunthergl.com/nbnode-cpu:0.9 .
# docker push registry.gunthergl.com/nbnode-cpu:0.9
# docker push registry.gunthergl.com/nbnode-cpu:latest