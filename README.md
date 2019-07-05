# MLOps with Azure DevOps

## Preparing the environment

### Infrastructure/Cloud Infrastructure

This repository contains the base structure for you to start developing your
Machine Learning project using:

|Technology|Objective/Reason|
|----------|----------------|
|Azure Machine Learning Service|Manage Machine Learning models with the power of Azure|
|Azure Databricks|Use its compute power as a Remote Compute for training models|
|Azure Container Instance|Deploy Machine Learning models as Docker containers|

To have all the resources set, leverage the following resource to get your
infrastructure ready:

- [Setting up your cloud infrastructure](docs/setup-cloud-infrastructure.md)

### Azure DevOps

This project is intended to demonstrate the benefits of adding DevOps practices
to a Machine Learning development cycle.

This structure already have some resources to help you set up your pipelines
to accelerate model training and deployment.

- [Setting up the training pipeline](docs/setup-training-pipeline.md)

## Sample project

This code sample reproduces the **Image Classification**, a *convolutional neural
network image classification*. For more details of the project structure,
check the [project structure](docs/project-structure.md) page.

This project structure was also based on the
[cookiecutter data science project template](https://drivendata.github.io/cookiecutter-data-science/).

### Running this project on your local development environment

This sample provides you two options to have your environment set up for developing
and debugging locally. The focus of these docs is on how to have all the
required environment variables set so you can invoke
and debug this code on your machine.

* See [this page](docs/vscode-launch-json.md) for details on how to debug
using **Visual Studio Code**
* See [this page](docs/bash-environment-variables.md) for looking into details on how
to have the enrironment variables set using **Bash**

## Project flow

### Starting point/Current state

A Machine Learning project being developed by a team of Data Engineers/Data
Analysts, using Python.

The team develops the code to train the Machine Learning model and they need
to orchestrate the way this code gets **tested, trained, packaged
and deployed**.

### Testing

Testing the code that generages a model is crucial to the success and accuracy
of the model being developed.

The code being developed will produce a Machine Learning model that will help
people to take decisions, when not being the main responsible for the
decisions itself.

That's why testing the units of the code to make sure it meets the requirements
is a fundamental piece of the development cycle.

You will achieve it using the following capabilities:

- Python Unit Testing frameworks
- Azure DevOps

### Training

This project is all about generating a Machine Learning model, which needs
to be trained. Training a model requires compute power and orchestration.

Compute power is commonly an expensive asset and that's why this project
leverages cloud workloads to optimize resource consumption and avoiding upfront
costs.

To enable this, the following capabilities will be used:

- Machine Learning Python SDKs
- Azure Databricks
- Azure DevOps

### Deploying

The resulting model from the training step needs to be deployed somewhere
so the edge can consume it. There are a few ways to achieve it and,
for this scenario, you will deploy this model as part of a Docker Container.

A container has the power of having all the dependencies the application needs
to run encapsulated within it. It is also easily portable to multiple different
platforms.

To take advantage of deploying the model to a container, you will use:

- Azure DevOps
- Azure Container Instances
- Azure Machine Learning Service
