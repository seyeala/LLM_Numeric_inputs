# LLM_Numeric_inputs

Numeric Learning Model Enhancement with PyTorch

1-Project Description

This project focuses on enhancing a numeric learning model by integrating custom projection layers into a pre-trained language model, specifically GPT-2, using PyTorch. The enhancements aim to improve the model's ability to handle numeric inputs through dedicated input and output projections, which are trainable while keeping the transformer layers static. This setup is intended to explore the potential of specialized projections in effectively integrating and processing numeric data within the context of a language model framework.

2-Code Repository Structure

wrapperNM.py: Contains the NumericLMWrapper class that extends a pre-trained GPT-2 model with custom projection layers for numeric input and output handling.
train.py: Main script to train the model using the configuration specified in the YAML file and command-line arguments.

config.yaml: Configuration file containing all training parameters such as batch size, learning rate, number of epochs, etc.

requirements.txt: Lists all the Python packages required to run the scripts.
alignmntNN.py creates instances of a given LLM and creates customized tokenizers with linear projectors. Next it trains the linear projectors for self alignment.

README.md: This file, providing an overview and instructions for using this repository.
Installation and Setup

3- How to run:

Clone the repository:
git clone 'https://github.com/seyeala/LLM_Numeric_inputs.git'

Install required packages:

pip install -r requirements.txt


To train the model with default settings defined in the /config/config.yaml file.

The first step is called self alignment of the projectors used in customized tokenizers:


python alignmentNN.py --config './config/config.yaml'

To override settings via command line, for example, changing the number of epochs:
python train.py --config './config/config.yaml' --num_epochs 100


4-Results and Observations

Training Loss Over Epochs

Observations
Effect of Learning Rate:
Projection Layer Impact:
Model Performance:

5-Conclusion

x
