# LLM_Numeric_inputs

Numeric Learning Model Enhancement with PyTorch

1-Project Description

This project focuses on enhancing a numeric learning model by integrating custom projection layers into a pre-trained language model, specifically GPT-2, using PyTorch. The enhancements aim to improve the model's ability to handle numeric inputs through dedicated input and output projections, which are trainable while keeping the transformer layers static. This setup is intended to explore the potential of specialized projections in effectively integrating and processing numeric data within the context of a language model framework.

2-Code Repository Structure

In the root the following files exist

wrapperNM.py: Contains the NumericLMWrapper class that extends a pre-trained GPT-2 model with custom projection layers for numeric input and output handling.




requirements.txt: Lists all the Python packages required to run the scripts.
alignmntNN.py creates instances of a given LLM and creates customized tokenizers with linear projectors. Next it trains the linear projectors for self alignment.

 alignmntTextN.py creates instances of a given LLM and creates customized tokenizers with linear projectors. Next it trainms the alignmentb of text to numbers ( at this stage only the output array is trained). It loads the prvious checkpoint stored in the chk folder and store the checkpoint in the end.

 alignmntMixedN.py creates instances of a given LLM and creates customized tokenizers with linear projectors. Next it trains the alignment or the LLM (if they are mad trainable in the code-I did not test this stage throughly ). It loads the prvious checkpoint stored in the chk folder and store a checkpoint in the end.


README.md: This file, providing an overview and instructions for using this repository.
Installation and Setup


config folder:

config.yaml: Configuration file containing all training parameters such as batch size, learning rate, number of epochs, etc. The parameters given as .arg will override config paramateres.

Dataset folder:

It includes two files ( jupyther notebook and a python file).Either of the can be used for generating the training dataset that are stored in this folder.

CHK:
Checkpoints for various states are stored Here

Beckmark training:

Includes excel files and other data gather that describe the benchmarks for the training that we observed.

3- How to run:

Clone the repository:
git clone 'https://github.com/seyeala/LLM_Numeric_inputs.git'

Install required packages:

cd LLM_Numeric_inputs
pip install -r requirements.txt


To train the model with default settings defined in the /config/config.yaml file.

The first self alignment of  the projectors and diffison layers when the input and output are the same numeric value you can run. In this stage only the linnear arrays (and not the tranformer) is trained

python alignmentNN.py --config './config/config.yaml' --num_epochs 2 --model_path_save ./chk/selfalignedNN.path

The second stage of  the projectors and diffison layers icnludes the input of strings of the numeric input with special tokens an the numeric output. In this stage only the linnear arrays (and not the tranformer) is further trained. Here the weight can be loaded from the prvious checkpoint and the outputs can be saved in anothe checkpoint.

python alignmentTextN.py --config './config/config.yaml' --num_epochs 2 --model_path_load ./chk/selfalignedNN.path --model_path_save ./chk/selfalignedTextN.path


You can sequentially reqpeat these steps. Everytime the input or output linear arrats are further optimized.



4-Results and Observations

We successfully performed training for Number to Number and Text to Number alignments. The best results on GPT-2 and GPT-2 Large showed close error rates of 2% and 25%, respectively. These results were achieved without any modifications to the LLM's weights, indicating promising potential for further stages (training with text and numbers to numbers). Additionally, we optimized the learning rate and memory utilization by determining the optimal batch size..




We noted that lscheduled learning rate starting from 0.003 and reducing by factor of 2 every 10 epoch woked well. We noticed that batch size of 8 provides the optimal perfomjance and best utilization of cuda memmory in both text to number and number to number alignments.


5-Conclusion

We note that the modular code was successfully tested on several transformers (GPT-2 and GPT-large). This modular approach allows us to conveniently scale up this test on larger networks. Additionally, we noticed promising results for the alignment of linear arrays on the LLM (2% error on numerical alignment). This indicates that using this approach on more resource-intensive datasets (Text to Text as well as Text/Number to Number) would require minimal changes to the LLM weights. While we are unsure about the ultimate success of this approach, the current findings so far strengthen our hypothesis that deep integration of numerical data in LLMs can be done efficiently for a new class of tasks, such as medical diagnosis or financial analysis, that involve unstructured number/text information.
