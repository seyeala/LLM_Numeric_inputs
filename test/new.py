import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr
import os
from torch import nn
import gc

torch.cuda.empty_cache()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class NumericToken(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NumericToken, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim).to(device)

    def forward(self, x):
        return self.linear(x)

modelname="distilbert/distilgpt2"
#modelname="lmsys/vicuna-7b-v1.5"




tokenizer = AutoTokenizer.from_pretrained(modelname)
model = AutoModelForCausalLM.from_pretrained(modelname).to(device)

embedding_dim = model.config.hidden_size
linear_layer_input = NumericToken(1, embedding_dim).to(device)
linear_layer_output = NumericToken(embedding_dim, 1).to(device)




inputn=linear_layer_input( torch.tensor([[12]], dtype=torch.float32).to(device))
inputT=tokenizer('12',return_tensors='pt')

print('inputn')

print('inputT')


input_ids = inputT['input_ids'].to(device)

print(input_ids)
outT=model.generate(input_ids, max_length=2)

print(outT,'outT')
