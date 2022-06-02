import torch
import transformers
from transformers import BertTokenizer, GPT2LMHeadModel, GPT2Config, GPT2Model, OpenAIGPTLMHeadModel, OpenAIGPTConfig
print(torch.__version__)

decode_config = GPT2Config.from_pretrained('./config.json')
decode_config.add_cross_attention=True
encode_config = OpenAIGPTConfig.from_pretrained('./config.json')
model_path = '/Users/zhouyang/Downloads/论文调研及数据集相关代码/CDial-gpt2-base'
# OpenAIGPTLMHeadModel GPT2LMHeadModel
gpt_encoder = OpenAIGPTLMHeadModel(encode_config)


tokenizer = BertTokenizer.from_pretrained(model_path)
token_ids = tokenizer.encode('今天我上街！')
token_ids = torch.tensor(token_ids, dtype=torch.long)
gpt_encoder(input_ids = token_ids)
gpt_encoder.generate()
print(tokenizer.encode('今天我上街！'))