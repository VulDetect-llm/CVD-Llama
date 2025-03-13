from transformers import TextGenerationPipeline, AutoTokenizer, AutoModel
from cvd_llama import MllamaForConditionalGeneration
from train_cvd_llama import get_cross_attention_token_mask, convert_sparse_cross_attention_mask_to_dense
import torch
import json
import random
import torch.nn.functional as F
from tqdm import tqdm
import os
import argparse

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def truncate(prompt: str, max_num_tokens: int, side: str, tokenizer) -> str:
    """Truncate prompt from side given the token budget"""

    tokens = tokenizer.tokenize(prompt)
    num_tokens = len(tokens)

    if num_tokens > max_num_tokens:
        if side == 'left':
            prompt_tokens = tokens[num_tokens - max_num_tokens:]
        elif side == 'right':
            prompt_tokens = tokens[:max_num_tokens]
        prompt = tokenizer.convert_tokens_to_string(prompt_tokens)
    return prompt

def preprocess(data, tokenizer, task_type):

    with open('prompts/detect_vul.txt', 'r') as f:
        task_prompt = f.read()
    
    ret = {}
    
    if task_type == 'code-doc-generation':
        #### cnl ####
        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n<|image|>Generate docs of this function.<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        ret['text'] = prompt
        ret['index_to_function_code'] = data['index_to_code']
        ret['adj'] = data['adj']
    
    elif task_type == 'cross-code-eval':
        #### cceval ####
        code = data['code']
        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n<|image|>According to the context, complete the following code:\n{truncate(code, 2048, 'left', tokenizer)}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{data['groundtruth']}<|eot_id|>"
        ret['text'] = prompt
        ret['index_to_function_code'] = data['index_to_code']
        ret['adj'] = data['adj']
    
    elif task_type == 'contextual-vul':
        #### contextual eval ####
        prompt = task_prompt
        if "vul_code" in data.keys():
            code = data['vul_code']
        else:
            code = data['index_to_code']['0']
        prompt = prompt.replace('<code>', truncate(code, 2048, 'left', tokenizer))
        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n<|image|>{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        text = prompt
        ret['text'] = text
        if len((data['index_to_code'])) > 1:
            del data['index_to_code']["0"]
            new_index_to_code = {str(int(key)-1): value for key, value in data['index_to_code'].items()}
            ret['index_to_function_code'] = new_index_to_code
        else:
            ret['index_to_function_code'] = data['index_to_code']
        ret['adj'] = data['adj']
        
    return ret

class PipeLine:
    def __init__(self, model_path):
        self.model = MllamaForConditionalGeneration.from_pretrained(model_path, device_map='cuda:1')
        self.gte = AutoModel.from_pretrained('/data/models/gte-large-en-v1.5', device_map='cuda:1', trust_remote_code=True)
        self.gte_tokenizer = AutoTokenizer.from_pretrained('/data/models/gte-large-en-v1.5', trust_remote_code=True)
        self.llama_tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.max_func_num = 10

    def preprocess(self, input, call_graph):
        cg = []
        for k, v in call_graph['index_to_function_code'].items():
            cg.append(v)
        
        inputs = input
        inputs = self.llama_tokenizer(inputs, max_length=8192, padding=False, truncation=True, return_tensors='pt')
        input_ids = inputs.input_ids
        cg = self.gte_tokenizer(cg, max_length=1024, padding='max_length', truncation=True, return_tensors='pt')
        code_input_ids = cg['input_ids']
        code_attention_mask = cg['attention_mask']
        code_embeddings = []
        
        for b in range(0, code_input_ids.size(0), 2):
            code_input_ids_batch = code_input_ids[b:b+2].to('cuda:1')
            code_attention_mask_batch = code_attention_mask[b:b+2].to('cuda:1')
            with torch.no_grad():
                code_outputs = self.gte(input_ids=code_input_ids_batch, attention_mask=code_attention_mask_batch).last_hidden_state
                code_embeddings.append(code_outputs)
        
        code_embeddings = [torch.cat(code_embeddings, dim=0)]
                
        cross_attention_mask =  [
            get_cross_attention_token_mask(token_ids, 128256) for token_ids in input_ids
        ]
        num_tiles = []
        for i in range(len(code_embeddings)):
            n = code_embeddings[i].shape[0]
            num_tiles.append([n])
        cross_attention_mask = convert_sparse_cross_attention_mask_to_dense(
            cross_attention_token_mask=cross_attention_mask,
            num_tiles=num_tiles,
            max_num_tiles=self.max_func_num,
            length=input_ids.shape[1]
        )
        cross_attention_mask = torch.tensor(cross_attention_mask).to('cuda:1')
        
        for i in range(len(code_embeddings)):
            code_id = code_embeddings[i]
            n = code_id.shape[0]
            if n < self.max_func_num:
                padding_size = self.max_func_num - n
                code_id = F.pad(code_id, (0, 0, 0, 0, 0, padding_size))
                
            elif n > self.max_func_num:
                code_id = code_id[:self.max_func_num, :, :]
            code_embeddings[i] = code_id
        
        code_embeddings = torch.stack(code_embeddings).to(f'cuda:1')

        inputs = inputs.to('cuda:1')
        return inputs, code_embeddings, cross_attention_mask
    
    def generate(self, inputs, call_graph):
        inputs, code_embeddings, cross_attention_mask = self.preprocess(inputs, call_graph)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        with torch.no_grad():
            self.model.eval()
            outputs = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, cross_attention_states=code_embeddings, cross_attention_mask=cross_attention_mask, max_new_tokens=50, use_cache=True, do_sample=True)
            decoded_output = self.llama_tokenizer.decode(outputs[0], skip_special_tokens=False)
            # print('--------------------------------------')
            # print(decoded_output)
        if '<|start_header_id|>assistant<|end_header_id|>' in decoded_output:
            return decoded_output.split('<|start_header_id|>assistant<|end_header_id|>')[-1].split('<|eot_id|>')[0]
        else:
            return "error"


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_type', type=str, default='cross-code-eval')
    parser.add_argument('--model_path', type=str)
    args = parser.parse_args()
    
    assert args.task_type in ['cross-code-eval', 'code-doc-generation', 'contextual-vul']
    
    # load model
    pipeline = PipeLine(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    task_type = args.task_type
    outputs = []
    # load data
    data = open('data/cross-code-eval/cceval_test.jsonl', 'r').readlines()
    data = [json.loads(d) for d in data]
    for d in tqdm(data):
        ret = preprocess(d, tokenizer, task_type)
        output = pipeline.generate(ret['text'], d)
        outputs.append(output)
    
    with open("experiments/ctx-eval-test.jsonl", 'w') as f:
        for d, response in tqdm(zip(data, outputs)):
            d['pred'] = response
            print(json.dumps(d), file=f, flush=True)
            
        
        
        
    

    

if __name__ == '__main__':
    run()
        

                
