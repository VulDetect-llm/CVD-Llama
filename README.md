# CVD-Llama: CVD-Llama: Advancing Code Security with Context-Enhanced Large Language Models

## Contextual Vul
ContextualVul is a new dataset that contains contextual information of vulnerable functions. 
### usage
The dataset format of Contextual Vul is as follows:
```json
[
    {
        "index_to_funcname":{"0": "func_1", "1": ".."},
        "index_to_code": {"0": "code_1", "1": ".."},
        "adj": [
            [
                0,
                1,
                0
            ],
            [
                0,
                0,
                0
            ],
            [
                1,
                0,
                0
            ]
        ],
        "reason": "why this function has vulnerabilities or not vulnerable",
        "vul_type": "Not Vulnerable/Vulnerable"
    }
]
```
The `index_to_code['0']` refers to the target function to detect. The other functions are the contextual informations. 

## Training
Using the following commands:
```
sh model/train_cvd.sh
```

## Inference
We provide inference pipline in `model/pipeline.py` and provide tree task type in our paper.

1. Code Document generation. We use [Code Bert](https://github.com/microsoft/CodeBERT) dataset to evaluate our model. You can obtain the dataset from the repo, or using the following commands:
```
pip install gdown
mkdir data data/code2nl
cd data/code2nl
gdown https://drive.google.com/uc?id=1rd2Tc6oUWBo7JouwexW3ksQ0PaOhUr6h
unzip Cleaned_CodeSearchNet.zip
rm Cleaned_CodeSearchNet.zip
cd ../..
```

2. CrossCodeEval: You can obtain the scripts and dataset in [cceval](https://github.com/amazon-science/cceval). We use the official code to eval the model.

3. ContextualVul

## Checkpoints
We will relase our model checkpoints soon.