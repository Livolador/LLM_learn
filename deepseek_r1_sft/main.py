
# 测试下微调之前的模型是否可用
import torch
torch.cuda.empty_cache()

from transformers import AutoTokenizer, AutoModelForCausalLM
from data_prepare import samples
import json
import copy

model_name = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')
print('----模型加载成功----')


# 制作数据集

with open('datasets.jsonl', 'w', encoding='utf-8') as f:
    for s in samples:
        json_line = json.dumps(s, ensure_ascii=False)
        f.write(json_line + '\n')

    else:
        print('----数据集制作成功----')

# 准本训练集和测试集
from datasets import load_dataset
dataset = load_dataset('json', data_files={'train': 'datasets.jsonl'}, split='train')
print('数据数量: ', len(dataset))

train_test_split = dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']
print('训练集数量: ', len(train_dataset))
print('测试集数量: ', len(test_dataset))

print('----数据集准备成功----')

# 编写tokenizer处理工具

def tokenize_function(examples):
    texts = [f'{prompt}\n {completion}' for prompt, completion in zip(examples['prompt'], examples['completion'])]
    tokens = tokenizer(texts, truncation=True, max_length=512, padding='max_length')
    tokens['labels'] = tokens['input_ids'].copy()
    return tokens

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)


print(tokenized_train_dataset.features)

print('----tokenizer处理成功----')

# print(tokenized_train_dataset[0])

# 量化设置

from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config, device_map='auto')
print('----量化设置成功----')

# lora微调设置

from peft import get_peft_model, LoraConfig, TaskType

lora_config = LoraConfig(
                        r = 8,
                        lora_alpha=16,
                        lora_dropout=0.05,
                        task_type=TaskType.CAUSAL_LM)

model = get_peft_model(model, lora_config)

model.print_trainable_parameters()
print('----lora微调设置成功----')

# 训练参数设置

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir='./finetuned_model',
    num_train_epochs=10,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    evaluation_strategy='steps',
    eval_steps=10,
    save_steps=100,
    learning_rate=2e-4,
    logging_steps=10,
    remove_unused_columns=False,
    fp16=True,
    logging_dir='./logs',
    run_name='deepseek-r1-distill-finetune'
)
print('----训练参数设置成功----')



from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset

)
print('----开始训练----')

trainer.train()
print('----训练完成✔✔✔----')

# 