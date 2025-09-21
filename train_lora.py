import json
import pandas as pd
import torch
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, TaskType, get_peft_model
import os
import swanlab


# 基础信息设置
swanlab.login(api_key="[你的API Key]", save=True) # 登录 SwanLab，保存 API Key 以便后续使用
os.environ["SWANLAB_PROJECT"] = "qwen3-sft-medical" # 设置进程级别的环境变量：SwanLab 项目名称
PROMPT = "你是一个医学专家，你需要根据用户的问题，给出带有思考的回答。" # 提示模板定义：模型角色+任务要求+要求展示思考过程
MAX_LENGTH = 2048 # 设置模型处理的最大 token 数量
swanlab.config.update({ # 设置 SwanLab 项目的配置信息
    "model": "Qwen/Qwen3-1.7B",
    "prompt": PROMPT,
    "data_max_length": MAX_LENGTH,
    })


# 将原始数据集转换为大模型微调所需数据格式的新数据集
def dataset_jsonl_transfer(origin_path, new_path):
    messages = []

    # 读取原有的JSONL文件
    with open(origin_path, "r") as file:
        for line in file:
            # 解析每一行的json数据
            data = json.loads(line)
            input = data["question"]
            think = data["think"]
            answer = data["answer"]
            output = f"<think>{think}</think> \n {answer}" # 将思考过程和答案拼接为输出，中间用特殊标记分隔
            
            # 构建符合要求的消息格式
            message = {
                "instruction": PROMPT, # 系统提示
                "input": f"{input}", # 用户输入
                "output": output, # 期望模型输出
            }
            messages.append(message)

    # 保存重构后的JSONL文件
    with open(new_path, "w", encoding="utf-8") as file:
        for message in messages:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")

# 将重构的数据集进行预处理
def process_func(example): 
    input_ids, attention_mask, labels = [], [], [] # 输入ID、注意力掩码、标签列表
    
    # 使用 Qwen 特定的对话格式构建
    instruction = tokenizer( # 构建指令部分（系统提示 + 用户输入
        f"<|im_start|>system\n{PROMPT}<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens =  False,
    )
    response = tokenizer(f"{example['output']}", add_special_tokens=False) # 构建响应部分（期望输出）
    
    # 拼接 input_ids, attention_mask, labels
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id] # 拼接输入ID，[指令token] + [响应token] + [padding token]
    attention_mask = ( # 拼接注意力掩码，[指令token全1] + [响应token全1] + [padding token全1]
        instruction["attention_mask"] + response["attention_mask"] + [1]
    )
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id] # 拼接标签，指令部分全 -100（不计算 loss），响应部分为实际 token ID，padding 部分为 pad token ID
    '''
    输入序列: [系统提示token] + [用户问题token] + [助手开始token] + [响应token] + [padding]
    注意力掩码: [1,1,1,...,1] (全是1，表示都是有效token)
    标签: [-100,-100,...,-100] + [响应token] + [padding]
    '''

    # 长度处理，做一个截断
    if len(input_ids) > MAX_LENGTH:  
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}   


# 预测函数
def predict(messages, model, tokenizer):
    device = "cuda"

    # 使用模型特定的对话格式化函数
    text = tokenizer.apply_chat_template( 
        messages,
        tokenize = False, # 不进行tokenize
        add_generation_prompt = True # 添加生成提示符
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device) # 将文本转换为模型输入张量，并移动到指定设备

    # 生成响应
    generated_ids = model.generate( 
        model_inputs.input_ids,
        max_new_tokens = MAX_LENGTH,
    )
    generated_ids = [ # 从输出中去除输入部分
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens = True)[0] # 解码生成的 ID 为文本，跳过特殊 token

    return response


# 在 modelscope 上下载 Qwen 模型到本地目录下
model_dir = snapshot_download("Qwen/Qwen3-1.7B", cache_dir="./", revision="master")
# Transformers 加载模型权重
tokenizer = AutoTokenizer.from_pretrained("./Qwen/Qwen3-1.7B", use_fast = False, trust_remote_code = True)
model = AutoModelForCausalLM.from_pretrained("./Qwen/Qwen3-1.7B", device_map = "auto", torch_dtype = torch.bfloat16)
model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

# 配置 lora
config = LoraConfig(
    task_type = TaskType.CAUSAL_LM, # 任务类型指定为因果语言模型
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], # 需要应用 LoRA 的注意力层和前馈层
    inference_mode = False,  # 训练模式
    r = 8,  # Lora 秩，常用 4, 8, 16, 32
    lora_alpha = 32,  # 缩放系数，控制 LoRA 适配器对原始权重的贡献程度
    lora_dropout = 0.1,
)
model = get_peft_model(model, config) # 将模型转换为 LoRA 模型


# 加载、处理数据集和测试集
train_dataset_path = "train.jsonl"
test_dataset_path = "val.jsonl"
train_jsonl_new_path = "train_format.jsonl" # 格式化后
test_jsonl_new_path = "val_format.jsonl" # 格式化后
# 检查文件存在性，避免重复处理：转换数据集
if not os.path.exists(train_jsonl_new_path):
    dataset_jsonl_transfer(train_dataset_path, train_jsonl_new_path)
if not os.path.exists(test_jsonl_new_path):
    dataset_jsonl_transfer(test_dataset_path, test_jsonl_new_path)

# 得到训练集
train_df = pd.read_json(train_jsonl_new_path, lines=True)
train_ds = Dataset.from_pandas(train_df)
train_dataset = train_ds.map(process_func, remove_columns=train_ds.column_names) # 应用预处理函数，移除原有列
# 得到验证集
eval_df = pd.read_json(test_jsonl_new_path, lines=True)
eval_ds = Dataset.from_pandas(eval_df)
eval_dataset = eval_ds.map(process_func, remove_columns=eval_ds.column_names)


# 训练参数配置
args = TrainingArguments(
    output_dir="./output/Qwen3-1.7B",
    # 有效批量大小 = per_device_train_batch_size × gradient_accumulation_steps × GPU数量
    per_device_train_batch_size = 1,
    per_device_eval_batch_size = 1,
    gradient_accumulation_steps = 4,
    # 每 100 步进行一次验证评估
    eval_strategy = "steps",
    eval_steps = 100,

    logging_steps = 10, # 每 10 步记录一次日志
    num_train_epochs = 2, # 训练总轮次
    save_steps = 400, # 每 400 步保存一次模型
    learning_rate = 1e-4,
    save_on_each_node = True, # 在分布式训练中，每个节点都保存检查点
    gradient_checkpointing = True, # 启用梯度检查点以节省内存

    report_to = "swanlab",
    run_name = "qwen3-1.7B",
)
# 创建 Trainer 实例
trainer = Trainer(
    model = model,
    args = args,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer, padding = True),
)
trainer.train()

# 用测试集的前3条，主观看模型
test_df = pd.read_json(test_jsonl_new_path, lines=True)[:3]

test_text_list = []
for index, row in test_df.iterrows():
    instruction = row['instruction'] # 系统提示
    input_value = row['input'] # 用户输入
    messages = [ # 构建对话消息列表
        {"role": "system", "content": f"{instruction}"},
        {"role": "user", "content": f"{input_value}"}
    ]

    response = predict(messages, model, tokenizer)
    # 格式化输出
    response_text = f"""
    Question: {input_value}

    LLM:{response}
    """
    
    test_text_list.append(swanlab.Text(response_text)) # 转为 SwanLab Text 对象
    print(response_text) # 控制台打印

swanlab.log({"Prediction": test_text_list})

swanlab.finish()