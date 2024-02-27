#在官方git仓库基础上新加入的文件，因为按照官方推理代码的话不知道为啥老OOM，就用accelerate切的
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import os
import torch
from accelerate import infer_auto_device_map, init_empty_weights, load_checkpoint_and_dispatch

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7'  #注意：此处若报错，export CUDA_VISIBLE_DEVICES对应进行更改
load_time=0
import time
start_time = time.time()

# 首先使用虚拟内存加载模型，此处还没使用显存
with init_empty_weights():
    model = AutoModelForCausalLM.from_pretrained("/root/Llama2-70b-chat-hf",
                                                 trust_remote_code=True, torch_dtype=torch.float16)

# The model weights are not tied.
# Please use the `tie_weights` method before using the `infer_auto_device` function.
model.tie_weights()

model = load_checkpoint_and_dispatch(model, checkpoint="/data/yi-34b-testTPS/Yi-34B-200K",
                    device_map='auto', offload_folder="offload", no_split_module_classes=["YiDecoderLayer"],
                    offload_state_dict=True, dtype=torch.float16).half()

end_time = time.time()
load_time=end_time-start_time
print("model loaded in"+str(load_time)+" s.")

while True: #获取单次输入，无历史
    model.model.fwd_num = 0
    model.model.seq_len = 0
    model.model.encode_time = 0
    model.model.decode_time = 0
    text_in = input('please input your question: ')
    tokenizer = AutoTokenizer.from_pretrained("/data/yi-34b-testTPS/Yi-34B-200K",trust_remote_code=True)

    inputs = tokenizer(text_in, return_tensors="pt").to(0)# to gpu 0

    outputs = model.generate(inputs["input_ids"], max_new_tokens=100)
    text_out = tokenizer.decode(outputs[0].tolist(), skip_special_tokens=True)
    print("\n")
    print("Tokens generated in 1 sec（第一秒生成token数）: %d" % (model.model.one_sec_tokens+1))
    print("First token time（生成第一个token耗时）: %.4f ms" % (model.model.encode_time))   #decoder-only，所以记录的“encode_time”就是first token time
    #print("Generated token count（总生成token数）:%d" % (model.model.fwd_num - 1))
    print("Time per token（平均生成每个token用时）:%.4f ms" % ((model.model.encode_time+model.model.decode_time)/(model.model.fwd_num)))
    model.model.one_sec_tokens=0
    print(text_out)
