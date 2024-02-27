# 部署推理说明
1. git clone https://github.com/01-ai/Yi.git
2. 下载Yi-34B模型在本地(新的hf仓库里似乎没有modeling_yi.py了，需要找6952531号commit，这个历史版本下载)
3. 环境：官方仓库用的micromamba，按照官方仓库说明进行；使用方法和miniconda一样
4. 修改Yi-34B模型目录里的modeling_yi.py为本仓库中的modeling_yi.py
5. 推理 进到demo目录然后新建一个my_text_generation_parallel.py贴入本仓库my_text_generation_parallel.py代码
