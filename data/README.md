把测试集 xlsx（含“语义”和“大模型agent”两个 sheet）放在这个目录，然后手动触发 jev-eval 工作流并选 `dataset=full`。

仓库是私有的，但 xlsx 一旦提交就留在 git 历史里。不希望测试集进仓库的话，只用 `dataset=smoke`，或者在本机跑 `build_dataset.py`。
