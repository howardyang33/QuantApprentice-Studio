# QuantApprentice Studio

这是 QuantApprentice 的产品工作台层，负责：

- 提供普通模式和专业模式前端；
- 封装数据接入、任务理解、工作流编排、Teacher Zoo、Lesson Set 和评分接口；
- 调用相邻目录中的 `QuantApprentice_clean` 核心研究流程；
- 通过环境变量连接本地或远程的 GPT-OSS / OpenAI-compatible 模型服务。

本目录是 GitHub clean 版本，不包含数据集、老师模型产物、lesson 产物、运行结果或测试报告。

## 启动

建议从仓库根目录阅读 `../README.md`。最小启动方式：

```bash
cd QuantApprentice_studio
conda env create -f environment.yml
conda activate quant_apprentice_studio

PYTHONPATH=src:../QuantApprentice_clean \
python -m quant_apprentice_studio.cli serve-api --host 0.0.0.0 --port 8010
```

打开：

```text
http://服务器IP:8010
```

## 可选环境变量

```bash
export QA_LIVE_MODEL_API_URL="http://127.0.0.1:2310/v1/chat/completions"
export QA_LIVE_MODEL_NAME="gpt-oss-20b"
export QA_LIVE_MODEL_API_KEY="EMPTY"
export TUSHARE_TOKEN="你的TushareToken"
```

请不要把真实 token、运行输出或本地数据提交到 GitHub。
