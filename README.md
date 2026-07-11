# QuantApprentice Studio

本目录包含 QuantApprentice Studio 产品工作台和它依赖的 clean 核心代码。

本目录不包含数据集、模型权重、运行输出、中间产物、导入论文资产或测试报告。使用时需要自行准备数据、模型服务和必要的环境变量。

## 目录结构

```text
QuantApprentice_studio_clean/
├── QuantApprentice_studio/   # Studio 后端、前端、工作流封装和产品界面
├── QuantApprentice_clean/    # clean 核心研究流程代码
├── README.md                 # 本说明
└── .gitignore                # GitHub 上传时忽略本地数据和输出
```

## 快速启动 Studio

```bash
cd /path/to/QuantApprentice_studio_clean/QuantApprentice_studio
conda env create -f environment.yml
conda activate quant_apprentice_studio

PYTHONPATH=src:../QuantApprentice_clean \
python -m quant_apprentice_studio.cli serve-api --host 0.0.0.0 --port 8010
```

然后在浏览器打开：

```text
http://服务器IP:8010
```

如果是在 Remote SSH 环境中使用，也可以通过 VS Code 端口转发打开 `8010`。

## 可选：连接本地 GPT-OSS 服务

Studio 默认可以在无模型模式下打开界面。若要启用本地 GPT-OSS 评分，需要先启动兼容 OpenAI 接口的本地模型服务，然后设置：

```bash
export QA_LIVE_MODEL_API_URL="http://127.0.0.1:2310/v1/chat/completions"
export QA_LIVE_MODEL_NAME="gpt-oss-20b"
export QA_LIVE_MODEL_API_KEY="EMPTY"
```

模型权重不包含在本仓库中，请在部署机器上自行准备。

## 可选：联网下载 A 股 K 线

如果使用 Studio 里的在线 K 线下载功能，需要安装相关依赖，并设置 Tushare token：

```bash
export TUSHARE_TOKEN="你的TushareToken"
```

## 数据说明

本目录没有附带：

- 历史行情 CSV / Parquet；
- demo 数据集；
- 论文复现实验导入资产；
- teacher / lesson 的运行产物；
- workflow 输出；
- smoke test 报告；
- 本地模型权重。

运行完整流程时，需要在 Studio 页面中上传数据，或在本地按项目约定放置数据资产。

## 开发说明

- `QuantApprentice_studio` 负责产品界面、API、工作流编排和 Studio wrapper。
- `QuantApprentice_clean` 保留核心研究流程代码，Studio 会通过 `PYTHONPATH=src:../QuantApprentice_clean` 复用它。
- 不建议把运行目录、下载数据、测试输出或本地密钥提交到 GitHub。
