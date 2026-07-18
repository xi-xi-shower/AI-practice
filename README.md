# ai_practice

## 1. 快速开始

### 1） 准备环境
- 复制 `.env.example` 为 `.env`
- 填写 `LLM_API_KEY`

### 2） 安装依赖
- 请在环境中自行安装依赖：

```bash
pip install -r requirements.txt
```

## 2. 数据来源

## 3. 技术栈

## 4. 运行方法

启动 FastAPI 服务：

```bash
uvicorn src.main:app --reload
```

服务启动后，默认访问地址为：

```text
http://127.0.0.1:8000
```

## 5. 接口调用示例

### 健康检查

```bash
curl http://127.0.0.1:8000/health
```

返回示例：

```json
{
  "status": "ok"
}
```

### 查看配置

```bash
curl http://127.0.0.1:8000/config
```

返回示例：

```json
{
  "model": "deepseek-chat",
  "base_url": "https://api.deepseek.com",
  "api_key_configured": true
}
```

### 调用聊天接口

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "请用一句话解释 RAG 是什么",
    "system_prompt": "You are a helpful assistant."
  }'
```

返回示例：

```json
{
  "answer": "RAG 是一种把外部知识检索结果与大语言模型生成能力结合起来的方法。"
}
```

## 6. 分析 / 模型流程

## 7. 结果展示

## 8. 问题与改进

## 9. 复盘记录