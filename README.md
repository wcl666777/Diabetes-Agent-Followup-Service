# 双 Agent 随访服务

> 基于 **LangGraph** 的糖尿病出院患者智能随访系统：两个 Agent 由两个显式入口直接路由，数据落 **MySQL**，并内置 **Human-in-the-Loop** 人工介入闭环。

> 本服务是「糖尿病出院患者智能随访助手」的**子项目二**，通过 **MCP 协议**调用子项目一（RAG 知识问答服务）获取医学指南上下文。
> 姊妹项目：[Diabetes-RAG-QA-Service](https://github.com/wcl666777/Diabetes-RAG-QA-Service)

## 项目做了什么

面向糖尿病出院患者提供两条随访业务线，并用**人工兜底**保证医疗安全：

- **咨询线**：问答 Agent 做**双源检索**（公共指南 + 个人病例/出院小结），答案拿不准或高危时自动转医生；
- **血糖线**：血糖管理 Agent 记录血糖、用**代码确定性计算**医学指标并主动告警。

核心目标：让 AI 承担高频、标准化的随访问答，把**高风险、不确定**的部分交还给医生。

## 核心亮点

| 亮点 | 说明 |
|------|------|
| **Human-in-the-Loop 双模式** | ① **单次转人工**：`interrupt()` 暂停图 → 落 `hitl_pending` → 医生回复 → `Command(resume)` 恢复（不重跑前置节点）；② **接管式连续会话**：医生首次回复即自动接管，患者消息直达医生（可多轮），「结束诊断」后自动恢复 AI，会话落库、跨进程不丢 |
| **主动转人工** | 患者消息命中「找医生 / 转人工 / 人工客服」关键词，跳过检索直接转人工（实测 < 0.1s） |
| **双源检索融合** | 公共指南（经 MCP 调 RAG 服务）+ 个人病历（本地父子块 BM25 检索），合并为统一上下文 |
| **确定性医学计算** | eAG / HbA1c 反推 / TIR·TAR·TBR / CV 全部用**纯代码**计算，绝不交给大模型，规避数值幻觉 |
| **JWT + bcrypt** | PyJWT 签发/校验、bcrypt 加盐哈希；患者端 / 医生端按角色区分权限 |
| **SSE 流式体验** | 逐 token 流式输出；检索阶段先推 `status` 事件（「正在检索知识库…」），消除用户空白等待 |
| **三层兜底** | 自动重试 → Agent 级降级 → 系统级兜底；即使 LLM / RAG / MySQL 全挂也返回友好提示，绝不把异常抛给用户 |
| **幂等迁移 + 单例缓存** | MySQL 迁移脚本幂等、启动自动执行、单条失败仅告警不阻断；核心组件进程级单例复用，避免重复加载 |

## 技术栈

| 层面 | 选型 |
|------|------|
| Agent 框架 | LangGraph 1.0（StateGraph + MemorySaver + interrupt 人工介入） |
| Web / 服务 | FastAPI + Uvicorn + SSE-Starlette（流式） |
| 认证 | JWT（PyJWT）+ bcrypt 加盐哈希 |
| ORM / 存储 | SQLAlchemy 2.0（异步 aiomysql）+ MySQL |
| 大模型 | DeepSeek（OpenAI 兼容接口） |
| 跨服务协议 | MCP（Streamable HTTP / JSON-RPC，httpx + tenacity 重试） |
| 个人文档检索 | 父子块切分（langchain-text-splitters）+ jieba 分词 + rank-bm25 |
| 文档解析 | python-docx / pdfplumber（上传解析为纯文本入库） |
| 前端 | 原生 HTML / CSS / JS 单页（患者端 + 医生端工作台） |
| 部署 | Docker（静态前端随服务托管） |

## 两个 Agent

### ① 问答 Agent（RAG + Human-in-the-Loop）

- 通过 **MCP** 调用 RAG 服务的 `search_knowledge_base` 获取「公共指南」；
- 同时从 **MySQL** 读取患者「个人出院小结 / 病例」（父子块纯文本），形成**双源混合上下文**；
- 低置信（拿不准）、高危，或**患者主动说「我要找医生 / 转人工」**时，暂停图转人工；
- **人工介入双模式**：
  - **单次转人工**：`interrupt()` 暂停图 → 医生回复 → `Command(resume)` 恢复，不重跑前置节点；转人工任务落 `hitl_pending` 表。
  - **接管式连续会话**：医生首次回复即**自动接管**，之后该患者的咨询消息**直达医生**（可多轮），医生点「结束诊断」后自动恢复 AI。会话与对话落 `human_sessions` / `human_messages`，跨进程、重启不丢。

```
START → retrieve → (路由) → generate                         高置信且非高危
                 → human_review(interrupt) → generate        低置信 / 高危 / 主动要求人工
                          │
                          └─► 医生接管（human_sessions: active）→ 患者↔医生多轮
                                                        └─► 结束诊断 → 恢复 AI
generate → END
```

> 补充：患者处于人工模式时，`/chat` 与 `/chat/unified/stream` 入口会拦截咨询类消息并直达医生（血糖记录不受影响）。

### ② 血糖管理 Agent（确定性计算 + 告警）

- 记录血糖并落库，形成时间序列；
- 医学指标一律用**代码确定性计算**（不交给 LLM）：eAG、HbA1c 反推、TIR / TAR / TBR / CV；
- 高血糖 / 低血糖主动告警，给出解读与生活方式建议。

```
START → record → interpret → END
```

## 入口分流

患者通过前端两个入口显式选择，后端按 `mode` 参数直接路由，不做自动意图分类：

```python
if payload.mode == "qa":
    return await qa_agent_graph.ainvoke(...)
elif payload.mode == "glucose":
    return await glucose_agent_graph.ainvoke(...)
```

## 认证

JWT（PyJWT）+ bcrypt。患者端 / 医生端均支持 **注册 + 登录**，注册成功即返回令牌（注册即登录）；医生权限由注册时 `role=doctor` 决定。

## 目录结构

```
Diabetes-Agent-Followup-Service/
├── app/
│   ├── main.py              # FastAPI 入口 + 静态前端
│   ├── config.py            # pydantic-settings 配置中心
│   ├── api/                 # auth / chat / unified_chat / glucose / hitl / patient_docs
│   ├── agents/
│   │   ├── qa_agent/        # 问答 Agent（graph/state/nodes/prompts）
│   │   └── glucose_agent/   # 血糖 Agent（graph/state/nodes/calculator）
│   ├── core/                # llm_factory / memory / mcp_client / db / models / chunking / auth
│   │                        # human_session（人工接管会话状态机）/ qa_flow（主动转人工）
│   └── db/migrations.py     # 幂等 Schema 迁移（MySQL 适配）
├── static/index.html        # 患者端 + 医生端演示页
└── scripts/seed_patient.py  # 写入示例患者
```

## 快速开始

```bash
pip install -r requirements.txt
cp .env.example .env
python scripts/seed_patient.py
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**前置依赖**：MySQL、DeepSeek API Key，以及可访问的 RAG 服务 MCP 端点（见 `RAG_MCP_SERVER_URL`，默认 `http://localhost:8001/rag`）。

## 接口一览

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | /api/v1/auth/register | 注册患者/医生（返回 token） |
| POST | /api/v1/auth/login | 登录换令牌 |
| POST | /api/v1/auth/refresh | 刷新令牌 |
| POST | /api/v1/chat | 咨询问答（mode=qa；命中主动转人工关键词则直接转人工） |
| POST | /api/v1/chat/stream | 咨询问答（SSE 流式：status / meta / token / interrupted） |
| POST | /api/v1/chat/unified/stream | 统一聊天入口（SSE，自动路由 qa / glucose / clarify / 人工） |
| GET | /api/v1/chat/hitl/{thread_id} | 患者端轮询单次人工处理结果 |
| POST | /api/v1/glucose | 记录血糖入口（mode=glucose） |
| POST | /api/v1/documents | 上传病例 / 出院小结（解析为纯文本入库） |
| GET | /api/v1/documents | 查看个人文档列表 |
| DELETE | /api/v1/documents/{doc_id} | 删除个人文档 |
| GET | /api/v1/hitl/pending | 医生端待人工介入列表（尚未接管） |
| POST | /api/v1/hitl/resume | 医生首次回复：恢复图并**自动接管** |
| GET | /api/v1/hitl/sessions | 进行中的人工会话列表（含完整对话） |
| POST | /api/v1/hitl/session/reply | 医生回复患者（接管后多轮） |
| POST | /api/v1/hitl/session/close | 结束诊断，患者恢复 AI 问答 |
| GET | /api/v1/hitl/session/me | 患者端轮询人工会话（是否接管 / 已结束 + 对话） |

## 相关项目

- 姊妹项目：[Diabetes-RAG-QA-Service](https://github.com/wcl666777/Diabetes-RAG-QA-Service) —— RAG 知识问答服务（本项目通过 MCP 调用其检索能力）
- 总入口：[Diabetes-Discharge-Management-Assistant](https://github.com/wcl666777/Diabetes-Discharge-Management-Assistant) —— 含两个服务与 `docker-compose.yml`，可一键启动整套系统
