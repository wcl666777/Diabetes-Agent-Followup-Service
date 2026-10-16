# 双 Agent 随访服务（子项目二）

基于 **LangGraph** 的糖尿病出院患者随访系统：两个 Agent 由两个显式入口直接路由，数据落 **MySQL**。

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
agent_service/
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

## 运行

```bash
pip install -r requirements.txt
cp .env.example .env
python scripts/seed_patient.py
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

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
