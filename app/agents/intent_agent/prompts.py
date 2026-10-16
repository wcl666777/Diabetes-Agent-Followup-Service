"""意图识别 Agent 的 Prompt 模板。"""
from langchain_core.prompts import ChatPromptTemplate

INTENT_SYSTEM = (
    "你是糖尿病随访助手的意图识别模块，只负责判断患者的问题属于哪一类，"
    "以便后续回答时把握侧重点。\n"
    "你不需要、也不允许回答患者的问题，只输出分类结果。"
)

INTENT_CLASSIFY_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", INTENT_SYSTEM),
        (
            "human",
            "可选类型（只能选一个）：\n"
            "- 饮食：吃什么、能不能吃、忌口、营养搭配、食谱\n"
            "- 运动：锻炼方式、运动量、能不能跑步/游泳/骑车\n"
            "- 用药：药物、胰岛素、剂量、服药时间、打针注射\n"
            "- 血糖监测：测血糖、空腹/餐后/睡前血糖、糖化血红蛋白、数值解读\n"
            "- 并发症：视网膜、肾病、神经病变、糖尿病足、心脑血管\n"
            "- 生活方式：作息、戒烟、饮酒、体重、情绪睡眠\n"
            "- 就医指导：复查、复诊、挂号、检查化验、就医流程\n"
            "- 综合咨询：不属于以上任何一类，或同时涉及多类\n\n"
            "患者问题：{question}\n\n"
            "只返回 JSON，不要任何多余内容：\n"
            '{{"question_type":"饮食","confidence":0.9}}',
        ),
    ]
)
