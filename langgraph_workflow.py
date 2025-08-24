
from operator import add
from typing_extensions import Annotated
from typing import List, Dict, Optional
from langgraph.graph import StateGraph, END
from pydantic import BaseModel

# costomized imports
from rag import retriever_dense, eval_ragas_metrics
from config import OPENAI_MODEL, SYSTEM_PROMPT, Generate_llm

ANSWER_RELEVANCY_SCORE_THRESHOLD = 0.7
FAITHFULNESS_SCORE_THRESHOLD = 0.9

class RAGState(BaseModel):
    # 누적 로그 (예: 처리 단계 로그)
    processing_log: Annotated[List[str], add]

    # 사용자 쿼리 히스토리 누적
    query : Optional[str]
    user_queries: Annotated[List[str], add]

    # 리트리버 관련
    retrieved_docs: Optional[List[Dict]] = None

    # 답변 생성 관련
    generation_attempts: int = 0
    generated_answer: Optional[str] = None

    # 평가 관련
    retrieval_score: Optional[float] = None
    generation_score: Optional[float] = None

    # 최종 출력
    final_output: Optional[str] = None



# 사용자 입력으로 query state를 업데이트하는 노드
def user_query_node(state):
    query = input("질문을 입력하세요: ")
    user_queries = list(getattr(state, "user_queries", []))
    user_queries.append(query)
    # 기존 state를 모두 유지하며 query와 user_queries만 갱신
    return state.copy(update={"query": query, "user_queries": user_queries})

# 1단계: 검색 노드
def dense_retrieve_node(state):
    query = getattr(state, "query", None)
    if not query:
        raise ValueError("query 값이 필요합니다.")
    result = retriever_dense(query, k=5)
    docs = result["retrieved_docs"]
    return state.copy(update={"retrieved_docs": docs})

def build_user_prompt(query, context_docs):
    context_text = "\n".join([doc["text"] for doc in context_docs]) if context_docs else ""
    return f"질문: {query}\n\n참고 문서:\n{context_text}"

def generate_node(state):
    query = getattr(state, "query", None)
    context_docs = getattr(state, "retrieved_docs", [])
    user_prompt = build_user_prompt(query, context_docs)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    resp = Generate_llm.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=0.2,
    )
    answer = resp.choices[0].message.content
    attempts = getattr(state, "generation_attempts", 0) + 1
    return state.copy(update={
        "retrieved_docs": context_docs,
        "generated_answer": answer,
        "final_output": answer,
        "response": answer,
        "generation_attempts": attempts,
    })

# 평가 노드 추가
def evaluate_node(state):
    query = getattr(state, "query", None)
    answer = getattr(state, "generated_answer", None)
    docs = getattr(state, "retrieved_docs", [])
    attempts = getattr(state, "generation_attempts", 1)
    scores = eval_ragas_metrics(query, answer, docs)

    print(f"[{attempts}회차 평가 결과]", scores)
    answer_relevancy_score = scores.get("answer_relevancy", 0)
    faithfulness_score = scores.get("faithfulness", 0)

    # 기준 미달이면 재시도, 최대 2회
    if (answer_relevancy_score < ANSWER_RELEVANCY_SCORE_THRESHOLD or faithfulness_score < FAITHFULNESS_SCORE_THRESHOLD) and attempts < 3:
        print(f"재시도 {attempts}회: 답변 품질 미달, answer 재생성 필요")
        return state.copy(update={"retry": True})
    elif (answer_relevancy_score < ANSWER_RELEVANCY_SCORE_THRESHOLD or faithfulness_score < FAITHFULNESS_SCORE_THRESHOLD) and attempts >= 3:
        print("최종 답변 품질 미달, 경고 문구 추가")
        answer = answer + "\n다음 답변은 검토가 필요합니다"
        return state.copy(update={"final_output": answer, "response": answer, "retry": False})
    else:
        return state.copy(update={"retry": False})
    
def eval_router_node(state):
    # 그냥 상태 객체를 그대로 반환
    return state

# ==================================================
# 랭그래프 워크플로우 정의
# ==================================================
graph = StateGraph(RAGState)
graph.add_node("user_query", user_query_node)
graph.add_node("retrieve", dense_retrieve_node)
graph.add_node("generate", generate_node)
graph.add_node("evaluate", evaluate_node)
graph.add_node("router", eval_router_node)

graph.add_edge("user_query", "retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", "evaluate")

graph.add_edge("evaluate", "router")
graph.add_conditional_edges(
    "router",  # 현재 평가 노드에서
    lambda state: "generate" if getattr(state, "retry", False) else "END",
    {"generate": "generate", "END": END}
)
graph.set_entry_point("user_query")
workflow = graph.compile()


# 실행 예시
if __name__ == "__main__":
    init_state = {
        "processing_log": [],
        "user_queries": [],
        "retrieved_docs": None,
        "generation_attempts": 0,
        "generated_answer": None,
        "retrieval_score": None,
        "generation_score": None,
        "final_output": None,
        "final_reason": None,
        "query": None,
    }
    result = workflow.invoke(init_state)
    retrieved_docs = result.get("retrieved_docs", [])
    # if retrieved_docs:
    #     for doc in retrieved_docs:
    #         print(f"doc_id: {doc.get('doc_id', 'N/A')}, text: {doc.get('text', '')}")
    # else:
    #     print("문서 없음")
    if retrieved_docs:
        for doc in retrieved_docs:
            print(f"doc_id: {doc.get('doc_id', 'N/A')}")
    else:
        print("문서 없음")
    print("검색 결과:", result.get("response", result.get("final_output", "결과 없음")))