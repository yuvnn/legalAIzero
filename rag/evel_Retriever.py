import os
from dotenv import load_dotenv
load_dotenv()

from typing import List, Dict, Any
from ragas import evaluate
from ragas.metrics import answer_relevancy, Faithfulness
from datasets import Dataset

from config import Ragas_llm

faithfulness = Faithfulness()

def eval_ragas_metrics(
    question: str,
    answer: str,
    retrieved_docs: List[Dict[str, Any]],
):
   
    # ragas용 row 생성
    contexts = [doc["text"] for doc in retrieved_docs]
    row = {
        "question": question,
        "answer": answer,
        "contexts": contexts,
    }
    ds = Dataset.from_list([row])
    result = evaluate(
        ds,
        metrics=[answer_relevancy, faithfulness],
        llm=Ragas_llm,
    )
   
    scores = {}
    for key in result._scores_dict.keys():
        v = result[key]
        if isinstance(v, dict) and "aggregate" in v:
            scores[key] = v["aggregate"]
        elif isinstance(v, (int, float)):
            scores[key] = float(v)
        elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], (int, float)):
            scores[key] = float(v[0])
        elif isinstance(v, list) and len(v) > 0 and hasattr(v[0], "item"):
            scores[key] = float(v[0].item())
        else:
            scores[key] = None
    return scores

if __name__ == "__main__":
    docs = [
        {'doc_id': '판례_23  37299', 'chunk_index': 2, 'score': 0.8989328145980835, 'filename': '판   례_237299.txt', 'text': '원칙 및  비례원칙을 위반한 것임은 물론 납세자의 재산권을 침해하는 위헌적인 것이다. 따라서 이 사건 처분 중 적법·정당하게 부과된 부분을 초과한 부분은 취소되 어야 한다.\n3. 관련 법령\n별지 기재 와 같다.\n4. 이 사건 처분의 적법 여 부\n가. 관련 규정의 개정 경과 등\n1) 종합부동산세법의 제정 배경 및 취지 우리 세법은 2005. 1. 종합부동산세가 도입되기 이전까지 부동산 보유세로서 지방세제 내에 재산세와 종합토지세를 두고 있었다. 종합토지세는 부동산 투 기와 과다한 토지 보유를 억제하여 지 가 안정과 토지 소유의 저변 확대를 목적으로 1990년부터 시행되었다.\n1997 년 외환위기의 영향으로 하락했던 부동산 가격이 2000년대 초반 국제통화기금(IMF)의 관리 체제에서 벗어나면서 수 도권 아파트를 중심으로 급등하자 종래 부동산 보유세제로는 투기 억제에 한 계가 있다고 보아 2005. 1. 5. 법률 제7328호로 종합부동산세법을 제정하여  종합부동산세를 도입하게 되었다.\n종 합부동산세법은 고액의 부동산 보유자 에 대하여 부동산보유세를 과세함에 있어서 지방세의 경우보다 높은 세율로  국세인 종합부동산세를 과세하여 부동 산 보유에 대한 조세부담의 형평성을  제고하고 부동산의 가격안정을 도모함 으로써 재방재정의 균형발전과 국민경 제의 건전한 발전을 기하려는 데 그 취지가 있다.\n2) 종합부동산세에서 공제되는 재산세액의 산정 방법에 관한 규 정 가) 이중과세의 조정 필요성 종합부동산세는 종합부동산세법상 과세기준일인 매년 6월 1일 현재 일정한 가액을  초과하는 주택과 토지 등 부동산을 보 유하는 자에 대하여 그 부동산 가액을 과세표준으로 삼아 과세하는 세금으로 서 일정한 재산의 소유라는 사실에 담 세력을 인정하여 부과하는 재산보유세 의 일종으로, 실질적으로는 토지 및 주택분 재산세 납세의무의 연장선상에 있다고 할 수 있다. 이에 따라 과세기준 금액 이하 구간은 재산세만 부과되지만, 과세기준금액 초과 구간은 재산세뿐 만 아니라 재산세보다 고율의 세율을  적용한 종합부동산세도 함께 부과되므 로, 재산세와 종합부동산세가 이중으로 과세되는 부분이 필연적으로 발생하게 된다.\n종합부동산세법에서는 이러한 이중과세의 문제를 조정하기 위하여 재산세액 공제제도를 도입하였고, 그에  관한 규정은 아래와 같이 변천되었다.\n나) 종합부동산세법상 재산세액 공제 제도의 변천 경과'},
        {'doc_id': '판례_71063', 'chunk_index': 4, 'scoree': 0.8976229429244995, 'filename': '판례_71063.txt', 'text': '부할 의 의무가 있다.\n제8조\n(과세표준)\n① 주 택에 대한 종합부동산세의 과세표준은 납세의무자별로 주택의 공시가격을 합 산한 금액에서 6억 원을 공제한 금액으로 한다. 다만, 그 금액이 영보다 작은 경우에는 영으로 본다.\n② 다음 각 호의 어느 하나에 해당하는 주택은\n제1 항\n의 규정에 의한 과세표준 합산의  대상이 되는 주택의 범위에 포함되지  아니하는 것으로 본다.\n1.\n임대주택 법 제2조 제1호 의 규정에 의한 임대주택 또는 대통령령이 정하는 다가구 임 대주택으로서 임대기간, 주택의 수, 가격, 규모 등을 감안하여 대통령령이 정하는 주택 2.\n제1호 의 주택외에 종업원의 주거에 제공하기 위한 기숙사 및 사원용 주택, 주택건설사업자가 건축하여 소유하고 있는 미분양주택 등 종합 부동산세를 부과하는 목적에 적합하지 아니한 것으로서 대통령령이 정하는 주택\n제9조\n(세율 및 세액)\n① 주택에 대한 종합부동산세는 과세표준에 다음 의 세율을 적용하여 계산한 금액을 그 세액(이하 ‘주택분 종합부동산세액’이 라 한다)으로 한다.\n〈과세표준〉 〈 세율〉3억 이하1천분의 103억 원 초과 14억 원 이하1천분의 15 14억 원 초과 94억 원 이하1천분의 20 94억 원 초과 1천분의 30 ② 주택분 종합부동산세액을 계산함에 있어 2006년부터 2008년까지의 기간에 납세의무가 성립하는 주택분 종합부동산세에 대하여는\n제1항\n의 규정에 의한 세율별 과세표준에 다음  각 호의 연도별 적용비율과\n제1항\n의 규정에 의한 세율을 곱하여 계산한 금액을 각각 당해연도의 세액으로 한다.\n1. 2006년 : 100분의 70\n2. 2007년 : 100분의 80\n3. 2008년 : 100분의 90'},
    ]
    question = "종합부동산세법의 제정 배경 및 취지와 공제되는 재산세액 산정 방법은?"
    answer = "ㅁㄴㅇㄻㄴㅇㄹ 고액의 부동산 보유자에 대해 형평성 있는 조세부담을 위해 제정되었으며, 공제되는 재산세된다."
    scores = eval_ragas_metrics(
        question=question,
        answer=answer,
        retrieved_docs=docs,
    )
    print("RAGAS 평가 결과:", scores)