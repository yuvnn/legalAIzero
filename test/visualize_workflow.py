import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from langgraph_workflow import workflow
import graphviz

if __name__ == "__main__":
    try:
        dot = workflow.to_dot()
        graphviz.Source(dot).render("workflow_graph", format="png", view=True)
        print("workflow_graph.png 파일로 워크플로우 구조가 시각화되었습니다.")
    except Exception as e:
        print("그래프 시각화 오류:", e)
