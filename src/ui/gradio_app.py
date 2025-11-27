# src/ui/gradio_app.py
from __future__ import annotations

import gradio as gr

from src.schemas import (
    ScalpCondition,
    UserProfile,
    ScalpAnalysisRequest,
)
from src.analysis.report_rules import simple_rule_based_analysis
from src.analysis.llm_agent import generate_dummy_llm_report


def run_analysis(
    gender: str,
    age_group: str,
    v1: int,
    v2: int,
    v3: int,
    v4: int,
    v5: int,
    v6: int,
) -> str:
    # 1) 더미 ScalpCondition 구성 (나중에 여기만 CNN 출력으로 바꿔주면 됨)
    condition = ScalpCondition(
        value_1=v1,
        value_2=v2,
        value_3=v3,
        value_4=v4,
        value_5=v5,
        value_6=v6,
    )

    # 2) 간단 UserProfile (필드 이름은 실제 정의에 맞게 조정)
    profile = UserProfile(
        gender=gender,
        age_group=age_group,
        # 나중에 샴푸/펌/염색 습관도 입력 받아서 넣어도 됨
    )

    req = ScalpAnalysisRequest(
        scalp_condition=condition,
        user_profile=profile,
    )

    # 3) rule-based + LLM 스타일 리포트 생성
    analysis = simple_rule_based_analysis(req)
    report_text = generate_dummy_llm_report(analysis)
    return report_text


def create_interface() -> gr.Blocks:
    with gr.Blocks() as demo:
        gr.Markdown(
            "# 🧠 Scalp Vision Agent (Demo)\n모델 없이 LLM 리포트 파이프라인 테스트"
        )

        with gr.Row():
            gender = gr.Radio(
                ["male", "female", "unknown"],
                value="unknown",
                label="성별",
            )
            age_group = gr.Dropdown(
                ["10대", "20대", "30대", "40대", "50대 이상", "unknown"],
                value="unknown",
                label="연령대",
            )

        gr.Markdown("### 증상 등급 입력 (0~3)")

        with gr.Row():
            v1 = gr.Slider(0, 3, step=1, value=0, label="value_1 (각질)")
            v2 = gr.Slider(0, 3, step=1, value=0, label="value_2 (피지)")
            v3 = gr.Slider(0, 3, step=1, value=0, label="value_3 (모낭 사이 홍반)")
        with gr.Row():
            v4 = gr.Slider(0, 3, step=1, value=0, label="value_4 (모낭 홍반/농포)")
            v5 = gr.Slider(0, 3, step=1, value=0, label="value_5 (비듬)")
            v6 = gr.Slider(0, 3, step=1, value=0, label="value_6 (탈모)")

        run_button = gr.Button("리포트 생성")

        output = gr.Textbox(
            lines=20,
            label="LLM 리포트 (데모)",
        )

        run_button.click(
            fn=run_analysis,
            inputs=[gender, age_group, v1, v2, v3, v4, v5, v6],
            outputs=[output],
        )

    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch()
