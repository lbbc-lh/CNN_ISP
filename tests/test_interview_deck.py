from pathlib import Path
import re


DECK_PATH = Path("docs/cnn-isp-interview-deck.html")


def _extract_h2_titles(html: str) -> list[str]:
    return re.findall(r"<h2>(.*?)</h2>", html)


def test_interview_deck_is_condensed_to_seven_slides():
    html = DECK_PATH.read_text()
    assert html.count('<section class="slide') == 7


def test_interview_deck_uses_the_condensed_storyline():
    html = DECK_PATH.read_text()
    assert _extract_h2_titles(html) == [
        "为什么做这个项目 + 目标边界",
        "系统架构与处理流程",
        "语义分区流程",
        "结构分区流程",
        "实验结果：从分区到建议",
        "难点与下一步",
    ]


def test_problem_slide_mentions_manual_isp_flow_and_goal_shift():
    html = DECK_PATH.read_text()
    for needle in [
        "传统 ISP 调试流程",
        "多设备准备",
        "评测是否通过?",
        "高度依赖人工经验",
        "评测不通过 -> 全流程重跑",
        "完全人工调参",
        "算法自动调一轮 + 人工微调",
        "自动完成 80% 调参工作",
    ]:
        assert needle in html


def test_cover_slide_is_simplified_without_highlight_panel():
    html = DECK_PATH.read_text()
    cover_html = html.split("<!-- ===========================================\n         SLIDE 2: PROBLEM + GOAL", 1)[0]
    assert "项目亮点" not in cover_html
    assert '<div class="hero-panel' not in cover_html


def test_core_design_slides_keep_flow_keywords_but_no_code_details_section():
    html = DECK_PATH.read_text()
    core_html = html.split("<!-- ===========================================\n         SLIDE 4: SEMANTIC FLOW", 1)[1]
    core_html = core_html.split("<!-- ===========================================\n         SLIDE 6: EXPERIMENT", 1)[0]
    for needle in [
        "SegFormer",
        "ADE20K",
        "person / sky / vegetation / background",
        "Sobel",
        "局部方差",
        "flat / edge / texture",
        "argmax",
        "上采样",
    ]:
        assert needle in core_html
    assert "代码里怎么做" not in core_html


def test_interview_deck_describes_the_segmentation_model_as_segformer_not_cnn():
    html = DECK_PATH.read_text()
    for needle in [
        "CNN + ISP + FastAPI",
        "<h1 class=\"reveal\">CNN ISP</h1>",
        "CNN + CV 指标",
    ]:
        assert needle not in html


def test_interview_deck_centers_non_cover_content_vertically():
    html = DECK_PATH.read_text()
    assert ".slide-shell {" in html
    assert "height: 100%;" not in html.split(".slide-shell {", 1)[1].split("}", 1)[0]
    assert "justify-content: center;" in html.split(".slide-shell {", 1)[1].split("}", 1)[0]
    assert ".title-slide .slide-content {" in html
