import gradio as gr
import pandas as pd
from collections import defaultdict
from dotenv import load_dotenv

from evaluation.eval import evaluate_all_retrieval, evaluate_all_answers

load_dotenv(override=True)

MRR_GREEN = 0.9
MRR_AMBER = 0.75
NDCG_GREEN = 0.9
NDCG_AMBER = 0.75
COVERAGE_GREEN = 90.0
COVERAGE_AMBER = 75.0

ANSWER_GREEN = 4.5
ANSWER_AMBER = 4.0

THEME = gr.themes.Base(
    primary_hue="indigo",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
).set(
    body_background_fill="#0f1117",
    body_background_fill_dark="#0f1117",
    block_background_fill="#161925",
    block_background_fill_dark="#161925",
    block_border_color="#262a38",
    block_border_color_dark="#262a38",
    body_text_color="#e5e7eb",
    body_text_color_dark="#e5e7eb",
    button_primary_background_fill="#5b5fef",
    button_primary_background_fill_hover="#4b4fd9",
    button_primary_text_color="#ffffff",
)

CUSTOM_CSS = """
footer { display: none !important; }
.gradio-container { max-width: 1300px !important; margin: auto; }
#header h1 { font-size: 1.4rem; font-weight: 600; margin-bottom: 0.15rem; }
#header p { color: #9ca3af; font-size: 0.9rem; margin-top: 0; }
"""


def get_color(value: float, metric_type: str) -> str:
    if metric_type == "mrr":
        thresholds = (MRR_GREEN, MRR_AMBER)
    elif metric_type == "ndcg":
        thresholds = (NDCG_GREEN, NDCG_AMBER)
    elif metric_type == "coverage":
        thresholds = (COVERAGE_GREEN, COVERAGE_AMBER)
    elif metric_type in ("accuracy", "completeness", "relevance"):
        thresholds = (ANSWER_GREEN, ANSWER_AMBER)
    else:
        return "#9ca3af"

    green, amber = thresholds
    if value >= green:
        return "#4ade80"
    elif value >= amber:
        return "#fbbf24"
    return "#f87171"


def format_metric_html(label, value, metric_type, is_percentage=False, score_format=False):
    color = get_color(value, metric_type)
    if is_percentage:
        value_str = f"{value:.1f}%"
    elif score_format:
        value_str = f"{value:.2f}/5"
    else:
        value_str = f"{value:.4f}"
    return f"""
    <div style="margin: 10px 0; padding: 15px; background-color: #1d2130; border-radius: 8px; border-left: 4px solid {color};">
        <div style="font-size: 13px; color: #9ca3af; margin-bottom: 5px;">{label}</div>
        <div style="font-size: 26px; font-weight: 600; color: {color};">{value_str}</div>
    </div>
    """


def run_retrieval_evaluation(progress=gr.Progress()):
    total_mrr = total_ndcg = total_coverage = 0.0
    category_mrr = defaultdict(list)
    count = 0

    for test, result, prog_value in evaluate_all_retrieval():
        count += 1
        total_mrr += result.mrr
        total_ndcg += result.ndcg
        total_coverage += result.keyword_coverage
        category_mrr[test.category].append(result.mrr)
        progress(prog_value, desc=f"Evaluating test {count}...")

    avg_mrr = total_mrr / count
    avg_ndcg = total_ndcg / count
    avg_coverage = total_coverage / count

    final_html = f"""
    <div style="padding: 0;">
        {format_metric_html("Mean Reciprocal Rank (MRR)", avg_mrr, "mrr")}
        {format_metric_html("Normalized DCG (nDCG)", avg_ndcg, "ndcg")}
        {format_metric_html("Keyword Coverage", avg_coverage, "coverage", is_percentage=True)}
        <div style="margin-top: 20px; padding: 10px; background-color: #12251c; border-radius: 6px; text-align: center; border: 1px solid #1f4d33;">
            <span style="font-size: 13px; color: #4ade80; font-weight: 600;">Evaluation complete — {count} tests</span>
        </div>
    </div>
    """

    category_data = [
        {"Category": category, "Average MRR": sum(scores) / len(scores)}
        for category, scores in category_mrr.items()
    ]
    return final_html, pd.DataFrame(category_data)


def run_answer_evaluation(progress=gr.Progress()):
    total_accuracy = total_completeness = total_relevance = 0.0
    category_accuracy = defaultdict(list)
    count = 0

    for test, result, prog_value in evaluate_all_answers():
        count += 1
        total_accuracy += result.accuracy
        total_completeness += result.completeness
        total_relevance += result.relevance
        category_accuracy[test.category].append(result.accuracy)
        progress(prog_value, desc=f"Evaluating test {count}...")

    avg_accuracy = total_accuracy / count
    avg_completeness = total_completeness / count
    avg_relevance = total_relevance / count

    final_html = f"""
    <div style="padding: 0;">
        {format_metric_html("Accuracy", avg_accuracy, "accuracy", score_format=True)}
        {format_metric_html("Completeness", avg_completeness, "completeness", score_format=True)}
        {format_metric_html("Relevance", avg_relevance, "relevance", score_format=True)}
        <div style="margin-top: 20px; padding: 10px; background-color: #12251c; border-radius: 6px; text-align: center; border: 1px solid #1f4d33;">
            <span style="font-size: 13px; color: #4ade80; font-weight: 600;">Evaluation complete — {count} tests</span>
        </div>
    </div>
    """

    category_data = [
        {"Category": category, "Average Accuracy": sum(scores) / len(scores)}
        for category, scores in category_accuracy.items()
    ]
    return final_html, pd.DataFrame(category_data)


def main():
    with gr.Blocks(title="RAG Evaluation Dashboard", theme=THEME, css=CUSTOM_CSS) as app:
        with gr.Column(elem_id="header"):
            gr.Markdown("# RAG Evaluation Dashboard")
            gr.Markdown("Retrieval and answer quality evaluation for the Insurellm RAG system.")

        gr.Markdown("### Retrieval Evaluation")
        retrieval_button = gr.Button("Run Evaluation", variant="primary", size="lg")

        with gr.Row():
            with gr.Column(scale=1):
                retrieval_metrics = gr.HTML(
                    "<div style='padding: 20px; text-align: center; color: #6b7280;'>Click 'Run Evaluation' to start</div>"
                )
            with gr.Column(scale=1):
                retrieval_chart = gr.BarPlot(
                    x="Category",
                    y="Average MRR",
                    title="Average MRR by Category",
                    y_lim=[0, 1],
                    height=400,
                )

        gr.Markdown("### Answer Evaluation")
        answer_button = gr.Button("Run Evaluation", variant="primary", size="lg")

        with gr.Row():
            with gr.Column(scale=1):
                answer_metrics = gr.HTML(
                    "<div style='padding: 20px; text-align: center; color: #6b7280;'>Click 'Run Evaluation' to start</div>"
                )
            with gr.Column(scale=1):
                answer_chart = gr.BarPlot(
                    x="Category",
                    y="Average Accuracy",
                    title="Average Accuracy by Category",
                    y_lim=[1, 5],
                    height=400,
                )

        retrieval_button.click(fn=run_retrieval_evaluation, outputs=[retrieval_metrics, retrieval_chart])
        answer_button.click(fn=run_answer_evaluation, outputs=[answer_metrics, answer_chart])

    app.launch(inbrowser=True)


if __name__ == "__main__":
    main()
