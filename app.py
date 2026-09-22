import os
import gradio as gr
from dotenv import load_dotenv

from implementation.answer import answer_question

load_dotenv(override=True)

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
    border_color_primary="#262a38",
    border_color_primary_dark="#262a38",
    body_text_color="#e5e7eb",
    body_text_color_dark="#e5e7eb",
    body_text_color_subdued="#9ca3af",
    button_primary_background_fill="#5b5fef",
    button_primary_background_fill_hover="#4b4fd9",
    button_primary_text_color="#ffffff",
    input_background_fill="#1d2130",
    input_background_fill_dark="#1d2130",
)

CUSTOM_CSS = """
footer { display: none !important; }
.gradio-container { max-width: 1200px !important; margin: auto; }
#header { margin-bottom: 0.5rem; }
#header h1 {
    font-size: 1.4rem;
    font-weight: 600;
    margin-bottom: 0.15rem;
}
#header p {
    color: #9ca3af;
    font-size: 0.9rem;
    margin-top: 0;
}
.score-tag {
    display: inline-block;
    font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    color: #a5b4fc;
    background: #1d2130;
    border: 1px solid #2f3346;
    border-radius: 4px;
    padding: 1px 6px;
    margin-left: 6px;
}
.source-label {
    font-size: 0.8rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}
"""


def format_context(context):
    if not context:
        return "No context retrieved yet. Ask a question to see the chunks used to answer it."

    blocks = []
    for i, doc in enumerate(context, start=1):
        source = doc.metadata.get("source", "unknown")
        score = doc.metadata.get("relevance_score")
        score_html = (
            f'<span class="score-tag">score {score:.2f}</span>'
            if score is not None
            else ""
        )
        blocks.append(
            f'<div class="source-label">Chunk {i} &middot; {source}{score_html}</div>\n\n'
            f"{doc.page_content}\n\n---\n"
        )
    return "\n".join(blocks)


def chat(history):
    last_message = history[-1]["content"]
    prior = history[:-1]
    answer, context = answer_question(last_message, prior)
    history.append({"role": "assistant", "content": answer})
    return history, format_context(context)


def main():
    def put_message_in_chatbot(message, history):
        return "", history + [{"role": "user", "content": message}]

    with gr.Blocks(
        title="Insurellm Assistant",
        theme=THEME,
        css=CUSTOM_CSS
    ) as ui:
        with gr.Column(elem_id="header"):
            gr.Markdown("# Insurellm Assistant")
            gr.Markdown(
                "Retrieval-augmented Q&A over the Insurellm knowledge base."
            )

        with gr.Row():
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=560,
                    type="messages",
                    show_copy_button=True,
                    avatar_images=None,
                )
                message = gr.Textbox(
                    placeholder="Ask a question about Insurellm...",
                    show_label=False,
                )

            with gr.Column(scale=1):
                context_markdown = gr.Markdown(
                    label="Retrieved context",
                    value=format_context([]),
                    container=True,
                    height=560,
                )

        message.submit(
            put_message_in_chatbot,
            inputs=[message, chatbot],
            outputs=[message, chatbot],
        ).then(
            chat,
            inputs=chatbot,
            outputs=[chatbot, context_markdown],
        )

    ui.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 10000)),
    )


if __name__ == "__main__":
    main()
