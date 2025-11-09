import gradio as gr
import os
from io import BytesIO

from docx import Document
from reflection_agent import AppState, build_agent

def parse_cv_file(cv_file):
    """
    Parse CV content from a Gradio File input.
    Supports:
      - .docx
      - .tex
    Accepts both NamedString (filepath wrapper) and plain str paths.
    Returns: (meta, text) or (None, None) on unsupported format.
    """
    if cv_file is None:
        return None, None

    # Gradio File usually returns a NamedString with .name = temp path
    # but handle str just in case.
    path = getattr(cv_file, "name", cv_file)
    filename = os.path.basename(path).lower()

    if filename.endswith(".docx"):
        # python-docx can take the path directly
        doc = Document(path)
        text = "\n".join(p.text for p in doc.paragraphs)
        fmt = "docx"

    elif filename.endswith(".tex"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        fmt = "tex"

    else:
        return None, None

    meta = {
        "format": fmt,
        "filename": os.path.basename(path),
    }
    return meta, text


def build_docx_from_text(text: str, filename: str = "document.docx") -> BytesIO:
    """
    Build an in-memory .docx file from plain text.
    """
    doc = Document()
    for line in text.splitlines():
        doc.add_paragraph(line)
    bio = BytesIO()
    doc.save(bio)
    bio.seek(0)
    bio.name = filename
    return bio


def run_langgraph_agent(agent, user_input: dict):
    """
    Plug your LangGraph reflection workflow here.

    Expected to return:
        cover_letter_text (str)
        improved_cv_text (str)

    For now this is a placeholder.
    """

    initial_state = AppState.from_user_input(user_input)
    final_state = agent.invoke(initial_state)
    return (
        final_state.get("cover_letter_final") or final_state.get("cover_letter_draft", ""),
        final_state.get("cv_final") or final_state.get("cv_draft", "")
    )

def process_submission(agent, job_description, cv_file):
    """
    Called when user clicks SUBMIT.
    - Validates inputs
    - Parses CV
    - Runs LangGraph agent (stub here)
    - Returns:
        - Cover letter textbox content
        - Downloadable cover letter .docx
        - Downloadable improved CV (.docx or .tex)
    """
    if not job_description:
        return (
            "Please paste the job description.",
            None,
            None,
        )

    if cv_file is None:
        return (
            "Please upload your CV in .docx or .tex format.",
            None,
            None,
        )

    meta, cv_text = parse_cv_file(cv_file)
    if meta is None:
        return (
            "Unsupported CV format. Please upload a .docx or .tex file.",
            None,
            None,
        )

    # This is the structured "user input" for the agent
    user_input = {
        "job_description": job_description,
        "cv_text": cv_text,
        "cv_format": meta["format"],
        "cv_filename": meta["filename"],
    }

    # Call your LangGraph reflection agent here
    cover_letter_text, improved_cv_text = run_langgraph_agent(agent, user_input)

    # Build downloadable cover letter (.docx)
    cl_docx = build_docx_from_text(
        cover_letter_text,
        filename="cover_letter.docx"
    )

    # Build downloadable refined CV in original format
    if meta["format"] == "docx":
        refined_cv = build_docx_from_text(
            improved_cv_text,
            filename=meta["filename"]  # preserve original name
        )
    else:  # .tex
        bio = BytesIO(improved_cv_text.encode("utf-8"))
        bio.seek(0)
        bio.name = meta["filename"]
        refined_cv = bio

    return cover_letter_text, cl_docx, refined_cv

def main():
    agent = build_agent()
    with gr.Blocks() as demo:
        gr.Markdown("# Job Application Assistant")

        with gr.Row():
            # LEFT: INPUTS
            with gr.Column():
                gr.Markdown("## Inputs")

                job_desc = gr.Textbox(
                    label="Job Description",
                    placeholder="Paste the job description here...",
                    lines=14,
                )

                cv_upload = gr.File(
                    label="CV Upload (.docx or .tex)",
                    file_types=[".docx", ".tex"],
                )

                submit_btn = gr.Button("Submit", variant="primary")

            # RIGHT: OUTPUTS
            with gr.Column():
                gr.Markdown("## Outputs")

                cover_letter_box = gr.Textbox(
                    label="Cover Letter (you can edit this)",
                    lines=18,
                    interactive=True,
                )

                download_cl_btn = gr.DownloadButton(
                    label="Download Cover Letter (.docx)",
                    value="cover_letter.docx"
                )

                download_cv_btn = gr.DownloadButton(
                    label="Download Refined CV (original format)",
                    value="refined_cv"
                )

        # Wire submit -> processing + outputs
        submit_btn.click(
            fn=process_submission,
            inputs=[agent, job_desc, cv_upload],
            outputs=[cover_letter_box, download_cl_btn, download_cv_btn],
        )

    demo.launch()

# this job requires experience in C++ and cmake

if __name__ == "__main__":
    main()