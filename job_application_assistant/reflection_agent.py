from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from langchain.chat_models.ollama import ChatOllama

llm = None

class AppState(TypedDict):
    job_description: str
    cv_text: str
    cv_format: str
    cv_filename: str
    cover_letter_draft: str
    cover_letter_feedback: str
    cover_letter_final: str
    cv_draft: str
    cv_feedback: str
    cv_final: str
    iteration: int

    @classmethod
    def from_user_input(cls, user_input: dict):
        """Factory to initialize state from parsed user input."""
        return cls(
            job_description=user_input["job_description"],
            cv_text=user_input["cv_text"],
            cv_format=user_input.get("cv_format", "docx"),
            cv_filename=user_input.get("cv_filename", "cv.docx"),
            cover_letter_draft="",
            cover_letter_feedback="",
            cover_letter_final="",
            cv_draft="",
            cv_feedback="",
            cv_final="",
            iteration=0,
        )

def init_cv_and_cl(state: AppState) -> AppState:
    """Initial drafts conditioned on JD + CV."""
    jd = state["job_description"]
    cv = state["cv_text"]

    cl_prompt = f"""
    Using the CV and job description, write a concise, tailored cover letter.
    Focus on the most relevant experiences and quantify impact.
    """
    cv_prompt = f"""
    Rewrite the CV content to better match this job description.
    Keep truthful, highlight the most relevant roles, skills, and results.
    """
    global llm
    cl = llm.invoke([HumanMessage(content=cl_prompt + "\n\nJOB:\n" + jd + "\n\nCV:\n" + cv)]).content
    cv_new = llm.invoke([HumanMessage(content=cv_prompt + "\n\nJOB:\n" + jd + "\n\nCV:\n" + cv)]).content

    state["cover_letter_draft"] = cl
    state["cv_draft"] = cv_new
    state["iteration"] = 0
    return state


def reflect_cover_letter(state: AppState) -> AppState:
    """Critic/reflector for cover letter."""
    review_prompt = """
    Review the cover letter against the job description and CV.
    List concrete, bullet-point suggestions to:
    - Improve relevance
    - Improve clarity/impact
    - Remove repetition
    - Keep length reasonable (max 1 page)
    """
    global llm
    feedback = llm.invoke([
        HumanMessage(content=review_prompt +
                     "\n\nJOB:\n" + state["job_description"] +
                     "\n\nCV:\n" + state["cv_text"] +
                     "\n\nCOVER LETTER:\n" + state["cover_letter_draft"])
    ]).content
    state["cover_letter_feedback"] = feedback
    return state


def revise_cover_letter(state: AppState) -> AppState:
    """Apply feedback to get improved cover letter."""
    prompt = """
    Apply the reviewer feedback to rewrite the cover letter.
    Output only the final improved letter.
    """
    global llm
    improved = llm.invoke([
        HumanMessage(content=prompt +
                     "\n\nJOB:\n" + state["job_description"] +
                     "\n\nCV:\n" + state["cv_text"] +
                     "\n\nCURRENT LETTER:\n" + state["cover_letter_draft"] +
                     "\n\nFEEDBACK:\n" + state["cover_letter_feedback"])
    ]).content
    state["cover_letter_draft"] = improved
    state["iteration"] += 1
    return state


def reflect_cv(state: AppState) -> AppState:
    """Critic/reflector for CV."""
    review_prompt = """
    Review the CV against the job description.
    Suggest:
    - Role reordering to emphasize relevance
    - Bullet improvements with metrics
    - Skills alignment
    - Anything that may be misleading: flag it to keep truthful.
    Reply with clear bullet points.
    """
    global llm
    fb = llm.invoke([
        HumanMessage(content=review_prompt +
                     "\n\nJOB:\n" + state["job_description"] +
                     "\n\nCV (current):\n" + state["cv_draft"])
    ]).content
    state["cv_feedback"] = fb
    return state


def revise_cv(state: AppState) -> AppState:
    """Apply CV feedback."""
    prompt = """
    Rewrite the CV text applying the feedback.
    Preserve factual accuracy. Output full CV text only.
    """
    global llm
    new_cv = llm.invoke([
        HumanMessage(content=prompt +
                     "\n\nJOB:\n" + state["job_description"] +
                     "\n\nCURRENT CV:\n" + state["cv_draft"] +
                     "\n\nFEEDBACK:\n" + state["cv_feedback"])
    ]).content
    state["cv_draft"] = new_cv
    return state


def decide_next(state: AppState) -> str:
    """
    Simple stopping rule:
    - run 2 refinement iterations, then finalize.
    You can instead parse scores from feedback.
    """
    if state["iteration"] >= 2:
        state["cover_letter_final"] = state["cover_letter_draft"]
        state["cv_final"] = state["cv_draft"]
        return END
    return "reflect_cover_letter"  # loop again through CL + CV improvements

def build_agent():

    global llm

    llm = ChatOllama(
        model="mistral",
        base_url="http://localhost:11434/",
        temperature=0.2
    )

    graph = StateGraph(AppState)
    graph.add_node("init", init_cv_and_cl)
    graph.add_node("reflect_cover_letter", reflect_cover_letter)
    graph.add_node("revise_cover_letter", revise_cover_letter)
    graph.add_node("reflect_cv", reflect_cv)
    graph.add_node("revise_cv", revise_cv)

    graph.set_entry_point("init")

    graph.add_edge("init", "reflect_cover_letter")
    graph.add_edge("reflect_cover_letter", "revise_cover_letter")
    graph.add_edge("revise_cover_letter", "reflect_cv")
    graph.add_edge("reflect_cv", "revise_cv")
    graph.add_conditional_edges("revise_cv", decide_next)

    app = graph.compile()

    return app