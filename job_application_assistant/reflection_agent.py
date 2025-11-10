from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

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

def build_agent():

    def init_cv_and_cl(state: AppState) -> AppState:
        jd = state["job_description"]
        cv = state["cv_text"]

        cl_prompt = """
        Using the CV and job description, write a concise, tailored cover letter.
        Focus heavily on the most relevant experience and concrete achievements.
        Keep it truthful and under 1 page.
        """
        cv_prompt = """
        Rewrite the CV content to better match this job description.
        - Reorder experiences by relevance.
        - Strengthen bullets with metrics when plausible.
        - Do NOT invent skills or roles.
        Output full CV text.
        """

        cl = draft_llm.invoke([
            HumanMessage(content=cl_prompt + "\n\nJOB:\n" + jd + "\n\nCV:\n" + cv)
        ]).content

        cv_new = draft_llm.invoke([
            HumanMessage(content=cv_prompt + "\n\nJOB:\n" + jd + "\n\nCV:\n" + cv)
        ]).content

        state["cover_letter_draft"] = cl
        state["cv_draft"] = cv_new
        state["iteration"] = 0
        return state

    # 2) Reviewer for cover letter using stronger reasoning model
    def reflect_cover_letter(state: AppState) -> AppState:
        jd = state["job_description"]
        cv = state["cv_text"]
        cl = state["cover_letter_draft"]

        review_prompt = """
        You are a critical reviewer.
        Assess the cover letter vs the job description and CV.

        Provide bullet-point feedback on:
        - Relevance to key requirements
        - Clarity & structure
        - Evidence / metrics
        - Tone & conciseness

        Be specific and actionable.
        """
        fb = revise_llm.invoke([
            HumanMessage(
                content=review_prompt
                + "\n\nJOB:\n" + jd
                + "\n\nCV:\n" + cv
                + "\n\nCOVER LETTER:\n" + cl
            )
        ]).content

        state["cover_letter_feedback"] = fb
        return state

    def revise_cover_letter(state: AppState) -> AppState:
        jd = state["job_description"]
        cv = state["cv_text"]
        cl = state["cover_letter_draft"]
        fb = state["cover_letter_feedback"]

        revise_prompt = """
        Rewrite the cover letter applying the feedback.
        Constraints:
        - 1 page max
        - Strong alignment with role
        - Concrete achievements
        - Truthful; do not add fictitious experience
        Output ONLY the final letter.
        """
        improved = revise_llm.invoke([
            HumanMessage(
                content=revise_prompt
                + "\n\nJOB:\n" + jd
                + "\n\nCV:\n" + cv
                + "\n\nCURRENT LETTER:\n" + cl
                + "\n\nFEEDBACK:\n" + fb
            )
        ]).content

        state["cover_letter_draft"] = improved
        state["iteration"] += 1
        return state

    # 3) Reviewer for CV using stronger model
    def reflect_cv(state: AppState) -> AppState:
        jd = state["job_description"]
        cv = state["cv_draft"]

        review_prompt = """
        You are optimizing a CV for this job.
        Provide bullet-point suggestions on:
        - Which roles/bullets to move up or down
        - Where to add or sharpen metrics
        - Where to trim irrelevant content
        - Skills section alignment
        Do NOT suggest anything untruthful.
        """
        fb = revise_llm.invoke([
            HumanMessage(
                content=review_prompt
                + "\n\nJOB:\n" + jd
                + "\n\nCV:\n" + cv
            )
        ]).content

        state["cv_feedback"] = fb
        return state

    def revise_cv(state: AppState) -> AppState:
        jd = state["job_description"]
        cv = state["cv_draft"]
        fb = state["cv_feedback"]

        revise_prompt = """
        Rewrite the CV applying the feedback.
        Keep all experience truthful.
        Emphasize achievements, impact, and relevance to the job.
        Output ONLY the full CV text.
        """
        new_cv = revise_llm.invoke([
            HumanMessage(
                content=revise_prompt
                + "\n\nJOB:\n" + jd
                + "\n\nCURRENT CV:\n" + cv
                + "\n\nFEEDBACK:\n" + fb
            )
        ]).content

        state["cv_draft"] = new_cv
        return state

    # 4) Simple stopping rule
    def decide_next(state: AppState) -> str:
        if state["iteration"] >= 2:
            state["cover_letter_final"] = state["cover_letter_draft"]
            state["cv_final"] = state["cv_draft"]
            return END
        return "reflect_cover_letter"

    # Fast drafter
    draft_llm = ChatOllama(
        model="phi3.5",
        temperature=0.7,
    )

    # Stronger reviser
    revise_llm = ChatOllama(
        model="qwen2.5:7b-instruct-q4_0",
        temperature=0.1,
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

AGENT = build_agent()
