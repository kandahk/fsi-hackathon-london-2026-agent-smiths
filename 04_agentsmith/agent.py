from langgraph.graph import END, START, StateGraph

from utils.nodes import (
    auto_clear,
    detect_chain,
    enrich_bank_a,
    enrich_bank_b,
    enrich_bank_c,
    human_review,
    load_trigger,
    route_decision,
    score_risk,
    write_audit,
)
from utils.state import AgentContext, AgentInput, AgentState


def build_agent(**kwargs):
    builder = StateGraph(
        state_schema=AgentState,
        context_schema=AgentContext,
        input_schema=AgentInput,
    )

    builder.add_node("load_trigger", load_trigger)
    builder.add_node("detect_chain", detect_chain)
    builder.add_node("enrich_bank_a", enrich_bank_a)
    builder.add_node("enrich_bank_b", enrich_bank_b)
    builder.add_node("enrich_bank_c", enrich_bank_c)
    builder.add_node("score_risk", score_risk)
    builder.add_node("human_review", human_review)
    builder.add_node("auto_clear", auto_clear)
    builder.add_node("write_audit", write_audit)

    builder.add_edge(START, "load_trigger")
    builder.add_edge("load_trigger", "detect_chain")
    builder.add_edge("detect_chain", "enrich_bank_a")
    builder.add_edge("enrich_bank_a", "enrich_bank_b")
    builder.add_edge("enrich_bank_b", "enrich_bank_c")
    builder.add_edge("enrich_bank_c", "score_risk")
    builder.add_conditional_edges("score_risk", route_decision)
    builder.add_edge("human_review", "write_audit")
    builder.add_edge("auto_clear", "write_audit")
    builder.add_edge("write_audit", END)

    return builder.compile(**kwargs)


agent = build_agent()
