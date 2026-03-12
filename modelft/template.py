IGNORE_INDEX = -100

user_tokens=[1786, 4194, 95388]  # <用户>

assistant_tokens=[1786, 10850, 95388] # <AI>

pythia_user_tokens=[29, 12335, 46136, 31]  # <用户>

pythia_assistant_tokens=[29, 18128, 31] # <AI>

RESPONSE_START_TOKEN_IDS = [128006, 78191, 128007]  # <|start_header_id|>assistant<|end_header_id|>

# PROMPT_DICT = {
#     "QA_querypassage_to_CoT": (
#         "Passages:{passages}\n"
#         "Based on these passages, answer the question below.\n"
#         "Question:{question}\n"
#         "Let's think step by step."
#     ),
#     "Mutichoice_querypassage_to_CoT": (
#         "Passages:{passages}\n"
#         "Based on these passages, please answer the multiple choice question below.\n"
#         "Question:{question}\n"
#         "Let's think step by step."
#     ),
# }

PROMPT_DICT = {
    "QA_querypassage_to_CoT": (
        "Passage:{passages}\n"
        "Based on these passages, answer the question below.\n"
        "If the passages don't contain the answer, use your own knowledge.\n"
        "Question:{question}\n"
        "Let's think step by step."
    ),
    "Mutichoice_querypassage_to_CoT": (
        "Passage:{passages}\n"
        "Based on these passages, answer the question below.\n"
        "If the passages don't contain the answer, use your own knowledge.\n"
        "Question:{question}\n"
        "Let's think step by step."
    ),
}

# PROMPT_DICT = {
#     "QA_querypassage_to_CoT": (
#         "Question: {question}\n"
#         "Passages: {passages}\n\n"
#         "First, judge if you can answer using only internal knowledge.\n"
#         "If yes, think step by step.\n"
#         "If no, use passages and think step by step.\n\n"
#         "Output format:\n"
#         "only_using_internal_knowledge:[yes/no]\n"
#         "COT: [step-by-step reasoning]\n"
#         "answer: A clear, brief paragraph that directly answers the question.\n\n\n\n"
#     ),
#     "Mutichoice_querypassage_to_CoT": (
#         "Question: {question}\n"
#         "Passages: {passages}\n\n"
#         "First, judge if you can answer using only internal knowledge.\n"
#         "If yes, think step by step.\n"
#         "If no, use passages and think step by step.\n\n"
#         "Output format:\n"
#         "only_using_internal_knowledge:[yes/no]\n"
#         "COT: [step-by-step reasoning]\n"
#         "answer:A clear, brief paragraph that directly answers the question.\n\n\n\n"
        # "Question: {question}\n"
        # "Passages: {passages}\n\n"
        # "First, judge if you can answer using only internal knowledge.\n"
        # "If yes, think step by step and provide CoT.\n"
        # "If no, use passages and include Source Document ID.\n\n"
        # "Output format (if using passages):\n"
        # "Source Document ID: [doc_id]\n"
        # "COT: [step-by-step reasoning]\n\n"
        # "Output format (if using internal knowledge):\n"
        # "COT: [step-by-step reasoning]\n\n"
#     ),
# }