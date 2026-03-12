IGNORE_INDEX = -100

user_tokens=[1786, 4194, 95388]  # <用户>

assistant_tokens=[1786, 10850, 95388] # <AI>

pythia_user_tokens=[29, 12335, 46136, 31]  # <用户>

pythia_assistant_tokens=[29, 18128, 31] # <AI>

RESPONSE_START_TOKEN_IDS = [128006, 78191, 128007]  # <|start_header_id|>assistant<|end_header_id|>

PROMPT_DICT = {
    "QA_querypassage_to_CoT": (
        "Passages:{passages}\n"
        "Based on these passages, answer the question below.\n"
        "If the passages don't contain the answer, use your own knowledge.\n"
        "Question:{question}\n"
        "Give a long-form ,complete and well-reasoned answer including all correct answers."
        "Let's think step by step."
    ),
    "QA_queryCoT_to_answer": (
        "Be concise and to the point\n"
        "Answer should be 1-3 sentences\n"
        # "Use the most relevant information from both sources\n"
        # "Avoid unnecessary details and repetition\n"
        "If the chain of thought don't work, please answer the question based on your own knowledge.\n"
        "Please answer the question and only output the answer."
        "Question: {question}\n"
        "Chain of Thought: {CoT}"
    ),
    # "QA_queryCoT_to_answer": (
    #     "Task Description:\n"
    #     "1. Read the given question and related chain of thought to gather relevant information.\n"
    #     "2. The content of the chain of thought is the thinking process that may be used to answer the question.\n"
    #     "3. If the chain of thought don't work, please answer the question based on your own knowledge.\n"
    #     "4. Please answer the question and only output the answer.\n "
    #     "Question:{question}\n"
    #     "Chain of Thought:{CoT}"
    # ),
    "QA_queryCoT_to_answer_forrouge": (
        "Task Description:\n"
        "Read the given question and related chain of thought to gather relevant information.\n"
        "The content of the chain of thought is the thinking process that may be used to answer the question.\n"
        "If the chain of thought don't work, please answer the question based on your own knowledge.\n"
        "Give a short ,clear and brief answer to a given question.\n"
        "Chain of Thought:{CoT}\n "
        "Question:{question}\n"
    ),
    "QA_queryCoT_to_answer_forasqa": (
        "Task Description:\n"
        "Read the given question and related chain of thought to gather relevant information.\n"
        "The content of the chain of thought is the thinking process that may be used to answer the question.\n"
        "If the chain of thought don't work, please answer the question based on your own knowledge.\n"
        "Answer the following question. The question may be ambiguous and have multiple correct answers,you have to provide a long-form answer including all correct answers.\n "
        "Question:{question}\n"
        "Chain of Thought:{CoT}"
    ),
    "vanilla_RAG":(
        "Background:{Passages}\nQuestion:{Question}Answer:"
    ),
}



# PROMPT_DICT = {
#     "QA_querypassage_to_CoT": (
#         # "Passages:{passages}\n"
#         # "Based on these passages, answer the question below.\n"
#         # "Question:{question}\n"
#         # "Let's think step by step."
        
#         # "Question: {question}\n"
#         # "Passages: {passages}\n\n"
#         # "First, judge if you can answer using only internal knowledge.\n"
#         # "If yes, think step by step.\n"
#         # "If no, use passages and think step by step.\n\n"
#         # "Output format:\n"
#         # "only_using_internal_knowledge:[yes/no]"
#         # "COT: [step-by-step reasoning]\n\n"
        
#         # "Passages:{passages}\n"
#         # "Based on these passages, answer the question below.\n"
#         # "If the passages don't contain the answer, use your own knowledge.\n"
#         # "Question:{question}\n"
#         # "Give a long-form ,complete and well-reasoned answer including all correct answers."
#         # "Let's think step by step."
        
#         # "Passage:{passages}\n"
#         # "Based on these passages, answer the question below.\n"
#         # "Question:{question}\n"
        
#         # "Let's think step by step."
        
        
#         # "Question: {question}\n"
#         # "Passages: {passages}\n\n"
#         # "First, judge if you can answer using only internal knowledge.\n"
#         # "If yes, think step by step.\n"
#         # "If no, use passages and think step by step.\n\n"
#         # "Output format:\n"
#         # "only_using_internal_knowledge:[yes/no]\n"
#         # "COT: [step-by-step reasoning]\n"
#         # "answer: A clear, brief paragraph that directly answers the question.\n\n\n\n"
#     ),
#     # "QA_queryCoT_to_answer": (
#     #     # "Task Description:\n"
#     #     # "1. Read the given question and related chain of thought to gather relevant information.\n"
#     #     # "2. The content of the chain of thought is the thinking process that may be used to answer the question.\n"
#     #     # "3. Extract the final answer from the chain of thought.\n"
#     #     # "4. If the chain of thought doesn't contain a clear answer, provide the answer based on your own knowledge.\n"
#     #     # "5. Only output the final answer without any additional text.\n\n"
#     #     # "Focus on the key information and provide a direct answer.\n\n"
#     #     # "Requirements:\n"
#     #     # "1. Be concise and to the point\n"
#     #     # "2. Answer should be 1-3 sentences maximum\n"
#     #     # "3. Use the most relevant information from both sources\n"
#     #     # "4. Avoid unnecessary details and repetition\n"
#     #     # "If the provided information is insufficient, you may use your own knowledge to answer the question.\n"
#     #     # "Format: A clear, brief paragraph that directly answers the question.\n\n"
        
#     #     # "Question: {question}\n"
#     #     # "Chain of Thought: {CoT}"
        
#     #     "Focus on the key information and provide a clear , direct answer.\n\n"
#     #     "Requirements:\n"
#     #     "1. Be concise and to the point\n"
#     #     "2. Answer should be 1-3 sentences\n"
#     #     # "2. Use the most relevant information from both sources\n"
#     #     "3. Avoid unnecessary details and repetition\n"
#     #     "If the provided information is insufficient, you may use your own knowledge to answer the question.\n"
#     #     # "Format: A clear, brief paragraph that directly answers the question.\n\n"
#     #     "Give a long-form ,complete and well-reasoned answer including all correct answers."
#     #     "Question: {question}\n"
#     #     "Chain of Thought: {CoT}"
#     # ),
#     "QA_queryCoT_to_answer": (
#         "Task Description:\n"
#         "1. Read the given question and related chain of thought to gather relevant information.\n"
#         "2. The content of the chain of thought is the thinking process that may be used to answer the question.\n"
#         "3. If the chain of thought don't work, please answer the question based on your own knowledge.\n"
#         "4. Please answer the question and only output the answer.\n "
#         "Question:{question}\n"
#         "Chain of Thought:{CoT}"
#     ),
#     "QA_queryCoT_to_answer_forrouge": (
#         "Task Description:\n"
#         "1. Read the given question and related chain of thought to gather relevant information.\n"
#         "2. The content of the chain of thought is the thinking process that may be used to answer the question.\n"
#         "3. If the chain of thought don't work, please answer the question based on your own knowledge.\n"
#         "4. Give a short ,clear and brief answer to a given question.\n"
#         "Chain of Thought:{CoT}\n "
#         "Question:{question}\n"
#     ),
#     "QA_queryCoT_to_answer_forasqa": (
#         "Task Description:\n"
#         "1. Read the given question and related chain of thought to gather relevant information.\n"
#         "2. The content of the chain of thought is the thinking process that may be used to answer the question.\n"
#         "3. If the chain of thought don't work, please answer the question based on your own knowledge.\n"
#         "4. Answer the following question. The question may be ambiguous and have multiple correct answers, and in that case, you have to provide a long-form answer including all correct answers.\n "
#         "Question:{question}\n"
#         "Chain of Thought:{CoT}"
#     ),
#     "vanilla_RAG":(
#         "Background:{Passages}\nQuestion:{Question}Answer:"
#     ),
# }