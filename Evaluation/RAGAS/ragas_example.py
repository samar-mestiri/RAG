from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

eval_data = Dataset.from_dict({
    "question": ["How many days off for a parent's 60th birthday in my first year?"],
    "answer": ["It's a general family event, so it doesn't qualify for special leave and must be taken as annual leave [HR/special_leave.md]."],
    "contexts": [["Family events such as a parent's 60th or 70th birthday do not qualify for special leave and must be taken as annual leave."]],
    "ground_truth": ["It must be taken as annual leave."],
})

result = evaluate(eval_data, metrics=[
    faithfulness, answer_relevancy, context_precision, context_recall])
print(result)