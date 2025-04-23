export OPENAI_PROVIDER=ollama
export OPENAI_MODEL=ensemble
export OLLAMA_ENDPOINT="http://triton:9000/v1"
export USE_INTERNAL_PROMPT_TEMPLATE=0
export DEFAULT_PROMPT=$(cat <<EOF
Given a block of medical text, generate several direct, succinct, and unique questions that stand alone, focusing on extracting specific medical information such as symptoms, diagnosis, treatment options, or patient management strategies. Each question should aim to elicit precise and informative responses without requiring additional context. The questions should cover diverse aspects of the medical content to ensure a comprehensive understanding. Ensure each question is clear and formulated to be self-contained. Here are examples to guide your question generation:

What are the common symptoms associated with [specific condition]?
How is [specific condition] diagnosed?
What treatment options are available for [specific condition]?
What are the potential side effects of [specific medication]?
What preventive measures are recommended for [specific condition]?

Use these examples as a template, tailoring questions to different parts of the text to maximize the dataset's utility and accuracy. Questions should be separated by a new line and not prefixed by any markers or numbers. Generate up to 5 questions. 
This is the text: {text}.
EOF

)
