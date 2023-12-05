from langchain.chat.models import ChatOpanAI
from langchain.prompts import ChatPromptTemplate

template = """Summarize the following question based on the context:

Question: {question}

Context

{context}"""

prompt = ChatPromptTemplate.from_template(template)

url = "https://blog.langchain.dev/announcing-langsmith/"