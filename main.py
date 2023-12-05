from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

template = """Summarize the following question based on the context:

Question: {question}

Context

{context}"""

prompt = ChatPromptTemplate.from_template(template)

def scrape_text(url: str):
    try:
        response = requests.get(url)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")

            page_text = soup.get_text(separator=" ", strip=True)

            return page_text
        else:
            return f"Failed to retrieve the webpage: Status code {response.status.code}"
    except Exception as e:
        print(e)
        return f"Failed to retrieve the webpage: {e}"
    
url = "https://blog.langchain.dev/announcing-langsmith/"

page_content = scrape_text(url)[:10000]

chain = prompt | ChatOpenAI(model="gpt-3.5-turbo-1106") | StrOutputParser()

chain.invoke(
    {
        "question": "what is langsmith",
        "context": page_content
    }
)