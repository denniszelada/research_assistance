from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

SUMMARY_TEMPLATE = """{text}

------

Using the above text, answer in short the following question:

> {question}

-------
if the question cannot be answered using the text, imply summarize the text. Include all factual information, numbers, stats, etc."""
SUMMARY_PROMPT = ChatPromptTemplate.from_template(SUMMARY_TEMPLATE)

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

chain = RunnablePassthrough.assign (
    text=lambda x: scrape_text(x["url"])[10000]
) | SUMMARY_PROMPT | ChatOpenAI(model="gpt-3.5-turbo-1106") | StrOutputParser()

chain.invoke(
    {
        "question": "what is langsmith",
        "url": url
    }
)