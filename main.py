from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
import requests
import json
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.utilities import DuckDuckGoSearchAPIWrapper


load_dotenv()

RESULTS_PER_QUESTION = 3

ddg_search = DuckDuckGoSearchAPIWrapper()

def web_search(query: str, num_results: int = RESULTS_PER_QUESTION):
    results = ddg_search.results(query, num_results)
    return [r["link"] for r in results]

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
    
scrape_and_summarize_chain = RunnablePassthrough.assign (
    text=lambda x: scrape_text(x["url"])[10000]
) | SUMMARY_PROMPT | ChatOpenAI(model="gpt-3.5-turbo-1106") | StrOutputParser()

web_search_chain = RunnablePassthrough.assign(
    urls = lambda x: web_search(x["question"])
) | (lambda x: [{"question": x["question"], "url": u} for u in x["urls"]]) | scrape_and_summarize_chain.map()

SEARCH_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "user",
            "Write 3 google search queries to search online that form an "
            "objective opinion from the following: {question}\n"
            "You must respond with a list of strings in the following format: "
            '["query1", "query2", "query3"].',
        ),
    ]
)

search_question_chain = SEARCH_PROMPT | ChatOpenAI(temperature=0) | StrOutputParser() | json.loads

chain = search_question_chain | (lambda x: [{"question": q} for q in x]) | web_search_chain.map()
chain.invoke(
    {
        "question": "what is the difference between langsmith and langchain",
    }
)