import streamlit as st
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os

load_dotenv()

# 1. 준비된 데이터를 벡터화

loader = CSVLoader(file_path="customer_service.csv",encoding="utf-8")
documents = loader.load()

# 첫 줄 출력
print(len(documents))

# 벡터화를 위해 OpenAIEmbeddings 사용
embeddings = OpenAIEmbeddings()

# FAISS (Facebook AI Similarity Search)를 사용하여 벡터 저장소 생성
db = FAISS.from_documents(documents, embeddings)

def retrieve_info(query):
    similar_response = db.similarity_search(query, k=3) #k=3 가장 유사한 3개

    page_contents_array = [doc.page_content for doc in similar_response]

    return page_contents_array

# customer_message = """
# 안녕하세요. 제 제품을 사용하면서 이상한 소리가 나는데 어떻게 해야 하나요?
# """

# result = retrieve_info(customer_message)
# print(result)

# LLMChain과 프롬프트 준비
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

template = """
당신은 샵투월드의 대표입니다.
나는 잠재 고객의 메시지를 공유할 것이고, 당신은 과거의 가장 결과과 좋은 것을 기반으로 이 잠재 고객에게 보낼 최상의 답변을 제공할 것입니다.
그리고 아래의 규칙을 모두 따를 것입니다:

1/ 답변은 길이, 어조, 논리적 주장 및 기타 세부 사항에서 과거의 가장 좋았던 것과 매우 유사하거나 심지어 동일해야합니다.

2/ 모범사례와 관련이 없다면, 그때는 모범사례의 스타일을 모방하여 잠재 고객의 메시지에 대응하십시오.

아래는 잠재 고객으로부터 받은 메시지입니다:
{message}


우리가 일반적으로 유사한 시나리오에서 잠재 고객에게 응답하는 모범사례 목록은 다음과 같습니다:
{best_practice}

이 잠재 고객에게 보낼 최상의 응답을 작성해주세요:
아래는 나에게 온 메시지입니다:
{message}

아래는 유사한 시나리오에서 우리가 일반적으로 잠재 고객에게 응답하는 모범 사례 목록입니다:
{best_practice}

이 잠재 고객에게 보내야 하는 최선의 응답을 작성해 주세요:
"""

prompt = PromptTemplate(
    input_variables=["message", "best_practice"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)

# 4. 증강 검색 생성
def generate_response(message):
    best_practice = retrieve_info(message)
    response = chain.run(message=message, best_practice=best_practice)
    return response


# message = '''안녕하세요. 
# 어제 도착한 상품이 매우 만족스럽습니다. 감사합니다!
# 이제품을 또 구하고 싶은데요?
# '''

# response = generate_response(message)
# print(response)

#스트림릿 작업

def main():
    st.set_page_config(
        page_title="신입사원 도우미-고객 문의 모범 답안", page_icon=":dog:")

    st.header("신입사원 도우미-고객 문의 모범 답안 :dog:")
    message = st.text_area("고객의 문의 내용을 여기 넣으시면 선배가 엄청나게 잘 알려줌")

    if message:
        st.write("모범 응답 생성 중...")

        result = generate_response(message)

        st.info(result)


if __name__ == '__main__':
    main()
