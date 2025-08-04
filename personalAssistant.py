import sys
import os
import streamlit as st
from logging import PlaceHolder
from sparkai.embedding.spark_embedding import Embeddingmodel
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms.sparkllm import SparkLLM
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel, RunnableBranch
from sparkModel import SparkModel
from sparkai_embedding import SparkAIEmbeddings
from dotenv import load_dotenv, find_dotenv
import re


"""
在检索链qa_history_chain中，数据的传递流程

假设初始输入如下
{
    "input": "南瓜书跟它有什么关系？",
    "chat_history": [
        ("human", "西瓜书是什么？"),
        ("ai", "西瓜书是指周志华老师的《机器学习》一书，是机器学习领域的经典入门教材之一。"),
    ]
}

一、第一步，RunnablePassthrough().assign(context=retrieve_docs)

1、retrieve_docs 分支判断，由于输入包含 chat_history，所以走第二个分支，
   执行 summarize_question_prompt | self.llm | StrOutputParser() | retriever
   
2、已知 summarize_question_prompt 如下
    summarize_question_prompt = ChatPromptTemplate([
        ("system", "请根据聊天记录总结用户最近的问题，如果没有多余的聊天记录，则返回用户的问题。"),
        ("human", "{input}"),
        ("placeholder", "{chat_history}")
    ])
    将输入数据注入到 summarize_question_prompt 中，得到如下
    {
        系统: 请根据聊天记录总结用户最近的问题，如果没有多余的聊天记录，则返回用户的问题。

        用户: 南瓜书跟它有什么关系？

        聊天记录: 
            用户: 西瓜书是什么？
            AI: 西瓜书是指周志华老师的《机器学习》一书，是机器学习领域的经典入门教材之一。
    }
    
3、将 summarize_question_prompt 发送给llm处理，llm可能会返回如下
    "南瓜书与西瓜书的关系是什么？"
    
4、将llm的输出通过StrOutputParser()转换为字符串，然后利用llm完善后的问题检索向量数据库
    retriever.invoke("南瓜书与西瓜书的关系是什么？")
    
5、假设检索返回文档如下
    [
        Document(page_content="南瓜书是《机器学习》的补充教材，与西瓜书形成配套...", metadata={...}),
        Document(page_content="南瓜书详细解释了西瓜书中的数学推导...", metadata={...})
    ]

6、最终，RunnablePassthrough().assign(context=retrieve_docs)的输出如下
    {
        "input": "南瓜书跟它有什么关系？",
        "chat_history": [
            ("human", "西瓜书是什么？"),
            ("ai", "西瓜书是指周志华老师的《机器学习》一书，是机器学习领域的经典入门教材之一。"),
        ],
        "context": [
            Document(page_content="南瓜书是《机器学习》的补充教材，与西瓜书形成配套...", metadata={...}),
            Document(page_content="南瓜书详细解释了西瓜书中的数学推导...", metadata={...})
        ]
    }

第二步，RunnablePassthrough().assign(answer=qa_chain)

1、已知 qa_chain = (
            RunnablePassthrough().assign(context=self.combine_docs)
            | qa_prompt
            | self.llm 
            | StrOutputParser()
        )

2、经过 combine_docs() 方法处理，将检索到的文档内容合并为一个字符串，如下
    "南瓜书是《机器学习》的补充教材，与西瓜书形成配套...

     南瓜书详细解释了西瓜书中的数学推导..."

3、已知 qa_prompt = ChatPromptTemplate([
            ("system", "你是一个问答任务的助手。
                        请使用检索到的上下文片段回答这个问题。
                        如果你不知道答案就说不知道。请使用简洁的话语回答用户。
                        \n\n{context}"),
            ("human", "{input}"),
            ("placeholder", "{chat_history}")
        ])

    将合并后的文档内容和原始输入数据注入到 qa_prompt 中，得到如下
    {
        系统: 你是一个问答任务的助手。请使用检索到的上下文片段回答这个问题。如果你不知道答案就说不知道。请使用简洁的话语回答用户。

        南瓜书是《机器学习》的补充教材，与西瓜书形成配套...

        南瓜书详细解释了西瓜书中的数学推导...

        用户: 南瓜书跟它有什么关系？

        聊天记录: 
            用户: 西瓜书是什么？
            AI: 西瓜书是指周志华老师的《机器学习》一书，是机器学习领域的经典入门教材之一。
    }
    
4、将 qa_prompt 发给 llm，llm基于上下文和聊天历史生成答案，比如
    "南瓜书是西瓜书的补充教材，专门用于详细解释西瓜书中复杂的数学推导过程，两者形成配套关系。"

5、最终，检索链qa_history_chain的输出如下
    {
        "input": "南瓜书跟它有什么关系？",
        "chat_history": [
            ("human", "西瓜书是什么？"),
            ("ai", "西瓜书是指周志华老师的《机器学习》一书，是机器学习领域的经典入门教材之一。"),
        ],
        "context": [
            Document(page_content="南瓜书是《机器学习》的补充教材，与西瓜书形成配套...", metadata={...}),
            Document(page_content="南瓜书详细解释了西瓜书中的数学推导...", metadata={...})
        ],
        "answer": "南瓜书是西瓜书的补充教材，专门用于详细解释西瓜书中复杂的数学推导过程，两者形成配套关系。"
    }
"""


class PersonalAssistant:
    def __init__(self):
        _ = load_dotenv(find_dotenv())
        self.spark_appid=os.environ.get("IFLYTEK_SPARK_APP_ID")
        self.spark_api_key=os.environ.get("IFLYTEK_SPARK_API_KEY")
        self.spark_api_secret=os.environ.get("IFLYTEK_SPARK_API_SECRET")
        self.spark_x1_url=os.environ.get("IFLYTEK_SPARK_X1_URL")
        self.spark_4ultra_url=os.environ.get("IFLYTEK_SPARK_4Ultra_URL")
        self.spark_4ultra_domain=os.environ.get("IFLYTEK_SPARK_4Ultra_DOMAIN")
        # 基于langchain调用大模型
        self.llm = SparkLLM(
            spark_app_id=self.spark_appid,
            spark_api_key=self.spark_api_key,
            spark_api_secret=self.spark_api_secret,
            spark_llm_domain=self.spark_4ultra_domain,
            spark_api_url=self.spark_4ultra_url,
            temperature=0.1
        )
    
    # 获取由向量数据库构建的检索器
    def get_retriever(self):
        # 星火文本向量化
        self.sparkEmbedding = SparkAIEmbeddings()
        self.database_directory = 'data_base/vector_db/chroma'
        vector_db = Chroma(
            embedding_function=self.sparkEmbedding,
            persist_directory=self.database_directory
        )
        # 通过as_retriever方法把向量数据库构造成检索器
        retriever = vector_db.as_retriever()
        return retriever
    
    # 处理检索器返回的文本
    def combine_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs["context"])
    
    # 构建一个带历史聊天记录的检索问答链
    def get_qa_history_chain(self):
        retriever = self.get_retriever()
        summarize_question_system_template = (
            "请根据聊天记录总结用户最近的问题，"
            "如果没有多余的聊天记录，则返回用户的问题。"
        )
        summarize_question_prompt = ChatPromptTemplate([
            ("system", summarize_question_system_template),
            ("human", "{input}"),
            ("placeholder", "{chat_history}")
        ])
        # 利用llm完善用户问题，用来检索向量数据库
        retrieve_docs = RunnableBranch(
            (lambda x : not x.get("chat_history", False), (lambda x : x.get("input") | retriever)),
            summarize_question_prompt | self.llm | StrOutputParser() | retriever
        )
        
        system_prompt = (
            "你是一个问答任务的助手。 "
            "请使用检索到的上下文片段回答这个问题。 "
            "如果你不知道答案就说不知道。 "
            "请使用简洁的话语回答用户。"
            "\n\n"
            "{context}"
        )
        qa_prompt = ChatPromptTemplate([
            ("system", system_prompt),
            ("human", "{input}"),
            ("placeholder", "{chat_history}")
        ])
        qa_chain = (
            RunnablePassthrough().assign(context=self.combine_docs)
            | qa_prompt
            | self.llm 
            | StrOutputParser()
        )
        # RunnablePassthrough().assign() 作用：保持所有原始输入数据不变，
        #                                      并为数据字典添加新的键值对（也可以覆盖现有字段的值）
        qa_history_chain = RunnablePassthrough().assign(
            context = retrieve_docs,
        ).assign(
            answer = qa_chain
        )
        return qa_history_chain
    
    # 接受检索问答链、用户输入及聊天历史，并以流式返回该链输出
    def gen_response(self, chain, input, chat_history):
        response = chain.stream({
            "input": input,
            "chat_history": chat_history
        })
        for res in response:
            if "answer" in res.keys():
                yield res["answer"]

def main():
    assistant = PersonalAssistant()
    st.markdown('### 🦜🔗 动手学大模型应用开发')
    # st.session_state可以存储用户与应用交互期间的状态与数据
    # 存储对话历史
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # 存储检索问答链
    if "qa_history_chain" not in st.session_state:
        st.session_state.qa_history_chain = assistant.get_qa_history_chain()
    # 建立容器 高度为500 px
    messages = st.container(height=550)
    # 显示整个对话历史
    for message in st.session_state.messages: # 遍历对话历史
            with messages.chat_message(message[0]): # messages指在容器下显示，chat_message显示用户及ai头像
                st.write(message[1]) # 打印内容
    if prompt := st.chat_input("Say something"):
        # 将用户输入添加到对话历史中
        st.session_state.messages.append(("human", prompt))
        # 显示当前用户输入
        with messages.chat_message("human"):
            st.write(prompt)
        # 生成回复
        answer = assistant.gen_response(
            chain=st.session_state.qa_history_chain,
            input=prompt,
            chat_history=st.session_state.messages
        )
        # 流式输出
        with messages.chat_message("ai"):
            output = st.write_stream(answer)
        # 将输出存入st.session_state.messages
        st.session_state.messages.append(("ai", output))


if __name__ == "__main__":
    main()
