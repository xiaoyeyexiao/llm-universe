import sys
import os
from sparkai.embedding.spark_embedding import Embeddingmodel
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms.sparkllm import SparkLLM
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
from sparkModel import SparkModel
from sparkai_embedding import SparkAIEmbeddings
from dotenv import load_dotenv, find_dotenv
import re

"""
1、invoke()的参数一般是字典
2、一个 ChatMessageTemplate 一般是一个列表，列表中的每一个元素都是一个二元组
3、LCEL中要求所有的组成元素都是Runnable类型，前面我们见过的ChatModel、PromptTemplate等都是继承自Runnable类
"""

class AgentTest:
    def __init__(self):
        # 读取本地/项目的环境变量。
        # find_dotenv()寻找并定位.env文件的路径
        # load_dotenv()读取该.env文件，并将其中的环境变量加载到当前的运行环境中  
        # 如果你设置的是全局的环境变量，这行代码则没有任何作用。
        _ = load_dotenv(find_dotenv())
        self.spark_appid=os.environ.get("IFLYTEK_SPARK_APP_ID")
        self.spark_api_key=os.environ.get("IFLYTEK_SPARK_API_KEY")
        self.spark_api_secret=os.environ.get("IFLYTEK_SPARK_API_SECRET")
        self.spark_x1_url=os.environ.get("IFLYTEK_SPARK_X1_URL")
        self.spark_x1_domain=os.environ.get("IFLYTEK_SPARK_X1_DOMAIN")
        self.spark_4ultra_url=os.environ.get("IFLYTEK_SPARK_4Ultra_URL")
        self.spark_4ultra_domain=os.environ.get("IFLYTEK_SPARK_4Ultra_DOMAIN")
        # 星火大模型
        self.sparkModel = SparkModel()
        # 星火文本向量化
        self.sparkEmbedding = SparkAIEmbeddings()
        self.database_directory = 'data_base/vector_db/chroma'
        # 基于langchain调用大模型
        self.llm = SparkLLM(
            spark_app_id=self.spark_appid,
            spark_api_key=self.spark_api_key,
            spark_api_secret=self.spark_api_secret,
            spark_llm_domain=self.spark_4ultra_domain,
            spark_api_url=self.spark_4ultra_url,
            temperature=0.1
        )

    # 调用模型，简单的对话
    def chat(self):
        print("欢迎使用 Agent 应用！输入 exit 退出。\n")
        while True:
            userInput = input("请输入你的问题：")
            if userInput.strip().lower() == "exit":
                print("再见！")
                break
            
            # 调用大模型API
            answer = self.sparkModel.chat(userInput)
            
            # 打印模型完整回答
            print("模型回答：", end="")
            print(answer, end="")
            print("\n")

    def test_embedding(self):
        xunfei_embedding = Embeddingmodel(
            spark_embedding_app_id=os.environ.get("IFLYTEK_SPARK_APP_ID"),
            spark_embedding_api_key=os.environ.get("IFLYTEK_SPARK_API_KEY"),
            spark_embedding_api_secret=os.environ.get("IFLYTEK_SPARK_API_SECRET"),
            spark_embedding_domain="para"
        )

        text = {"content":'要生成 embedding 的输入文本。',"role":"user"}
        response = xunfei_embedding.embedding(text=text)
        
        print(f'生成的embedding长度为：{len(response)}')
        print(f'embedding（前10）为: {response[:10]}')

    def data_process(self):
        """
        * 数据读取
        """
        # 创建一个 PyMuPDFLoader Class 实例，输入为待加载的 pdf 文档路径
        pdf_loader = PyMuPDFLoader("临时文件/组会3.pdf")
        # 调用 PyMuPDFLoader Class 的函数 load 对 pdf 文件进行加载
        pdf_pages = pdf_loader.load()
        pdf_page = pdf_pages[1]
        
        """
        * 数据清洗
        * 
        """
        # 正则表达式模式：匹配非中文字符之间的换行符
        # [^\u4e00-\u9fff] 表示非中文字符，(\n) 表示换行符
        # 这个模式会找到两个非中文字符之间的换行符，并将其删除
        pattern = re.compile(r'[^\u4e00-\u9fff](\n)[^\u4e00-\u9fff]', re.DOTALL)
        pdf_page.page_content = re.sub(pattern, lambda match: match.group(0).replace('\n', ''), pdf_page.page_content)
        # 删除文本中的项目符号 "•"
        pdf_page.page_content = pdf_page.page_content.replace('•', '')
        # 删除文本中的所有空格字符
        pdf_page.page_content = pdf_page.page_content.replace(' ', '')
        
        # print(f"载入后的变量类型为：{type(pdf_pages)}，该 PDF 一共包含 {len(pdf_pages)} 页",
        #       f"每一个元素的类型：{type(pdf_page)}.", 
        #       f"该文档的描述性数据：{pdf_page.metadata}", 
        #       f"查看该文档的内容:\n{pdf_page.page_content}", 
        #       sep="\n------\n")
        
        ''' 
        * 文档分割
        * RecursiveCharacterTextSplitter 递归字符文本分割
          RecursiveCharacterTextSplitter 将按不同的字符递归地分割(按照这个优先级["\n\n", "\n", " ", ""])，
          这样就能尽量把所有和语义相关的内容尽可能长时间地保留在同一位置
          RecursiveCharacterTextSplitter需要关注的是4个参数：

        * separators - 分隔符字符串数组
        * chunk_size - 每个文档的字符数量限制
        * chunk_overlap - 两份文档重叠区域的长度
        * length_function - 长度计算函数
        '''
        # 知识库中单段文本长度
        CHUNK_SIZE = 500
        # 知识库中相邻文本重合长度
        OVERLAP_SIZE = 50
        
        # 使用递归字符文本分割器
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=OVERLAP_SIZE
        )
        text_splitter.split_text(pdf_page.page_content[0:1000])
        split_docs = text_splitter.split_documents(pdf_pages)
        print(f"切分后的文件数量：{len(split_docs)}")
        print(f"切分后的字符数（可以用来大致评估 token 数）：{sum([len(doc.page_content) for doc in split_docs])}")

    def create_database(self):
        
        print("appId: ", os.environ.get("IFLYTEK_SPARK_APP_ID"))
        
        file_paths = []
        folder_path = '临时文件'
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_paths.append(file_path)
        print(file_paths[:3])
        
        # 遍历文件路径并把实例化的loader存放在loaders里
        loaders = []
        for file_path in file_paths:
            file_type = file_path.split('.')[-1]
            if file_type == 'pdf':
                loaders.append(PyMuPDFLoader(file_path))
            elif file_type == 'md':
                loaders.append(UnstructuredMarkdownLoader(file_path))
                
        # 下载文件并存储到text
        texts = []
        for loader in loaders: texts.extend(loader.load())

        # 切分文档
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50)
        split_docs = text_splitter.split_documents(texts)
        
        # 构建Chrome向量库
        # 定义持久化路径
        vector_db = Chroma.from_documents(
            documents=split_docs,
            embedding=self.sparkEmbedding,
            persist_directory=self.database_directory  # 允许我们将persist_directory目录保存到磁盘上
        )
        
        print(f"向量库中存储的数量：{vector_db._collection.count()}")

    def query_database(self):
        vector_db = Chroma(
            persist_directory=self.database_directory, 
            embedding_function=self.sparkEmbedding
        )
        question="文章"
        
        # 按余弦相似度排序进行搜索
        sim_docs = vector_db.similarity_search(question,k=3)
        print(f"检索到的内容数：{len(sim_docs)}")
        for i, sim_doc in enumerate(sim_docs):
            print(f"检索到的第{i}个内容: \n{sim_doc.page_content[:200]}", end="\n--------------\n")
        
        # 最大边际相关性 (MMR, Maximum marginal relevance) 可以帮助我们在保持相关性的同时，增加内容的丰富度。
        # 核心思想是在已经选择了一个相关性高的文档之后，再选择一个与已选文档相关性较低但是信息丰富的文档。
        # 这样可以在保持相关性的同时，增加内容的多样性，避免过于单一的结果。
        mmr_docs = vector_db.max_marginal_relevance_search(question,k=3)
        for i, sim_doc in enumerate(mmr_docs):
            print(f"MMR 检索到的第{i}个内容: \n{sim_doc.page_content[:200]}", end="\n--------------\n")

    def langchain_chain(self):
        # system提示词模板和用户输入提示词模板
        system_template = "你是一个翻译助手，可以帮助我将 {input_language} 翻译成 {output_language}。\
                           请最终给出一条最符合的翻译结果，不要任何解释或其他内容。"
        human_template = "{text}"
        # 提示词模板，一个 ChatPromptTemplate 是一个 ChatMessageTemplate 的列表。
        # 每个 ChatMessageTemplate 包含格式化该聊天消息的说明（其角色以及内容）。
        chat_prompt = ChatPromptTemplate([
            ("system", system_template),
            ("human", human_template),
        ])
        
        # 基于langchain调用大模型
        llm = SparkLLM(
            spark_app_id=self.spark_appid,
            spark_api_key=self.spark_api_key,
            spark_api_secret=self.spark_api_secret,
            spark_llm_domain=self.spark_4ultra_domain,
            spark_api_url=self.spark_4ultra_url,
            temperature=0.1
        )

        # 输出解析器，将模型的输出解析为需要的格式，例如字符串、JSON 等
        output_parser = StrOutputParser()
        
        # LCEL 将不同的组件拼凑成一个链，前一个组件的输出作为下一个组件的输入
        # 用法示例：chain = prompt | model | output_parser
        # 该链将获取输入变量 text，将这些变量传递给提示模板 chat_prompt 以创建提示，
        # 再将提示传递给语言模型 llm，然后通过（可选）输出解析器 output_parser 传递输出
        chain = chat_prompt | llm | output_parser
        
        # 用户输入的内容
        input_language = "中文"
        output_language = "英文"
        text = "我带着比身体重的行李，\
                游入尼罗河底，\
                经过几道闪电 看到一堆光圈，\
                不确定是不是这里。" 
        
        # 将用户输入的内容传递给chain，调用大模型并返回结果
        output  = chain.invoke(
            {
                "input_language": input_language, 
                "output_language": output_language, 
                "text": text
            }
        )
        print(output)


    # 构建检索问答链
    def paper_reading(self):
        # 加载向量数据库
        vector_db = Chroma(
            persist_directory = self.database_directory,
            embedding_function = self.sparkEmbedding
        )
        # print(f"向量库中存储的数量：{vector_db._collection.count()}")
        
        # 通过as_retriever方法把向量数据库构造成检索器。
        # 如下代码会在向量数据库中根据相似性进行检索，返回前 k 个最相似的文档
        retriever = vector_db.as_retriever(search_kwargs={"k": 3})
        # question = "什么是SSL？"
        # docs = retriever.invoke(question)
        # for i, doc in enumerate(docs):
        #     print(f"检索到的第{i}个内容: \n {doc.page_content}", 
        #           end="\n-----------------------------------------------------\n")

        # 处理检索到的文档
        def combine_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        # LCEL中要求所有的组成元素都是Runnable类型，前面我们见过的ChatModel、PromptTemplate等都是继承自Runnable类。
        combiner = RunnableLambda(combine_docs)
        # 检索链retrieval_chain是由检索器retriever及组合器combiner组成的，由|符号串连，数据从左向右传递，
        # 即问题先被retriever检索得到检索结果，再将检索结果给combiner()进一步处理并输出。
        retrieval_chain = retriever | combiner
        # output = retrieval_chain.invoke(question)
        
        # 基于langchain调用大模型
        llm = SparkLLM(
            spark_app_id=self.spark_appid,
            spark_api_key=self.spark_api_key,
            spark_api_secret=self.spark_api_secret,
            spark_llm_domain=self.spark_4ultra_domain,
            spark_api_url=self.spark_4ultra_url,
            temperature=0.1
        )
        
        template = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
        案。最多使用三句话。尽量使答案简明扼要。请你在回答的最后说“谢谢你的提问！”。
        {context}
        问题: {input}
        """
        # 将template通过 PromptTemplate 转为可以在LCEL中使用的类型
        prompt = PromptTemplate(template=template)
        # 把刚才定义的检索链retrieval_chain当作子链作为prompt的context，
        # 再使用RunnablePassthrough存储用户的问题作为prompt的input。
        # 又因为这两个操作是并行的，所以我们使用RunnableParallel来将他们并行运行
        """
        1、假如用户输入问题 “什么是强化学习”，RunnableParallel 会并行执行两个操作: 
            (1) retrieval_chain通过invoke()方法执行检索链，从向量数据库中获取相关文档
            (2) RunnablePassthrough直接传递用户的原始输入
           之后得到: 
                {
                    "context": "强化学习是机器学习的一个分支...",  # 检索到的文档内容
                    "input": "什么是强化学习？"  # 用户的原始问题
                }
        2、将上述结果传递给prompt，prompt会通过invoke()方法将context和input代入到template中，
           生成给大模型的提示词模板
        3、将生成的提示词模板prompt传递给llm，llm同样通过invoke()方法调用大模型生成最终的答案
        4、将llm的输出结果传递给StrOutputParser，StrOutputParser会将输出解析为字符串格式
        """
        qa_chain = (
            RunnableParallel({"context": retrieval_chain, "input": RunnablePassthrough()})
            | prompt
            | llm
            | StrOutputParser()
        )
        
        question1 = "什么是SSL？"
        output = qa_chain.invoke(question1)
        print(output)

    # 传递聊天记录，让ai具有记忆功能
    def historical_record(self):
        # 系统prompt
        # 问答链的系统prompt
        system_prompt = (
            "你是一个问答任务的助手。 "
            "请使用检索到的上下文片段回答这个问题。 "
            "如果你不知道答案就说不知道。 "
            "请使用简洁的话语回答用户。"
            "\n\n"
            "{context}"
        )
        # 设置prompt模板
        qa_prompt = ChatPromptTemplate(
            [
                ("system", system_prompt),
                ("human", "{question}"),
                ("placeholder", "{chat_history}")
            ]
        )
        # 无历史记录情况
        no_history_messages = qa_prompt.invoke(
            {
                "context": "",
                "chat_history": [],
                "question": "你在哪个城市？"
            }
        )
        # 有历史记录情况
        history_messages = qa_prompt.invoke(
            {
                "context": "",
                "chat_history": [
                    ("human", "你在哪个城市？"),
                    ("ai", "我在中国的北京。")
                ],
                "question": "这个城市位于中国的哪个省份？"
            }
        )
        for message in history_messages.messages:
            print(message.content)
        # TODO 让ai帮忙翻译，并根据上下文完善问题，然后再到向量数据库中查询
        question = "FixMatch_Simplifying Semi-Supervised Learning这篇文章提出了什么方法"
        vector_db = Chroma(
            persist_directory = self.database_directory,
            embedding_function = self.sparkEmbedding
        )
        retriever = vector_db.as_retriever(search_kwargs={"k": 1})
        def combine_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        combiner = RunnableLambda(combine_docs)
        retrieval_chain = retriever | combiner
        output = retrieval_chain.invoke(question)
        print(output)

if __name__ == "__main__":

    agentTest = AgentTest()
    # agentTest.chat()
    # agentTest.test_embedding()
    # agentTest.data_process()
    # agentTest.create_database()
    agentTest.query_database()
    # agentTest.langchain_chain()
    # agentTest.paper_reading()
    # agentTest.historical_record()
