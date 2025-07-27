import sys
import os
from sparkai.embedding.spark_embedding import Embeddingmodel
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from spark_x1 import SparkModel
from sparkai_embedding import SparkAIEmbeddings
from dotenv import load_dotenv, find_dotenv
import re


class AgentTest:
    def __init__(self):
        # 读取本地/项目的环境变量。
        # find_dotenv()寻找并定位.env文件的路径
        # load_dotenv()读取该.env文件，并将其中的环境变量加载到当前的运行环境中  
        # 如果你设置的是全局的环境变量，这行代码则没有任何作用。
        _ = load_dotenv(find_dotenv())
        # 星火大模型
        self.sparkModel = SparkModel()
        # 星火文本向量化
        self.sparkEmbedding = SparkAIEmbeddings()
        self.persist_directory = 'data_base/vector_db/chroma'

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
            persist_directory=self.persist_directory  # 允许我们将persist_directory目录保存到磁盘上
        )
        
        print(f"向量库中存储的数量：{vector_db._collection.count()}")

    def query_database(self):
        vector_db = Chroma(persist_directory=self.persist_directory, embedding_function=self.sparkEmbedding)
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



if __name__ == "__main__":

    agentTest = AgentTest()
    # agentTest.chat()
    # agentTest.test_embedding()
    # agentTest.data_process()
    # agentTest.create_database()
    agentTest.query_database()
