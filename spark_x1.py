import _thread as thread
import base64
import datetime
import hashlib
import hmac
import json
import os
from urllib.parse import urlparse
import ssl
from datetime import datetime
from time import mktime
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time
from sparkai.embedding.spark_embedding import Embeddingmodel
from typing import Optional

import websocket  # 使用websocket_client

answer = ""
isFirstcontent = False

class Ws_Param(object):
    # 初始化
    def __init__(self, APPID, APIKey, APISecret, Spark_url):
        self.APPID = APPID
        self.APIKey = APIKey
        self.APISecret = APISecret
        self.host = urlparse(Spark_url).netloc
        self.path = urlparse(Spark_url).path
        self.Spark_url = Spark_url

    # 生成url
    def create_url(self):
        # 生成RFC1123格式的时间戳
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        # 拼接字符串
        signature_origin = "host: " + self.host + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + self.path + " HTTP/1.1"

        # 进行hmac-sha256进行加密
        signature_sha = hmac.new(self.APISecret.encode('utf-8'), signature_origin.encode('utf-8'),
                                 digestmod=hashlib.sha256).digest()

        signature_sha_base64 = base64.b64encode(signature_sha).decode(encoding='utf-8')

        authorization_origin = f'api_key="{self.APIKey}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha_base64}"'

        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')

        # 将请求的鉴权参数组合为字典
        v = {
            "authorization": authorization,
            "date": date,
            "host": self.host
        }
        # 拼接鉴权参数，生成url
        url = self.Spark_url + '?' + urlencode(v)
        # 此处打印出建立连接时候的url,参考本demo的时候可取消上方打印的注释，比对相同参数时生成的url与自己代码生成的url是否一致
        return url


# 讯飞星火大模型
class SparkModel:
    def __init__(self, appid: Optional[str] = None, api_key: Optional[str] = None, 
                 api_secret: Optional[str] = None, Spark_url: Optional[str] = None, 
                 domain: Optional[str] = None):
        self.appid = appid or os.environ.get("IFLYTEK_SPARK_APP_ID")
        self.api_key = api_key or os.environ.get("IFLYTEK_SPARK_API_KEY")
        self.api_secret = api_secret or os.environ.get("IFLYTEK_SPARK_API_SECRET")
        self.Spark_url = Spark_url or os.environ.get("IFLYTEK_SPARK_X1_URL")
        self.domain = domain or os.environ.get("IFLYTEK_SPARK_DOMAIN")
        self.text = []
        self.answer = ""
        self.isFirstcontent = False
        self._ws = None
        self._response_collected = []
        self._finished = False
        print("appid: {}".format(self.appid))
        print("api_key: {}".format(self.api_key))
        print("api_secret: {}".format(self.api_secret))
        print("-----------")

    def on_error(self, ws, error):
        print("### error:", error)
        self._finished = True

    def on_close(self, ws, *args):
        self._finished = True

    def on_open(self, ws):
        thread.start_new_thread(self.run, (ws,))

    def run(self, ws, *args):
        data = json.dumps(self.gen_params(self.appid, self.domain, ws.question))
        ws.send(data)

    def on_message(self, ws, message):
        data = json.loads(message)
        code = data['header']['code']
        content = ''
        if code != 0:
            print(f'请求错误: {code}, {data}')
            ws.close()
        else:
            choices = data["payload"]["choices"]
            status = choices["status"]
            text = choices['text'][0]
            if ('reasoning_content' in text and '' != text['reasoning_content']):
                reasoning_content = text["reasoning_content"]
                # print(reasoning_content, end="")
                self.isFirstcontent = True
            if('content' in text and '' != text['content']):
                content = text["content"]
                if(True == self.isFirstcontent):
                    # print("\n*******************以上为思维链内容，模型回复内容如下********************\n")
                    pass
                # print(content, end="")
                self.isFirstcontent = False
            self.answer += content
            self._response_collected.append(content)
            if status == 2:
                ws.close()
                self._finished = True

    def gen_params(self, appid, domain, question):
        data = {
            "header": {
                "app_id": appid,
                "uid": "1234",
            },
            "parameter": {
                "chat": {
                    "domain": domain,
                    "temperature": 0.3,
                    "max_tokens": 32768
                }
            },
            "payload": {
                "message": {
                    "text": question
                }
            }
        }
        return data

    def getText(self, role, content):
        jsoncon = {}
        jsoncon["role"] = role
        jsoncon["content"] = content
        self.text.append(jsoncon)
        return self.text

    def getlength(self, text):
        length = 0
        for content in text:
            temp = content["content"]
            leng = len(temp)
            length += leng
        return length

    def checklen(self, text):
        while (self.getlength(text) > 8000):
            del text[0]
        return text

    def chat(self, prompt):
        self.answer = ""
        self._response_collected = []
        self._finished = False
        # 补充历史提问
        question = self.checklen(self.getText("user", prompt))
        wsParam = Ws_Param(self.appid, self.api_key, self.api_secret, self.Spark_url)
        websocket.enableTrace(False)
        wsUrl = wsParam.create_url()
        ws = websocket.WebSocketApp(
            wsUrl,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
            on_open=self.on_open
        )
        ws.appid = self.appid
        ws.question = question
        ws.domain = self.domain
        self._ws = ws
        ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
        # 等待直到 _finished
        import time
        while not self._finished:
            time.sleep(0.05)
        # 补充历史回答
        self.getText("assistant", self.answer)
        # 返回完整回答
        return self.answer


if __name__ == '__main__':
    # 以下密钥信息从服务管控页面获取：https://console.xfyun.cn/services/bmx1
    appid = "504016fa"  # 填写控制台中获取的 APPID 信息
    api_secret = "OWNmN2UxZWJlZTdlOTMwMDEzODQ0ODNm"  # 填写控制台中获取的 APISecret 信息
    api_key = "dfdc894a00dcb6b74ed22014b6e8d8f5"  # 填写控制台中获取的 APIKey 信息
    domain = "x1"       #控制请求的模型版本
    # 服务地址
    Spark_url = "wss://spark-api.xf-yun.com/v1/x1"  #查看接口文档  https://www.xfyun.cn/doc/spark/X1ws.html
    sparkModel = SparkModel(appid, api_key, api_secret, Spark_url, domain)

    while (1):
        Input = input("\n" + "我:")
        print("星火:", end="")
        answer = sparkModel.chat(Input)
        print(answer, end="")
        