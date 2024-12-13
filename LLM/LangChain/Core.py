from langchain.agents import Tool, AgentExecutor, initialize_agent
from langchain.chains.llm import LLMChain
import langchain_llama
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain.prompts.prompt import PromptTemplate

def search_restaurant(cuisine: str, location: str, price: str) -> str:
    """根据菜系、地点和价格范围搜索餐厅。"""
    #  这里可以调用第三方 API 或数据库进行实际的餐厅搜索
    #  为了演示，我们直接返回模拟结果
    return f"已找到以下{cuisine}餐厅：餐厅 A ({price})、餐厅 B ({price})"

def get_restaurant_details(restaurant_name: str) -> str:
    """获取餐厅的详细信息，如评分、地址、营业时间等。"""
    #  这里可以调用第三方 API 或数据库进行实际的信息获取
    #  为了演示，我们直接返回模拟结果
    return f"{restaurant_name}：评分 4.5 分，地址：XX 路 XX 号，营业时间：10:00-22:00"

if __name__ == '__main__':
    # 设置模型路径
    llama = LlamaCppEmbeddings(model_path='''D:\LLM\MODELs\llama3-8B\Meta-Llama-3-8B-Instruct\Meta-Llama-3-8B-Instruct-Q4_0.gguf''')
    text = "This is a test document."

    # 查询嵌入
    query_result = llama.embed_query(text)
    print("Query Embedding:", query_result)
    
    # 文档嵌入
    doc_result = llama.embed_documents([text])
    print("Document Embedding:", doc_result)
    #llm = langchain_llama.(model="gpt-4", temperature=0)
    #
    #prompt_template = """
    #你是一位智能餐厅推荐助手，你需要根据用户的需求，推荐合适的餐厅。
    #你可以使用以下工具：
    #- search_restaurant: 搜索餐厅，参数：菜系，地点，价格范围
    #- get_restaurant_details: 获取餐厅详细信息，参数：餐厅名称
    #
    #以下是用户的需求：
    #{user_request}
    #
    #请根据用户的需求，给出你的建议，并调用相应的工具获取必要的信息。
    #"""
    #
    #prompt = PromptTemplate(template=prompt_template, input_variables=["user_request"])
    #llm_chain = LLMChain(llm=llm, prompt=prompt)
    #
    #tools = [search_restaurant, get_restaurant_details]
    #agent = initialize_agent(tools, llm_chain, agent="zero-shot-react-description", verbose=True)
    #
    ## Test
    #user_request = "我想在周五晚上和朋友一起去吃湘菜，最好环境优雅，人均消费在 150 元左右。"
    #agent.run(user_request)