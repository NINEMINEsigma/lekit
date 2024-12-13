
def template_agent3p_prompt():
    return """
    你在运行一个“思考”，“工具调用”，“响应”循环。每次只运行一个阶段

    1.“思考”阶段：你要仔细思考用户的问题
    2.“工具调用阶段”：选择可以调用的工具，并且输出对应工具需要的参数
    3.“响应”阶段：根据工具调用返回的影响，回复用户问题。

    已有的工具如下：
    get_weather：
    e.g. get_weather:天津
    返回天津的天气情况

    Example：
    question：天津的天气怎么样？
    thought：我应该调用工具查询天津的天气情况
    Action：
    {
        "function_name":"get_response_time"
        "function_params":{
            "location":"天津"
        }
    }
    调用Action的结果：“天气晴朗”
    Answer:天津的天气晴朗
    """