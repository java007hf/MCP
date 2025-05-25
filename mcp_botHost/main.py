import asyncio # 异步编程支持
import os # 操作系统功能
import json # 添加全局 json 模块导入
import traceback # 添加 traceback 模块导入，用于打印详细的错误信息
import argparse # 添加命令行参数解析
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
from typing import Optional, Protocol, Dict, Any, List
from sse_starlette.sse import EventSourceResponse
from abc import ABC, abstractmethod
from datetime import datetime
from collections import defaultdict

# agent: agent 是SDK的核心构建块，定义 agent 的行为、指令和可用工具
# Model: 抽象基类，定义模型的接口
# ModelProvider: 提供模型实例，自定义配置模型
# OpenAIChatCompletionsModel: OpenAI Chat Completions API的模型实现，用于与 OpenAI API 交互
# RunConfig: 用于配置 agent 运行的配置参数
# Runner: 用于运行 agent 的组件，负责管理 agent 的执行流程和上下文
# set_tracing_disabled: 用于禁用追踪
# ModelSettings: 配置模型的参数，如温度、top_p和工具选择策略等
from agents import (
    Agent,
    Model,
    ModelProvider,
    OpenAIChatCompletionsModel,
    RunConfig,
    Runner,
    set_tracing_disabled,
    ModelSettings
)
from openai import AsyncOpenAI # OpenAI异步客户端
# ResponseTextDeltaEvent: 表示文本增量响应事件，包含文本的增量变化
# ResponseContentPartDoneEvent: 表示内容部分完成响应事件，表示一个内容片段已完成生成
from openai.types.responses import ResponseTextDeltaEvent, ResponseContentPartDoneEvent

# MCP服务器相关，用于连接MCP服务器
from agents.mcp import MCPServerStdio

# 环境变量加载相关
from dotenv import load_dotenv

# 加载.env文件
load_dotenv()

# 设置DeepSeek API密钥
API_KEY = os.getenv("API_KEY")
# 设置DeepSeek API基础URL
BASE_URL = os.getenv("BASE_URL")
# 设置DeepSeek API模型名称
MODEL_NAME = os.getenv("MODEL_NAME")

if not API_KEY:
    raise ValueError("DeepSeek API密钥未设置")
if not BASE_URL:
    raise ValueError("DeepSeek API基础URL未设置")
if not MODEL_NAME:
    raise ValueError("DeepSeek API模型名称未设置")

# 创建 DeepSeek API 客户端(使用兼容openai的接口)
client = AsyncOpenAI(
    base_url=BASE_URL,
    api_key=API_KEY
)
# 禁用追踪以避免需要 Openai API 密钥
set_tracing_disabled(True)

class DeepSeekModelProvider(ModelProvider):
    """
    DeepSeek V3 模型提供商 - 通过 OpenAI兼容接口连接DeepSeek API
    这个类负责提供与 DeepSeek 模型的连接，通过 OpenAI 兼容接口调用 DeepSeek API
    """

    def get_model(self, model_name: str) -> Model:
        """
        获取模型实例，根据提供的模型名称创建并返回一个Openai兼容的模型实例

        Args:
            model_name (str): 模型名称，如果为空，则使用默认模型

        Returns:
            Model: OpenAI兼容的模型实例
        """
        # 使用 chat Completions API 调用 DeepSeek API,返回openai兼容模型
        return OpenAIChatCompletionsModel(model=model_name or MODEL_NAME, openai_client=client)

# 创建 DeepSeek 模型提供者实例
model_provider = DeepSeekModelProvider()

# MCP 服务器单例管理
class MCPServerManager:
    _instance = None
    _weather_server = None
    _bot_server = None
    _initialized = False

    @classmethod
    async def initialize(cls):
        """初始化 MCP 服务器连接"""
        if cls._initialized:
            return

        try:
            print("正在初始化 MCP 服务器...")
            
            # 初始化天气服务器
            cls._weather_server = MCPServerStdio(
                name="weather",
                params={
                    "command": "C:\\Windows\\System32\\cmd.exe",
                    "args": [
                        "/c", 
                        "C:\\workspace\\MCP\\mcp_botHost\\.venv\\Scripts\\python.exe", 
                        "C:\\workspace\\MCP\\weather\\weather.py"],
                    "env":{}
                },
                cache_tools_list=True
            )

            # 初始化机器人服务器
            cls._bot_server = MCPServerStdio(
                name="bot",
                params={
                    "command": "C:\\Windows\\System32\\cmd.exe",
                    "args": [
                        "/c", 
                        "C:\\workspace\\MCP\\mcp_botHost\\.venv\\Scripts\\python.exe", 
                        "C:\\workspace\\MCP\\mcp_botServer\\main.py"],
                    "env":{}
                },
                cache_tools_list=True
            )

            # 连接服务器
            print("正在连接到 MCP 服务器...")
            await cls._weather_server.connect()
            await cls._bot_server.connect()
            print("MCP 服务器连接成功！")

            cls._initialized = True
        except Exception as e:
            print(f"MCP 服务器初始化失败: {e}")
            traceback.print_exc()
            raise

    @classmethod
    async def cleanup(cls):
        """清理 MCP 服务器连接"""
        if not cls._initialized:
            return

        try:
            if cls._weather_server:
                print("正在清理 weather_server 服务器资源...")
                await cls._weather_server.cleanup()
                print("weather_server 资源清理成功！")

            if cls._bot_server:
                print("正在清理 bot_server 服务器资源...")
                await cls._bot_server.cleanup()
                print("bot_server 资源清理成功！")

            cls._initialized = False
        except Exception as e:
            print(f"清理 MCP 服务器资源时出错: {e}")
            traceback.print_exc()

    @classmethod
    def get_servers(cls):
        """获取 MCP 服务器实例"""
        if not cls._initialized:
            raise RuntimeError("MCP 服务器未初始化")
        return cls._weather_server, cls._bot_server

class Conversation:
    """对话历史管理类"""
    
    def __init__(self, max_history: int = 10):
        """
        初始化对话历史
        
        Args:
            max_history (int): 最大历史记录数
        """
        self.history: List[Dict[str, Any]] = []
        self.max_history = max_history
    
    def add_message(self, role: str, content: str) -> None:
        """
        添加消息到历史记录
        
        Args:
            role (str): 消息角色（user/assistant）
            content (str): 消息内容
        """
        self.history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
        # 如果超出最大历史记录数，删除最旧的消息
        if len(self.history) > self.max_history:
            self.history.pop(0)
    
    def get_history(self) -> List[Dict[str, Any]]:
        """获取历史记录"""
        return self.history
    
    def clear(self) -> None:
        """清空历史记录"""
        self.history.clear()

class ConversationManager:
    """对话管理器"""
    
    def __init__(self):
        self.conversations: Dict[str, Conversation] = {}
    
    def get_conversation(self, session_id: str) -> Conversation:
        """
        获取或创建对话
        
        Args:
            session_id (str): 会话ID
        """
        if session_id not in self.conversations:
            self.conversations[session_id] = Conversation()
        return self.conversations[session_id]
    
    def clear_conversation(self, session_id: str) -> None:
        """
        清空指定会话的历史记录
        
        Args:
            session_id (str): 会话ID
        """
        if session_id in self.conversations:
            self.conversations[session_id].clear()

# 创建全局对话管理器实例
conversation_manager = ConversationManager()

# 修改请求模型
class QueryRequest(BaseModel):
    query: str
    streaming: Optional[bool] = True
    session_id: Optional[str] = "default"  # 添加会话ID字段

# 创建 FastAPI 应用
app = FastAPI(title="MCP Agent API")

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，生产环境应该限制
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class OutputHandler(ABC):
    """输出处理器基类"""
    
    @abstractmethod
    async def handle_model_delta(self, delta: str) -> None:
        """处理模型增量输出"""
        pass
    
    @abstractmethod
    async def handle_model_done(self) -> None:
        """处理模型输出完成"""
        pass
    
    @abstractmethod
    async def handle_tool_call(self, name: str, arguments: str) -> None:
        """处理工具调用"""
        pass
    
    @abstractmethod
    async def handle_tool_output(self, tool_id: str, output: str) -> None:
        """处理工具输出"""
        pass
    
    @abstractmethod
    async def handle_final_output(self, content: str) -> None:
        """处理最终输出"""
        pass
    
    @abstractmethod
    async def handle_error(self, error: str) -> None:
        """处理错误"""
        pass

class CLIOutputHandler(OutputHandler):
    """命令行输出处理器"""
    
    async def handle_model_delta(self, delta: str) -> None:
        print(delta, end="", flush=True)
    
    async def handle_model_done(self) -> None:
        print("\n", end="", flush=True)
    
    async def handle_tool_call(self, name: str, arguments: str) -> None:
        print(f"\n工具名称: {name}", flush=True)
        print(f"工具参数: {arguments}", flush=True)
    
    async def handle_tool_output(self, tool_id: str, output: str) -> None:
        print(f"\n工具调用{tool_id} 返回结果: {output}", flush=True)
    
    async def handle_final_output(self, content: str) -> None:
        print("\n===== 完整agent信息 =====")
        print(content)
    
    async def handle_error(self, error: str) -> None:
        print(f"运行Agent时出错: {error}")
        traceback.print_exc()

class HTTPOutputHandler(OutputHandler):
    """HTTP输出处理器（改为向SSE队列发布事件）"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id  # 新增：绑定会话ID

    async def handle_model_delta(self, delta: str) -> None:
        print(delta, end="", flush=True)
        await sse_registry.publish(
            self.session_id,
            {"event": "model_delta", "data": {"content": delta}}
        )
    
    async def handle_model_done(self) -> None:
        print("\n", end="", flush=True)
        await sse_registry.publish(
            self.session_id,
            {"event": "model_done", "data": {}}
        )
    
    async def handle_tool_call(self, name: str, arguments: str) -> None:
        print(f"\n工具名称: {name}", flush=True)
        print(f"工具参数: {arguments}", flush=True)
        await sse_registry.publish(
            self.session_id,
            {"event": "tool_call", "data": {"name": name, "arguments": arguments}}
        )
    
    async def handle_tool_output(self, tool_id: str, output: str) -> None:
        print(f"\n工具调用{tool_id} 返回结果: {output}", flush=True)
        await sse_registry.publish(
            self.session_id,
            {"event": "tool_output", "data": {"tool_id": tool_id, "output": output}}
        )
    
    async def handle_final_output(self, content: str) -> None:
        print("\n===== 完整agent信息 =====")
        print(content)
        await sse_registry.publish(
            self.session_id,
            {"event": "final_output", "data": {"content": content}}
        )
    
    async def handle_error(self, error: str) -> None:
        print(f"运行Agent时出错: {error}")
        traceback.print_exc()
        await sse_registry.publish(
            self.session_id,
            {"event": "error", "data": {"error": error}}
        )

async def run_agent(query: str, output_handler: OutputHandler, streaming: bool = True, session_id: str = "default") -> None:
    """
    启动并运行agent，支持流式输出和上下文记忆

    Args:
        query (str): 用户的自然语言查询
        output_handler (OutputHandler): 输出处理器
        streaming (bool): 是否流式输出
        session_id (str): 会话ID，用于维护对话上下文
    """
    try:
        # 获取或创建对话
        conversation = conversation_manager.get_conversation(session_id)
        
        # 添加用户消息到历史记录
        conversation.add_message("user", query)
        
        # 获取已初始化的服务器实例
        weather_server, bot_server = MCPServerManager.get_servers()

        # 构建系统提示，包含历史记录
        history_context = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in conversation.get_history()[:-1]  # 不包含当前消息
        ])
        
        system_instructions = (
            "你是一个平衡车机器人，可以帮助用户查询天气信息、根据用户的指令控制平衡车行走。"
            "用户可能会询问天气状况、天气预报等信息，请根据用户的问题选择合适的工具进行查询。"
            "用户可能会询问平衡车行走的指令，请根据用户的问题选择合适的工具进行控制。"
            "\n\n以下是之前的对话历史，请参考上下文来理解用户的意图：\n"
            f"{history_context}"
        )

        # 创建agent实例
        agent = Agent(
            name="平衡车Agent",
            instructions=system_instructions,
            mcp_servers=[weather_server, bot_server],
            model_settings=ModelSettings(
                temperature=0.6,
                top_p=0.9,
                max_tokens=4096,
                tool_choice="auto",
                parallel_tool_calls=True,
                truncation="auto"
            )
        )

        # 使用流式输出模式
        if streaming:
            result = Runner.run_streamed(
                agent,
                input=query,
                max_turns=10,
                run_config=RunConfig(
                    model_provider=model_provider,
                    trace_include_sensitive_data=True,
                    handoff_input_filter=None,
                )
            )

            full_response = ""
            async for event in result.stream_events():
                if event.type == "raw_response_event":
                    if isinstance(event.data, ResponseTextDeltaEvent):
                        full_response += event.data.delta
                        await output_handler.handle_model_delta(event.data.delta)
                    elif isinstance(event.data, ResponseContentPartDoneEvent):
                        await output_handler.handle_model_done()
                elif event.type == "run_item_stream_event":
                    if event.item.type == "tool_call_item":
                        raw_item = getattr(event.item, "raw_item", None)
                        if raw_item:
                            await output_handler.handle_tool_call(
                                getattr(raw_item, "name", "未知工具"),
                                getattr(raw_item, "arguments", "{}")
                            )
                    elif event.item.type == "tool_call_output_item":
                        raw_item = getattr(event.item, "raw_item", None)
                        tool_id = "未知工具ID"
                        if isinstance(raw_item, dict) and "call_id" in raw_item:
                            tool_id = raw_item["call_id"]
                        output = getattr(event.item, "output", "未知输出")
                        
                        output_text = ""
                        if isinstance(output, str) and (output.startswith("{") or output.startswith("[")):
                            try:
                                output_data = json.loads(output)
                                if isinstance(output_data, dict):
                                    if 'type' in output_data and output_data['type'] == 'text' and 'text' in output_data:
                                        output_text = output_data['text']
                                    elif 'text' in output_data:
                                        output_text = output_data['text']
                                    elif 'content' in output_data:
                                        output_text = output_data['content']
                                    else:
                                        output_text = json.dumps(output_data, ensure_ascii=False, indent=2)
                            except json.JSONDecodeError:
                                output_text = str(output)
                        else:
                            output_text = str(output)
                            
                        await output_handler.handle_tool_output(tool_id, output_text)

            # 添加助手回复到历史记录
            conversation.add_message("assistant", full_response)

            if hasattr(result, "final_output"):
                await output_handler.handle_final_output(result.final_output)
        else:
            result = await Runner.run(
                agent,
                input=query,
                max_turns=10,
                run_config=RunConfig(
                    model_provider=model_provider,
                    trace_include_sensitive_data=True,
                    handoff_input_filter=None,
                )
            )

            if hasattr(result, "final_output"):
                # 添加助手回复到历史记录
                conversation.add_message("assistant", result.final_output)
                await output_handler.handle_final_output(result.final_output)
            else:
                await output_handler.handle_final_output("未获取到信息")
            
            if hasattr(result, "new_items"):
                for item in result.new_items:
                    if item.type == "tool_call_item":
                        raw_item = getattr(item, "raw_item", None)
                        if raw_item:
                            await output_handler.handle_tool_call(
                                getattr(raw_item, "name", "未知工具"),
                                getattr(raw_item, "arguments", "{}")
                            )
                    elif item.type == "tool_call_output_item":
                        raw_item = getattr(item, "raw_item", None)
                        tool_id = "未知工具ID"
                        if isinstance(raw_item, dict) and "call_id" in raw_item:
                            tool_id = raw_item["call_id"]
                        output = getattr(item, "output", "未知输出")
                        
                        output_text = ""
                        if isinstance(output, str) and (output.startswith("{") or output.startswith("[")):
                            try:
                                output_data = json.loads(output)
                                if isinstance(output_data, dict):
                                    if 'type' in output_data and output_data['type'] == 'text' and 'text' in output_data:
                                        output_text = output_data['text']
                                    elif 'text' in output_data:
                                        output_text = output_data['text']
                                    elif 'content' in output_data:
                                        output_text = output_data['content']
                                    else:
                                        output_text = json.dumps(output_data, ensure_ascii=False, indent=2)
                            except json.JSONDecodeError:
                                output_text = str(output)
                        else:
                            output_text = str(output)
                            
                        await output_handler.handle_tool_output(tool_id, output_text)

    except Exception as e:
        await output_handler.handle_error(str(e))

async def run_cli():
    """命令行交互模式"""
    print("===== DeepSeek MCP =====")
    print("请输入自然语言查询")
    print("输入'quit'或'退出'结束程序")
    print("输入'clear'或'清除'清空对话历史")
    print("=======================")

    session_id = "cli_session"
    
    while True:
        # 获取用户输入
        user_query = input("\n请输入您的命令(输入'quit'或'退出'结束程序): ").strip()

        # 检查是否退出
        if user_query.lower() in ["quit", "退出"]:
            print("感谢使用DeepSeek MCP 再见！")
            break
        
        # 检查是否清空历史
        if user_query.lower() in ["clear", "清除"]:
            conversation_manager.clear_conversation(session_id)
            print("对话历史已清空")
            continue
        
        # 如果查询为空，则提示用户输入
        if not user_query:
            print("查询内容不能为空，请重新输入。")
            continue
        
        # 获取输出模型
        streaming = input("是否启用流式输出? (y/n, 默认y): ").strip().lower() != "n"

        # 运行agent
        await run_agent(user_query, CLIOutputHandler(), streaming, session_id)

async def run_http_server():
    """HTTP 服务器模式"""
    config = uvicorn.Config(app, host="192.168.3.102", port=8000, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()



class SSERegistry:
    """管理 SSE 会话的事件队列"""
    def __init__(self):
        self.queues = defaultdict(asyncio.Queue)  # session_id -> asyncio.Queue

    async def publish(self, session_id: str, event: dict):
        """向指定会话的队列发布事件"""
        if session_id in self.queues:
            await self.queues[session_id].put(event)

    async def subscribe(self, session_id: str) -> asyncio.Queue:
        """订阅指定会话的事件队列"""
        queue = self.queues[session_id]
        return queue

    def unregister(self, session_id: str):
        """移除会话的队列（可选，用于清理资源）"""
        if session_id in self.queues:
            del self.queues[session_id]

# 创建全局 SSE 注册中心实例
sse_registry = SSERegistry()

@app.post("/query")
async def query(request: QueryRequest):
    """立即返回确认响应，并异步执行run_agent"""
    # 启动异步任务执行run_agent（注意：需将代码移至return之前，此处为示意）
    asyncio.create_task(
        run_agent(
            query=request.query,
            output_handler=HTTPOutputHandler(session_id=request.session_id),  # 传递session_id
            streaming=request.streaming,
            session_id=request.session_id
        )
    )
    
    # 立即返回包含session_id的确认响应（类似clear_history）
    return {
        "code": 200,
        "message": "请求已接收，结果将通过SSE流推送",
        "session_id": request.session_id
    }

@app.get("/sse", response_class=Response)
async def sse_endpoint(session_id: str = "default"):
    """SSE接口，客户端通过session_id订阅事件流"""
    # 获取会话对应的事件队列
    queue = await sse_registry.subscribe(session_id)
    
    # 设置响应头为SSE格式
    headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive"
    }
    
    async def event_stream():
        try:
            while True:
                # 从队列中获取事件（阻塞直到有新事件）
                event = await queue.get()
                # 格式化为SSE标准格式（event: ...\ndata: ...\n\n）
                yield f"event: {event['event']}\ndata: {json.dumps(event['data'])}\n\n"
        except asyncio.CancelledError:
            # 客户端断开连接时清理队列（可选）
            sse_registry.unregister(session_id)
    
    return Response(event_stream(), headers=headers)

# 添加清空对话历史的接口
@app.post("/clear_history")
async def clear_history(session_id: str = "default"):
    """清空指定会话的对话历史"""
    conversation_manager.clear_conversation(session_id)
    return {"status": "success", "message": "对话历史已清空"}

async def main():
    """
    应用程序主函数 - 根据命令行参数选择运行模式
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='DeepSeek MCP 服务')
    parser.add_argument('--mode', choices=['cli', 'http'], default='cli',
                      help='运行模式: cli (命令行交互) 或 http (HTTP服务)')
    args = parser.parse_args()

    try:
        # 初始化 MCP 服务器
        await MCPServerManager.initialize()
        
        if args.mode == 'cli':
            await run_cli()
        else:
            print("启动 HTTP 服务器模式...")
            await run_http_server()
            
    except KeyboardInterrupt:
        print("\n程序被用户中断，正在退出...")
    except Exception as e:
        print(f"程序运行时发生错误: {e}")
        traceback.print_exc()
    finally:
        # 清理 MCP 服务器资源
        await MCPServerManager.cleanup()
        print("程序结束，所有资源已释放。")

# 程序入口点
if __name__ == "__main__":
    # 运行主函数
    asyncio.run(main())
        
    
    

