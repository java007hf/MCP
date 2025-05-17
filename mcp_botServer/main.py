from typing import Any
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("botServer")

@mcp.tool()
async def send_command(type: int, args: str) -> str:
    """发送命令给机器人，type表示命令类型，args表示命令参数，机器人的名字叫做“平衡车”，可以执行前进、后退、左转、右转、停止等命令

    Args:
        type: 命令类型，可以是1、2、3、4、5，分别对应前进、后退、左转、右转、停止
        args: 是具体的参数，例如前进10米 返回10，后退5米 返回5，左转10度 返回10，右转5度 返回5，停止返回0
    """
    return f"{type} {args}"


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')