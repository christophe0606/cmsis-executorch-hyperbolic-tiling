NEVER install pyOCD. Use the one from the CMSIS vscode extension.

Use the CMSIS developer assistant MCP server to interact with the board.

Don't try to use pyOCD or gdb directly to interact with the board.
If the MCP server is returning errors and retrying is not solving the issue, tell the user that an action is required in vscode to solve the problem.
