NEVER install pyOCD. Use the one from the CMSIS vscode extension.

Use the CMSIS developer assistant MCP server to interact with the board.

Don't try to use pyOCD or gdb directly to interact with the board.
If the MCP server is returning errors and retrying is not solving the issue, tell the user that an action is required in vscode to solve the problem.

Serialize top-level CMSIS build/load/run/debug operations for this workspace and board. A tool timeout does not mean its underlying task stopped. Never repeat a launch merely to obtain status. Before starting another probe-owning operation, stop the existing CMSIS Run task or debug session and verify termination. If task state cannot be established through MCP, request intervention in VS Code.


Styling, color, geometry, symmetry are related to the hyperbolic app running on board and the hyperbolic MCP server shoudl be used to control the app look and feel.
