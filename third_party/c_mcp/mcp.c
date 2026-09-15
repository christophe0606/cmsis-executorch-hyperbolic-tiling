#include "mcp.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

struct argument
{
    const char *name;
    enum type type;          // "str", "int", "float", "bool"
    const char *description; // optional
    struct argument *next;
};

struct tool
{
    const char *name;
    const char *description;
    struct argument *arguments; // JSON schema as string
    struct tool *next;
};

static struct tool *tool_list = NULL; // linked list of registered tools

void add_argument(struct tool *tool,
                  const char *name,
                  enum type type,
                  const char *description)
{
    struct argument *arg = malloc(sizeof(struct argument));
    if (!tool || !arg) { free(arg); return; }
    arg->name = name;
    arg->type = type;
    arg->description = description;
    arg->next = tool->arguments;
    tool->arguments = arg;
}

void free_arguments(struct argument *arg_list)
{
    struct argument *arg = arg_list;
    while (arg)
    {
        struct argument *next = arg->next;
        free(arg);
        arg = next;
    }
}

struct tool *add_tool(const char *name,
                      const char *description)
{
    struct tool *t = malloc(sizeof(struct tool));
    if (!t) return NULL;
    t->name = name;
    t->description = description;
    t->arguments = NULL;
    t->next = tool_list;
    tool_list = t;
    return (t);
}

void free_tools()
{
    struct tool *t = tool_list;
    while (t)
    {
        struct tool *next = t->next;
        free_arguments(t->arguments);
        free(t);
        t = next;
    }
    tool_list = NULL;
}

void add_arguments(cJSON *props,
                   cJSON *req,
                   struct tool *tool)
{
    struct argument *arg = tool->arguments;
    while (arg)
    {
        cJSON *jsonArg = cJSON_CreateObject();
        if (arg->type == TYPE_STR)
            cJSON_AddStringToObject(jsonArg, "type", "string");
        else if (arg->type == TYPE_INT)
            cJSON_AddStringToObject(jsonArg, "type", "integer");
        else if (arg->type == TYPE_FLOAT)
            cJSON_AddStringToObject(jsonArg, "type", "number");
        else if (arg->type == TYPE_BOOL)
            cJSON_AddStringToObject(jsonArg, "type", "boolean");
        else
            cJSON_AddStringToObject(jsonArg, "type", "string"); // default to string

        if (arg->description != NULL)
        {
            cJSON_AddStringToObject(jsonArg, "description", arg->description);
        }
        cJSON_AddItemToObject(props, arg->name, jsonArg);
        cJSON_AddItemToArray(req, cJSON_CreateString(arg->name));
        arg = arg->next;
    }
}

cJSON *get_json_for_tool(struct tool *tool)
{

    cJSON *t = cJSON_CreateObject();
    cJSON_AddStringToObject(t, "name", tool->name);
    cJSON_AddStringToObject(t, "description", tool->description);
    cJSON *schema = cJSON_CreateObject();
    cJSON_AddStringToObject(schema, "type", "object");
    cJSON *props = cJSON_CreateObject();
    cJSON *req = cJSON_CreateArray();
    add_arguments(props, req, tool);
    cJSON_AddItemToObject(schema, "properties", props);
    cJSON_AddItemToObject(schema, "required", req);
    cJSON_AddItemToObject(t, "inputSchema", schema); // camelCase per spec
    return (t);
}

#define PROTOCOL_VERSION "2025-06-18" // match spec


static void send_json(cJSON *obj,int cfd)
{
    (void)cfd;
    char *s = cJSON_PrintUnformatted(obj); // single line, no pretty \n
    if (!s) return;
    fputs(s, stdout);
    fputc('\n', stdout); // newline = message boundary
    fflush(stdout);
    free(s);
}

cJSON *ok(cJSON *id, cJSON *result)
{
    cJSON *m = cJSON_CreateObject();
    cJSON_AddStringToObject(m, "jsonrpc", "2.0");
    cJSON_AddItemToObject(m, "id", cJSON_Duplicate(id, 1));
    cJSON_AddItemToObject(m, "result", result);
    return m;
}

cJSON *err(cJSON *id, int code, const char *msg)
{
    cJSON *m = cJSON_CreateObject();
    cJSON_AddStringToObject(m, "jsonrpc", "2.0");
    cJSON_AddItemToObject(m, "id", id ? cJSON_Duplicate(id, 1) : cJSON_CreateNull());
    cJSON *e = cJSON_CreateObject();
    cJSON_AddNumberToObject(e, "code", code);
    cJSON_AddStringToObject(e, "message", msg);
    cJSON_AddItemToObject(m, "error", e);
    return m;
}

cJSON *handle_fetch()
{
    cJSON *result = cJSON_CreateObject();
    cJSON_AddStringToObject(result, "mcpVersion", "0.1");
    cJSON_AddStringToObject(result, "name", "hyperbolic");
    cJSON_AddStringToObject(result, "version", "0.1.0");
    cJSON *cap = cJSON_CreateObject();
    cJSON_AddBoolToObject(cap, "tools", 1);
    cJSON_AddItemToObject(result, "capabilities", cap);

    return result;

}

static cJSON *handle_initialize(cJSON *id, cJSON *params)
{
    (void)params;
    cJSON *result = cJSON_CreateObject();
    cJSON_AddStringToObject(result, "protocolVersion", PROTOCOL_VERSION);

    cJSON *caps = cJSON_CreateObject();
    cJSON *tools = cJSON_CreateObject();
    cJSON_AddBoolToObject(tools, "listChanged", 0);
    cJSON_AddItemToObject(caps, "tools", tools);
    cJSON_AddItemToObject(result, "capabilities", caps);

    cJSON *serverInfo = cJSON_CreateObject();
    cJSON_AddStringToObject(serverInfo, "name", "hyperbolic-c-mcp-uart");
    cJSON_AddStringToObject(serverInfo, "version", "0.2.0");
    cJSON_AddItemToObject(result, "serverInfo", serverInfo);

    return ok(id, result);
}

static cJSON *handle_tools_list(cJSON *id)
{
    cJSON *result = cJSON_CreateObject();
    cJSON *tools = cJSON_CreateArray();

    struct tool *tool = tool_list;
    while (tool)
    {
        cJSON *t = get_json_for_tool(tool);
        cJSON_AddItemToArray(tools, t);
        tool = tool->next;
    }

    cJSON_AddItemToObject(result, "tools", tools);
    return ok(id, result);
}

static cJSON *handle_ping(cJSON *id)
{
    return ok(id, cJSON_CreateObject());
}

cJSON *create_result_text(const char *text)
{
    cJSON *res = cJSON_CreateObject();
    cJSON *content = cJSON_CreateArray();
    cJSON *item = cJSON_CreateObject();
    cJSON_AddStringToObject(item, "type", "text");
    cJSON_AddStringToObject(item, "text", text);
    cJSON_AddItemToArray(content, item);
    cJSON_AddItemToObject(res, "content", content);
    return res;
}

void dispatch(const char *line,int cfd)
{
    cJSON *root = cJSON_ParseWithOpts(line, NULL, 1);
    if (!root)
    {
        cJSON *e = err(NULL, MCP_PARSE_ERROR, "Parse error");
        send_json(e,cfd);
        cJSON_Delete(e);
        return;
    }
    
    cJSON *id = cJSON_GetObjectItemCaseSensitive(root, "id"); // may be NULL for notifications
    cJSON *method = cJSON_GetObjectItemCaseSensitive(root, "method");
    cJSON *params = cJSON_GetObjectItemCaseSensitive(root, "params");

    cJSON *version = cJSON_GetObjectItemCaseSensitive(root, "jsonrpc");
    if (!cJSON_IsObject(root) || !cJSON_IsString(version) ||
        strcmp(version->valuestring, "2.0") != 0 ||
        !cJSON_IsString(method) ||
        (id && !cJSON_IsString(id) && !cJSON_IsNumber(id) && !cJSON_IsNull(id)))
    {
        cJSON *e = err(NULL, MCP_INVALID_REQUEST, "Invalid Request");
        send_json(e,cfd);
        cJSON_Delete(e);
        cJSON_Delete(root);
        return;
    }

    /* Notifications never have responses; only requests may mutate settings. */
    if (!id) { cJSON_Delete(root); return; }

    cJSON *resp = NULL;
    const char *m = method->valuestring;

    if (strcmp(m, "initialize") == 0)
    {
        resp = handle_initialize(id, params);
    }
    else if (strcmp(m, "ping") == 0)
    {
        resp = handle_ping(id);
    }
    else if (strcmp(m, "tools/list") == 0)
    {
        resp = handle_tools_list(id);
    }
    else if (strcmp(m, "tools/call") == 0)
    {
        resp = handle_tools_call(id, params);
    }
    else
    {
        resp = err(id, MCP_METHOD_NOT_FOUND, "Method not found");
    }

    send_json(resp,cfd);
    cJSON_Delete(resp);
    cJSON_Delete(root);
}
