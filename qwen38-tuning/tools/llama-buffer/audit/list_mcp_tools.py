"""List every tool of every MCP server the local session loads, and tokenize the
schemas with the served model's tokenizer. Output: mcp_tools.json + a summary.

Real data: the schemas come from the servers themselves, the token count from
llama-server /tokenize. What it cannot see: claude.ai-hosted connectors."""
import asyncio, json, os, sys, time, urllib.request
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

OUT = os.path.join(os.path.dirname(__file__), "mcp_tools-" + time.strftime("%Y-%m-%d") + ".json")
CFG = json.load(open(os.path.expanduser("~/.claude.json"), encoding="utf-8"))
PROJ = json.load(open("D:/Github/Agentic Framework/.mcp.json", encoding="utf-8"))["mcpServers"]

servers = dict(CFG["mcpServers"])
for k, v in PROJ.items():
    servers[k] = v  # project scope overrides user scope on a name clash, as Claude Code does

def tok(text):
    req = urllib.request.Request("http://localhost:8080/tokenize",
                                 data=json.dumps({"content": text}).encode(),
                                 headers={"Content-Type": "application/json"})
    return len(json.load(urllib.request.urlopen(req, timeout=120))["tokens"])

def render(server, t):
    # the shape Claude Code hands the model: name, description, input schema
    return json.dumps({"name": f"mcp__{server}__{t.name}",
                       "description": t.description or "",
                       "input_schema": t.inputSchema}, ensure_ascii=False)

async def list_one(name, cfg):
    kind = cfg.get("type", "stdio")
    if kind == "http":
        async with streamablehttp_client(cfg["url"], headers=cfg.get("headers") or {}) as (r, w, _):
            async with ClientSession(r, w) as s:
                await s.initialize()
                return (await s.list_tools()).tools
    env = dict(os.environ); env.update(cfg.get("env") or {})
    params = StdioServerParameters(command=cfg["command"], args=cfg.get("args", []), env=env)
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            return (await s.list_tools()).tools

async def main():
    results = {}
    for name, cfg in servers.items():
        t0 = time.time()
        try:
            tools = await asyncio.wait_for(list_one(name, cfg), timeout=150)
            rendered = [render(name, t) for t in tools]
            text = "\n".join(rendered)
            results[name] = {"ok": True, "n_tools": len(tools), "chars": len(text),
                             "tokens": tok(text) if text else 0,
                             "tools": [t.name for t in tools], "secs": round(time.time() - t0, 1)}
        except Exception as e:
            results[name] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}",
                             "secs": round(time.time() - t0, 1)}
        r = results[name]
        print(f"{name:24s} {'OK ' if r['ok'] else 'ERR'} tools={r.get('n_tools','-'):>4} tokens={r.get('tokens','-'):>6} {r.get('error','')}  [{r['secs']}s]", flush=True)
        json.dump(results, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    ok = [v for v in results.values() if v["ok"]]
    print(f"\nservers listed: {len(ok)}/{len(results)}  tools: {sum(v['n_tools'] for v in ok)}  tokens: {sum(v['tokens'] for v in ok)}")

asyncio.run(main())
