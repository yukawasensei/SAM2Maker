# 使用github - mcp访问仓库指南

## 一、前提条件
1. **安装Docker**：要在容器中运行服务器，需要安装Docker。可从[Docker官网](https://www.docker.com/)下载并安装。
2. **创建GitHub个人访问令牌**：MCP服务器可以使用许多GitHub API，因此需要创建一个GitHub个人访问令牌，并启用相应的权限。创建步骤如下：
    - 登录GitHub后，点击页面右上角的头像，选择 "Settings"。
    - 在左侧菜单中选择 "Developer settings"。
    - 点击左侧菜单中的 "Personal access tokens"，然后选择 "Fine - grained tokens"。
    - 点击 "Generate new token" 按钮。
    - 为令牌设置以下信息：
        - 名称：输入一个描述性名称，如 "mcp - access"。
        - 过期时间：根据需要选择（建议选择较长时间，如90天或1年）。
        - 权限范围：至少选择 `repo`（所有仓库权限），如果需要操作GitHub Actions，还需选择 `workflow`。
    - 点击页面底部的 "Generate token" 按钮，生成后立即复制显示的令牌，因为它只会显示一次。

## 二、配置MCP服务器
### 2.1 使用VS Code配置
为了快速安装，可以使用README顶部的一键安装按钮。手动安装时，将以下JSON块添加到VS Code的用户设置（JSON）文件中。可以通过按 `Ctrl + Shift + P` 并键入 `Preferences: Open User Settings (JSON)` 来完成此操作。可选地，也可以在工作区中添加一个名为 `.vscode/mcp.json` 的文件，这将允许与其他人共享配置。注意：在 `.vscode/mcp.json` 文件中不需要 `mcp` 键。
```json
{
    "mcp": {
        "inputs": [
            {
                "type": "promptString",
                "id": "github_token",
                "description": "GitHub个人访问令牌",
                "password": true
            }
        ],
        "servers": {
            "github": {
                "command": "docker",
                "args": [
                    "run",
                    "-i",
                    "--rm",
                    "-e",
                    "GITHUB_PERSONAL_ACCESS_TOKEN",
                    "ghcr.io/github/github - mcp - server"
                ],
                "env": {
                    "GITHUB_PERSONAL_ACCESS_TOKEN": "${input:github_token}"
                }
            }
        }
    }
}
```
### 2.2 使用Claude Desktop配置
```json
{
    "mcpServers": {
        "github": {
            "command": "docker",
            "args": [
                "run",
                "-i",
                "--rm",
                "-e",
                "GITHUB_PERSONAL_ACCESS_TOKEN",
                "ghcr.io/github/github - mcp - server"
            ],
            "env": {
                "GITHUB_PERSONAL_ACCESS_TOKEN": "<YOUR_TOKEN>"
            }
        }
    }
}
```
### 2.3 从源代码构建
如果没有Docker，可以使用 `go` 在 `cmd/github - mcp - server` 目录中构建二进制文件，并使用 `github - mcp - server stdio` 命令，同时将 `GITHUB_PERSONAL_ACCESS_TOKEN` 环境变量设置为您的令牌。

## 三、使用GitHub MCP Server访问仓库
配置完成后，就可以使用GitHub MCP Server提供的工具来访问和操作仓库了，例如：
- `get_me`：获取已认证用户的详细信息。
- `get_issue`：获取仓库中一个问题的内容。
- `create_issue`：在GitHub仓库中创建一个新问题。
- `add_issue_comment`：向问题添加评论。
- `list_issues`：列出并过滤仓库问题。
- `update_issue`：更新GitHub仓库中的现有问题。

具体使用时，根据需求调用相应的工具，并传入必要的参数。