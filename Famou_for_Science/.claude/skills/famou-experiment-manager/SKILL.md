---
name: famou-experiment-manager
description: 管理 famou 进化实验任务的工作流技能。当用户提到"提交实验"、"查看实验状态"、"删除实验"、"获取实验结果"、"famou 实验"、"上传实验"、"config.yaml 实验"或需要使用 famou-ctl 管理实验任务时，必须使用此技能。即使用户只说"提交"或"跑实验"，只要上下文涉及 famou 平台，也应触发此技能。
---

# Famou 任务管理 Skill

用于通过 `famou-ctl-sdk` 提交和管理实验任务的完整工作流。

---

## 前置环境检查

### 1.1 检查 famou-ctl-sdk 是否已安装

```bash
famou-ctl --version
```

- 若输出版本号，继续下一步
- 若命令未找到，执行安装：`pip install famou-sdk --pre`

安装完成后再次验证 `famou-ctl --version`，若仍失败，停止并提示用户检查 Python 环境或 pip 源配置。

### 1.2 检查和配置 API 配置

使用辅助脚本 `scripts/config.py` 检查和配置 API 设置。

**检查 API 配置**

```bash
python3 scripts/config.py read
```

| 检查结果 | 操作 |
|---------|------|
| `status: "ok"` | 配置完整，跳过配置 |
| `status: "missing"` | 执行配置 API Key |

**配置 API：**

提示用户输入有效的 **API_KEY**，然后执行配置命令

```bash
python3 scripts/config.py write <YOUR_API_KEY>
```

---

## 提交 FaMou 实验

### 2.1 查找 config.yaml 文件

在当前工作目录下递归查找所有 `config.yaml` 文件：

```bash
find . -name "config.yaml" -type f 2>/dev/null | sort
```

**处理结果：**

| 情况 | 操作 |
|------|------|
| 找到 1 个 | 直接使用，告知用户路径，继续下一步 |
| 找到多个 | 使用 `ask_user` 工具询问用户选择哪个 |
| 未找到 | **报告用户并提示用户创建 config.yaml** |

config.yaml 模板：

```yaml
evolve_config:
  max_iterations: 100
  population_size: 100
  num_islands: 2
initial_program: "init.py"
evaluator: "evaluator.py"
system_message: "prompt.md"
```

### 2.2 确认实验目录

将所选 `config.yaml` 的父目录（绝对路径）作为实验目录：

```bash
# 示例：若 config.yaml 路径为 ./experiments/my_exp/config.yaml，则实验目录是 /absolute/path/to/experiments/my_exp
realpath $(dirname <config.yaml路径>)
```

### 2.3 获取实验名称

使用 `ask_user` 工具或直接在对话中请求用户输入实验名称 `experiment_name`, **提示用户实验名称只能包含字母、数字和下划线，且长度不超过20个字符**

### 2.4 提交实验

```bash
famou-ctl experiment create \
  --config <config.yaml绝对路径> \
  --experiment-name <experiment_name> \
  --json
```

**处理输出：**
- 命令成功：解析 JSON 输出，展示实验 ID、状态等关键信息
- 命令失败：展示错误信息，提示用户检查配置或网络连接

### 2.5 实验状态查询

每间隔10s，查询一下实验状态，检查是否通过线上的验证并且正常进入 famou 进化（验证需要消耗一些时间，因此需要轮训检查实验状态）
  
- 验证失败：修复评估器和初始解，删除失败的实验，重新提交
- 验证成功：输出实验状态并结束

--- 

## FaMou 实验其他操作

### 步骤 1：确认 experiment-id

- 查找上下文到 experiment-id，直接使用
- 若未提供，使用 `ask_user` 工具请求用户输入

### 步骤 2: 执行相应的命令

```bash
famou-ctl experiment status <experiment-id> --json  # 查看实验状态
famou-ctl experiment cancel <experiment-id> --json  # 取消实验
famou-ctl experiment delete <experiment-id> --json  # 删除实验
famou-ctl experiment logs <experiment-id> --follow/-f --output <file-path> --api-url <url> --json  # 查看并保存实验日志
famou-ctl experiment results <experiment-id> --output <file-path> --json  # 查看实验结果
```

**处理输出：**
- 命令成功：解析 JSON，清晰展示实验状态、进度、创建时间等信息
- 命令失败：展示错误信息，提示检查 experiment-id 是否正确或网络连接
