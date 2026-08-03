import os
import re
import json
import asyncio
import logging
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv

from semantic_kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent, AgentGroupChat
from semantic_kernel.agents.strategies import TerminationStrategy
from semantic_kernel.connectors.ai.open_ai import (
    AzureChatCompletion,
    OpenAIChatPromptExecutionSettings,
)
from semantic_kernel.functions import KernelArguments


# ============================================================
# AgentQuant: AI-Powered Data Analysis and Reporting Workflow
# ============================================================


# ============================================================
# Workflow Paths
# ============================================================

class WorkflowPaths:
    """
    Central file and folder path manager for the project.
    """

    def __init__(self, root_path: Path):
        self.project_root = root_path
        self.data_dir = self.project_root / "data"
        self.logs_dir = self.project_root / "logs"
        self.specs_dir = self.project_root / "specs"
        self.artifacts_dir = self.project_root / "artifacts"

        self.agent_log_file = self.logs_dir / "agent_chat.log"
        self.cleaned_data_file = self.project_root / "data-cleaned.json"
        self.visualization_code_file = self.artifacts_dir / "data_visualization_code.py"
        self.visualization_file = self.artifacts_dir / "data_visualization.png"
        self.final_report_file = self.artifacts_dir / "final_report.md"

    def ensure_directories(self) -> None:
        """
        Create required folders if missing.
        """
        for directory in [
            self.data_dir,
            self.logs_dir,
            self.specs_dir,
            self.artifacts_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)


PROJECT_ROOT = Path(__file__).parent.resolve()
PATHS = WorkflowPaths(PROJECT_ROOT)
PATHS.ensure_directories()


# ============================================================
# Logging
# ============================================================

class AgentMessageLogger:
    """
    Handles structured logging of agent communication.
    Logs are written to logs/agent_chat.log.
    """

    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.logger = logging.getLogger("agent_logger")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        if not self.logger.handlers:
            handler = logging.FileHandler(self.log_file, mode="a", encoding="utf-8")
            formatter = logging.Formatter(
                fmt="%(asctime)s | %(levelname)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def log_agent_message(self, role: str, name: str, content: str) -> None:
        """
        Log role, name, and content.
        """
        try:
            safe_content = str(content).replace("\n", "\\n")
            self.logger.info(
                f"role={role} | name={name} | content={safe_content}"
            )
        except Exception as exc:
            print(f"[Logging Error] {exc}")


AGENT_LOGGER = AgentMessageLogger(PATHS.agent_log_file)


def log_agent_message(role: str, name: str, content: str) -> None:
    """
    Wrapper function required by rubric.
    """
    AGENT_LOGGER.log_agent_message(role, name, content)


# ============================================================
# Artifact Manager
# ============================================================

class WorkflowArtifacts:
    """
    Handles loading and saving workflow files.
    """

    def __init__(self, paths: WorkflowPaths):
        self.paths = paths

    def save_cleaned_data(self, data: Any) -> None:
        try:
            with open(self.paths.cleaned_data_file, "w", encoding="utf-8") as file:
                json.dump(data, file, indent=2, ensure_ascii=False)
            print(f"Cleaned data saved to: {self.paths.cleaned_data_file}")
        except Exception as exc:
            raise RuntimeError(f"Failed to save cleaned data: {exc}") from exc

    def save_final_report(self, report_content: str) -> None:
        try:
            with open(self.paths.final_report_file, "w", encoding="utf-8") as file:
                file.write(report_content)
            print(f"Final report saved to: {self.paths.final_report_file}")
        except Exception as exc:
            raise RuntimeError(f"Failed to save final report: {exc}") from exc

    def load_logs(self) -> str:
        try:
            if not self.paths.agent_log_file.exists():
                return "No agent logs found."
            return self.paths.agent_log_file.read_text(encoding="utf-8")
        except Exception as exc:
            return f"Could not load logs: {exc}"

    def load_spec_file(self, file_name: str, fallback_text: str) -> str:
        try:
            file_path = self.paths.specs_dir / file_name
            if not file_path.exists():
                return fallback_text
            return file_path.read_text(encoding="utf-8")
        except Exception as exc:
            return f"{fallback_text}\n\nSpec file could not be loaded: {exc}"


ARTIFACTS = WorkflowArtifacts(PATHS)


# ============================================================
# Environment Setup
# ============================================================

load_dotenv()

API_KEY = os.getenv("AZURE_OPENAI_KEY")
BASE_URL = os.getenv("URL")
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4.1")
API_VERSION = "2024-05-01-preview"

if not API_KEY:
    raise ValueError("AZURE_OPENAI_KEY is missing from .env file.")

if not BASE_URL:
    raise ValueError("URL is missing from .env file.")


# ============================================================
# Semantic Kernel Setup
# ============================================================

kernel = Kernel()

chat_service = AzureChatCompletion(
    service_id="azure_openai_chat",
    deployment_name=DEPLOYMENT_NAME,
    endpoint=BASE_URL,
    api_key=API_KEY,
    api_version=API_VERSION,
)

kernel.add_service(chat_service)


# ============================================================
# Helper Functions
# ============================================================

def get_csv_name() -> str:
    """
    Lists CSV files in the data folder and allows user selection.
    """
    try:
        csv_files = sorted(
            [file for file in PATHS.data_dir.iterdir() if file.suffix.lower() == ".csv"]
        )

        if not csv_files:
            raise FileNotFoundError(
                f"No CSV files found in {PATHS.data_dir}. Add one CSV file inside data folder."
            )

        print("\nAvailable CSV files:")
        for index, file in enumerate(csv_files, start=1):
            print(f"{index}. {file.name}")

        while True:
            selected = input("\nSelect CSV file number: ").strip()

            if selected.isdigit():
                selected_index = int(selected) - 1

                if 0 <= selected_index < len(csv_files):
                    selected_file = csv_files[selected_index]
                    print(f"Selected CSV file: {selected_file.name}")
                    return selected_file.name

            print("Invalid selection. Please enter a valid number.")

    except Exception as exc:
        raise RuntimeError(f"CSV selection failed: {exc}") from exc


def load_csv_file(csv_name: str) -> pd.DataFrame:
    """
    Loads a selected CSV file from the data folder.
    """
    try:
        csv_path = PATHS.data_dir / csv_name

        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        df = pd.read_csv(csv_path)
        print(f"\nCSV loaded successfully: {csv_path}")
        print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        return df

    except Exception as exc:
        raise RuntimeError(f"Failed to load CSV file: {exc}") from exc


def load_quality_instructions() -> str:
    """
    Loads data quality instructions from specs folder.
    """
    fallback = """
Data Quality Instructions:
1. Verify that missing values are handled.
2. Verify that duplicate rows are removed.
3. Verify that numeric outliers are identified and removed.
4. Verify that statistics are calculated after data cleaning.
5. Verify that cleaned data is suitable for visualization.
"""
    return ARTIFACTS.load_spec_file("Data_Quality_Instructions.txt", fallback)


def load_reports_instructions() -> str:
    """
    Loads report instructions from specs folder.
    """
    fallback = """
Report Instructions:
1. Use markdown format.
2. Include Executive Summary.
3. Include Data Cleaning Summary.
4. Include Statistical Analysis.
5. Include Visualization Summary.
6. Include Agent Workflow Log Summary.
7. Include Validation Summary.
8. Include Final Conclusion.
"""
    return ARTIFACTS.load_spec_file("Report_Instructions.txt", fallback)


def load_logs() -> str:
    """
    Loads agent interaction logs.
    """
    return ARTIFACTS.load_logs()


def save_final_report(report_content: str) -> None:
    """
    Saves final markdown report.
    """
    ARTIFACTS.save_final_report(report_content)


def dataframe_to_prompt(df: pd.DataFrame, max_rows: int = 40) -> str:
    """
    Converts dataframe preview into agent-readable prompt.
    """
    preview = df.head(max_rows).to_csv(index=False)
    columns = ", ".join(df.columns)
    shape = f"{df.shape[0]} rows x {df.shape[1]} columns"

    return (
        f"Dataset shape: {shape}\n"
        f"Columns: {columns}\n\n"
        f"CSV Preview:\n{preview}"
    )


def extract_json_from_text(text: str) -> Optional[Any]:
    """
    Extracts JSON object or list from agent response.
    """
    try:
        return json.loads(text)
    except Exception:
        pass

    try:
        match = re.search(r"```json\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if match:
            return json.loads(match.group(1).strip())
    except Exception:
        pass

    try:
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            return json.loads(match.group(1).strip())
    except Exception:
        pass

    try:
        match = re.search(r"(\[.*\])", text, re.DOTALL)
        if match:
            return json.loads(match.group(1).strip())
    except Exception:
        pass

    return None


def extract_python_code(text: str) -> str:
    """
    Extracts raw Python code from agent output.
    """
    python_block = re.search(
        r"```python\s*(.*?)```",
        text,
        re.DOTALL | re.IGNORECASE,
    )

    if python_block:
        return python_block.group(1).strip()

    generic_block = re.search(r"```\s*(.*?)```", text, re.DOTALL)

    if generic_block:
        return generic_block.group(1).strip()

    return text.strip()


# ============================================================
# Python Executor
# ============================================================

class PythonExecutor:
    """
    Executes Python code generated by PythonExecutorAgent.
    """

    def __init__(self, paths: WorkflowPaths):
        self.paths = paths

    def run(self, code: str) -> Dict[str, Any]:
        """
        Executes generated Python code and returns result dictionary.
        """
        try:
            self.paths.artifacts_dir.mkdir(parents=True, exist_ok=True)

            clean_code = extract_python_code(code)

            with open(self.paths.visualization_code_file, "w", encoding="utf-8") as file:
                file.write(clean_code)

            execution_globals = {
                "__name__": "__main__",
                "PROJECT_ROOT": str(self.paths.project_root),
                "DATA_DIR": str(self.paths.data_dir),
                "ARTIFACTS_DIR": str(self.paths.artifacts_dir),
                "CLEANED_DATA_FILE": str(self.paths.cleaned_data_file),
                "VISUALIZATION_FILE": str(self.paths.visualization_file),
            }

            execution_locals: Dict[str, Any] = {}

            exec(clean_code, execution_globals, execution_locals)

            if not self.paths.visualization_file.exists():
                return {
                    "success": False,
                    "output": "",
                    "error": f"Code ran but did not create {self.paths.visualization_file}",
                }

            return {
                "success": True,
                "output": f"Code executed successfully. Plot saved to {self.paths.visualization_file}",
                "error": "",
            }

        except Exception:
            return {
                "success": False,
                "output": "",
                "error": traceback.format_exc(),
            }


# ============================================================
# Termination Strategy
# ============================================================

class ApprovalTerminationStrategy(TerminationStrategy):
    """
    Approval-based termination strategy used by AgentGroupChat.
    """

    async def should_agent_terminate(self, agent, history) -> bool:
        try:
            if not history:
                return False

            last_message = str(history[-1]).lower()

            approval_keywords = [
                "approved",
                "validation_status",
                "report_status",
                "success",
            ]

            return any(keyword in last_message for keyword in approval_keywords)

        except Exception:
            return False


# ============================================================
# Agent Configuration
# ============================================================

QUALITY_INSTRUCTIONS = load_quality_instructions()
REPORT_INSTRUCTIONS = load_reports_instructions()

AGENT_CONFIG = {
    "DataCleaning": {
        "temperature": 0.7,
        "instructions": """
You are a Data Cleaning Assistant.

Your task:
- Analyze the raw CSV data.
- Present a short cleaning plan inside JSON.
- Handle missing values where reasonable.
- Remove duplicate rows.
- Identify and remove clear numeric outliers.
- Preserve important columns.
- Return final cleaned data only as valid JSON.

Expected JSON format:
{
  "cleaning_plan": ["step 1", "step 2"],
  "cleaning_summary": {
    "missing_values_handled": true,
    "duplicates_removed": true,
    "outliers_removed": true
  },
  "cleaned_data": [
    {"column1": "value", "column2": 123}
  ]
}
""",
    },
    "DataStatistics": {
        "temperature": 0.5,
        "instructions": """
You are a Data Statistics Assistant.

Your task:
- Receive cleaned data.
- Generate descriptive statistics.
- Include mean, median, min, max, and standard deviation for numeric columns.
- Include frequency counts for categorical columns where useful.
- Return only valid JSON.
- Do not include commentary outside JSON.

Expected JSON format:
{
  "numeric_statistics": {
    "column_name": {
      "mean": 0,
      "median": 0,
      "min": 0,
      "max": 0,
      "std": 0
    }
  },
  "categorical_statistics": {
    "column_name": {
      "category": 0
    }
  }
}
""",
    },
    "AnalysisChecker": {
        "temperature": 0.2,
        "instructions": f"""
You are a Data Validation Auditor.

Your task:
- Verify that data cleaning was completed before statistics.
- Verify that outlier removal was attempted.
- Verify that statistics were calculated from cleaned data.
- Use these quality instructions:

{QUALITY_INSTRUCTIONS}

Return only valid JSON.

Expected JSON format:
{{
  "validation_status": "approved",
  "checks": [
    {{"check": "cleaning performed", "status": "pass"}},
    {{"check": "statistics calculated", "status": "pass"}}
  ],
  "issues": []
}}
""",
    },
    "PythonExecutorAgent": {
        "temperature": 0.1,
        "instructions": """
You are a Python code generation agent.

Your task:
- Generate runnable Python code for data visualization.
- Use pandas and matplotlib.
- Read cleaned data from data-cleaned.json.
- Read original CSV from the file path provided in the prompt.
- Plot original data in blue.
- Plot cleaned data in green.
- Use a single line chart where possible.
- Save the plot to artifacts/data_visualization.png.
- Ensure the artifacts directory exists.
- Output only raw Python code.
- Do not include explanations.
- Do not include markdown fences.
- Do not include comments.
""",
    },
    "ReportGenerator": {
        "temperature": 1.0,
        "instructions": f"""
You are a Report Generator.

Your task:
- Generate a structured markdown report.
- Use cleaned data summary, statistics, validation result, visualization path, and agent logs.
- Follow these report instructions:

{REPORT_INSTRUCTIONS}

The report must include:
# AI-Powered Data Analysis Report
## Executive Summary
## Data Cleaning Summary
## Statistical Analysis
## Visualization Summary
## Agent Workflow Log Summary
## Validation Summary
## Final Conclusion
""",
    },
    "ReportChecker": {
        "temperature": 0.2,
        "instructions": f"""
You are a Report Validation Auditor.

Your task:
- Review the generated markdown report.
- Check completeness, clarity, accuracy, and formatting.
- Validate against these instructions:

{REPORT_INSTRUCTIONS}

Return only valid JSON.

Expected JSON format:
{{
  "report_status": "approved",
  "checks": [
    {{"check": "executive summary present", "status": "pass"}},
    {{"check": "statistics section present", "status": "pass"}},
    {{"check": "visualization section present", "status": "pass"}}
  ],
  "issues": []
}}
""",
    },
}


# ============================================================
# Agent Factory
# ============================================================

AGENT_ARGUMENTS: Dict[str, KernelArguments] = {}


def create_agent(name: str, instructions: str, temperature: Optional[float] = None) -> ChatCompletionAgent:
    """
    Creates a ChatCompletionAgent with optional execution settings.
    """
    if temperature is not None:
        settings = OpenAIChatPromptExecutionSettings(
            service_id="azure_openai_chat",
            temperature=temperature,
        )
        AGENT_ARGUMENTS[name] = KernelArguments(settings=settings)

    try:
        agent = ChatCompletionAgent(
            service=chat_service,
            name=name,
            instructions=instructions,
        )
        return agent

    except TypeError:
        agent = ChatCompletionAgent(
            kernel=kernel,
            name=name,
            instructions=instructions,
        )
        return agent


# ============================================================
# Agents
# ============================================================

DataCleaning = create_agent(
    name="DataCleaning",
    instructions=AGENT_CONFIG["DataCleaning"]["instructions"],
    temperature=AGENT_CONFIG["DataCleaning"]["temperature"],
)

DataStatistics = create_agent(
    name="DataStatistics",
    instructions=AGENT_CONFIG["DataStatistics"]["instructions"],
    temperature=AGENT_CONFIG["DataStatistics"]["temperature"],
)

AnalysisChecker = create_agent(
    name="AnalysisChecker",
    instructions=AGENT_CONFIG["AnalysisChecker"]["instructions"],
    temperature=AGENT_CONFIG["AnalysisChecker"]["temperature"],
)

PythonExecutorAgent = create_agent(
    name="PythonExecutorAgent",
    instructions=AGENT_CONFIG["PythonExecutorAgent"]["instructions"],
    temperature=AGENT_CONFIG["PythonExecutorAgent"]["temperature"],
)

ReportGenerator = create_agent(
    name="ReportGenerator",
    instructions=AGENT_CONFIG["ReportGenerator"]["instructions"],
    temperature=AGENT_CONFIG["ReportGenerator"]["temperature"],
)

ReportChecker = create_agent(
    name="ReportChecker",
    instructions=AGENT_CONFIG["ReportChecker"]["instructions"],
    temperature=AGENT_CONFIG["ReportChecker"]["temperature"],
)


# ============================================================
# Group Chats
# ============================================================

analysis_chat = AgentGroupChat(
    agents=[
        DataCleaning,
        DataStatistics,
        AnalysisChecker,
    ],
    termination_strategy=ApprovalTerminationStrategy(),
)

code_chat = AgentGroupChat(
    agents=[
        PythonExecutorAgent,
    ],
    termination_strategy=ApprovalTerminationStrategy(),
)

report_chat = AgentGroupChat(
    agents=[
        ReportGenerator,
        ReportChecker,
    ],
    termination_strategy=ApprovalTerminationStrategy(),
)


# ============================================================
# Agent Invocation
# ============================================================

async def invoke_agent(agent: ChatCompletionAgent, prompt: str) -> str:
    """
    Invokes a single agent and returns text response.
    """
    log_agent_message("user", "User", prompt)

    arguments = AGENT_ARGUMENTS.get(agent.name)

    try:
        response_chunks: List[str] = []

        try:
            if arguments is not None:
                async for response in agent.invoke(messages=prompt, arguments=arguments):
                    response_text = str(getattr(response, "content", response))
                    response_chunks.append(response_text)
            else:
                async for response in agent.invoke(messages=prompt):
                    response_text = str(getattr(response, "content", response))
                    response_chunks.append(response_text)

        except TypeError:
            try:
                if arguments is not None:
                    result = await agent.invoke(prompt, arguments=arguments)
                else:
                    result = await agent.invoke(prompt)

                response_chunks.append(str(getattr(result, "content", result)))

            except TypeError:
                result = await agent.invoke(prompt)
                response_chunks.append(str(getattr(result, "content", result)))

        final_response = "\n".join(response_chunks).strip()

        log_agent_message("assistant", agent.name, final_response)

        return final_response

    except Exception as exc:
        error_text = f"Agent invocation failed for {agent.name}: {exc}"
        log_agent_message("error", agent.name, error_text)
        raise RuntimeError(error_text) from exc


# ============================================================
# Workflow Phases
# ============================================================

async def run_analysis_phase(raw_data_prompt: str) -> Dict[str, Any]:
    """
    Runs DataCleaning, DataStatistics, and AnalysisChecker.
    """
    print("\n====================================")
    print("PHASE 1: ANALYSIS CHAT")
    print("====================================")

    cleaning_prompt = (
        "Clean the following raw CSV data.\n\n"
        f"{raw_data_prompt}"
    )

    cleaning_output = await invoke_agent(DataCleaning, cleaning_prompt)
    cleaned_json = extract_json_from_text(cleaning_output)

    if cleaned_json is None:
        raise ValueError("DataCleaning did not return valid JSON.")

    cleaned_data = cleaned_json.get("cleaned_data", cleaned_json)

    statistics_prompt = (
        "Generate descriptive statistics for this cleaned data.\n\n"
        f"{json.dumps(cleaned_data, indent=2, ensure_ascii=False)}"
    )

    statistics_output = await invoke_agent(DataStatistics, statistics_prompt)
    statistics_json = extract_json_from_text(statistics_output)

    if statistics_json is None:
        raise ValueError("DataStatistics did not return valid JSON.")

    checker_prompt = (
        "Validate this analysis workflow.\n\n"
        f"Cleaning output:\n{json.dumps(cleaned_json, indent=2, ensure_ascii=False)}\n\n"
        f"Statistics output:\n{json.dumps(statistics_json, indent=2, ensure_ascii=False)}"
    )

    checker_output = await invoke_agent(AnalysisChecker, checker_prompt)
    checker_json = extract_json_from_text(checker_output)

    if checker_json is None:
        checker_json = {
            "validation_status": "manual_review_required",
            "raw_output": checker_output,
        }

    return {
        "cleaning_output": cleaned_json,
        "cleaned_data": cleaned_data,
        "statistics_output": statistics_json,
        "analysis_validation": checker_json,
    }


async def run_code_phase(cleaned_data: Any, original_csv_name: str) -> Dict[str, Any]:
    """
    Runs PythonExecutorAgent and executes code with retry logic.
    """
    print("\n====================================")
    print("PHASE 2: CODE CHAT")
    print("====================================")

    executor = PythonExecutor(PATHS)

    base_prompt = (
        "Generate Python visualization code.\n\n"
        f"Original CSV path: {PATHS.data_dir / original_csv_name}\n"
        f"Cleaned data JSON path: {PATHS.cleaned_data_file}\n"
        f"Visualization output path: {PATHS.visualization_file}\n\n"
        "Requirements:\n"
        "- Use matplotlib.\n"
        "- Plot original data in blue and cleaned data in green where possible.\n"
        "- Save the final plot to artifacts/data_visualization.png.\n"
        "- Output only raw Python code."
    )

    last_error = ""

    for attempt in range(1, 11):
        print(f"\nCode attempt {attempt}/10")

        if last_error:
            prompt = (
                f"The previous Python code failed with this error:\n{last_error}\n\n"
                "Generate corrected raw Python code only."
            )
        else:
            prompt = base_prompt

        generated_code = await invoke_agent(PythonExecutorAgent, prompt)

        result = executor.run(generated_code)

        log_agent_message(
            "executor",
            "PythonExecutor",
            json.dumps(result, indent=2, ensure_ascii=False),
        )

        if result["success"]:
            print("Python code executed successfully.")
            return {
                "success": True,
                "attempts": attempt,
                "code": extract_python_code(generated_code),
                "execution_result": result,
            }

        print("Python code failed. Retrying...")
        last_error = result["error"]

    raise RuntimeError(f"Python code failed after 10 retries.\n{last_error}")


async def run_report_phase(
    analysis_results: Dict[str, Any],
    code_results: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Runs ReportGenerator and ReportChecker.
    """
    print("\n====================================")
    print("PHASE 3: REPORT CHAT")
    print("====================================")

    logs = load_logs()

    report_prompt = (
        "Generate the final markdown report using the following data.\n\n"
        f"Analysis results:\n{json.dumps(analysis_results, indent=2, ensure_ascii=False)}\n\n"
        f"Code results:\n{json.dumps(code_results, indent=2, ensure_ascii=False)}\n\n"
        f"Visualization path: {PATHS.visualization_file}\n\n"
        f"Agent logs:\n{logs}"
    )

    report_output = await invoke_agent(ReportGenerator, report_prompt)

    save_final_report(report_output)

    checker_prompt = (
        "Validate this final markdown report.\n\n"
        f"{report_output}"
    )

    checker_output = await invoke_agent(ReportChecker, checker_prompt)
    checker_json = extract_json_from_text(checker_output)

    if checker_json is None:
        checker_json = {
            "report_status": "manual_review_required",
            "raw_output": checker_output,
        }

    return {
        "report": report_output,
        "report_validation": checker_json,
    }


# ============================================================
# Main Workflow
# ============================================================

async def main() -> None:
    """
    Main end-to-end workflow.
    """
    try:
        print("\n====================================")
        print("AgentQuant AI Data Analysis Workflow")
        print("====================================")

        csv_name = get_csv_name()

        raw_df = load_csv_file(csv_name)

        raw_data_prompt = dataframe_to_prompt(raw_df)

        analysis_results = await run_analysis_phase(raw_data_prompt)

        ARTIFACTS.save_cleaned_data(analysis_results["cleaned_data"])

        print("\n====================================")
        print("HUMAN APPROVAL CHECKPOINT")
        print("====================================")
        print("Analysis phase completed.")
        print(f"Cleaned data saved to: {PATHS.cleaned_data_file}")
        print("\nValidation result:")
        print(json.dumps(analysis_results["analysis_validation"], indent=2, ensure_ascii=False))

        approval = input("\nType 'yes' to approve and continue to visualization: ").strip().lower()

        log_agent_message("human", "HumanApproval", approval)

        if approval != "yes":
            print("Workflow stopped by human approval checkpoint.")
            return

        code_results = await run_code_phase(
            cleaned_data=analysis_results["cleaned_data"],
            original_csv_name=csv_name,
        )

        report_results = await run_report_phase(
            analysis_results=analysis_results,
            code_results=code_results,
        )

        print("\n====================================")
        print("WORKFLOW COMPLETED SUCCESSFULLY")
        print("====================================")
        print(f"Cleaned data: {PATHS.cleaned_data_file}")
        print(f"Visualization code: {PATHS.visualization_code_file}")
        print(f"Visualization image: {PATHS.visualization_file}")
        print(f"Final report: {PATHS.final_report_file}")
        print(f"Agent logs: {PATHS.agent_log_file}")

        print("\nReport validation:")
        print(json.dumps(report_results["report_validation"], indent=2, ensure_ascii=False))

    except Exception as exc:
        error_message = f"Workflow failed: {exc}"
        print(error_message)
        print(traceback.format_exc())
        log_agent_message("error", "main", error_message)


if __name__ == "__main__":
    asyncio.run(main())