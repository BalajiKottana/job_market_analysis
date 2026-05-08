from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import (
    JobSettings, Task, SparkPythonTask, TaskDependency, JobEnvironment,
)
from databricks.sdk.service.compute import Environment

w = WorkspaceClient()
base = "/Workspace/Users/balaji.kottana05@gmail.com/job_market_analysis"
env_key = "default_env"

# Get current job to preserve existing tasks
current = w.jobs.get(job_id=294804090840260)
existing_tasks = list(current.settings.tasks)

# Add new scraper task
existing_tasks.append(Task(
    task_key="scrape_microsoft",
    description="Scrape Microsoft Careers job postings",
    environment_key=env_key,
    spark_python_task=SparkPythonTask(
        python_file=f"{base}/scrape_microsoft_careers.py",
    ),
))

# Update clean_data to also depend on the new scraper
for task in existing_tasks:
    if task.task_key == "clean_data":
        task.depends_on.append(TaskDependency(task_key="scrape_microsoft"))

w.jobs.update(
    job_id=294804090840260,
    new_settings=JobSettings(tasks=existing_tasks),
)