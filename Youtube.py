import os
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_MODEL_NAME"] = "gpt-4o-mini"

from crewai import Crew, Agent, Task
from crewai_tools import SerperDevTool, ScrapeWebsiteTool, YoutubeChannelSearchTool

# ──────────────────────────── 사용 툴 ────────────────────────────
search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()
youtube_tool = YoutubeChannelSearchTool()

# ──────────────────────────── 에이전트 정의 ────────────────────────────
researcher = Agent(
    role="         ",
    goal="             ",
    backstory="""
    
    """,
    allow_delegation=False,
    tools=[search_tool, scrape_tool, youtube_tool],
    max_iter=10,
    verbose=True,
)

marketer = Agent(
    role="   ",
    goal="             ",
    backstory="""
    
    """,
    verbose=True,
)

writer = Agent(
    role="     ",
    goal="            ",
    backstory="""
    
    """,
    verbose=True,
)

# ──────────────────────────── 태스크 정의 ────────────────────────────
brainstorm_task = Task(
    description="{industry} ",
    agent=marketer,
    expected_output="""
   
   
    """,
    output_file="ideas_task.md",
    human_input=True,
)

selection_task = Task(
    description="            ",
    agent=writer,
    expected_output="""
   
    """,
    human_input=True,
    context=[brainstorm_task],
    output_file="selection_task.md",
)

research_task = Task(
    description="               ",
    agent=researcher,
    expected_output="             ",
    async_execution=True,
    context=[selection_task],
    output_file="research_task.md",
)

competitors_task = Task(
    description="{industry}                ",
    expected_output="            ",
    agent=researcher,
    async_execution=True,
    context=[selection_task],
    output_file="competitors_task.md",
)

inspiration_task = Task(
    description="                          ",
    expected_output="               ",
    agent=researcher,
    async_execution=True,
    context=[selection_task],
    output_file="inspiration_task.md",
)

script_task = Task(
    description="{industry}           ",
    expected_output="""
    
    """,
    agent=writer,
    context=[selection_task, research_task, competitors_task, inspiration_task],
    output_file="script_task.md",
)

# ──────────────────────────── Crew 구성 및 실행 ────────────────────────────
crew = Crew(
    agents=[researcher, writer, marketer],
    tasks=[
        brainstorm_task,
        selection_task,
        research_task,
        inspiration_task,
        competitors_task,
        script_task,
    ],
    verbose=2,
)

result = crew.kickoff(
    inputs=dict(
        industry="                ",
    ),
)

result
