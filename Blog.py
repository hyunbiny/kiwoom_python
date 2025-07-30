import os
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_MODEL_NAME"] = "gpt-4o-mini"

from pydantic import BaseModel
from typing import List
from crewai import Crew, Agent, Task
from crewai_tools import SerperDevTool, ScrapeWebsiteTool

# ---------- 블로그 포스트 데이터 모델 ----------
class SubSection(BaseModel):
    title: str
    content: str

class BlogPost(BaseModel):
    title: str
    introduction: str
    sections: List[SubSection]
    sources: List[str]
    hashtags: List[str]

# ---------- 사용 툴 ----------
search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()

# ---------- 에이전트 정의 ----------
researcher = Agent(
    role="     ",
    goal="                  ",
    backstory="""
    
    """,
    allow_delegation=False,   #
    verbose=True, #
    tools=[search_tool, scrape_tool],
    max_iter=10,
)

editor = Agent(
    role="          ",
    goal="                    ",
    backstory="""
    
    
    """,
    verbose=True,
)

# ---------- 태스크 정의 ----------
task = Task(
    description="주제 {topic} 에 대해 블로그 글을 작성하세요.",
    agent=editor,
    expected_output="""
    
        """,
    output_file="blog_post.md",    
)

# ---------- Crew 구성 ----------
crew = Crew(
    agents=[researcher, editor],
    tasks=[task],
    verbose=2,
)

# ---------- 실행 ----------
result = crew.kickoff(
    inputs=dict(topic="       "),
)
