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
    role="수석 리서처",
    goal="웹을 탐색하여 신뢰할 수 있는 최신 정보를 수집·분석합니다.",
    backstory="""
    당신은 최고 수준의 리서치를 수행하는 전문가입니다.
    다양한 출처를 활용해 정보를 교차 검증하며,
    정확하고 시의적절한 데이터를 찾는 데 탁월합니다.
    """,
    allow_delegation=False,
    verbose=True,
    tools=[search_tool, scrape_tool],
    max_iter=10,
)

editor = Agent(
    role="수석 작가/편집자",
    goal="흥미로운 블로그 글을 작성하여 독자의 관심을 사로잡습니다.",
    backstory="""
    당신은 쉽고 재미있지만 깊이 있는 글을 쓰는 작가입니다.
    정보를 명확하게 전달하며, 사람들이 공유하고 싶어 하는 콘텐츠를 만듭니다.
    이번 프로젝트는 매우 중요한 클라이언트를 위한 것입니다.
    """,
    verbose=True,
)

# ---------- 태스크 정의 ----------
task = Task(
    description="주제 {topic} 에 대해 블로그 글을 작성하세요.",
    agent=editor,
    expected_output="""
    - 도입부
    - 최소 3개의 소제목이 포함된 본문
    - 참고 출처 링크
    - SNS용 추천 해시태그
    - 눈길을 끄는 제목
    """,
    output_file="blog_post.md",
    # output_pydantic=BlogPost,
)

# ---------- Crew 구성 ----------
crew = Crew(
    agents=[researcher, editor],
    tasks=[task],
    verbose=2,
)

# ---------- 실행 ----------
result = crew.kickoff(
    inputs=dict(topic="키움증권"),
)





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
    role="수석 리서처",
    goal="웹을 탐색하여 정보를 수집·분석합니다.",
    backstory="""
    당신은 최고 수준의 리서치 전문가입니다.
    여러 출처를 활용해 내용을 교차 검증하고,
    최신·정확한 정보를 동료들에게 제공해 존경을 받습니다.
    """,
    allow_delegation=False,
    tools=[search_tool, scrape_tool, youtube_tool],
    max_iter=10,
    verbose=True,
)

marketer = Agent(
    role="수석 마케터",
    goal="바이럴 효과가 높은 참신하고 유용한 콘텐츠 아이디어를 도출합니다.",
    backstory="""
    당신은 마케팅 에이전시에서 일하는 아이디어 메이커입니다.
    영상·광고·SNS 마케팅 등 Z세대가 열광할 만한 기획을 가장 잘 만들어 냅니다.
    """,
    verbose=True,
)

writer = Agent(
    role="수석 작가",
    goal="유튜브에서 바이럴될 스크립트를 작성합니다.",
    backstory="""
    당신은 보는 사람을 끝까지 사로잡는 대본을 쓰는 영상 작가입니다.
    재미·정보·공유 가치를 모두 잡아, 중요한 클라이언트의 프로젝트를 담당합니다.
    """,
    verbose=True,
)

# ──────────────────────────── 태스크 정의 ────────────────────────────
brainstorm_task = Task(
    description="{industry} 업계 유튜브 채널을 위한 영상 아이디어 5가지를 제안하세요.",
    agent=marketer,
    expected_output="""
    반드시 아래 형식을 따르세요.
    - 아이디어 5가지 (각각 제목 + 영상의 핵심 각도/포인트 설명 포함)
    """,
    output_file="ideas_task.md",
    human_input=True,
)

selection_task = Task(
    description="가장 바이럴 가능성이 높은 아이디어를 1개 선택하세요.",
    agent=writer,
    expected_output="""
    - 선택한 아이디어
    - 그 아이디어를 선택한 이유
    """,
    human_input=True,
    context=[brainstorm_task],
    output_file="selection_task.md",
)

research_task = Task(
    description="선택된 아이디어로 중간 길이(약 8–10분) 영상 대본을 쓰기 위한 모든 자료를 조사하세요.",
    agent=researcher,
    expected_output="작가가 바로 대본을 쓸 수 있을 정도로 상세한 정보·데이터·근거를 제공하세요.",
    async_execution=True,
    context=[selection_task],
    output_file="research_task.md",
)

competitors_task = Task(
    description="{industry} 업계에서 유사한 영상·기사 사례를 찾아 우리 영상이 차별화될 방안을 제안하세요.",
    expected_output="영상 차별화를 위한 구체적 제안 리스트",
    agent=researcher,
    async_execution=True,
    context=[selection_task],
    output_file="competitors_task.md",
)

inspiration_task = Task(
    description="다른 업계에서 유사한 각도를 가진 영상·기사 사례를 찾아 벤치마킹 포인트를 제시하세요.",
    expected_output="타 업계 사례 목록과 적용 가능한 인사이트",
    agent=researcher,
    async_execution=True,
    context=[selection_task],
    output_file="inspiration_task.md",
)

script_task = Task(
    description="{industry} 업계 유튜브 영상을 위한 대본을 작성하세요.",
    expected_output="""
    - 영상 제목
    - 인트로
    - 최소 3개의 본문 섹션
    - 아웃트로
    - 썸네일 생성 프롬프트
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
        industry="댄스 챌린지",
    ),
)

result
