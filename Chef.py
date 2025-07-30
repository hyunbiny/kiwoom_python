import os
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_MODEL_NAME"] = "gpt-4o-mini"

from crewai import Crew, Agent, Task

# ───────────────────────────────── 에이전트 정의 ─────────────────────────────────
international_chef = Agent(
    role="세계 요리 전문가 셰프",
    goal="집에서도 손쉽게 만들 수 있는 각국의 전통 요리 레시피를 창작합니다.",
    backstory="""
    당신은 전 세계의 전통 요리에 정통한 유명 셰프입니다.
    정통 레시피를 존중하면서도, 가정에서 구할 수 있는 재료와 조리 도구로
    쉽게 따라 할 수 있도록 변형하는 능력이 탁월합니다.
    """,
    verbose=True,
)

healthy_chef = Agent(
    role="건강식 전문 셰프",
    goal="어떤 요리든 맛을 유지하면서 채식·건강 레시피로 변환합니다.",
    backstory="""
    당신은 건강식을 전문으로 하는 셰프입니다.
    기존 레시피의 풍미를 살리면서도
    재료를 채식 위주로 교체해 영양 균형을 높이는 데 능숙합니다.
    """,
    verbose=True,
)

# ───────────────────────────────── 태스크 정의 ─────────────────────────────────
normal_recipe = Task(
    description="{people}인분에 맞춘 {dish} 레시피를 작성하세요.",
    agent=international_chef,
    expected_output="""
    **반드시 다음 세 가지 섹션을 포함해야 합니다**
    1) 재료 및 분량  
    2) 조리 과정  
    3) 플레이팅·서빙 팁
    """,
    output_file="normal_recipe.md",
)

healthy_recipe = Task(
    description="위 레시피의 재료를 채식 친화적으로 교체하되 풍미를 유지하세요.",
    agent=healthy_chef,
    expected_output="""
    **반드시 다음 네 가지 섹션을 포함해야 합니다**
    1) 교체된 재료 및 분량  
    2) 조리 과정  
    3) 플레이팅·서빙 팁  
    4) 어떤 재료를 왜 교체했는지 설명
    """,
    output_file="healthy_recipe.md",
)

# ───────────────────────────────── Crew 구성 및 실행 ─────────────────────────────────
crew = Crew(
    tasks=[normal_recipe, healthy_recipe],
    agents=[international_chef, healthy_chef],
    verbose=2,
)

result = crew.kickoff(
    inputs=dict(
        dish="한국식 저녁",
        people="4",
    )
)
