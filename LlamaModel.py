import lmstudio as lms
from utils import google_translate

SERVER_URL = "localhost:1234"
lms.configure_default_client(SERVER_URL)
print(lms.list_loaded_models())
model = lms.llm("meta-llama-3-8b-instruct")

async def explain_concept(word):
    word = await google_translate(word, src='en', dest='ko')
    prompt = f"\"{word}\"의 개념을 설명해주셍요. 당신은 전문가 설명자입니다. 항상 가장 중요한 정보에만 초점을 맞춰 2-3개의 간결한 문장으로 답변하세요. 불필요한 세부 사항은 피하세요. 일반 청중이 이해하기 쉽게 설명할 수 있도록 명확하고 이해하기 쉽게 하세요. 그냥 대답하세요, 저에게 아무것도 물어볼 필요 없어요. 설명으로 바로 넘어가세요. 서론, 번역 안내, 제목, 요청 반복 없이, 설명만 출력하세요. 한국어를 쓰고 영어로 번역하지마세요."
    
    result = model.complete_stream(
        prompt, 
        config={
            "temperature": 0
        }
    )

    parenthesesIsOpened = False

    for fragment in result:
        content = fragment.content

        if not content: continue

        if (content.strip() == "("): 
            parenthesesIsOpened = True
            continue

        if content.strip() == "Translation":
            break
        else:
            if parenthesesIsOpened:
                content = "(" + content
                parenthesesIsOpened = False

        yield f"data: {content}\n\n"

