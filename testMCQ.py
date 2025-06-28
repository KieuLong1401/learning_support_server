import os
import asyncio
import requests

from pprint import pprint

from Questgen import main
from googletrans import Translator

async def google_translate(text, src='ko', dest='en'):
    translator = Translator()
    result = await translator.translate(text, src=src, dest=dest)
    return result.text

text = """
운영체제(Operating System, OS)는 컴퓨터 시스템의 핵심 소프트웨어로서, 사용자와 하드웨어 간의 중재자 역할을 수행하며, 시스템 자원의 효율적 
관리와 프로그램 실행 환경을 제공합니다. 운영체제의 주요 기능에는 프로세스 관리, 메모리 관리, 파일 시스템, 입출력(I/O) 제어, 장치 관리, 보안 및 
사용자 인터페이스 제공 등이 포함됩니다. 프로세스 관리는 실행 중인 프로그램들의 상태를 제어하고, CPU 스케줄링을 통해 공정하고 효율적인 자원 
배분을 보장합니다. 메모리 관리는 주소 공간 분할, 가상 메모리, 페이지 교체 알고리즘 등을 통해 제한된 물리적 메모리를 최적화하며, 멀티태스킹 및 
다중 사용자 환경에서 핵심적인 역할을 합니다. 파일 시스템은 데이터의 저장, 접근, 보안 및 무결성을 유지하기 위한 구조화된 접근 방식을 제공하며, 
계층적 디렉토리 구조 및 접근 제어 리스트(ACL) 등을 통해 관리됩니다. 운영체제는 크게 커널(kernel)과 사용자 공간(user space)으로 나뉘며, 커널은 
시스템 콜을 통해 응용 프로그램과 직접적으로 상호작용하고 하드웨어 자원을 제어합니다. 운영체제의 설계 방식은 모놀리식 커널(monolithic kernel), 
마이크로커널(microkernel), 하이브리드 커널 등으로 다양하며, 각각의 구조는 안정성, 확장성, 성능 측면에서 고유한 장단점을 가집니다. 대표적인 
운영체제로는 Windows, macOS, Linux, 그리고 모바일 환경의 Android와 iOS 등이 있으며, 이들은 각각 다른 커널 구조와 시스템 호출 인터페이스, 
보안 정책을 기반으로 설계되어 다양한 사용자 요구를 충족시킵니다. 최근에는 클라우드 컴퓨팅, IoT, 엣지 컴퓨팅 등의 발전과 함께 운영체제의 경량화, 
보안성 강화, 실시간 처리 능력이 점점 더 중요해지고 있습니다.
"""
text = text.replace('\n', ' ')
translated_text = asyncio.run(google_translate(text, src='ko', dest='en'))

payload = {
    "input_text": translated_text,
}

qg = main.QGen()
output = qg.predict_mcq(payload)
questions = output["questions"]

translated_questions = []
for question in questions:
    question_statement = asyncio.run(google_translate(question["question_statement"], src='en', dest='ko'))
    answer = asyncio.run(google_translate(question["answer"], src='en', dest='ko'))
    options = [asyncio.run(google_translate(option, src='en', dest='ko')) for option in question["options"]]
    extra_options = [asyncio.run(google_translate(option, src='en', dest='ko')) for option in question["extra_options"]]
    context = asyncio.run(google_translate(question["context"], src='en', dest='ko'))

    translated_questions.append({
        "question_statement": question_statement,
        "answer": answer,
        "options": options,
        "extra_options": extra_options,
        "context": context
    })

pprint(translated_questions)