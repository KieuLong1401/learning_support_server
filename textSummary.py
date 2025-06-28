from transformers import pipeline
from summa.summarizer import summarize


model = pipeline(
    "summarization",
    model="digit82/kobart-summarization",
    tokenizer="digit82/kobart-summarization"
)

text = """
일론 머스크는 단지 트윗만으로도 디지털 화폐 시장에 영향을 줄 수 있다는 것을 다시 한 번 보여주었습니다. 그는 자신이 운영하는 전기차 제조 회사 
테슬라가 환경 문제로 인해 비트코인 결제를 받지 않겠다고 발표한 뒤, 도지코인 개발자들과 함께 거래 시스템의 효율성을 개선하기 위해 협력하고 있다고 
트윗했습니다. 이러한 두 가지 상반된 발언 이후, 세계 최대 암호화폐인 비트코인은 2개월 만에 최저치를 기록했고, 반면 도지코인은 약 20% 급등했습니다. 
스페이스X의 CEO인 머스크는 최근 몇 달간 도지코인을 지지하는 트윗을 자주 올려왔지만, 비트코인에 대해서는 거의 언급하지 않았습니다. 최근의 한 
트윗에서 머스크는 테슬라 명의로 발표한 성명을 올렸으며, 해당 성명에서는 비트코인 채굴과 거래에 있어 화석연료, 특히 석탄 사용이 급격히 증가하고 
있다는 점을 우려하며, 이로 인해 암호화폐를 이용한 차량 구매를 일시 중단한다고 밝혔습니다. 하루 뒤, 그는 다시 트윗을 통해 “명확히 하자면, 나는 
암호화폐를 강하게 믿지만, 그것이 화석연료 사용의 급증, 특히 석탄 사용을 유발해서는 안 된다”고 말했습니다. 이 발언은 비트코인 가격의 급락을 
초래했지만, 이후 시장은 점차 안정세를 되찾았습니다. 많은 트위터 사용자들이 머스크의 발언을 환영했으며, 한 사용자는 “이제 사람들이 도지코인이 
계속 존재할 것이라는 사실을 인식해야 할 때”라고 말했고, 또 다른 사용자는 머스크가 예전에 암호화폐가 세계의 미래 통화가 될 수 있다고 주장한 것을 
언급했습니다.
"""
text = text.replace('\n', ' ')

summary = model(
            text,
            max_length=len(text.split()) * 2,
            min_length=len(text.split()),
            truncation=True,
            do_sample=False
        )[0]['summary_text']
print(summary)