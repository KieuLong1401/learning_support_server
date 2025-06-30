from googletrans import Translator

translator = Translator()
async def google_translate(texts, src='ko', dest='en'):
    if isinstance(texts, list):
        results = await translator.translate(texts, src=src, dest=dest)
        return [r.text for r in results]
    else:
        result = await translator.translate(texts, src=src, dest=dest)
        return result.text