from langchain.document_loaders import UnstructuredURLLoader
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download()
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

urls = ["https://www.moneycontrol.com/news/","https://www.moneycontrol.com/news/"]
loader = UnstructuredURLLoader(urls=urls)
data = loader.load()
print(data)