from langchain.document_loaders import UnstructuredURLLoader
import nltk
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

urls = ["https://www.moneycontrol.com/news/","https://www.moneycontrol.com/news/"]
loader = UnstructuredURLLoader(urls=urls)
data = loader.load()
print(data)