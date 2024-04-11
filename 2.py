# import
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain import PromptTemplate
from langchain.schema import StrOutputParser
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI, HarmBlockThreshold, HarmCategory,GoogleGenerativeAIEmbeddings


load_dotenv("C:\\Users\\navan\\PycharmProjects\\ark1\\env.env")
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
# load the document and split it into chunks

doc_path='C:\\Users\\navan\\Downloads\\Waves.txt'
loader = TextLoader(doc_path,encoding="utf8")
documents = loader.load()

# split it into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
docs= text_splitter.split_documents(documents)

# create the open-source embedding function
embedding_function =GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")

# load it into Chroma
db = Chroma.from_documents(docs, embedding_function,persist_directory="chroma_db")

# query it
query =input("Type the topic you want summarized:")

retriever = db.as_retriever()
docs = retriever.invoke(query)

doc1=str(docs[-1])[14:]
doc2=str(docs[-2])[14:]
doc3=str(docs[-3])[14:]
a=0
b=0
c=0
for i in range(len(doc1)):
    if doc1[i]=="{":
        a=i
        break
for i in range(len(doc2)):
    if doc2[i]=="{":
        b=i
        break
for i in range(len(doc3)):
    if doc3[i]=="{":
        c=i
        break
doc1=doc1[:a-11]
doc2=doc2[:b-11]
doc3=doc3[:c-11]

context=doc1+doc2+doc3
context.join(context.split("\n"))


llm = GoogleGenerativeAI(model="gemini-pro",safety_settings={HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,})
# To extract data from WebBaseLoader
doc_prompt = PromptTemplate.from_template("{page_content}")

# To query Gemini
llm_prompt_template = """Write a concise summary of the following:
"{text}"
CONCISE SUMMARY:"""
llm_prompt = PromptTemplate.from_template(llm_prompt_template)


stuff_chain = (
    # Extract data from the documents and add to the key `text`.
    {
        "text": lambda context: "\n\n".join(context)
    }
    | llm_prompt         # Prompt for Gemini
    | llm                # Gemini function
    | StrOutputParser()  # output parser
)
output=(stuff_chain.invoke(context))
print(output)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertForMaskedLM, BertModel
from bert_score import BERTScorer
from rouge import Rouge
from rouge_score import rouge_scorer

scorer = BERTScorer(model_type='bert-base-uncased')
P, R, F1 = scorer.score([output], [context])
print(f"BERTScore Precision: {P.mean():.4f}, Recall: {R.mean():.4f}, F1: {F1.mean():.4f}")
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
scores = scorer.score(output, context)
print(f"ROUGE-1 Precision: {scores['rouge1'].precision:.4f}, Recall: {scores['rouge1'].recall:.4f}, F1: {scores['rouge1'].fmeasure:.4f}")
print(f"ROUGE-2 Precision: {scores['rouge2'].precision:.4f}, Recall: {scores['rouge2'].recall:.4f}, F1: {scores['rouge2'].fmeasure:.4f}")
print(f"ROUGE-L Precision: {scores['rougeL'].precision:.4f}, Recall: {scores['rougeL'].recall:.4f}, F1: {scores['rougeL'].fmeasure:.4f}")

