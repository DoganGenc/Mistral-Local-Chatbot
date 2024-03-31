import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import warnings
warnings.filterwarnings("ignore")
import pickle

def chat_histroy_tempalte(memory):
    chat_history=""
    for i in range(0,len(memory.chat_memory.messages),2):
        human="Human: "+memory.chat_memory.messages[i].content
        ai="AI: "+memory.chat_memory.messages[i+1].content
        conversation=human+"\n"+ai+"\n\n"
        chat_history=chat_history+conversation
    return chat_history
    
# Save db
def save_db(db):
  with open("faiss_index.pkl", "wb") as f:
      pickle.dump(db, f)

# Load db
def load_db():
  with open("faiss_index.pkl", "rb") as f:
      loaded_db = pickle.load(f)
      return loaded_db
  
def get_completion(question: str,  context: str,chat_history: str, model, tokenizer) -> str:
  device = "cuda:0"

  prompt_template = """
  <s>
  [INST]
    Answer the question based only on the following context:
    {context}
    
    Chat_history
    {chat_history}
    Question: {question}
  [/INST]
  </s>
  <s>

  """
  prompt = prompt_template.format(question=question,context=context,chat_history=chat_history)
  encodeds = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
  model_inputs = encodeds.to(device)

  generated_ids = model.generate(**model_inputs, max_new_tokens=1000, do_sample=True, pad_token_id=tokenizer.eos_token_id)
  decoded = tokenizer.batch_decode(generated_ids)
  return (decoded[0])

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model_id = "mistralai/Mistral-7B-Instruct-v0.2"
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0})
tokenizer = AutoTokenizer.from_pretrained(model_id, add_eos_token=True)
tokenizer.pad_token = tokenizer.eos_token

# Load data
try:
    db=load_db()
except:
    loader = PyPDFLoader("path\\to\\your\\pdf")
    pages = loader.load_and_split()

    db = FAISS.from_documents(pages, 
                            HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2'))
    save_db(db)

# Connect query to FAISS index using a retriever
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={'k': 4}
)


from langchain.memory import ConversationBufferMemory

# Instantiate ConversationBufferMemory
memory = ConversationBufferMemory(
 return_messages=True, output_key="answer", input_key="question"
)
inputs = {"question": "My name is Optimus Prime can you remember it"}
memory.save_context(inputs, {"answer": "of course I can remember"})

chat_history= chat_histroy_tempalte(memory)

while True:
    question = str(input("You: "))
    if question.lower() in ["quit","exit"] :
       break
    context = retriever.get_relevant_documents(question)
    all_context=''
    for i in context:
        all_context =  all_context + i.page_content
    result = get_completion(question=question,context=all_context, chat_history=chat_history, model=model, tokenizer=tokenizer)
    result=result.split("</s>")
    print("AI: ",result[-2])
    memory.save_context({'question':question},{'answer':result[-2]})
    chat_history=chat_histroy_tempalte(memory)