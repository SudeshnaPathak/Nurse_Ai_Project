from dotenv import load_dotenv
import os
import openai
import langchain
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
# from test import vectorise

llm = ChatOpenAI(model_name='gpt-3.5-turbo-16k')
agent = ConversationChain(llm=llm, verbose=True)
global mdm_value



langchain.debug = False
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")

res = openai.chat.completions.create(model = "gpt-3.5-turbo", messages=[{"role":"user", "content":"What is the current exchange rate of dollar to naira?"}])
output = res.choices[0].message.content
print(output)

def vectorise(folder_path, filename_without_extension):

    from langchain_community.document_loaders import DirectoryLoader
    from langchain_community.document_loaders import TextLoader
    from langchain_community.vectorstores import FAISS
    from langchain.embeddings.openai import OpenAIEmbeddings
    from langchain_openai import OpenAIEmbeddings
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import PyPDFLoader
    import os
    

    # loader1 = DirectoryLoader(f'{folder_path}', glob=f"./{filename_without_extension}.docx", loader_cls=TextLoader, loader_kwargs=dict(encoding="utf-8"))
    loader = PyPDFLoader("data/upload_files/medical_test.pdf")
    document1 = loader.load()
    
    if not os.path.exists("data/vectors"):
        os.makedirs("data/vectors")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    
    pages = text_splitter.split_documents(document1)
    text = []
    doc_contents = []
    for page in pages:
        text.append(str(page))
        doc_content = '\n'.join(text)
        doc_contents.append(doc_content)
    # if doc_contents:
    #     print('\n'.join(doc_contents))
    # else:
    #     print("No .docx files found in the specified directory.")
    # for page in pages:
    #     print(page)

    embeddings = OpenAIEmbeddings()
    
    vector_folder_name = "data/vectors/"

    docs_db = FAISS.from_documents(pages, embeddings)
    docs_db.save_local(f"data/vectors/{vector_folder_name}")
    print("Successfully vectorised")

    one_word_propmt = '''Classify into "MINIMAL", "LOW", "MODERATE", "HIGH". You should ALWAYS give only ONE option and NOTHING ELSE.
ONLY return ONE option. AWAYS answer in ONLY one word. DO NOT give full sentences.
ALWAYS give the answer in ALL UPPER CASE. DO NOT EVER give answer in any other case.'''

    #####################################################

    instruction1 = '''You are an expert at calculating the Risk of Complications and/or Morbidity or Mortality of the Patient Management Decisions Made at Visit (choose highest) You are provided with a detailed medical report. Classify it into "MINIMAL", "LOW", "MODERATE", and "HIGH" on the basis of the Given rules.

MINIMAL
Minimal risk of morbidity from additional diagnostic testing or treatment
Examples only
• Rest
• Gargles
• Elastic bandages
Superficial dressings

LOW
Low risk of morbidity from additional diagnostic testing or treatment
Examples only
OTC drugs
• Minor surgery w/no identified risk factors
• Physical/Occtherapy

MODERATE
MODERATE risk of morbidity from additional diagnostic testing or treatment
Examples only
Prescription drug management
Decision regarding minor surgery with identified patient or procedure risk factors
Decision regarding elective major surgery without identified patient or procedure risk factors
Diagnosis or treatment significantly limited by social determinants of health

HIGH
HIGH risk of morbidity from additional diagnostic testing or treatment
Examples only
Parenteral controlled substances (DEA controlled substance given by route other than digestive tract)
Drug therapy requiring intensive monitoring for toxicity
Decision regarding elective major surgery with identified patient or procedure risk factors
Decision regarding emergency major surgery
Decision regarding hospitalization or escalation of hospital level care (i.e. transfer to ICU)
Decision not to resuscitate or to deescalate care because of poor prognosis


Here is the Report '''

    instruction1 += "\n\n"
    instruction1 += doc_content
    instruction1 += "\n\n"
    instruction1 += "Study it and just provide your answer"
    response1 = agent.predict(input=instruction1)

    instruction2 = one_word_propmt
    instruction2 += "\n\n"
    instruction2 += response1
    response2 = agent.predict(input=instruction2)   #main

    #####################################################

    instruction3 = '''Calculate Amount and/or Complexity of Data to be Reviewed & Analyzed (choose highest criteria met). You are provided with a detailed medical report. Classify it into "MINIMAL", "LOW", "MODERATE", and "HIGH" on the basis of the Given rules.


CATEGORY 1
1. Review of prior external note(s) from each unique source (each unique source counted once, regardless of # of notes reviewed)
2. Review of the result(s) of each unique test
3. Ordering of each unique test (includes review of result, do not count in #2)
4. Assessment requiring an Independent historian

CATEGORY 2
Independent interpretation of tests performed by another physician/other qualified healthcare professional (not separately reported)
Do not count independent interpretation for a test billed or ordered by a colleague in the same specialty

CATEGORY 3
Discussion of management or test interpretation- with external physician/other qualified health care professional/ appropriate source (not separately reported)
Requires direct interactive exchange (not via intermediaries or notes)



MINIMAL
If Minimal or No Data Reviewed


LOW
if it meets any combination of 2 from items 1-3 Or Meet item 4 (independent historian)
1. Review of prior external note(s) from each unique source (each unique source counted once, regardless of # of notes reviewed)
2. Review of the result(s) of each unique test
3. Ordering of each unique test (includes review of result, do not count in #2)
4. Assessment requiring an Independent historian


MODERATE
Meet 1 of 3 categories below
Category 1: Meet any combination of 3 from items 1-4
Category 2: Independent interpretation of test
Category 3: Discussion management, or test interpretation (external)


HIGH
Meet 2 of 3 categories below
Category 1: Meet any combination of 3 from items 1-4
Category 2: Independent interpretation of test
Category 3: Discussion management, or test interpretation (external)


Here is the Report '''

    instruction3 += "\n\n"
    instruction3 += doc_content
    instruction3 += "\n\n"
    instruction3 += "Study it and just provide your answer"
    response3 = agent.predict(input=instruction3)

    instruction4 = one_word_propmt
    instruction4 += "\n\n"
    instruction4 += response3
    response4 = agent.predict(input=instruction4)   #main

    #####################################################

    instruction5 = '''You are an expert at calc
    ulating the complexity of medical problems based on Medical reports. You are provided with a detailed medical report. Classify it into "MINIMAL", "LOW", "MODERATE", and "HIGH" on the basis of the Given rules.

MINIMAL
1 self- limited or minor problem
(runs a definite or prescribed course, is transient in nature, and is not likely to permanently alter health status)

LOW
2 or more self-limited or minor problems; or
1 stable chronic illness (chronic illness which is at treatment goal for the specific patient); or
1 acute, uncomplicated illness or injury (full recovery w/out functional impairment is expected); or
Stable, acute illness (treatment newly or recently initiated, resolution may not be complete, but condition stable); or
Acute, uncomplicated illness or injury requiring hospital inpatient or observation level care (little to no risk of mortality with treatment, but treatment required is delivered in inpt or obs setting)

MODERATE
1 or more chronic illnesses with ex- acerbation, progression, or side effects of treatment (requires supportive care or attention to treatment for side effects); or
2 or more stable chronic illnesses; or
1 undiagnosed new problem with uncertain prognosis (likely to result in high risk of morbidity w/out tx); or
1 acute illness with systemic symptoms (illness that causes systemic symptoms and has high risk of morbidity without treatment); or
1 acute complicated injury (eval of body systems not part of injured organ, extensive injury, or multiple tx options are multiple and/ or associated with risk of morbidity


HIGH
• 1 or more chronic illness- es with severe exacerbation, progression, or side effects of treatment (significant risk of morbidity: may require escalation in level of care); or
1 acute or chronic illness or injury that poses a threat to life or bodily
function (in the near term without treatment e.g. AMI, pulmonary embolus, severe respiratory distress psychiatric illness with potential threat to self or others, peritonitis, acute renal failure)

Here is the Report '''

    instruction5 += "\n\n"
    instruction5 += doc_content
    instruction5 += "\n\n"
    instruction5 += "Study it and just provide your answer"
    response5 = agent.predict(input=instruction5)

    instruction6 = one_word_propmt
    instruction6 += "\n\n"
    instruction6 += response5
    response6 = agent.predict(input=instruction6)   #main

    ######################################################

    inputs = [response2, response4, response6]
    sorted_inputs = sorted(inputs, key=lambda x: ["MINIMAL", "LOW", "MODERATE", "HIGH"].index(x))
    # print(sorted_inputs)
    mdm_value = sorted_inputs[1]
    # print(mdm_value)
    # print(mdm_value)
    return (mdm_value)

# def checking():
#     name = vectorise("data/upload_files/", "medical_test")
#     print(name)


mdm_value = vectorise("data/upload_files/", "medical_test")
def get_code1(input):

    global mdm_value

    if input=="NEW" and mdm_value=="MINIMAL":
        hcpcs_code="99202"
    elif input=="NEW" and mdm_value=="LOW":
        hcpcs_code="99203"
    elif input=="NEW" and mdm_value=="MODERATE":
        hcpcs_code="99204"
    elif input=="NEW" and mdm_value=="HIGH":
        hcpcs_code="99205"

    elif input=="ESTABLISHED" and mdm_value=="MINIMAL":
        hcpcs_code="99212"
    elif input=="ESTABLISHED" and mdm_value=="LOW":
        hcpcs_code="99213"
    elif input=="ESTABLISHED" and mdm_value=="MODERATE":
        hcpcs_code="99214"
    elif input=="ESTABLISHED" and mdm_value=="HIGH":
        hcpcs_code="99215"

    return hcpcs_code

def get_code2(input):

    if input=="20 min":
       hcpcs_code = "99242"
    elif input=="30 min":
       hcpcs_code = "99243"
    elif input=="40 min":
       hcpcs_code = "99244"
    elif input=="55 min":
       hcpcs_code = "99245"

    return hcpcs_code



# checking()
