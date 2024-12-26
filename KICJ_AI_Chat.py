import streamlit as st
import os

from langsmith import Client
from langchain_core.runnables import RunnableConfig
from langchain_core.tracers import LangChainTracer
from langchain_core.tracers.run_collector import RunCollectorCallbackHandler
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from streamlit_feedback import streamlit_feedback

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')
LANGCHAIN_PROJECT = os.environ.get('LANGCHAIN_PROJECT')

# Langsmith 클라이언트 객체생성
client = Client()
ls_tracer = LangChainTracer(project_name=LANGCHAIN_PROJECT, client=client)
run_collector = RunCollectorCallbackHandler()
cfg = RunnableConfig()
cfg["callbacks"] = [ls_tracer, run_collector]
cfg["configurable"] = {"session_id": "any"}

# 세션에 저장될 객체들
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "store" not in st.session_state:
    st.session_state["store"] = dict()
if "last_run" not in st.session_state:
    st.session_state["last_run"] = None

# 페이지 툴바 설정
st.set_page_config(page_title="KICJ AI Chat", page_icon=":smiling_face_with_3_hearts:")

# 헤더 스타일 설정
st.markdown(
    """
    <style>
        .sub-header {
            font-size: 18px;
            font-weight: bold;
            color: #696363;
            text-align: center;
        }
        .warning-text {
            font-size: 18px;
            color: red;
            font-weight: bold;
            text-align: center;
        }
        .center-text {
            text-align: center;
            font-size: 50px;
            font-weight: bold;
        }
        .custom-divider-rainbow {
        margin: 2em 0;
        height: 2px;
        background: linear-gradient(to right, 
            #FF0000, #FFA500, #FFFF00, 
            #008000, #0000FF, #4B0082, #EE82EE); /* 무지개색 */
        }
        .custom-divider-gray {
        margin: 2em 0;
        height: 2px;
        background: #e8e7e6;
        }
    </style>
    """
    , unsafe_allow_html=True)

# 헤더 삽입
st.markdown('<div class="center-text">🥰 KICJ AI Chat</div>', unsafe_allow_html=True)
st.markdown('<div class="custom-divider-rainbow"></div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">이 프로그램은 GPT, Gemini, Claude 모델을 선택하여 대화할 수 있는 사이트입니다.</div>',
            unsafe_allow_html=True)
st.markdown('<div class="warning-text">AI는 허위정보나 오류를 포함할 수 있으니, 항상 추가 검증을 진행하시기 바랍니다.</div>', unsafe_allow_html=True)
st.markdown('<div class="custom-divider-rainbow"></div>', unsafe_allow_html=True)

# 유저의 설정 부분
model_choice = st.selectbox(
    "AI Model (Optional)",
    ('GPT-3.5-Turbo', 'GPT-4o', 'Gemini-1.5-Pro', 'Gemini-1.5-Flash', 'Claude-3.5-Sonnet', 'Claude-3-Sonnet',
     'Claude-3-Opus', 'Cluade-3-Haiku'),
    help="GPT=범용성↑, Gemini=최신성↑, Claude=독해력↑"
)

# 온도 슬라이더
# temperature = st.slider("창의력 (Optional)", 0.0, 1.0, 0.7)

# 기본 메시지 설정
default_system_message = "질문에 대하여 자세하게 답변해 주세요."

# Instruction 수정 필드
system_message = st.text_input("Instruction (Optional)", default_system_message,
                               help="AI에게 임무를 부여할 수 있습니다. (Ex. 답변을 영어로만 대답해.)")

# 헤더 삽입
st.markdown('<div class="custom-divider-gray"></div>', unsafe_allow_html=True)


# GPT 사용시 streaming을 위한 클래스
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


# 화면 출력을 위한 대화 기록
def print_messages():
    if "messages" in st.session_state and len(st.session_state["messages"]) > 0:
        for chat_message in st.session_state["messages"]:
            st.chat_message(chat_message.role).write(chat_message.content)


# 프롬프트에 전달할 누적 채팅 히스토리
def get_session_history(session_ids: str) -> BaseChatMessageHistory:
    if session_ids not in st.session_state["store"]:
        st.session_state["store"][session_ids] = ChatMessageHistory()
    return st.session_state["store"][session_ids]


# 화면에 대화출력
print_messages()

if user_input := st.chat_input("질문을 시작하세요!"):
    # 사용자가 입력한 내용
    st.chat_message("user").write(f"{user_input}")
    st.session_state["messages"].append(ChatMessage(role="user", content=user_input))
    # AI 답변
    with st.chat_message("assistant"):
        with st.spinner("생성중.."):
            stream_handler = StreamHandler(st.empty())
            if model_choice == 'GPT-3.5-Turbo':
                model_choice = 'gpt-3.5-turbo'
            elif model_choice == 'GPT-4o':
                model_choice = 'gpt-4o'
            elif model_choice == 'Gemini-1.5-Pro':
                model_choice = 'gemini-1.5-pro'
            elif model_choice == 'Gemini-1.5-Flash':
                model_choice = 'gemini-1.5-flash'
            elif model_choice == 'Claude-3.5-Sonnet':
                model_choice = 'claude-3-5-sonnet-20240620'
            elif model_choice == 'Claude-3-Sonnet':
                model_choice = 'claude-3-sonnet-20240229'
            elif model_choice == 'Claude-3-Opus':
                model_choice = 'claude-3-opus-20240229'
            elif model_choice == 'Cluade-3-Haiku':
                model_choice = 'claude-3-haiku-20240307'

            if model_choice.startswith('gemini'):
                llm = ChatGoogleGenerativeAI(model=model_choice,
                                             # temperature=temperature,
                                             )
            elif model_choice.startswith('gpt'):
                llm = ChatOpenAI(model=model_choice,
                                 # temperature=temperature,
                                 streaming=True,
                                 callbacks=[stream_handler])
            elif model_choice.startswith('claude'):
                llm = ChatAnthropic(model=model_choice,
                                    # temperature=temperature,
                                    streaming=True,
                                    callbacks=[stream_handler])

            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_message),
                    MessagesPlaceholder(variable_name="history"),
                    ("human", "{question}"),
                ]
            )
            chain = prompt | llm

            chain_with_memory = RunnableWithMessageHistory(
                chain,
                get_session_history,
                input_messages_key="question",
                history_messages_key="history"
            )

            try:
                response = chain_with_memory.invoke(
                    {"question": user_input}, cfg
                )
                st.session_state["messages"].append(ChatMessage(role="assistant", content=response.content))
                st.session_state.last_run = run_collector.traced_runs[0].id
                st.rerun()

            except Exception as e:
                st.write(f"에러발생: {e}")

if st.session_state.get("last_run"):
    feedback = streamlit_feedback(
        feedback_type="thumbs",
        optional_text_label="[선택] 의견을 입력해주세요.",
        key=f"feedback_{st.session_state.last_run}",
    )

    if feedback:
        scores = {"👍": 1, "👎": 0}
        client.create_feedback(
            st.session_state.last_run,
            feedback["type"],
            score=scores[feedback["score"]],
            comment=feedback.get("text", None),
        )
        st.toast("피드백이 전송되었습니다.", icon="🥰")