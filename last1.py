import os
import re
import json
import streamlit as st
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pytubefix import YouTube
from pytubefix.cli import on_progress
from moviepy.editor import VideoFileClip, concatenate_videoclips  

# 환경 변수 로드
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# segments 폴더 생성
SEGMENT_DIR = "segments"
if not os.path.exists(SEGMENT_DIR):
    os.makedirs(SEGMENT_DIR)

# 영상 다운로드 파일 저장 디렉토리 설정
DOWNLOAD_DIR = "downloaded_videos"
if not os.path.exists(DOWNLOAD_DIR):
    os.makedirs(DOWNLOAD_DIR)

# 세션 초기화: 실행 여부, 영상 다운로드 결과, 자막, 프롬프트, LLM 응답, 병합 결과 저장
if "has_run" not in st.session_state:
    st.session_state.has_run = False
if "download_file_name" not in st.session_state:
    st.session_state.download_file_name = None
if "download_video_bytes" not in st.session_state:
    st.session_state.download_video_bytes = None
if "transcript" not in st.session_state:
    st.session_state.transcript = None
if "prompt_template" not in st.session_state:
    st.session_state.prompt_template = None
if "llm_response" not in st.session_state:
    st.session_state.llm_response = None
if "merged_file" not in st.session_state:      
    st.session_state.merged_file = None

# 자막 관련 함수
def extract_video_id(url):
    parsed_url = urlparse(url)
    hostname = parsed_url.hostname
    if hostname in ('www.youtube.com', 'youtube.com'):
        qs = parse_qs(parsed_url.query)
        return qs.get('v', [None])[0]
    elif hostname == "youtu.be":
        return parsed_url.path.lstrip("/")
    return None

def seconds_to_hms(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def fetch_transcript(url):
    video_id = extract_video_id(url)
    if not video_id:
        return "올바른 유튜브 URL을 입력해주세요."

    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        try:
            transcript = transcript_list.find_transcript(['ko'])
            st.info("한국어 원본 자막을 찾았습니다.")
            transcript_data = transcript.fetch()
        except Exception as original_error:
            st.info("한국어 자막이 없으므로 영어 자막을 자동 번역합니다.")
            transcript = transcript_list.find_transcript(['en'])
            transcript = transcript.translate('ko')
            transcript_data = transcript.fetch()

        transcript_lines = []
        for snippet in transcript_data:
            start = snippet.start
            duration = snippet.duration if hasattr(snippet, "duration") else 0
            end = start + duration
            start_str = seconds_to_hms(start)
            end_str = seconds_to_hms(end)
            transcript_lines.append(f"[{start_str} ~ {end_str}] {snippet.text}")
        transcript_text = "\n".join(transcript_lines)
        return transcript_text

    except Exception as e:
        return "자막을 가져오는 중 오류 발생: " + str(e)

# ----- 영상 다운로드 관련 함수 -----
def sanitize_filename(filename):
    return re.sub(r'[\\/*?:"<>|]', "", filename)

def get_youtube_video_id(url):
    regex_patterns = [
        r"(?:https?:\/\/)?(?:www\.)?youtu\.be\/([\w\-]{11})",
        r"(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([\w\-]{11})"
    ]
    for pattern in regex_patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def download_video_to_bytes(url):
    video_id = get_youtube_video_id(url)
    if not video_id:
        st.error("유효한 유튜브 URL이 아닙니다.")
        return None, None

    try:
        yt = YouTube(url, on_progress_callback=on_progress)
        st.write("영상 제목:", yt.title)
        stream = yt.streams.get_highest_resolution()
        if stream is None:
            st.error("다운로드 가능한 스트림을 찾을 수 없습니다.")
            return None, None

        raw_file_name = f"{yt.title}.mp4"
        file_name = sanitize_filename(raw_file_name)
        file_path = os.path.join(DOWNLOAD_DIR, file_name)
        stream.download(output_path=DOWNLOAD_DIR, filename=file_name)

        with open(file_path, "rb") as file:
            video_bytes = file.read()
        return video_bytes, file_name
    except Exception as e:
        st.error(f"에러 발생: {e}")
        return None, None

def split_video_by_segments():
    llm_response = st.session_state.llm_response
    if not llm_response:
        st.error("LLM 응답이 없습니다.")
        return

    llm_response = llm_response.strip()
    pattern = r"^```(?:json)?\s*(\{.*\})\s*```$"
    match = re.search(pattern, llm_response, re.DOTALL)
    if match:
        pure_json_str = match.group(1)
    else:
        pure_json_str = llm_response

    try:
        segments_dict = json.loads(pure_json_str)
        segments = segments_dict.get("segments", [])
    except json.JSONDecodeError as e:
        st.error("LLM 응답 파싱 중 오류: " + str(e))
        st.write("파싱 대상 문자열:", pure_json_str)
        return

    video_file_path = os.path.join(DOWNLOAD_DIR, st.session_state.download_file_name)
    if not os.path.exists(video_file_path):
        st.error("다운로드된 영상 파일을 찾을 수 없습니다.")
        return

    try:
        video_clip = VideoFileClip(video_file_path)
    except Exception as e:
        st.error("영상을 열 수 없습니다: " + str(e))
        return

    video_duration = video_clip.duration
    output_files = []
    for idx, segment in enumerate(segments):
        start_time = segment.get("start_time")
        end_time = segment.get("end_time")
        if start_time is None or end_time is None:
            continue

        if end_time > video_duration:
            end_time = video_duration

        try:
            subclip = video_clip.subclip(start_time, end_time)
        except Exception as e:
            st.error(f"Segment {idx+1} 자르는 중 에러: " + str(e))
            continue

        # 파일 이름에 3자리 패딩을 적용하여 정렬 문제 해결 (예: _segment_001.mp4)
        base_name, ext = os.path.splitext(st.session_state.download_file_name)
        output_filename = f"{base_name}_segment_{idx+1:03d}{ext}"
        output_path = os.path.join(SEGMENT_DIR, output_filename)

        subclip.write_videofile(output_path, codec="libx264", audio_codec="aac")
        st.write(f"Segment {idx+1} 저장 완료: {output_filename}")
        output_files.append(output_filename)

    video_clip.close()
    return output_files

def merge_video_segments(segment_dir, output_filename="merged_video.mp4"):
    segment_files = [f for f in os.listdir(segment_dir) if f.endswith(".mp4")]
    if not segment_files:
        st.error("병합할 세그먼트 파일이 없습니다.")
        return None
    # 자연 정렬을 위해 세그먼트 번호 추출 후 정렬
    def sort_key(f):
        m = re.search(r"segment_(\d+)", f)
        return int(m.group(1)) if m else 0

    segment_files = sorted(segment_files, key=sort_key)

    clips = []
    for f in segment_files:
        file_path = os.path.join(segment_dir, f)
        clip = VideoFileClip(file_path)
        clips.append(clip)

    final_clip = concatenate_videoclips(clips)
    final_clip.write_videofile(output_filename, codec="libx264", audio_codec="aac")

    final_clip.close()
    for clip in clips:
        clip.close()

    return output_filename

# ----- Streamlit 앱 구성 -----
st.title("유튜브 영상 다운로드 및 자막 분석 앱")

with st.form(key="input_form"):
    video_url = st.text_input("유튜브 URL을 입력하세요:")
    summary_ratio = st.number_input("자막 및 영상 요약 축약 비율(%)", min_value=1, max_value=100, value=5)
    submitted = st.form_submit_button("실행")

if submitted and video_url:
    st.header("1. 영상 다운로드 및 미리보기")
    video_bytes, filename = download_video_to_bytes(video_url)
    if video_bytes is not None:
        st.session_state.download_file_name = filename
        st.session_state.download_video_bytes = video_bytes
    else:
        st.session_state.download_file_name = None
        st.session_state.download_video_bytes = None

    transcript = fetch_transcript(video_url)
    if transcript is None or transcript.startswith("자막을 가져오는 중 오류") or transcript.startswith("올바른"):
        st.error(transcript)
        st.session_state.transcript = None
    else:
        st.session_state.transcript = transcript
        # 영상 총 길이 (초) 계산: 다운로드 완료 후 영상 파일에서
        video_file_path = os.path.join(DOWNLOAD_DIR, st.session_state.download_file_name)
        try:
            clip = VideoFileClip(video_file_path)
            total_duration = clip.duration  # 초 단위
            clip.close()
        except Exception as e:
            st.error("다운로드된 영상에서 길이 확인 실패: " + str(e))
            total_duration = None

        if total_duration is None:
            st.error("영상 길이 정보를 가져올 수 없어 요약 프롬프트 생성에 실패했습니다.")
        else:
            
            summary_duration = total_duration * (summary_ratio / 100)
            # 새로운 프롬프트 템플릿 (두 단계 접근)
            prompt_template = f"""다음은 유튜브 동영상의 전체 자막 데이터입니다.
전체 영상 길이: {total_duration:.2f}초
요약 시간: {summary_duration:.2f}초  (전체 영상 길이의 {summary_ratio}%에 해당)

먼저, 전체 자막을 꼼꼼하게 분석하여 영상에서 정말 중요한 핵심 키워드들을 추출하세요.
주의사항:
1. 반드시 핵심적인 키워드만 선택하세요. (예: 강의의 주요 개념, 중요한 용어 등)
2. 키워드 사이의 부수적인 내용은 배제하고 오직 핵심 내용만 고려하세요.

그리고 추출한 핵심 키워드에 기반하여, 각 키워드가 가장 명확하게 설명되는 구간을 아래 조건에 따라 선택하세요.
- 선택된 구간들의 총 길이가 반드시 {summary_duration:.2f}초(전체 영상 길이의 {summary_ratio}%에 해당)여야 합니다.
- 각 구간은 해당 핵심 키워드가 가장 집중된 순간이어야 하며, 관련성이 낮은 잡담이나 부수적 내용은 반드시 배제하세요.
- 만약 지정한 요약 길이를 초과할 우려가 있다면, 가장 핵심적인 구간들만 남겨 총 길이가 정확히 {summary_duration:.2f}초를 넘지 않도록 조절하세요.

각 핵심 구간은 아래 JSON 구조로 표현되어야 합니다.

$$
{{
    "segments": [
        {{
            "start_time": float,  // 구간 시작 시간 (초)
            "end_time": float,    // 구간 종료 시간 (초)
            "duration": float,    // 구간 길이 (초)
            "content": str        // 해당 구간의 핵심 내용에 대한 간결한 설명
        }}
        // 추가 구간...
    ]
}}
$$

응답은 반드시 오직 위 JSON 형식만 포함해야 하며, 추가 설명 없이 순수 JSON만 반환해 주세요.

아래에 전체 자막 내용이 주어집니다:
::::
{transcript}
::::"""
            st.session_state.prompt_template = prompt_template

            st.info("LangChain을 통해 LLM 모델에 프롬프트 전달 중입니다. 잠시 기다려주세요...")
            llm = ChatOpenAI(
                model_name="o3-mini",
                openai_api_key=api_key,
            )
            try:
                response = llm.invoke(prompt_template)
                st.session_state.llm_response = response.content
            except Exception as e:
                st.error("LLM 호출 중 오류 발생: " + str(e))
                st.session_state.llm_response = None

    st.session_state.has_run = True

if st.session_state.has_run:
    st.header("영상 다운로드 결과")
    if st.session_state.download_video_bytes and st.session_state.download_file_name:
        st.write("다운로드된 영상 파일:", st.session_state.download_file_name)
        st.download_button(
            label="영상 파일 다운로드",
            data=st.session_state.download_video_bytes,
            file_name=st.session_state.download_file_name,
            mime="video/mp4"
        )
    else:
        st.info("영상 다운로드에 실패했습니다.")

    st.header("저장된 영상 목록 및 미리보기")
    videos = [f for f in os.listdir(DOWNLOAD_DIR) if f.endswith(".mp4")]
    if videos:
        selected_video = st.selectbox("영상 선택", videos)
        if st.button("선택한 영상 미리보기", key="view_video"):
            file_path = os.path.join(DOWNLOAD_DIR, selected_video)
            try:
                with open(file_path, "rb") as f:
                    video_bytes = f.read()
                st.video(video_bytes)
            except Exception as e:
                st.error("영상을 불러오는 데 실패했습니다: " + str(e))
    else:
        st.info("저장된 영상이 없습니다.")

    st.header("2. 자막 분석 결과")
    st.subheader("LLM 응답 결과 (요약)")
    if st.session_state.llm_response:
        st.text_area("응답", st.session_state.llm_response, height=300)
    else:
        st.info("LLM 응답 없음")

    st.header("3. 영상 분할 (세그먼트 적용)")
    if st.button("영상 분할 실행"):
        output_files = split_video_by_segments()
        if output_files:
            st.write("분할된 영상 파일 목록:")
            for file in output_files:
                st.write(file)
                file_path = os.path.join(SEGMENT_DIR, file)
                with open(file_path, "rb") as f:
                    video_bytes = f.read()
                st.download_button(
                    label=f"{file} 다운로드",
                    data=video_bytes,
                    file_name=file,
                    mime="video/mp4"
                )

    st.header("4. 영상 병합")
    if st.button("영상 병합 실행"):
        merged_file = merge_video_segments(SEGMENT_DIR, output_filename="merged_video.mp4")
        if merged_file:
            st.session_state.merged_file = merged_file
            st.success("영상 병합 완료!")

    if st.session_state.get("merged_file"):
        file_path = os.path.join(".", st.session_state.merged_file)
        try:
            with open(file_path, "rb") as f:
                video_bytes = f.read()
            st.download_button(
                label="병합된 영상 다운로드",
                data=video_bytes,
                file_name=st.session_state.merged_file,
                mime="video/mp4"
            )
        except Exception as e:
            st.error("병합된 영상을 불러오지 못했습니다: " + str(e))