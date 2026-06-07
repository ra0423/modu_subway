import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pydeck as pdk  # 💡 지도 시각화를 위해 누락되었던 pydeck 추가
from google import genai
from google.genai import types
from supabase import create_client, Client

# 한글 폰트 깨짐 방지 설정 (Style 설정)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# 0. 페이지 설정
st.set_page_config(
    page_title="고령층 지하철 패턴 및 역세권 분석 시스템",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 1. 시크릿 및 클라이언트 초기화
@st.cache_resource
def init_clients():
    try:
        gemini_key = st.secrets["GEMINI_API_KEY"]
        supabase_url = st.secrets["SUPABASE_URL"]
        supabase_key = st.secrets["SUPABASE_KEY"]
        
        # 최신 google-genai SDK 클라이언트 생성
        ai_client = genai.Client(api_key=gemini_key)
        # Supabase 클라이언트 생성
        supabase_client = create_client(supabase_url, supabase_key)
        return ai_client, supabase_client
    except Exception as e:
        st.error(f"시크릿 로드 또는 클라이언트 초기화 실패: {e}")
        return None, None

ai_client, supabase_client = init_clients()

# 2. 사이드바 - 데이터 업로드 가이드 및 입력 일괄 관리
st.sidebar.title("데이터 동기화 및 관리")
st.sidebar.info("💡 Supabase DB 연동 또는 CSV 직접 업로드를 통해 데이터를 실시간으로 시각화합니다.")

uploaded_senior = st.sidebar.file_uploader("1. 노인이용.csv 업로드", type=["csv"])
uploaded_congestion = st.sidebar.file_uploader("2. 시간별혼잡도.csv 업로드", type=["csv"])
uploaded_infra = st.sidebar.file_uploader("3. 역세권.csv 업로드", type=["csv"])
uploaded_geo = st.sidebar.file_uploader("4. 역좌표.csv 업로드", type=["csv"]) # 💡 업로더 통합 이동

# 데이터 로드 도우미 함수
@st.cache_data
def load_data(file, default_name):
    if file is not None:
        return pd.read_csv(file)
    return None

df_senior = load_data(uploaded_senior, "노인이용")
df_congestion = load_data(uploaded_congestion, "시간별혼잡도")
df_infra = load_data(uploaded_infra, "역세권")
df_geo = load_data(uploaded_geo, "역좌표") # 💡 load_data_auto 대신 올바른 함수명으로 수정

# 메인 타이틀
st.title("👵 고령층 지하철 이용 패턴 및 역세권 혼잡도 분석 시스템")
st.markdown("본 시스템은 서울시 지하철 데이터를 기반으로 **고령층의 유동인구 패턴**, **역세권 주요 인프라**, 그리고 **시간대별 열차 혼잡도**를 융합 분석하여 실효성 있는 교통 복지 정책을 제안합니다.")

# 데이터 준비 체크 확인
if df_senior is None or df_congestion is None or df_infra is None:
    st.warning("⚠️ 분석을 시작하려면 사이드바에서 [노인이용.csv], [시간별혼잡도.csv], [역세권.csv] 3개 파일을 모두 업로드해 주세요.")
    st.stop()

# --- 탭 구성 부분 확장 ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "⏱️ Tab 1: 역별 무임승차 제한 권고 시간 도출", 
    "🔍 Tab 2: 인프라 유형별 제한 정책 필터링", 
    "📈 Tab 3: 역세권 특성별 혼잡도 분석 및 AI 정책 리포트",
    "🌐 Tab 4: 고령층 이동 흐름 & 인프라 지도",
    "📊 Tab 5: 시나리오별 적자 회복 시뮬레이터"
])

# 공통 시간대 변수 통일 (6시부터 24시까지 데이터 슬라이싱용)
hours_all = [f"{i}시" for i in range(6, 25)]

# ==========================================
# Tab 1: 고령층 교통 안전 및 복지 최적화
# ==========================================
with tab1:
    st.header("📍 고령층 유동인구 vs 열차 혼잡도 융합 분석")
    st.caption("선택한 역의 시간대별 노인 승하차 수요와 전체 열차의 혼잡 지수를 비교하여 고위험 밀집 시간대를 파악합니다.")
    
    available_stations = sorted(list(set(df_senior['역명'].unique()) & set(df_congestion['출발역'].unique())))
    
    col1, col2 = st.columns([1, 3])
    with col1:
        selected_station = st.selectbox("🎯 분석할 지하철역 선택", available_stations)
        
    station_senior = df_senior[df_senior['역명'] == selected_station]
    station_congest = df_congestion[df_congestion['출발역'] == selected_station]
    
    if not station_senior.empty and not station_congest.empty:
        senior_ride = station_senior[station_senior['승하차'] == '승차'][hours_all].sum().values
        senior_alight = station_senior[station_senior['승하차'] == '하차'][hours_all].sum().values
        senior_total = senior_ride + senior_alight
        
        congest_values = station_congest[hours_all].mean().values
        
        fig, ax1 = plt.subplots(figsize=(12, 5))
        color1 = '#3498db'
        ax1.set_xlabel('시간대', fontweight='bold')
        ax1.set_ylabel('고령층 이용객 수 (명)', color=color1, fontweight='bold')
        bars = ax1.bar(hours_all, senior_total, color=color1, alpha=0.6, label='노인 이용객 합계(승+하차)')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, axis='x', linestyle=':', alpha=0.6)
        
        ax2 = ax1.twinx()
        color2 = '#e74c3c'
        ax2.set_ylabel('열차 혼잡도 (%)', color=color2, fontweight='bold')
        line = ax2.plot(hours_all, congest_values, color=color2, marker='o', linewidth=2.5, label='열차 혼잡도')
        ax2.tick_params(axis='y', labelcolor=color2)
        
        plt.title(f"[{selected_station}역] 시간대별 고령층 유동인구 및 열차 혼잡도 비교", fontsize=14, pad=15, fontweight='bold')
        fig.tight_layout()
        st.pyplot(fig)
        
        senior_threshold = np.percentile(senior_total, 70)
        danger_hours = []
        for idx, h in enumerate(hours_all):
            if senior_total[idx] >= senior_threshold and congest_values[idx] >= 35.0:
                danger_hours.append(f"{h} (혼잡도: {congest_values[idx]:.1f}%, 노인인구: {int(senior_total[idx])}명)")
                
        if danger_hours:
            st.warning(f"⚠️ **[{selected_station}역 안전 경고]** 고령층 유동인구와 열차 혼잡도가 동시에 높은 **위험 시간대**가 감지되었습니다. 실버 안전 요원 배치 및 안내 방송 강화가 필요합니다.\n\n" + "\n".join([f"- {dh}" for dh in danger_hours]))
        else:
            st.success("✅ 해당 역은 고령층 유동인구 대비 혼잡도가 안정적인 편입니다.")
    else:
        st.info("선택한 역의 데이터 매칭 결과가 존재하지 않습니다.")

# ==========================================
# Tab 2: 노인 인기 역세권 시설 Top 5 분석
# ==========================================
with tab2:
    st.header("🛍️ 노인 유동인구 상위 역의 역세권 인프라 분석")
    st.caption("고령층이 가장 많이 이용하는 상위 5개 역을 정의하고, 주변 주요 시설을 모니터링 및 검색합니다.")
    
    df_senior['총이용량'] = df_senior[hours_all].sum(axis=1)
    top_stations = df_senior.groupby('역명')['총이용량'].sum().nlargest(5).index.tolist()
    
    st.subheader("🔝 고령층 이용객 수 기준 상위 5개 역사")
    st.write(", ".join([f"**{idx+1}위: {st_name}역**" for idx, st_name in enumerate(top_stations)]))
    
    df_top_infra = df_infra[df_infra['역명'].isin(top_stations)][['역명', '호선', '역주변']].dropna()
    
    col_t1, col_t2 = st.columns([2, 1])
    with col_t1:
        st.markdown("#### 🏢 상위 5개 역 주변 주요 인프라 현황")
        st.dataframe(df_top_infra, use_container_width=True)
    with col_t2:
        st.markdown("#### 📊 상위 역 주변 시설 빈도")
        if not df_top_infra.empty:
            infra_counts = df_top_infra['역명'].value_counts()
            st.bar_chart(infra_counts)
            
    st.markdown("---")
    st.subheader("🔍 역세권 시설 맞춤형 키워드 필터링")
    search_keyword = st.text_input("💡 검색하고 싶은 시설 키워드를 입력해 주세요 (예: 병원, 공원, 복지관, 학교)", "공원")
    
    if search_keyword:
        filtered_infra = df_infra[df_infra['역주변'].str.contains(search_keyword, na=False, case=False)]
        unique_filtered_stations = filtered_infra['역명'].unique()
        
        senior_filtered_sum = df_senior[df_senior['역명'].isin(unique_filtered_stations)].groupby('역명')['총이용량'].sum().reset_index()
        senior_filtered_sum = senior_filtered_sum.sort_values(by='총이용량', ascending=False)
        
        st.success(f"🔑 **'{search_keyword}'** 키워드가 포함된 역세권 보유 역은 총 **{len(unique_filtered_stations)}개**입니다.")
        
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.markdown(f"##### 📈 '{search_keyword}' 인근 고령층 총이용량 순위")
            st.dataframe(senior_filtered_sum.rename(columns={'총이용량': '고령층 총이용량(명)'}), use_container_width=True, hide_index=True)
        with col_f2:
            st.markdown("##### 📥 필터링된 데이터 내보내기")
            st.write("해당 인프라를 포함하는 상세 데이터를 CSV 파일로 다운로드하여 정책 수립 자료로 활용할 수 있습니다.")
            
            csv_data = filtered_infra.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📄 필터링된 역세권 데이터 CSV 다운로드",
                data=csv_data,
                file_name=f"역세권_필터링_{search_keyword}.csv",
                mime="text/csv"
            )

# ==========================================
# Tab 3: 역세권 특성별 혼잡도 분석 및 AI 리포트
# ==========================================
with tab3:
    st.header("📊 역세권 특성별 혼잡도 패턴 및 AI 정책 리포트")
    st.caption("인프라 키워드를 기반으로 역세권 유형을 정의하고, Gemini 2.5 Flash가 실시간 맞춤형 실버 교통 솔루션을 제공합니다.")
    
    col_b1, col_b2 = st.columns([1, 1])
    
    with col_b1:
        st.subheader("🔄 역세권 인프라 특성별 혼잡도 트렌드")
        
        def categorize_infra(text):
            if any(k in str(text) for k in ['공원', '복지관', '병원', '의원', '노인']): return '복지/의료/휴양형'
            if any(k in str(text) for k in ['시장', '백화점', '쇼핑', '상가']): return '상업/중심지형'
            if any(k in str(text) for k in ['학교', '초등학교', '고등학교', '대학교']): return '교육/주거형'
            return '기반시설형'
            
        df_infra['역세권유형'] = df_infra['역주변'].apply(categorize_infra)
        
        station_type_map = df_infra.groupby('역명')['역세권유형'].first().to_dict()
        df_congestion_typed = df_congestion.copy()
        df_congestion_typed['역세권유형'] = df_congestion_typed['출발역'].map(station_type_map).fillna('기반시설형')
        
        hours_congest = [f"{i}시" for i in range(6, 25)] # 데이터 범위와 결합되도록 6시~24시 조정
        type_congest_profile = df_congestion_typed.groupby('역세권유형')[hours_congest].mean().T
        
        st.line_chart(type_congest_profile)
        st.caption("💡 주거/교육형은 출퇴근 시간에, 복지/의료형은 낮 시간대(11시~14시)에 노인층 유동 유입과 결합 시 혼잡 지수가 대폭 상승하는 패턴을 보입니다.")
        
    with col_b2:
        st.subheader("🤖 Gemini 2.5 Flash 실시간 AI 실버 리포트")
        ai_station = st.selectbox("🔮 AI 진단을 진행할 역사 선택", available_stations, key="ai_select")
        
        if st.button("🚀 AI 분석 리포트 생성"):
            if ai_client is None:
                st.error("Gemini API Key가 설정되지 않았습니다. 시크릿 설정을 확인해 주세요.")
            else:
                with st.spinner("Gemini AI가 역세권 환경 및 혼잡 위험도를 실시간 분석 중입니다..."):
                    target_infra = df_infra[df_infra['역명'] == ai_station]['역주변'].tolist()[:8]
                    target_congest = df_congestion[df_congestion['출발역'] == ai_station][hours_all].mean().to_dict()
                    
                    prompt = f"""
                    지하철역 [{ai_station}역]에 대한 복합 데이터 분석을 기반으로 고령층 맞춤형 교통 안전 대책을 수립해 주세요.
                    
                    1. 역세권 주요 주변 시설: {', '.join(map(str, target_infra))}
                    2. 시간대별 평균 열차 혼잡도 현황: {target_congest}
                    
                    위 데이터를 면밀히 검토하고, 최신 실버 교통 복지 사례를 참고하여 이 역만을 위한 '고령층 안전 대책 및 교통 복지 개선 방안'에 대해 명확한 **3줄 요약 리포트**를 작성해 주세요. 문장은 공손하고 전문적인 어조(~합니다)로 작성해 주세요.
                    """
                    
                    try:
                        response = ai_client.models.generate_content(
                            model='gemini-2.5-flash',
                            contents=prompt,
                            config=types.GenerateContentConfig(
                                system_instruction="너는 대한민국 지하철 교통 정책을 수립하는 고령사회 대응 전담 교통 데이터 과학자야.",
                                temperature=0.2,
                                tools=[types.Tool(google_search=types.GoogleSearch())]
                            )
                        )
                        
                        st.markdown("#### 📋 3줄 요약 핵심 진단 리포트")
                        st.info(response.text)
                        
                        if supabase_client:
                            try:
                                supabase_client.table('subway_analysis_logs').insert({
                                    "station_name": ai_station,
                                    "analysis_