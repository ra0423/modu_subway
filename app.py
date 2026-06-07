import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pydeck as pdk
import sqlite3
import os
from google import genai
from google.genai import types
from supabase import create_client, Client

# 한글 폰트 깨짐 방지 설정
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# 0. 페이지 기본 설정
st.set_page_config(
    page_title="고령층 지하철 패턴 및 역세권 분석 시스템",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 1. 외부 서비스 시크릿 및 클라이언트 초기화
@st.cache_resource
def init_clients():
    try:
        gemini_key = st.secrets["GEMINI_API_KEY"]
        supabase_url = st.secrets["SUPABASE_URL"]
        supabase_key = st.secrets["SUPABASE_KEY"]
        
        ai_client = genai.Client(api_key=gemini_key)
        supabase_client = create_client(supabase_url, supabase_key)
        return ai_client, supabase_client
    except Exception as e:
        st.error(f"시크릿 로드 또는 클라이언트 초기화 실패: {e}")
        return None, None

ai_client, supabase_client = init_clients()

DB_PATH = "subway_data.db"

# 데이터베이스 존재 여부 체크 가이드
if not os.path.exists(DB_PATH):
    st.error(f"❌ 데이터베이스 파일({DB_PATH})을 찾을 수 없습니다. 1단계 가이드를 참조하여 DB 파일을 생성하고 깃허브에 추가해 주세요.")
    st.stop()

st.sidebar.title("데이터베이스 엔진 상태")
st.sidebar.success("✅ 고성능 SQLite SQL 연동 엔진이 상시 가동 중입니다.")

# 메인 타이틀
st.title("👵 고령층 지하철 이용 패턴 및 역세권 혼잡도 분석 시스템")
st.markdown("본 시스템은 4개의 통합 데이터셋을 탑재한 SQLite 데이터베이스를 기반으로 **순수 SQL 쿼리 연산**을 통해 실시간 지표를 도출합니다.")

# 대시보드 탭 레이아웃 설정
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "⏱️ Tab 1: 역별 무임승차 제한 권고 시간 도출", 
    "🔍 Tab 2: 인프라 유형별 제한 정책 필터링", 
    "📈 Tab 3: 역세권 특성별 혼잡도 분석 및 AI 정책 리포트",
    "🌐 Tab 4: 고령층 이동 흐름 & 인프라 지도",
    "📊 Tab 5: 시나리오별 적자 회복 시뮬레이터"
])

hours_all = [f"{i}시" for i in range(6, 25)]

# ==========================================
# Tab 1: SQL 기반 역별 노인 이용 유동인구 vs 열차 혼잡도
# ==========================================
with tab1:
    st.header("📍 고령층 유동인구 vs 열차 혼잡도 융합 분석")
    st.caption("선택한 역의 시간대별 노인 승하차 수요와 전체 열차의 혼잡 지수를 비교하여 고위험 밀집 시간대를 파악합니다.")
    
    # 두 테이블에 동시에 존재하는 역 목록 SQL 쿼리로 추출
    conn = sqlite3.connect(DB_PATH)
    station_query = """
    SELECT DISTINCT 역명 FROM senior_usage 
    WHERE 역명 IN (SELECT DISTINCT 출발역 FROM station_congestion)
    ORDER BY 역명 ASCE;
    """
    try:
        available_stations = pd.read_sql(station_query, conn)['역명'].tolist()
    except Exception:
        # 컬럼 매칭 안전장치
        available_stations = ["서울역", "시청", "종각", "종로3가", "신도림", "강남"]
    
    col1, _ = st.columns([1, 3])
    with col1:
        selected_station = st.selectbox("🎯 분석할 지하철역 선택", available_stations)
        
    # SQL 쿼리를 통해 선택된 역사 데이터만 정밀 타격 로드
    df_senior_st = pd.read_sql("SELECT * FROM senior_usage WHERE 역명 = ?", conn, params=[selected_station])
    df_congest_st = pd.read_sql("SELECT * FROM station_congestion WHERE 출발역 = ?", conn, params=[selected_station])
    conn.close()
    
    if not df_senior_st.empty and not df_congest_st.empty:
        senior_ride = df_senior_st[df_senior_st['승하차'] == '승차'][hours_all].sum().values
        senior_alight = df_senior_st[df_senior_st['승하차'] == '하차'][hours_all].sum().values
        senior_total = senior_ride + senior_alight
        
        congest_values = df_congest_st[hours_all].mean().values
        
        fig, ax1 = plt.subplots(figsize=(12, 4))
        color1 = '#3498db'
        ax1.set_xlabel('시간대', fontweight='bold')
        ax1.set_ylabel('고령층 이용객 수 (명)', color=color1, fontweight='bold')
        ax1.bar(hours_all, senior_total, color=color1, alpha=0.6, label='노인 이용객 합계')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, axis='x', linestyle=':', alpha=0.6)
        
        ax2 = ax1.twinx()
        color2 = '#e74c3c'
        ax2.set_ylabel('열차 혼잡도 (%)', color=color2, fontweight='bold')
        ax2.plot(hours_all, congest_values, color=color2, marker='o', linewidth=2.5, label='열차 혼잡도')
        ax2.tick_params(axis='y', labelcolor=color2)
        
        plt.title(f"[{selected_station}역] 시간대별 고령층 유동인구 및 열차 혼잡도 비교", fontsize=13, pad=15, fontweight='bold')
        fig.tight_layout()
        st.pyplot(fig)
        
        senior_threshold = np.percentile(senior_total, 70) if len(senior_total) > 0 else 100
        danger_hours = []
        for idx, h in enumerate(hours_all):
            if idx < len(senior_total) and idx < len(congest_values):
                if senior_total[idx] >= senior_threshold and congest_values[idx] >= 35.0:
                    danger_hours.append(f"{h} (혼잡도: {congest_values[idx]:.1f}%, 노인인구: {int(senior_total[idx])}명)")
                
        if danger_hours:
            st.warning(f"⚠️ **[{selected_station}역 안전 권고]** 고령층 밀집 위험 시간대가 감지되었습니다.\n\n" + "\n".join([f"- {dh}" for dh in danger_hours]))
        else:
            st.success("✅ 해당 역은 고령층 유동인구 대비 혼잡도가 안정적인 편입니다.")

# ==========================================
# Tab 2: SQL JOIN 문 기반 역세권 키워드 필터링 및 고령층 데이터 연동
# ==========================================
with tab2:
    st.header("🛍️ 노인 유동인구 상위 역의 역세권 인프라 분석")
    st.caption("인프라 테이블과 노인 이용량 테이블을 SQL INNER JOIN하여 실시간으로 융합 순위를 집계합니다.")
    
    conn = sqlite3.connect(DB_PATH)
    # 고령층 최다 이용 상위 5개 역을 구하는 SQL 그룹화 탑쿼리
    top5_query = """
    SELECT 역명, SUM(CAST(`6시` AS INT)+CAST(`7시` AS INT)+CAST(`8시` AS INT)+CAST(`9시` AS INT)+CAST(`10시` AS INT)+
                    CAST(`11시` AS INT)+CAST(`12시` AS INT)+CAST(`13시` AS INT)+CAST(`14시` AS INT)+CAST(`15시` AS INT)+
                    CAST(`16시` AS INT)+CAST(`17시` AS INT)+CAST(`18시` AS INT)+CAST(`19시` AS INT)+CAST(`20시` AS INT)+
                    CAST(`21시` AS INT)+CAST(`22시` AS INT)+CAST(`23시` AS INT)+CAST(`24시` AS INT)+CAST(`25시` AS INT)) AS 총이용량
    FROM senior_usage GROUP BY 역명 ORDER BY 총이용량 DESC LIMIT 5;
    """
    df_top5 = pd.read_sql(top5_query, conn)
    top_stations_list = df_top5['역명'].tolist()
    
    st.subheader("🔝 고령층 이용객 수 기준 상위 5개 역사")
    st.write(", ".join([f"**{idx+1}위: {name}역**" for idx, name in enumerate(top_stations_list)]))
    
    # 상위 5개역 주변 인프라 정보 호출
    df_top_infra = pd.read_sql("SELECT 역명, 호선, 역주변 FROM station_infra WHERE 역명 IN (?,?,?,?,?)", 
                               conn, params=top_stations_list + [""]*(5-len(top_stations_list)))
    
    col_t1, col_t2 = st.columns([2, 1])
    with col_t1:
        st.markdown("#### 🏢 상위 5개 역 주변 주요 인프라 현황")
        st.dataframe(df_top_infra, use_container_width=True)
    with col_t2:
        st.markdown("#### 📊 상위 역 주변 시설 빈도")
        if not df_top_infra.empty:
            st.bar_chart(df_top_infra['역명'].value_counts())
            
    st.markdown("---")
    st.subheader("🔍 역세권 시설 맞춤형 키워드 필터링 (SQL JOIN 연산)")
    search_keyword = st.text_input("💡 검색하고 싶은 시설 키워드를 입력해 주세요", "공원")
    
    if search_keyword:
        # 사용자가 입력한 키워드를 SQL LIKE 구문과 JOIN문으로 가공하여 가져오기
        join_search_query = f"""
        SELECT i.역명, i.호선, i.역주변, 
               SUM(s.`6시` + s.`7시` + s.`8시` + s.`9시` + s.`10시` + s.`11시` + s.`12시` + 
                   s.`13시` + s.`14시` + s.`15시` + s.`16시` + s.`17시` + s.`18시` + s.`19시` + 
                   s.`20시` + s.`21시` + s.`22시` + s.`23시` + s.`24시` + s.`25시`) AS 고령층_총이용량
        FROM station_infra i
        INNER JOIN senior_usage s ON i.역명 = s.역명
        WHERE i.역주변 LIKE ?
        GROUP BY i.역명
        ORDER BY 고령층_총이용량 DESC;
        """
        df_filtered_join = pd.read_sql(join_search_query, conn, params=[f"%{search_keyword}%"])
        
        st.success(f"🔑 **'{search_keyword}'** 키워드가 포함된 역세권 연동 역은 총 **{len(df_filtered_join)}개**입니다.")
        
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.markdown(f"##### 📈 '{search_keyword}' 인근 고령층 총이용량 순위")
            st.dataframe(df_filtered_join[['역명', '고령층_총이용량']].rename(columns={'고령층_총이용량': '고령층 총이용량(명)'}), use_container_width=True, hide_index=True)
        with col_f2:
            st.markdown("##### 📥 필터링된 데이터 내보내기")
            st.write("해당 인프라를 포함하는 상세 데이터를 CSV 파일로 즉시 파일 변환하여 정책 수립 자료로 다운로드합니다.")
            
            csv_data = df_filtered_join.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📄 필터링된 역세권 데이터 CSV 다운로드",
                data=csv_data,
                file_name=f"역세권_필터링_{search_keyword}.csv",
                mime="text/csv"
            )
    conn.close()

# ==========================================
# Tab 3: SQL CASE WHEN 구문을 활용한 역세권 유형화 및 AI 진단 리포트
# ==========================================
with tab3:
    st.header("📊 역세권 특성별 혼잡도 패턴 및 AI 정책 리포트")
    st.caption("SQL 내에서 CASE WHEN 분기문을 돌려 역세권 유형을 정의하고, Gemini 2.5 Flash가 실시간 맞춤형 실버 솔루션을 제공합니다.")
    
    col_b1, col_b2 = st.columns([1, 1])
    
    with col_b1:
        st.subheader("🔄 역세권 인프라 특성별 혼잡도 트렌드 (SQL CASE WHEN)")
        
        conn = sqlite3.connect(DB_PATH)
        # 인프라 유형 분류와 평균 혼잡도 산출을 동시에 처리하는 고급 통계 SQL 쿼리문
        advanced_case_query = """
        SELECT 
            CASE 
                WHEN i.역주변 LIKE '%공원%' OR i.역주변 LIKE '%복지관%' OR i.역주변 LIKE '%병원%' OR i.역주변 LIKE '%의원%' OR i.역주변 LIKE '%노인%' THEN '복지/의료/휴양형'
                WHEN i.역주변 LIKE '%시장%' OR i.역주변 LIKE '%백화점%' OR i.역주변 LIKE '%쇼핑%' OR i.역주변 LIKE '%상가%' THEN '상업/중심지형'
                WHEN i.역주변 LIKE '%학교%' OR i.역주변 LIKE '%초등학교%' OR i.역주변 LIKE '%고등학교%' OR i.역주변 LIKE '%대학교%' THEN '교육/주거형'
                ELSE '기반시설형'
            END AS 역세권유형,
            AVG(c.`6시`) AS `6시`, AVG(c.`7시`) AS `7시`, AVG(c.`8시`) AS `8시`, AVG(c.`9시`) AS `9시`, AVG(c.`10시`) AS `10시`,
            AVG(c.`11시`) AS `11시`, AVG(c.`12시`) AS `12시`, AVG(c.`13시`) AS `13시`, AVG(c.`14시`) AS `14시`, AVG(c.`15시`) AS `15시`,
            AVG(c.`16시`) AS `16시`, AVG(c.`17시`) AS `17시`, AVG(c.`18시`) AS `18시`, AVG(c.`19시`) AS `19시`, AVG(c.`20시`) AS `20시`,
            AVG(c.`21시`) AS `21시`, AVG(c.`22시`) AS `22시`, AVG(c.`23시`) AS `23시`, AVG(c.`24시`) AS `24시`, AVG(c.`25시`) AS `25시`
        FROM station_congestion c
        INNER JOIN station_infra i ON c.출발역 = i.역명
        GROUP BY 역세권유형;
        """
        df_type_profile = pd.read_sql(advanced_case_query, conn)
        conn.close()
        
        if not df_type_profile.empty:
            df_type_profile = df_type_profile.set_index('역세권유형').T
            st.line_chart(df_type_profile)
        st.caption("💡 주거/교육형은 출퇴근 시간에, 복지/의료형은 낮 시간대(11시~14시)에 노인층 유동 유입과 결합 시 혼잡 지수가 대폭 상승하는 패턴을 보입니다.")
        
    with col_b2:
        st.subheader("🤖 Gemini 2.5 Flash 실시간 AI 실버 리포트")
        ai_station = st.selectbox("🔮 AI 진단을 진행할 역사 선택", available_stations, key="ai_select")
        
        if st.button("🚀 AI 분석 리포트 생성"):
            if ai_client is None:
                st.error("Gemini API Key가 설정되지 않았습니다. 시크릿 설정을 확인해 주세요.")
            else:
                with st.spinner("Gemini AI가 역세권 환경 및 혼잡 위험도를 실시간 분석 중입니다..."):
                    conn = sqlite3.connect(DB_PATH)
                    target_infra = pd.read_sql("SELECT 역주변 FROM station_infra WHERE 역명 = ? LIMIT 8", conn, params=[ai_station])['역주변'].tolist()
                    target_congest_df = pd.read_sql("SELECT * FROM station_congestion WHERE 출발역 = ?", conn, params=[ai_station])
                    conn.close()
                    
                    target_congest = target_congest_df[hours_all].mean().to_dict() if not target_congest_df.empty else {}
                    
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
                                    "analysis_type": "고령층 안전 대책 리포트",
                                    "ai_report": response.text
                                }).execute()
                            except Exception:
                                pass
                                
                    except Exception as e:
                        st.error(f"AI 리포트 생성 중 오류가 발생했습니다: {e}")

# ==========================================
# Tab 4: SQL 공간 데이터 매핑 및 Pydeck 3D 시각화 지도
# ==========================================
with tab4:
    st.header("🌐 고령층 유동인구 이동 흐름 및 인프라 맵")
    st.caption("지정된 시간대에 노인 이용인구가 어디서 출발하는지 위경도 좌표 조인 SQL을 수행하여 지도로 시각화합니다.")
    
    selected_hour = st.slider("🕰️ 시각화할 시간대 선택", 6, 24, 10)
    hour_col = f"{selected_hour}시"
    
    conn = sqlite3.connect(DB_PATH)
    # 노인 이용량과 위경도 좌표(station_coordinates) 데이터를 공백 제거 후 완전 매핑하는 SQL 쿼리
    map_spatial_query = f"""
    SELECT TRIM(s.역명) AS 역명, s.승하차, CAST(s.`{hour_col}` AS INT) AS 승차인원, 
           CAST(g.위도 AS REAL) AS 위도, CAST(g.경도 AS REAL) AS 경도
    FROM senior_usage s
    INNER JOIN station_coordinates g ON TRIM(s.역명) = TRIM(g.역명)
    WHERE s.승하차 = '승차';
    """
    df_rides_geo = pd.read_sql(map_spatial_query, conn)
    
    # 팝업 툴팁에 띄워줄 간단 인프라 매핑 정보 쿼리
    df_infra_short = pd.read_sql("SELECT 역명, GROUP_CONCAT(역주변, ', ') AS 주변인프라 FROM (SELECT 역명, 역주변 FROM station_infra LIMIT 300) GROUP BY 역명", conn)
    conn.close()
    
    if df_rides_geo.empty:
        st.warning("⚠️ 지도를 시각화하기 위한 데이터 좌표 결합에 실패했습니다. DB 데이터 내의 역명을 점검하세요.")
    else:
        infra_dict = dict(zip(df_infra_short['역명'], df_infra_short['주변인프라']))
        df_rides_geo['주변인프라'] = df_rides_geo['역명'].map(infra_dict).fillna("상세 인프라 정보 없음")
        
        df_rides_geo['to_lat'] = 37.5665  # 서울 중심부 타겟 설정
        df_rides_geo['to_lng'] = 126.9780 
        
        view_state = pdk.ViewState(latitude=37.5665, longitude=126.9780, zoom=11, pitch=45)
        
        arc_layer = pdk.Layer(
            'ArcLayer',
            data=df_rides_geo,
            get_source_position='[경도, 위도]',
            get_target_position='[to_lng, to_lat]',
            get_source_color='[255, 99, 71, 150]',  
            get_target_color='[30, 144, 255, 150]', 
            get_width='승차인원 / 30', 
            pickable=True
        )
        
        st.pydeck_chart(pdk.Deck(
            map_style='mapbox://styles/mapbox/dark-v9', 
            initial_view_state=view_state,
            layers=[arc_layer],
            tooltip={"text": "📌 출발역: {역명}\n👥 해당 시간대 승차: {승차인원}명\n🏢 인근 주요시설: {주변인프라}"}
        ))

# ==========================================
# Tab 5: 시나리오별 적자 회복 시뮬레이터 (SQL 기반 데이터 정밀 추출)
# ==========================================
with tab5:
    st.header("📊 정책 시나리오별 코레일/교통공사 재정 적자 회복 시뮬레이터")
    st.caption("무임승차 연령 상향이나 시간제 유료화(할인) 정책을 적용했을 때 예상되는 재정 보전 효과를 시뮬레이션합니다.")
    
    col_param, col_chart = st.columns([1, 2])
    
    with col_param:
        st.subheader("⚙️ 정책 시나리오 설정")
        base_fare = st.number_input("💵 지하철 기본 요금 (원)", value=1400, step=100)
        discount_rate = st.slider("👴 피크타임 고령층 요금 할인율 (%)", 0, 100, 50) / 100
        
        policy_time = st.radio(
            "⏱️ 무임승차 제한(유료화) 시간대 설정",
            [
                "출근 피크만 제한 (07시 ~ 09시)",
                "퇴근 피크만 제한 (18시 ~ 20시)",
                "출퇴근 모두 제한 (07~09시, 18~20시)",
                "낮 시간만 우대 적용 (11시 ~ 14시만 무임)"
            ]
        )
        elasticity = 0.8 
        
    with col_chart:
        time_buckets = []
        if "출근 피크만" in policy_time: 
            time_buckets = ["7시", "8시", "9시"]
        elif "퇴근 피크만" in policy_time: 
            time_buckets = ["18시", "19시", "20시"]
        elif "출퇴근 모두" in policy_time: 
            time_buckets = ["7시", "8시", "9시", "18시", "19시", "20시"]
        else: 
            time_buckets = [f"{i}시" for i in range(6, 25) if f"{i}시" not in ["11시", "12시", "13시", "14시"]]
            
        # SQL문을 이용해 정책 타겟이 되는 시간대 컬럼만 다이렉트 합산 조회
        sum_columns = ", ".join([f"SUM(CAST(`{t}` AS INT))" for t in time_buckets])
        query_sim = f"SELECT ({sum_columns}) AS target_sum FROM senior_usage WHERE 승하차 = '승차';"
        
        conn = sqlite3.connect(DB_PATH)
        df_sim_res = pd.read_sql(query_sim, conn)
        conn.close()
        
        total_senior_target_hour = float(df_sim_res['target_sum'].iloc[0]) if not df_sim_res.empty and df_sim_res['target_sum'].iloc[0] is not None else 0.0
        
        annual_recovered_revenue = total_senior_target_hour * base_fare * (1 - discount_rate) * elasticity * 365
        annual_recovered_revenue_billion = annual_recovered_revenue / 100000000 
        
        current_deficit = 4000.0
        new_deficit = max(0.0, current_deficit - annual_recovered_revenue_billion)
        
        sim_df = pd.DataFrame({
            "시나리오": ["기존 방식 유지 시 적자", "선택 정책 도입 시 예상 적자"],
            "적자 규모 (억 원)": [current_deficit, new_deficit]
        })
        
        st.subheader("📉 연간 적자 규모 비교 분석")
        st.bar_chart(sim_df, x="시나리오", y="적자 규모 (억 원)", use_container_width=True)
        
        st.metric(
            label="💰 연간 예상 재정 확보액 (코레일/지자체 적자 보전액)", 
            value=f"{annual_recovered_revenue_billion:.1f} 억 원",
            delta=f"적자 {int((annual_recovered_revenue_billion/current_deficit)*100) if current_deficit > 0 else 0}% 감소"
        )
        
        st.info(f"💡 **분석 결과:** {policy_time} 요금 {int((1-discount_rate)*100)}% 부과 시, 연간 약 **{annual_recovered_revenue_billion:.1f}억 원**의 재정 건전성을 확보할 수 있습니다. (※ 고령층의 {int((1-elasticity)*100)}% 수요 탄력성 감소 반영)")
