"""연금저축 전략 가이드 + 주문방식 + 수수료/세금 탭."""
import streamlit as st
import pandas as pd
from src.utils.formatting import _fmt_etf_code_name, _code_only
from src.ui.components.triggers import render_strategy_trigger_tab


def render_pension_guide_tab(dm_settings: dict, vaa_settings: dict,
                             pen_cfg: dict):
    """Tab 5: 전략 가이드."""
    st.header("연금저축 전략 가이드")
    st.caption("전략별 하위탭에서 원하는 전략 설명을 바로 확인할 수 있습니다.")
    guide_tab_laa, guide_tab_dm, guide_tab_vaa = st.tabs(
        ["LAA", "듀얼모멘텀", "VAA"]
    )

    with guide_tab_laa:
        st.subheader("LAA (Lethargic Asset Allocation) 전략")
        st.markdown("""
**LAA**는 Keller & Keuning이 제안한 게으른 자산배분 전략입니다.

- **코어 자산 (75%)**: IWD(미국 가치주), GLD(금), IEF(미국 중기채) 각 25%
- **리스크 자산 (25%)**: SPY가 200일 이동평균선 위 → QQQ, 아래 → SHY(단기채)
- **리밸런싱**: 월 1회 (월말 기준)
""")
        st.subheader("의사결정 흐름")
        st.markdown("""
```
매월 말 기준:
  1. SPY 종가 vs SPY 200일 이동평균
  2. SPY > 200일선 → 리스크 자산 = QQQ (공격)
     SPY < 200일선 → 리스크 자산 = SHY (방어)
  3. 코어 3종목 25%씩 + 리스크 자산 25% 배분
  4. 목표 비중 대비 괴리 > 3%p이면 리밸런싱 실행
```
""")
        st.subheader("국내 ETF 매핑")
        st.dataframe(pd.DataFrame([
            {"미국 티커": "IWD", "국내 ETF": "TIGER 미국S&P500 (360750)", "역할": "코어 - 미국 가치주"},
            {"미국 티커": "GLD", "국내 ETF": "ACE KRX금현물 (411060)", "역할": "코어 - 금"},
            {"미국 티커": "IEF", "국내 ETF": "TIGER 미국채10년선물 (305080)", "역할": "코어 - 중기채"},
            {"미국 티커": "QQQ", "국내 ETF": "TIGER 미국나스닥100 (133690)", "역할": "리스크 공격"},
            {"미국 티커": "SHY", "국내 ETF": "TIGER 미국달러단기채권액티브 (329750)", "역할": "리스크 방어"},
        ]), use_container_width=True, hide_index=True)
        st.caption("연금저축 계좌에서 해외 ETF 직접 매매 불가 → 국내 ETF로 대체 실행")

    with guide_tab_dm:
        _guide_dm_map = (dm_settings.get("kr_etf_map", {}) if isinstance(dm_settings, dict) else {}) or {}
        _guide_dm_spy = str(_guide_dm_map.get("SPY", _code_only(pen_cfg.get("pen_dm_kr_spy", "360750"))))
        _guide_dm_efa = str(_guide_dm_map.get("EFA", _code_only(pen_cfg.get("pen_dm_kr_efa", "195930"))))
        _guide_dm_agg = str(_guide_dm_map.get("AGG", _code_only(pen_cfg.get("pen_dm_kr_agg", "305080"))))
        _guide_dm_bil = str(_guide_dm_map.get("BIL", _code_only(pen_cfg.get("pen_dm_kr_bil", "329750"))))
        _guide_dm_w = (dm_settings.get("momentum_weights", {}) if isinstance(dm_settings, dict) else {}) or {}
        _w1 = float(_guide_dm_w.get("m1", pen_cfg.get("pen_dm_w1", 12.0)))
        _w3 = float(_guide_dm_w.get("m3", pen_cfg.get("pen_dm_w3", 4.0)))
        _w6 = float(_guide_dm_w.get("m6", pen_cfg.get("pen_dm_w6", 2.0)))
        _w12 = float(_guide_dm_w.get("m12", pen_cfg.get("pen_dm_w12", 1.0)))
        _lb = int(dm_settings.get("lookback", pen_cfg.get("pen_dm_lookback", 12)) if isinstance(dm_settings, dict) else pen_cfg.get("pen_dm_lookback", 12))
        _td = int(dm_settings.get("trading_days_per_month", pen_cfg.get("pen_dm_trading_days", 22)) if isinstance(dm_settings, dict) else pen_cfg.get("pen_dm_trading_days", 22))

        st.subheader("듀얼모멘텀 (GEM) 전략")
        st.markdown(f"""
**듀얼모멘텀(GEM)**은 상대모멘텀 + 절대모멘텀을 결합한 월간 리밸런싱 전략입니다.

- **공격 자산(2개)**: SPY, EFA 중 모멘텀 점수 상위 1개 선택
- **방어 자산(1개)**: AGG
- **카나리아(1개)**: BIL
- **리밸런싱**: 월 1회 (월말 기준)

모멘텀 점수식:
`((1개월수익률 × {_w1:g}) + (3개월수익률 × {_w3:g}) + (6개월수익률 × {_w6:g}) + (12개월수익률 × {_w12:g})) ÷ 4`

절대모멘텀 기준:
`카나리아 룩백 {_lb}개월 수익률` (월 환산 거래일 `{_td}`일 기준)
""")
        st.subheader("의사결정 흐름")
        st.markdown(f"""
```
매월 말 기준:
  1. 공격 자산(SPY, EFA)의 가중 모멘텀 점수 계산
  2. 카나리아(BIL) 룩백 {_lb}개월 수익률 계산
  3. 공격 1위 점수 > 카나리아 수익률  → 공격 1위 100%
     공격 1위 점수 <= 카나리아 수익률 → 방어(AGG) 100%
  4. 목표 비중 대비 괴리 발생 시 리밸런싱 실행
```
""")
        st.subheader("국내 ETF 매핑")
        st.dataframe(pd.DataFrame([
            {"전략 키": "SPY", "국내 ETF": _fmt_etf_code_name(_guide_dm_spy), "역할": "공격 자산 1"},
            {"전략 키": "EFA", "국내 ETF": _fmt_etf_code_name(_guide_dm_efa), "역할": "공격 자산 2"},
            {"전략 키": "AGG", "국내 ETF": _fmt_etf_code_name(_guide_dm_agg), "역할": "방어 자산"},
            {"전략 키": "BIL", "국내 ETF": _fmt_etf_code_name(_guide_dm_bil), "역할": "카나리아"},
        ]), use_container_width=True, hide_index=True)
        st.caption("듀얼모멘텀도 연금저축 계좌에서 국내 ETF로 시그널/실매매를 수행합니다.")
        st.subheader("운용 특성")
        st.markdown("""
- **시장 대응**: 상승장에서는 공격 자산, 하락장/둔화장에서는 방어 자산으로 자동 전환
- **리스크 관리**: 절대모멘텀(카나리아) 기준으로 하락 추세 회피
- **운용 빈도**: 월 1회 리밸런싱으로 과도한 매매 방지
- **과세이연**: 연금저축 계좌 내 매매차익 과세이연 효과로 복리 운용에 유리
""")

    with guide_tab_vaa:
        st.subheader("VAA (Vigilant Asset Allocation) 전략")
        st.markdown("""
**VAA**는 Wouter Keller가 제안한 경계적 자산배분 전략으로, **13612W 모멘텀 스코어**를 활용하여
공격/방어 자산을 동적으로 전환합니다.

- **공격 자산 (4개)**: SPY(미국), EFA(선진국), EEM(이머징), AGG(채권)
- **방어 자산 (3개)**: LQD(회사채), IEF(중기채), SHY(단기채)
- **선택 규칙**: 공격 자산 중 모멘텀 양수인 것 최고 1개 선택 → 전부 음수 시 방어 자산 최고 1개
- **리밸런싱**: 월 1회 (월말 기준)

**13612W 모멘텀 스코어 계산식:**
`(1개월수익률 × 12 + 3개월수익률 × 4 + 6개월수익률 × 2 + 12개월수익률 × 1) ÷ 19`

단기(1개월)에 높은 가중치를 부여하여 추세 반전에 빠르게 대응합니다.
""")
        st.subheader("의사결정 흐름")
        st.markdown("""
```
매월 말 기준:
  1. 공격 자산 4개(SPY, EFA, EEM, AGG)의 13612W 모멘텀 스코어 계산
  2. 방어 자산 3개(LQD, IEF, SHY)의 13612W 모멘텀 스코어 계산
  3. 공격 자산 중 모멘텀 > 0 인 것이 있으면:
     → 양수 모멘텀 중 최고 스코어 1개에 100% 투자 (공격 모드)
  4. 모든 공격 자산 모멘텀 ≤ 0 이면:
     → 방어 자산 중 최고 스코어 1개에 100% 투자 (방어 모드)
  5. 목표 대비 괴리 발생 시 리밸런싱 실행
```
""")
        st.subheader("국내 ETF 매핑")
        _guide_vaa_map = (vaa_settings.get("kr_etf_map", {}) if isinstance(vaa_settings, dict) else {}) or {}
        st.dataframe(pd.DataFrame([
            {"전략 키": "SPY", "국내 ETF": _fmt_etf_code_name(str(_guide_vaa_map.get("SPY", "379800"))), "유형": "공격", "역할": "미국 주식"},
            {"전략 키": "EFA", "국내 ETF": _fmt_etf_code_name(str(_guide_vaa_map.get("EFA", "195930"))), "유형": "공격", "역할": "선진국 주식"},
            {"전략 키": "EEM", "국내 ETF": _fmt_etf_code_name(str(_guide_vaa_map.get("EEM", "195980"))), "유형": "공격", "역할": "이머징 주식"},
            {"전략 키": "AGG", "국내 ETF": _fmt_etf_code_name(str(_guide_vaa_map.get("AGG", "305080"))), "유형": "공격", "역할": "미국 채권"},
            {"전략 키": "LQD", "국내 ETF": _fmt_etf_code_name(str(_guide_vaa_map.get("LQD", "329750"))), "유형": "방어", "역할": "회사채"},
            {"전략 키": "IEF", "국내 ETF": _fmt_etf_code_name(str(_guide_vaa_map.get("IEF", "305080"))), "유형": "방어", "역할": "중기채"},
            {"전략 키": "SHY", "국내 ETF": _fmt_etf_code_name(str(_guide_vaa_map.get("SHY", "329750"))), "유형": "방어", "역할": "단기채"},
        ]), use_container_width=True, hide_index=True)


def render_pension_order_info_tab():
    """Tab 6: 주문방식 안내."""
    st.header("KIS 국내 ETF 주문방식 안내")
    st.dataframe(pd.DataFrame([
        {"구분": "시장가", "API": 'ord_dvsn="01"', "설명": "즉시 체결 (최우선 호가)"},
        {"구분": "지정가", "API": 'ord_dvsn="00"', "설명": "원하는 가격에 주문"},
        {"구분": "동시호가 매수", "API": '상한가(+30%) 지정가', "설명": "15:20~15:30 동시호가 참여 → 60초 대기 → 미체결 시 시간외 재주문"},
        {"구분": "동시호가 매도", "API": '하한가(-30%) 지정가', "설명": "15:20~15:30 동시호가 참여 → 60초 대기 → 미체결 시 시간외 재주문"},
        {"구분": "시간외 종가", "API": 'ord_dvsn="06"', "설명": "15:40~16:00 당일 종가로 체결"},
    ]), use_container_width=True, hide_index=True)

    st.subheader("호가단위")
    st.dataframe(pd.DataFrame([
        {"가격대": "~5,000원", "호가단위": "5원"},
        {"가격대": "5,000~10,000원", "호가단위": "10원"},
        {"가격대": "10,000~50,000원", "호가단위": "50원"},
        {"가격대": "50,000원~", "호가단위": "100원"},
    ]), use_container_width=True, hide_index=True)

    st.subheader("자동매매 흐름 (GitHub Actions)")
    st.markdown("""
0. 로컬 PC에서 직접 주문하지 않고, GitHub Actions에서만 주문 실행
1. 매월 25~31일 평일 KST 15:20 실행 (`TRADING_MODE=kis_pension`)
2. 국내 ETF(SPY/IWD/GLD/IEF/QQQ/SHY 매핑) 일봉 조회
3. SPY vs 200일선 → 리스크 자산 결정 (QQQ or SHY)
4. 목표 배분 vs 현재 보유 비교 → 리밸런싱 필요 여부 판단
5. 매도 → `smart_sell_all_closing()` (동시호가+시간외)
6. 매수 → `smart_buy_krw_closing()` (동시호가+시간외)
""")


def render_pension_fee_tab():
    """Tab 7: 수수료/세금 안내."""
    st.header("연금저축 수수료 및 세금 안내")

    st.subheader("1. 매매 수수료")
    st.dataframe(pd.DataFrame([
        {"증권사": "한국투자증권", "매매 수수료": "0.0140396%", "비고": "나무 온라인 (현재 사용)"},
        {"증권사": "키움증권", "매매 수수료": "0.015%", "비고": "영웅문 온라인"},
        {"증권사": "미래에셋", "매매 수수료": "0.014%", "비고": "m.Stock 온라인"},
    ]), use_container_width=True, hide_index=True)

    st.subheader("2. 매매 대상 ETF 보수")
    st.dataframe(pd.DataFrame([
        {"ETF": "TIGER 미국S&P500", "코드": "360750", "총보수": "0.07%", "추종": "SPY 대용"},
        {"ETF": "ACE KRX금현물", "코드": "411060", "총보수": "0.05%", "추종": "GLD 대용"},
        {"ETF": "TIGER 미국채10년선물", "코드": "305080", "총보수": "0.10%", "추종": "IEF 대용"},
        {"ETF": "TIGER 미국나스닥100", "코드": "133690", "총보수": "0.07%", "추종": "QQQ 대용"},
        {"ETF": "TIGER 미국달러단기채권액티브", "코드": "329750", "총보수": "0.09%", "추종": "SHY 대용"},
    ]), use_container_width=True, hide_index=True)

    st.subheader("3. 연금저축 세제혜택")
    st.markdown("""
| 항목 | 내용 |
|------|------|
| 세액공제 | 연 최대 600만원 (IRP 합산 900만원) |
| 공제율 | 총급여 5,500만원 이하: 16.5% / 초과: 13.2% |
| 과세이연 | 매매차익·배당 세금 인출 시까지 이연 |
| 연금 수령 시 | 3.3~5.5% 연금소득세 (일반 15.4% 대비 유리) |
| 중도 인출 시 | 16.5% 기타소득세 (불이익) |
    """)
    st.caption("LAA 월간 리밸런싱 매매차익이 모두 과세이연되어 복리 효과 극대화 (일반 계좌 대비 연 1~2% 추가 수익)")
