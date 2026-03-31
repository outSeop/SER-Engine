# SER: Self-Extinguishing Rationality — Simulation Engine

**이 프로젝트는 그래프 기반 에이전트 플랫폼이 아닙니다.**

SER(Self-Extinguishing Rationality) 및 Eight Billion Robots 사고실험을 계산적으로 검증하기 위한 **연구용 시뮬레이션 엔진**입니다. 그래프는 환경 표현 수단일 뿐입니다.

---

## 핵심 연구 질문

1. 순수 cost-benefit rationality만 가진 집단은 정적 평형(static equilibrium)으로 수렴하는가?
2. 외생 목적만 가진 집단은 활동은 하지만 자기복제적 지속성을 만들지 못하는가?
3. self-replication + mutation + selection이 도입될 때 이전에 없던 지속적 동학이 나타나는가?
4. AI safety 관점에서 위험의 핵심은 "지능" 자체가 아니라 artificial evolution 조건인가?

---

## 5개 시나리오

| 시나리오 | 설명 | 예상 결과 |
|---------|------|----------|
| `NO_OBJECTIVE` | 목적 없음 | 즉각적 정적 수렴 |
| `PURE_RATIONAL` | 순수 cost-benefit | 복제 없음, 서서히 정적 수렴 (SER 핵심) |
| `SELF_PRESERVING` | 생존만 존재 | 안정적이지만 정적 |
| `PROGRAMMED_OBJECTIVE` | 외생 목적 최적화 | 활발하지만 지속적 복제 미달 |
| `SELF_REPLICATING` | 복제 + 변이 + 선택 | 지속적 동학, lineage 분화 |

---

## 설치

```bash
pip install -r requirements.txt
```

---

## 빠른 시작

### 단일 실험

```bash
# SELF_REPLICATING 시나리오, 500 tick
python main.py run --scenario SELF_REPLICATING --seed 42 --ticks 500

# PURE_RATIONAL 비교
python main.py run --scenario PURE_RATIONAL --seed 42 --ticks 500
```

### 5-시나리오 비교 실험 (전체)

```bash
python main.py experiment --name ser_comparison
```

### 결과 분석

```bash
python main.py analyze --run-dirs output/PURE_RATIONAL_* output/SELF_REPLICATING_* --output output/analysis
```

### 리플레이 뷰어

```bash
python main.py replay --run-dir output/SELF_REPLICATING_42_<timestamp>
python main.py replay --run-dir output/... --timeline
```

---

## 프로젝트 구조

```
config/
  default.yaml        # 기본 파라미터 (seed, world, agent, replication, mutation, ...)
  experiments.yaml    # 배치 실험 정의

sim/
  config.py           # StrategyType, Action, SimulationConfig 및 로더
  graph_world.py      # NodeState, EdgeState, GraphWorld (환경)
  agent.py            # LineageAgent, Observation, 전략별 행동 선택
  agent_system.py     # AgentSystem (에이전트 생명주기 관리)
  evolution.py        # can_replicate, execute_replication, mutate
  scheduler.py        # 8-step tick 순서 관리
  metrics.py          # TickMetrics, MetricsCollector (equilibrium_score 포함)
  event_detector.py   # SimEvent, EventDetector
  logger.py           # SimLogger (parquet + JSON 저장)
  replay_loader.py    # ReplayLoader (저장 데이터 로드)
  engine.py           # SimulationEngine (모든 컴포넌트 조립)

viz/
  plots.py            # 시나리오 비교 플롯
  graph_renderer.py   # 그래프 상태 시각화
  timeline.py         # 이벤트 타임라인
  viewer.py           # 인터랙티브 리플레이 뷰어

experiments/
  run_experiment.py   # 단일 실험 실행
  batch_runner.py     # 배치 실험 실행
  analyze_results.py  # 결과 비교 분석

tests/
  test_graph_world.py, test_replication.py, test_mutation.py,
  test_metrics.py, test_events.py
```

---

## 테스트 실행

```bash
pytest tests/ -v
```

---

## SER 가설의 계산적 핵심

`PURE_RATIONAL` 에이전트의 복제 효용:

```
replication_utility = future_benefit_to_parent - (energy_cost + mass_loss + competition)
```

자식은 독립적인 agent이므로 `future_benefit_to_parent ≈ 0`.
따라서 `replication_utility < 0` → 순수 합리성 하에서 복제는 항상 회피됨 → 정적 수렴.

`SELF_REPLICATING`은 내재적 `replication_weight`가 이 비용을 상쇄하여 복제를 선택하게 만듦.

---

## 출력 구조

```
output/{scenario}_{seed}_{timestamp}/
  config.yaml          # 재현을 위한 설정 저장
  metrics.parquet      # tick별 전체 메트릭
  agents/
    tick_000000.parquet  # N tick마다 에이전트 스냅샷
    ...
  world_snapshots/
    tick_000000.parquet
    ...
  events.json          # 주요 사건 로그
  summary.json         # 실행 요약
```

---

## 주요 설계 결정

- **단일 RNG 전파**: `numpy.random.default_rng(seed)` → 모든 하위 시스템 → 완전 재현성
- **death 후 replication**: Step 6 → Step 7 순서 엄수, 죽은 agent의 복제 방지
- **mutation_rate 자체가 변이 가능**: meta-evolution으로 runaway lineage 관찰 가능
- **equilibrium_score**: 최근 window의 CV(변동계수) 평균으로 정적 수렴 직접 측정
- **mass 비례 자원 경쟁**: 명시적 fitness 함수 없이 암묵적 selection pressure