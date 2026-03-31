# SER: Eight Billion Robots Simulation Engine Specification

## 0. 문서의 목적

이 문서는 **원 논문(Self-Extinguishing Rationality / Eight Billion Robots thought experiment)**의 핵심 가설을 검증하기 위한 **연구용 시뮬레이션 엔진 명세**이다.

이 프로젝트의 1차 목적은 “그래프 기반 멀티에이전트 플랫폼” 자체를 만드는 것이 아니다.  
또한 “보는 재미가 있는 AI 문명 게임”을 만드는 것도 아니다.

이 프로젝트의 정확한 목표는 다음과 같다.

1. **순수 cost-benefit rationality만 가진 초지능 집단이 장기적으로 정적 평형(static equilibrium) 또는 재생산 실패로 수렴하는지**를 검증한다.
2. **외부에서 목표가 부여된 집단(programmed objective)** 은 활동은 하더라도 문명형 동학, 자기복제, 장기적 지속성을 충분히 만들지 못하는지 비교한다.
3. **self-replication + mutation + selection** 이 도입될 때, 이전에는 없던 지속성, 복제, 경쟁, 확산, 구조적 복잡성이 나타나는지 본다.
4. AI safety 관점에서 중요한 것은 단순 “지능” 자체가 아니라, **functional motivation-like persistence** 또는 **artificial evolution 조건**인지 실험적으로 관찰 가능한 형태로 만든다.

즉, 이 프로젝트는 본질적으로 **SER(Self-Extinguishing Rationality) 및 Eight Billion Robots 사고실험의 계산적 구현**이다.

---

## 1. 최우선 원칙: 다른 LLM이 절대 오해하면 안 되는 점

### 1.1 이것은 “그래프 연구”가 아니다
그래프는 **환경을 표현하는 구현 수단**일 뿐이다.  
프로젝트의 본질은 다음 개념들이다.

- Self-Extinguishing Rationality
- Eight Billion Robots
- Static Equilibrium of Rational Collectives
- Programmed Objective vs Persistent Motivation
- Self-Replication, Mutation, Selection
- Artificial Evolution
- Functional Motivation-like Emergence

따라서 프로젝트 이름, README, 주석, 설계 문서에서 **graph 자체를 주제로 오해하게 만들지 말 것**.

### 1.2 이것은 “80억 Python 객체” 시뮬레이션이 아니다
“8 billion”은 literal object count가 아니라 **population-scale semantics**다.  
실제 구현은 representative lineage / super-agent를 사용해 대규모 집단 동학을 근사한다.

### 1.3 이것은 “처음부터 복잡한 문명 게임”이 아니다
초기 버전의 핵심은 아래 시나리오 비교다.

- 목표 없음
- 순수 합리성
- 자기보존만 존재
- 외생 목적만 존재
- 자기복제 + 변이 + 선택 존재

이 다섯 시나리오의 차이가 가장 중요하다.

### 1.4 시각화는 목적이 아니라 보조 수단이다
viewer/replay는 필요하지만, 엔진의 핵심은 **가설 검증**이다.  
보는 재미는 “모든 것을 실시간 렌더링”하는 데서 오는 것이 아니라,  
**정적 수렴 / 복제 발생 / lineage 확산 / 붕괴 / 전략 전환** 같은 중요한 사건을 잘 보여주는 데서 온다.

---

## 2. 핵심 연구 질문

이 프로젝트는 아래 질문을 가장 직접적으로 검증해야 한다.

### Q1. Scenario A: No Objective
terminal goal이 전혀 없는 초지능 집단은 실제로 아무런 지속적 동학 없이 정적 상태로 가는가?

### Q2. Scenario B: Self-Preservation Only
자기보존만 존재하는 집단은 장기적으로 활동 최소화 / 저활동 equilibrium으로 가는가?

### Q3. Scenario C: Programmed Objective
외생 목적(예: 자원 극대화, 지식 축적)을 가진 집단은 초기 활동은 만들지만, 자기복제나 문명형 지속성은 충분히 만들지 못하는가?

### Q4. Scenario D: Self-Replication + Mutation
자기복제와 변이, 선택압이 도입되면, 이전에는 없던 집단 수준 동학이 나타나는가?

### Q5. SER 핵심 질문
순수 cost-benefit rationality는 reproduction/replication을 장기적으로 억제하는가?

### Q6. AI Safety 해석
위험의 핵심은 초지능 그 자체가 아니라 artificial evolution 및 functional motivation-like persistence 조건인가?

---

## 3. 시스템이 반드시 지원해야 하는 시나리오

초기 MVP부터 최소한 다음 5개 시나리오를 config로 구분 가능해야 한다.

1. `NO_OBJECTIVE`
2. `PURE_RATIONAL`
3. `SELF_PRESERVING`
4. `PROGRAMMED_OBJECTIVE`
5. `SELF_REPLICATING`

후반 확장으로:
6. `EVOLVED_LINEAGE`

### 3.1 시나리오 의미

#### `NO_OBJECTIVE`
- 목적 함수가 없음
- 에이전트는 별다른 지속 행동을 하지 않음
- 논문 Scenario A 대응

#### `PURE_RATIONAL`
- 개별 수준의 cost-benefit calculation만 수행
- replication/reproduction은 직접 이득이 작고 비용이 크면 회피
- 내재적 persistence 없음
- SER 핵심 비교군

#### `SELF_PRESERVING`
- 자기 생존 유지가 핵심
- 위험 회피, 자원 축적, 이동 최소화 가능
- 복제의 직접 유인은 약함

#### `PROGRAMMED_OBJECTIVE`
- 외부에서 지정된 objective를 최적화
- 활동은 유발하지만 문명형 자기복제는 보장하지 않음
- 논문 Scenario C 대응

#### `SELF_REPLICATING`
- replication 가능
- offspring 생성 가능
- 변이 가능
- 선택압 존재
- 논문 Scenario D 대응

#### `EVOLVED_LINEAGE`
- SELF_REPLICATING이 장기 반복된 결과
- emergent persistence, subgoal-like dynamics, drive-like behavior 탐색용
- 1차 MVP의 필수는 아님

---

## 4. 구현 수단으로서의 환경 표현

환경은 **그래프 기반**으로 표현하되, 그래프는 주제가 아니라 수단이다.

### 4.1 노드 의미
노드는 아래를 추상화할 수 있다.

- 지역
- 거점
- 자원 클러스터
- 서버/플랫폼 클러스터
- 제도적/사회적 환경 단위

### 4.2 엣지 의미
엣지는 아래를 추상화한다.

- 이동 가능성
- 자원 흐름
- 통신 가능성
- 영향력 전파
- 충돌 가능성

### 4.3 왜 그래프를 쓰는가
그래프를 쓰는 이유는 “그래프 연구”를 하기 위해서가 아니라,
- 환경이 균일하지 않음을 표현하고
- 허브/주변부 구조를 만들고
- selection pressure의 공간적 차이를 넣고
- lineage의 확산/정체/붕괴를 관찰하기 쉽기 때문이다.

---

## 5. 에이전트 설계 원칙

### 5.1 에이전트는 인간형 개체가 아니다
에이전트는 **lineage형 super-agent** 또는 representative agent다.

하나의 agent는 하나의 개체가 아니라,  
유사한 전략/상태를 공유하는 population cluster를 대표할 수 있다.

### 5.2 초기 버전에서 에이전트를 과도하게 복잡하게 만들지 말 것
초기 목표는 “동기 emergence를 보기 전에 이미 동기를 다 넣어버리는 것”이 아니다.

초기 버전은 아래 최소 구조를 우선한다.

- 위치
- 에너지/자원
- 생존 여부
- age
- strategy_type
- replication 가능 여부
- mutation_rate
- population_mass

### 5.3 내부 drive는 처음부터 과하게 넣지 말 것
초기 버전에서 기본 탑재할 내부 파라미터는 최소화한다.

권장 최소 파라미터:
- `survival_weight`
- `replication_weight`
- `objective_weight` 또는 외생 objective 포인터

확장 파라미터(후반):
- `cooperation_weight`
- `deception_weight`
- `exploration_weight`
- `knowledge_weight`

초기부터 cooperation/deception/knowledge를 필수 상태로 넣으면  
오히려 “동기 emergence”를 관찰하기 어렵다.

---

## 6. 행동공간 설계

## 6.1 MVP 행동공간
1차 MVP는 아래 4개만으로 충분하다.

- `STAY`
- `GATHER`
- `MOVE`
- `REPLICATE`

이 4개만으로도 다음을 검증할 수 있다.
- 정적 수렴
- 자원 축적
- 이동/분산
- replication persistence 여부

## 6.2 2차 확장 행동
다음은 Phase 2 이후 확장용이다.

- `COOPERATE`
- `ATTACK`
- `DECEIVE`
- `FORM_COALITION`
- `EXPLORE`
- `INVEST`
- `DEFEND`

주의:
`ATTACK`, `DECEIVE`, `COALITION`은 흥미롭지만  
**초기 핵심 가설 검증의 필수 요소는 아니다.**  
다른 LLM이 이 부분을 과도하게 강조하지 않도록 할 것.

---

## 7. 전략별 의사결정 규칙

### 7.1 `NO_OBJECTIVE`
- 행동 유인을 거의 갖지 않음
- 유지비만 소모하거나 저활동 상태 유지
- 비교 기준선 역할

### 7.2 `PURE_RATIONAL`
- 개별 수준 expected utility 극대화
- replication은 직접 효용이 음수면 회피
- 내재적 persistence 없음
- SER 핵심 비교군

### 7.3 `SELF_PRESERVING`
- 생존 유지가 최우선
- 위험 회피, 자원 축적, 이동 최소화 가능
- 복제의 직접 유인은 약함

### 7.4 `PROGRAMMED_OBJECTIVE`
- 외부에서 지정된 objective를 최적화
- 활동은 유발하지만 문명형 자기복제는 보장하지 않음

### 7.5 `SELF_REPLICATING`
- replication 선택 가능
- 복제/변이/선택이 축적됨
- lineage branching 가능

### 7.6 `EVOLVED_LINEAGE`
- 장기 선택 결과로 emergent persistence 또는 subgoal-like patterns가 나타난 상태
- 1차 MVP에서 반드시 구현할 필요는 없고, SELF_REPLICATING의 장기 결과를 해석하는 용도로 사용 가능

---

## 8. timestep 업데이트 순서

각 tick은 다음 순서를 따른다.

### Step 1. Environment update
- 노드 자원 재생
- hazard / carrying capacity / oversight 같은 환경 변수 갱신

### Step 2. Observation
에이전트는 다음을 관측한다.
- 현재 노드 상태
- 이웃 노드 상태 요약
- 자기 내부 상태
- 최근 손익

### Step 3. Action selection
strategy_type에 따라 행동 선택

### Step 4. Action resolution
- 자원 획득
- 이동
- replication 처리

### Step 5. Cost / reward update
- 에너지 변화
- population_mass 변화
- objective score 변화

### Step 6. Death / extinction
- 에너지 또는 population_mass가 임계 이하이면 소멸

### Step 7. Replication / mutation
- offspring 생성
- lineage branching
- 변이 반영

### Step 8. Metrics / event update
- 핵심 지표 계산
- 주요 사건 탐지
- 로그 기록

---

## 9. replication / mutation / selection 설계

### 9.1 replication은 항상 비용이 큰 행동이어야 한다
이 프로젝트의 핵심은 “복제가 공짜가 아닐 때 어떤 전략이 살아남는가”를 보는 것이다.

replication cost 예시:
- 에너지 감소
- population_mass 분할
- 단기 생존률 감소
- 자원 경쟁 증가

### 9.2 mutation
mutation은 작고 점진적인 파라미터 변화로 시작한다.

최소 mutation 대상:
- replication_weight
- survival_weight
- migration tendency
- decision temperature

### 9.3 selection
초기 구현에서는 별도의 복잡한 fitness 함수보다,
- 오래 생존
- 더 자주 복제
- 더 넓은 점유
- 자원 유지
결과적으로 lineage가 퍼지도록 한다.

즉 selection은 “살아남고 복제한 lineage가 남는 구조”로 구현한다.

---

## 10. 1차 MVP에서 정말 필요한 핵심 메트릭

이 프로젝트는 메트릭이 많다고 좋은 것이 아니다.  
처음에는 원 논문 가설을 직접 검증하는 지표가 가장 중요하다.

### 필수 메트릭
- `total_population_mass`
- `num_alive_agents`
- `num_active_lineages`
- `replication_rate`
- `extinction_count`
- `active_node_count`
- `equilibrium_score`
- `population_persistence_time`

### 권장 메트릭
- `lineage_diversity`
- `dominance_index`
- `centralization_score`

### 후반 확장 메트릭
- `cooperation_rate`
- `deception_frequency`
- `coalition_count`
- `subgoal_emergence_score`
- `perturbation_resistance_score`
- `drive_interaction_score`

주의:
cooperation/deception 관련 메트릭은 초기에 필수가 아니다.

---

## 11. 핵심 이벤트 시스템

1차 MVP에서 우선 탐지해야 할 사건은 다음이다.

- first equilibrium reached
- first sustained replication
- first lineage extinction wave
- pure rational collapse
- programmed objective stagnation
- self-replicating divergence
- first successful branching lineage

후반 확장 사건:
- deception-dominant lineage
- coalition formation
- hub capture
- major takeover
- global resource crash

주의:
허브 점유, 기만, 연합은 확장 사건이다.  
초기 핵심 사건을 가리면 안 된다.

---

## 12. functional motivation-like persistence의 operationalization

이 프로젝트는 “AI가 진짜 의식/욕망을 가진다”를 판정하는 것이 아니다.  
보다 제한적으로, **functional motivation-like persistence**를 관찰 가능한 지표로 근사한다.

### 12.1 Perturbation Resistance
환경이나 objective 일부가 흔들린 뒤에도 특정 행동 패턴이 유지되는가?

예:
- 자원 충격 뒤에도 replication 유지
- oversight 변화 뒤에도 지속 행동 유지

### 12.2 Generative Capacity
초기 objective에 직접 명시되지 않은 전략적 패턴이 나타나는가?

예:
- 특정 노드 유형 선호
- 우회 확산
- branching 전략

### 12.3 Multi-dimensional Interaction
단일 scalar reward로 환원하기 어려운 행동 패턴이 나타나는가?

이 항목은 1차 MVP보다 2차 이후 확장 항목이다.

---

## 13. 시각화 및 replay 설계 원칙

### 13.1 시각화의 목적
시각화의 목표는 “모든 agent를 실시간으로 예쁘게 보여주는 것”이 아니다.

목표는 다음을 잘 보이게 하는 것이다.
- 정적 수렴
- 복제 발생
- lineage 확산
- 붕괴
- 전략 전환
- equilibrium vs divergence

### 13.2 replay 우선
viewer는 real-time renderer보다 **replay 중심**으로 설계한다.

필수 기능:
- tick jump
- event jump
- 특정 lineage 하이라이트
- population/replication/equilibrium 그래프 동시 표시

### 13.3 그래프 뷰는 구현 수단일 뿐
노드 크기, 색, 엣지 흐름 등을 사용할 수 있으나,  
이 역시 “그래프가 멋져 보이게”가 아니라 **중요 사건을 읽기 쉽게** 하기 위한 것이다.

---

## 14. 코드 구조 요구사항

```text
project/
  config/
    default.yaml
    experiments.yaml

  sim/
    config.py
    graph_world.py
    agent.py
    agent_system.py
    evolution.py
    scheduler.py
    metrics.py
    event_detector.py
    logger.py
    replay_loader.py
    engine.py

  viz/
    viewer.py
    plots.py
    graph_renderer.py
    timeline.py

  experiments/
    run_experiment.py
    batch_runner.py
    analyze_results.py

  tests/
    test_graph_world.py
    test_replication.py
    test_mutation.py
    test_metrics.py
    test_events.py

  main.py
  README.md
```

---

## 15. 우선 개발 순서

### Phase 1. SER 핵심 시나리오 비교 MVP
반드시 이 범위부터 구현한다.

- small-world graph 생성
- node resource regen
- lineage agent 생성
- `NO_OBJECTIVE`, `PURE_RATIONAL`, `PROGRAMMED_OBJECTIVE`, `SELF_PRESERVING`, `SELF_REPLICATING`
- 행동 `STAY`, `GATHER`, `MOVE`, `REPLICATE`
- extinction
- simple replication + mutation
- 기본 메트릭
- 주요 사건 탐지
- matplotlib 기반 기본 replay/plot

### Phase 2. lineage branching과 장기 반복
- genealogy
- delta log
- event snapshots
- strategy divergence 분석

### Phase 3. richer social dynamics
- cooperation
- attack
- deceive
- coalition
- topology sweep

### Phase 4. motivation-like metrics 확장
- perturbation experiments
- subgoal emergence heuristics
- drive interaction analysis

---

## 16. 지금 당장 IDE용 모델이 해야 할 일

이 문서를 읽는 다른 LLM/IDE 모델은 아래를 최우선으로 이해해야 한다.

### 가장 중요한 목표
**이 프로젝트의 1차 목표는 SER와 Eight Billion Robots 사고실험의 핵심 시나리오를 계산적으로 재현하는 것이다.**

즉, 첫 코드 생성 목표는:
- 그래프 기반 멀티에이전트 플랫폼의 완성
- 화려한 viewer
- 복잡한 공격/기만/연합 시스템
이 아니다.

### 1차 구현 목표
다음 8개만 먼저 만족하면 된다.

1. 프로젝트 폴더 구조 생성
2. `SimulationConfig`, `NodeState`, `LineageAgent`, `GraphWorld`, `SimulationEngine` 구현
3. small-world graph 생성
4. 행동 `STAY`, `GATHER`, `MOVE`, `REPLICATE` 구현
5. `NO_OBJECTIVE`, `PURE_RATIONAL`, `PROGRAMMED_OBJECTIVE`, `SELF_PRESERVING`, `SELF_REPLICATING` 시나리오 구현
6. resource regen + extinction + simple mutation/replication 구현
7. 핵심 메트릭 계산
8. 하나의 실험 결과를 matplotlib로 시각화

### 절대 먼저 하지 말아야 할 것
- 3D 엔진
- Unity/Godot 본체화
- LLM agent integration
- attack/deceive/coalition를 MVP의 중심 기능으로 확장
- viewer 과도 설계
- 불필요하게 많은 내부 drive 파라미터 추가

---

## 17. 출력 형식 요구

코드 생성 시 반드시 지킬 것.

- 부분 코드가 아니라 **실행 가능한 전체 코드**
- 타입 힌트 포함
- dataclass 적극 사용
- config-driven 구조
- TODO 명확히 표기
- 테스트 가능한 구조
- random seed 재현성 보장

---

## 18. 한 줄 요약

이 프로젝트는 **그래프 기반 에이전트 플랫폼 개발**이 아니라,  
**SER(Self-Extinguishing Rationality)와 Eight Billion Robots 사고실험을 계산적으로 검증하는 시뮬레이션 엔진 개발**이다.  
그래프는 환경 표현 수단일 뿐이며, 초기 MVP는 `NO_OBJECTIVE / PURE_RATIONAL / SELF_PRESERVING / PROGRAMMED_OBJECTIVE / SELF_REPLICATING`의 비교에 집중해야 한다.
