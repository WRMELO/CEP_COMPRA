flowchart TD
  %% =========================
  %% FONTES / SSOT
  %% =========================
  subgraph SSOT["SSOT / Dados (somente passado até D)"]
    MKT["Master (índice) + Universo (preços, volume)"]
    PORT["Carteira (holdings, equity, custos, turnover)"]
    BURN["Burners por ativo (stress/local)"]
    RULES["Regras/constantes (custos, quarentena, limites)"]
  end

  %% =========================
  %% OBSERVADOR CONTÍNUO
  %% =========================
  subgraph OBS["Observador Contínuo (estado multidimensional)"]
    AENV["Ambiente/Master: vol_curta/longa, slope_10/20/60, DD_master, dispersão, breadth"]
    APOR["Carteira: DD_port, underwater_time, underwater_slope/persist, exposição, concentração, turnover/custo, U_t_agregado"]
    ABUR["Burners agregados: percentis de stress (holdings), contagem stress alto, oportunidades top-K (candidatos)"]
    AEXE["Execução/fricção: liquidez relativa, risco de slippage (proxys), book em baixa liquidez"]
    STATE["Estado S(D) = concat(AENV, APOR, ABUR, AEXE)"]
  end

  %% =========================
  %% PLANTA: CORE DE SELEÇÃO
  %% =========================
  subgraph CORE["Planta (Core M3 congelado)"]
    RANK["Ranking/Seleção: lista de candidatos + scores (oportunidade)"]
    SIGS["Sinais potenciais: compras/vendas sugeridas (não executadas ainda)"]
  end

  %% =========================
  %% POLÍTICA: RL SELETOR DE MODO
  %% =========================
  subgraph RL["Política (RL) — Seletor de Modo Operacional"]
    POLICY["π(S): escolhe modo do dia"]
    ACTS["Ações (modo): exposição alvo; cap turnover; severidade defensiva; rampa reentrada; cap concentração"]
  end

  %% =========================
  %% GUARDRAILS / SEGURANÇA
  %% =========================
  subgraph SAFE["Guardrails Determinísticos (não violáveis)"]
    GR1["Venda compulsória por stress extremo do burner (trilho separado)"]
    GR2["Limites de risco: MDD/underwater críticos, concentração, custos/turnover, quarentena"]
    GR3["Regras de execução: rampas graduais, evitar histerese (saída mais fácil que entrada)"]
  end

  %% =========================
  %% EXECUÇÃO
  %% =========================
  subgraph EXEC["Executor (ordens e atualização)"]
    EXECUTE["Executar trades: aplicar modo + sinais do core sob guardrails"]
    UPDATE["Atualizar carteira: equity, custos, turnover, holdings"]
  end

  %% =========================
  %% OBJETIVOS / FEEDBACK
  %% =========================
  subgraph OBJ["Objetivos (feedback)"]
    OBJ1["Reduzir queda continuada: menor duração/área underwater e persistência negativa"]
    OBJ2["Reduzir bloqueio: menor tempo subexposto quando há oportunidade retrospectiva"]
    REWARD["Reward do RL: U_t líquido - penalidade(underwater) - penalidade(bloqueio) - penalidade(custo/turnover/concentração)"]
  end

  %% =========================
  %% CONEXÕES
  %% =========================
  MKT --> AENV
  PORT --> APOR
  BURN --> ABUR
  RULES --> APOR
  RULES --> AEXE
  MKT --> AEXE

  AENV --> STATE
  APOR --> STATE
  ABUR --> STATE
  AEXE --> STATE

  MKT --> RANK
  PORT --> RANK
  BURN --> RANK
  RANK --> SIGS

  STATE --> POLICY
  POLICY --> ACTS

  SIGS --> EXECUTE
  ACTS --> EXECUTE

  GR1 --> EXECUTE
  GR2 --> EXECUTE
  GR3 --> EXECUTE

  EXECUTE --> UPDATE
  UPDATE --> PORT

  UPDATE --> OBJ1
  UPDATE --> OBJ2
  OBJ1 --> REWARD
  OBJ2 --> REWARD
  REWARD --> POLICY