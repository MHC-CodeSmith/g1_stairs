# Relatório: repositórios usados e como o projeto os junta (2026-09-28)

Resumo do que cada repositório tem, como treina, se tem Docker e como entra no `g1_stairs`.
Os números vêm de `docs/SCORECARD.md` (mesmo robô G1 29-DoF + mãos Dex3 no MuJoCo, mesmos testes para todos).

## Como a junção funciona

Nenhum desses projetos usa o formato de outro: cada um tem ordem de juntas, ganhos PD, escalas, relógio de marcha e
histórico próprios. O projeto junta tudo em três passos:

1. **Adaptador por repositório** (`arena/policies.py`, `arena/trackers.py`, `arena/sonic.py` no MuJoCo;
   `skills/adapters.py` no Isaac Lab). Cada adaptador reconstrói a observação e a decodificação de ação originais.
   `arena/check_adapters.py` roda o código do próprio repositório no mesmo estado e compara: 9 de 9 batem com
   diferença máxima de 7,4e-6.
2. **Placar** (`arena/scorecard.py`): todas as políticas nos mesmos testes (plano, sem mãos, terreno irregular, braços
   mexendo, empurrões, escada, agachar) e os rastreadores de movimento em 9 clipes. Isso diz quem é bom em quê.
3. **Destilação multi-professor** (`g1_rl/body.py`, `g1_rl/distill.py`, Isaac Lab): um aluno único aprende com PPO
   mais imitação do professor certo para cada situação. Hoje: G1DWAQ para andar/escada, GR00T WBC para agachar.
   O aluno liberado (`checkpoints/g1_body.pt`) sobe a escada, agacha até 0,52 m (erro 3 mm) e aguenta 1000 N.

Sobre **MPC**: nenhum dos repositórios usa MPC. Todos são aprendizado por reforço (PPO, SAC), às vezes com
professor-aluno (DAgger), aprendizado não supervisionado ou modelos generativos. O único controle "clássico" é a
cinemática inversa dos braços no GR00T WBC desacoplado.

## Tabela rápida

| repositório | o que dá | treino | simulador | Docker | licença | no placar |
|---|---|---|---|---|---|---|
| G1DWAQ_Lab (TienKung-Lab) | subir escada às cegas | PPO + VAE (DreamWaQ) | Isaac Lab | não (fizemos `docker/Dockerfile.train`) | BSD-3 | único professor que atravessa a escada; 1000 N |
| WBC-AGILE (NVIDIA) | andar + altura da pelve | PPO com histórico | Isaac Lab | não usado | Apache-2.0 | melhor em velocidade (0,016 m/s); agacha só até 0,62 m |
| GR00T-WholeBodyControl (NVIDIA) | WBC desacoplado + SONIC | PPO (WBC); rastreamento + planejador generativo (SONIC) | Isaac Lab / MuJoCo | sim (deploy, ROS 2) | código Apache-2.0, pesos NVIDIA Open Model | WBC agacha até 0,50 m; SONIC é o melhor rastreador |
| holosoma (Amazon FAR) | andar 29-DoF, rastreamento | PPO e FastSAC | IsaacGym, IsaacSim, MJWarp | sim (3 Dockerfiles) | Apache-2.0 | anda em terreno irregular; 700 N |
| unitree_rl_gym | andar só pernas (12) | PPO (rsl_rl) | Isaac Gym | não | BSD-3 | erra giro (1,14 rad/s); cai com braços mexendo |
| mujoco_playground (DeepMind) | andar 29-DoF | PPO (Brax/JAX, MJX) | MuJoCo MJX / Warp | não | Apache-2.0 | mediano; cai no irregular |
| g1_walk_isaaclab_mujoco | andar G1 + Dex3 (37 juntas) | PPO (Isaac Lab), ajuste robusto | Isaac Lab | não | MIT | cai no nosso modelo; precisa do robô dele |
| GMT | rastrear qualquer movimento | professor PPO → aluno (DAgger), mistura de especialistas | Isaac Gym | não | Apache-2.0 | bom nas juntas, perde a direção em clipes longos |
| TWIST | rastrear movimento (teleoperação) | professor PPO → aluno RL + BC | Isaac Gym | não (usa Redis) | MIT | pior dos três rastreadores |
| BFM-Zero (LeCAR) | modelo de comportamento "promptável" | RL não supervisionado (forward-backward) | Isaac Sim / MuJoCo | não | CC-BY-NC (não comercial) | baixado, ainda não adaptado |
| mujoco_menagerie | modelos MuJoCo do G1 (com e sem mãos) | — | MuJoCo | não | BSD-3 | modelo base da arena |

## Um por um

### G1DWAQ_Lab (liuyufei-nubot) — a base do vídeo da escada
- **O que tem**: fork do TienKung-Lab (Isaac Lab) com a tarefa `g1_dwaq` e um checkpoint treinado (`model_9999.pt`) que
  sobe escadas de 0 a 23 cm sem câmera. Também `LeggedLabDeploy` para o robô real.
- **Como funciona**: DreamWaQ. Um codificador (VAE) lê os últimos 5 quadros de sensores e estima a velocidade do corpo
  (3) e um código latente do terreno (16). O ator recebe [código 19, observação 100] e dá 29 alvos de junta.
- **Treino**: PPO (fork do rsl_rl) + perda do VAE; currículo de terreno com escadas.
- **Docker**: não tem. O nosso `docker/Dockerfile.train` (imagem `g1-isaaclab`) roda ele.
- **No projeto**: professor de andar/escada (`skills.adapters.DwaqPolicy`) e ponto de partida do aluno `g1_body`.

### WBC-AGILE (NVIDIA Isaac)
- **O que tem**: políticas de corpo inteiro do G1; usamos `Velocity-Height-G1-History-v0` (só pernas, 12 juntas),
  com comando [vx, vy, giro, altura da pelve].
- **Treino**: PPO no Isaac Lab, 5 quadros de histórico; exportado em ONNX/TorchScript.
- **No projeto**: foi o primeiro professor de agachar, mas no nosso ambiente para em 0,62 m. Foi trocado pelo GR00T WBC.
  Continua o melhor rastreador de velocidade no plano.

### GR00T-WholeBodyControl (NVIDIA NVlabs) — o mais importante para o futuro
- **WBC desacoplado** (`decoupled_wbc/`): duas redes, Walk e Balance, para pernas + cintura (15 juntas). Comando:
  velocidade, altura da pelve, roll/pitch/yaw do tronco. Os braços ficam livres (IK ou teleoperação). Treinado com PPO
  e um estimador de velocidade/latente sobre 6 quadros. Foi treinado com as mãos Dex3 no modelo.
- **SONIC** (`gear_sonic/`, `gear_sonic_deploy/`, pesos no HF `nvidia/GEAR-SONIC`): controlador de rastreamento universal.
  - Um **planejador** gera movimento (27 modos: andar, correr, agachar, ajoelhar, engatinhar, boxe, andares estilizados)
    a partir de direção e velocidade.
  - Um **codificador** transforma o movimento de referência em 64 "tokens".
  - Um **decodificador** transforma tokens + estado em 29 alvos de junta.
  - Treino: rastreamento de movimento em grande escala (RL) com aprendizado da representação latente; o planejador é um
    modelo generativo cinemático.
- **Docker**: sim (`Dockerfile.deploy`, `Dockerfile.ros2`), voltado ao robô real e ROS 2.
- **No projeto**: WBC é o professor de agachar do `g1_body`; SONIC é o melhor rastreador e a interface de "tokens" que o
  GR00T N1.7 usa (`UNITREE_G1_SONIC`). O SONIC só tem referência em C++, então o adaptador foi validado pela qualidade de
  rastreamento, não por comparação numérica.

### holosoma (Amazon FAR)
- **O que tem**: framework completo de treino e deploy para G1 e Booster T1: locomoção (PPO e FastSAC), rastreamento de
  corpo inteiro e retargeting de movimento.
- **Treino**: PPO ou FastSAC; IsaacGym, IsaacSim ou MuJoCo Warp.
- **Docker**: sim (`docker/isaacgym`, `isaacsim`, `mujoco`).
- **No projeto**: duas políticas de andar (FastSAC e PPO). Estão entre as poucas que não caem no terreno irregular:
  bom candidato a professor de terreno irregular.

### unitree_rl_gym (Unitree)
- **O que tem**: exemplos oficiais de RL (Go2, H1, G1) no Isaac Gym, com sim2sim no MuJoCo e deploy real.
- **Treino**: PPO (rsl_rl). A política do G1 controla só as 12 juntas das pernas.
- **No projeto**: referência básica. É fraca no nosso placar: erra o giro e cai quando os braços mexem.

### mujoco_playground (Google DeepMind)
- **O que tem**: ambientes em MuJoCo MJX/Warp, treino em JAX/Brax na GPU, incluindo o G1 com joystick.
- **Treino**: PPO (Brax), muito rápido na GPU.
- **No projeto**: política 29-DoF mediana. Serve de alternativa ao Isaac para treinar direto no MuJoCo.

### g1_walk_isaaclab_mujoco (yezzzzye)
- **O que tem**: projeto didático Isaac Lab → MuJoCo com o G1 + Dex3 (37 juntas), baseline e versão robusta.
- **Treino**: PPO no Isaac Lab, depois ajuste em terreno difícil.
- **No projeto**: usa um G1 antigo (sem cintura roll/pitch e punhos). No nosso modelo ele cai, então não serve como professor.

### GMT — humanoid-general-motion-tracking (zixuan417)
- **O que tem**: rastreador geral de movimento do G1 (23 juntas) e sim2sim no MuJoCo, com clipes de exemplo.
- **Treino**: professor com informação privilegiada (PPO) → aluno por DAgger, com mistura de especialistas e amostragem
  adaptativa de movimentos; Isaac Gym.
- **No projeto**: rastreia bem as juntas, mas perde a direção em clipes longos (3 m de erro numa caminhada de 38 s).

### TWIST (YanjieZe)
- **O que tem**: sistema de teleoperação de corpo inteiro: rastreador (23 juntas + punhos) e servidores que conversam
  por Redis.
- **Treino**: professor PPO → aluno com RL + imitação (BC); Isaac Gym.
- **No projeto**: pior dos três rastreadores no nosso placar.

### BFM-Zero (LeCAR Lab)
- **O que tem**: modelo de comportamento que se usa por "prompt" (alcançar uma pose, seguir um movimento, otimizar uma
  recompensa) sem retreinar.
- **Treino**: RL não supervisionado (representações forward-backward, FB-CPR); Isaac Sim ou MuJoCo.
- **Licença**: CC-BY-NC, **não comercial**.
- **No projeto**: baixado, ainda não adaptado.

### mujoco_menagerie (Google DeepMind)
- Modelos MuJoCo curados; usamos o G1 com e sem mãos. Não treina nada.

## Repositórios adicionados depois (testados na arena)
Resultados completos, GIFs e gráficos em [`docs/ARENA_REPORT.md`](ARENA_REPORT.md).
- **Safe100Humanoid** (lzqw, Apache-2.0): escada com CBF-RL (mjlab + MuJoCo-Warp + PPO). No simulador dele sobe
  16/16 escadas; no MuJoCo de CPU cai em menos de 1 s, mesmo com o modelo compilado dele (dependência do simulador).
- **unitree_rl_lab** (Unitree, Apache-2.0, tem Docker): política de andar 29-DoF (anda bem, não sobe escada) e duas
  danças (rastreiam o próprio clipe com 0,08 rad de erro).
- **HumanoidBench**: o percurso de escada e a recompensa dele viraram um teste da arena (as políticas dele são para o H1).
- **GRAIL** (NVIDIA): rastreador de escada derivado do SONIC com mapa de altura; inspecionado, ainda não rodado.
- **unitree_mujoco, unitree_sim_isaaclab, RLinf, Unitree-G1-Humanoid-Robot-Tasks**: não têm política de G1 para testar
  (simulador, cenas de teleoperação, infraestrutura de RL, coleção de tarefas).

## O que ler primeiro para estudar a junção
1. `skills/adapters.py`: como políticas de origens diferentes ganham a mesma interface, e `gain_equivalent_target`
   (converte o alvo de um conjunto de ganhos PD para outro).
2. `g1_rl/body.py` e `g1_rl/distill.py`: como o aluno escolhe o professor e como a imitação entra na perda do PPO.
3. `arena/check_adapters.py`: como provar que um adaptador é fiel ao repositório original.
4. `docs/SCORECARD.md` e `docs/HANDOFF.md`: resultados e próximos passos.
