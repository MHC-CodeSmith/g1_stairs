const pptxgen = require('pptxgenjs');
const D = '/home/hipolito/g1_stairs/docs/';
const pres = new pptxgen(); pres.layout = 'LAYOUT_WIDE'; // 13.33 x 7.5
pres.title = 'G1 arena: stairs e locomoção';
const NAVY='0B1F33', INK='1B2733', ORANGE='F26B1D', TEAL='1C7C8C', MUTED='5B6B7A', LIGHT='F4F7FA', WHITE='FFFFFF', GREEN='2E8B57', RED='C0392B';
const HF='Cambria', BF='Calibri';

function base(title, sub){
  const s = pres.addSlide(); s.background={color:WHITE};
  s.addText(title,{x:0.6,y:0.35,w:12.1,h:0.8,fontFace:HF,fontSize:32,bold:true,color:NAVY,isTextBox:true,margin:0,valign:'top'});
  if(sub) s.addText(sub,{x:0.6,y:1.1,w:12.1,h:0.4,fontFace:BF,fontSize:16,color:MUTED,isTextBox:true,margin:0});
  return s;
}
function bullets(s, items, o){
  const arr = items.map((t,i)=>({text:t,options:{bullet:true,breakLine:i<items.length-1,paraSpaceAfter:8}}));
  s.addText(arr,Object.assign({fontFace:BF,fontSize:16,color:INK,valign:'top',isTextBox:true,margin:0},o));
}
function take(s, text, o){
  o=o||{};
  s.addShape(pres.shapes.ROUNDED_RECTANGLE,{x:o.x||0.6,y:o.y||6.35,w:o.w||12.1,h:o.h||0.7,fill:{color:NAVY},rectRadius:0.08});
  s.addText(text,{x:(o.x||0.6)+0.25,y:o.y||6.35,w:(o.w||12.1)-0.5,h:o.h||0.7,fontFace:BF,fontSize:o.fs||16,bold:true,color:WHITE,valign:'middle',isTextBox:true,margin:0});
}
function gif(s,f,x,y,w,cap){
  const h=w*204/360;
  s.addImage({path:D+'media/'+f,x,y,w,h});
  if(cap) s.addText(cap,{x,y:y+h+0.03,w,h:0.3,fontFace:BF,fontSize:12,color:MUTED,align:'center',isTextBox:true,margin:0});
  return h;
}
function stat(s,x,y,w,big,label,color){
  s.addShape(pres.shapes.ROUNDED_RECTANGLE,{x,y,w,h:2.3,fill:{color:LIGHT},rectRadius:0.1});
  s.addText(big,{x,y:y+0.2,w,h:1.1,fontFace:HF,fontSize:54,bold:true,color:color||ORANGE,align:'center',isTextBox:true,margin:0});
  s.addText(label,{x:x+0.2,y:y+1.3,w:w-0.4,h:0.9,fontFace:BF,fontSize:15,color:INK,align:'center',valign:'top',isTextBox:true,margin:0});
}
const H=(t,o)=>({text:t,options:Object.assign({bold:true,color:WHITE,fill:{color:NAVY},fontFace:BF,fontSize:13,align:'center',valign:'middle'},o||{})});
const C=(t,o)=>({text:t,options:Object.assign({fontFace:BF,fontSize:13,color:INK,align:'center',valign:'middle'},o||{})});
const OK=t=>C(t,{color:GREEN,bold:true}), NO=t=>C(t,{color:RED}), NA=t=>C(t,{color:'9AA5B1'});

// ===== helpers =====
const fs = require('fs');
function gifGrid(s, items, o){
  const cols=o.cols, w=o.w, h=w*204/360, gx=o.gx||0.2, gy=o.gy||0.4;
  items.forEach((g,i)=>{ const x=o.x+(i%cols)*(w+gx), y=o.y+Math.floor(i/cols)*(h+gy);
    s.addImage({path:D+'media/'+g[0],x,y,w,h});
    s.addText(g[1],{x,y:y+h+0.02,w,h:0.3,fontFace:BF,fontSize:o.fs||11,color:MUTED,align:'center',isTextBox:true,margin:0}); });
}
function tbl(s, head, rows, o){
  const fsz=o.fs||12;
  const hr=head.map((t,i)=>H(t,{fontSize:fsz,align:i===0&&o.leftFirst!==false?'left':'center'}));
  const body=rows.map(r=>r.map((c,i)=>{
    if(typeof c==='object') { c.options=Object.assign({fontSize:fsz},c.options); return c; }
    return C(c,{fontSize:fsz,align:(i===0||o.left&&o.left.includes(i))?'left':'center',bold:i===0}); }));
  s.addTable([hr].concat(body),{x:o.x||0.6,y:o.y||1.6,w:o.w||12.1,colW:o.colW,rowH:o.rowH||0.34,border:{type:'solid',color:'D5DCE3',pt:0.75}});
}
function testSlide(title, sub, items, how, res){
  const s=base(title,sub);
  gifGrid(s,items,{x:0.6,y:1.65,w:3.9,cols:2,gx:0.2,gy:0.35});
  s.addText('Como é o teste',{x:9.3,y:1.65,w:3.4,h:0.35,fontFace:HF,fontSize:18,bold:true,color:NAVY,isTextBox:true,margin:0});
  s.addText(how,{x:9.3,y:2.05,w:3.4,h:1.8,fontFace:BF,fontSize:14,color:INK,valign:'top',isTextBox:true,margin:0});
  s.addText('Resultado',{x:9.3,y:3.9,w:3.4,h:0.35,fontFace:HF,fontSize:18,bold:true,color:ORANGE,isTextBox:true,margin:0});
  bullets(s,res,{x:9.3,y:4.3,w:3.4,h:2.5,fs:13});
  return s;
}
const fell=(t)=>t;
// 1
{ const s=pres.addSlide(); s.background={color:NAVY};
  s.addText('Qual política do Unitree G1 sobe escadas melhor?',{x:0.8,y:1.7,w:11.7,h:1.6,fontFace:HF,fontSize:46,bold:true,color:WHITE,isTextBox:true,margin:0,valign:'top'});
  s.addText('Benchmark de 18 políticas pré-treinadas de 12 repositórios, no mesmo robô, no mesmo simulador, com os mesmos testes',{x:0.8,y:3.5,w:9.5,h:1,fontFace:BF,fontSize:20,color:'CADCFC',isTextBox:true,margin:0,valign:'top'});
  s.addText('G1 29-DoF + Dex3  |  MuJoCo 3.6  |  controle 50 Hz  |  set/2026',{x:0.8,y:6.5,w:9,h:0.4,fontFace:BF,fontSize:14,color:'9FB3C8',isTextBox:true,margin:0});
  gif(s,'stairs_g1dwaq_stairs.gif',9.3,4.3,3.5);
  s.addNotes('Relatório completo: docs/ARENA_REPORT.md');
}
// 2
{ const s=base('O que foi feito','Mesmo robô, mesmos testes: a única variável é a política');
  bullets(s,[
   'Cada política roda via um adapter que reconstrói observação, ordem de juntas, gains e action decoding do repo original',
   '10 adapters verificados número a número contra o loop do próprio repo (diferença máxima 1,2e-5)',
   'Testes: flat, rough, arms waving, pushes até 1000 N, crouch, 3 protocolos de stairs, tracking de 11 clips',
   'Stairs: HumanoidBench (curso + reward), step-height sweep (8 a 24 cm), staircase do Safe100 (16 seeds)',
   'Política própria: g1_body, distilada de 2 teachers (G1DWAQ + GR00T WBC)'],{x:0.6,y:1.8,w:6.4,h:4.3});
  stat(s,7.5,1.8,2.5,'18','políticas pré-treinadas');
  stat(s,10.2,1.8,2.5,'12','repositórios',TEAL);
  stat(s,7.5,4.3,2.5,'3','protocolos de stairs',TEAL);
  stat(s,10.2,4.3,2.5,'11','clips de motion tracking');
  take(s,'Limitação: tudo em MuJoCo. Uma política pode depender do simulador em que foi treinada (caso Safe100).');
}
{ const s=base('Os testes, todos iguais para todas as políticas','Mesmo robô, mesmo simulador: só muda a política');
  tbl(s,['teste','o que acontece','score'],[
   ['flat','20 s: parado, 0,5 e 1,0 m/s, anda + gira 0,5 rad/s, lateral 0,3 m/s, para','erro médio |v − comando|'],
   ['sem mãos','mesmo teste no G1 sem as mãos Dex3','idem'],
   ['rough','mesmo teste em terreno aleatório de 6 cm','caiu ou não; erro de velocidade'],
   ['braços balançando','braços que a política não controla balançam a 0,5 Hz','idem'],
   ['push','andando a 0,5 m/s, empurrões laterais na pelve de 100 a 1000 N por 0,1 s a cada 3 s','maior empurrão sobrevivido'],
   ['crouch','altura da pelve comandada: 0,62 / 0,52 / 0,70 m (políticas com entrada de altura)','erro de altura; altura mais baixa'],
   ['stairs (scorecard)','10 degraus de 15 cm subindo, plataforma, 10 descendo, a 0,6 m/s','cruzou ou não'],
   ['stairs HumanoidBench','4 pirâmides de 5 degraus de 18 cm, 0,6 m de piso; reward do benchmark, 1000 passos','retorno (máx. 1000)'],
   ['step sweep','8 degraus subindo / plataforma / 8 descendo, alturas de 8 a 24 cm','maior altura cruzada'],
   ['stairs Safe100','6 degraus de 13 cm, piso de 0,35 m, 16 inícios aleatórios','% terminando no topo'],
   ['tracking','clip de referência desde o primeiro frame, 11 clips','erro de juntas, posição da raiz e heading']],
   {colW:[2.2,6.8,3.1],fs:12,rowH:0.42,left:[1,2]});
}
{ const s=base('Verificação dos adapters','Cada adapter roda ao lado do loop de controle original do repositório');
  tbl(s,['adapter','referência','dif. máx. obs','dif. máx. ação'],[
   ['unitree_rl_gym','seu deploy_mujoco.py','1,5e-6','3,9e-7'],
   ['mujoco_playground','seu OnnxController','9,2e-7','4,6e-7'],
   ['gr00t_wbc','seu GearWbcController','0','0'],
   ['g1_walk37 (2)','seu run_grid','0','6,0e-8'],
   ['holosoma (2)','seu LocomotionPolicy','4,2e-6','1,0e-6'],
   ['GMT','seu sim2sim.py','7,4e-6','3,7e-6'],
   ['TWIST','seus servidores de sim e de movimento','3,9e-6','1,9e-6'],
   ['Safe100 (CBF)','seu env mjlab em MuJoCo-Warp (GPU)','1,2e-5','n/a'],
   ['G1DWAQ, AGILE, GR00T WBC (lado Isaac)','tests/test_*_adapter.py','≤ 1,7e-4',''],
   ['SONIC, unitree_rl_lab, GRAIL','sem loop em Python (C++ ou só Isaac Lab)','verificados por comportamento','']],
   {colW:[3.6,4.6,2.2,1.7],fs:13,rowH:0.4,left:[1]});
  take(s,'Quando uma política cai, não é erro de adapter: observação e ação batem com o repositório original.',{y:6.3,h:0.7,fs:15});
}
// 3
{ const s=base('Escadas: resposta curta','Só 2 de 14 políticas de locomoção sobem degraus no arena');
  stat(s,0.6,1.8,3.8,'22 cm','G1DWAQ_Lab: maior degrau cruzado (sobe, passa a plataforma e desce)');
  stat(s,4.75,1.8,3.8,'622','g1_body (nosso): maior retorno no curso do HumanoidBench (G1DWAQ: 594; máx. 1000)',TEAL);
  stat(s,8.9,1.8,3.8,'12 de 14','políticas caem no primeiro degrau e pontuam 16 a 53 no HumanoidBench',RED);
  bullets(s,[
    'G1DWAQ: 100% de sucesso na staircase do Safe100; g1_body: 94% (13 cm)',
    'g1_body sobe até 20 cm: perde em altura máxima, ganha em retorno e ainda agacha a 0,52 m e aguenta 1000 N',
    'Nenhum outro repo sobe um único degrau neste simulador'],{x:0.6,y:4.4,w:12.1,h:1.8});
  take(s,'Melhor base para estudar escadas: G1DWAQ_Lab (teacher) e g1_body (student com mais habilidades).');
}
// 4
{ const s=base('Escadas 1: curso do HumanoidBench','4 pirâmides de 5 degraus de 18 cm; reward do próprio benchmark; 1000 passos');
  s.addImage({path:D+'figures/stairs_humanoidbench.png',x:0.6,y:1.6,w:5.9,h:5.9*761/975});
  s.addTable([[H('#'),H('política'),H('retorno'),H('subiu [m]')],
    [C('1'),C('g1_body',{bold:true}),OK('622'),C('0,87')],
    [C('2'),C('g1dwaq_stairs',{bold:true}),OK('594'),C('0,88')],
    [C('3'),C('unitree_rl_gym'),NO('53 (caiu)'),C('0,00')],
    [C('4'),C('gr00t_wbc'),NO('52 (caiu)'),C('0,00')],
    [C('5'),C('holosoma_ppo'),NO('51 (caiu)'),C('0,00')],
    [C('…'),C('demais 9'),NO('16 a 50'),C('0,00')]],
    {x:7.0,y:1.7,w:5.7,colW:[0.5,2.3,1.5,1.4],rowH:0.45,border:{type:'solid',color:'D5DCE3',pt:0.75}});
  s.addText('O baseline do HumanoidBench é para o H1; aqui só o curso e a fórmula de reward foram reaproveitados (nenhuma política treinou nele).',{x:7.0,y:5.1,w:5.7,h:0.9,fontFace:BF,fontSize:13,italic:true,color:MUTED,isTextBox:true,margin:0,valign:'top'});
  take(s,'Dois grupos: quem sobe (~600) e quem cai no primeiro degrau (~16 a 53). Não existe meio-termo.',{y:6.5,h:0.6,fs:15});
}
{ const s=base('Curso do HumanoidBench em vídeo','Mesma pista, reward do HumanoidBench');
  gifGrid(s,[['hb_g1_body.gif','g1_body: retorno 622'],['hb_g1dwaq_stairs.gif','g1dwaq_stairs: 594'],['hb_sonic.gif','sonic: 41 (cai)'],
    ['hb_agile_vel_height.gif','agile_vel_height: 50 (cai)'],['hb_holosoma_fastsac.gif','holosoma_fastsac: 47 (cai)'],['hb_safe100_cbf.gif','safe100_cbf: 16 (cai)']],
    {x:0.6,y:1.6,w:3.0,cols:3,gx:0.2,gy:0.45,fs:11});
  s.addText('4 pirâmides de 5 degraus de 18 cm com piso de 0,6 m. A reward vem do HumanoidBench: cabeça alta, torques pequenos, avanço até 1 m/s. As políticas nunca treinaram com ela.',{x:10.1,y:1.6,w:2.7,h:4.3,fontFace:BF,fontSize:14,color:INK,isTextBox:true,margin:0,valign:'top'});
  take(s,'Quem sobe chega a 0,87 m de altura (5 degraus); o resto cai no primeiro degrau.',{y:6.5,h:0.6,fs:15});
}
// 5
{ const s=base('Escadas 2: qual a altura máxima?','8 degraus subindo, plataforma, 8 descendo; altura de 8 a 24 cm; 0,5 m/s');
  s.addImage({path:D+'figures/stairs_sweep.png',x:0.6,y:1.6,w:5.9,h:5.9*787/975});
  s.addTable([[H('política'),H('8'),H('10'),H('12'),H('14'),H('16'),H('18'),H('20'),H('22'),H('24 cm')],
    [C('g1dwaq_stairs',{bold:true,align:'left'}),OK('✓'),OK('✓'),OK('✓'),OK('✓'),OK('✓'),OK('✓'),OK('✓'),OK('✓'),NA('·')],
    [C('g1_body',{bold:true,align:'left'}),NA('·'),OK('✓'),OK('✓'),OK('✓'),OK('✓'),OK('✓'),OK('✓'),NA('·'),NA('·')],
    [C('outras 12',{align:'left'}),NA('·'),NA('·'),NA('·'),NA('·'),NA('·'),NA('·'),NA('·'),NA('·'),NA('·')]],
    {x:6.9,y:1.8,w:5.9,colW:[1.7,0.46,0.46,0.46,0.46,0.46,0.46,0.46,0.46,0.58],rowH:0.5,border:{type:'solid',color:'D5DCE3',pt:0.75}});
  bullets(s,['G1DWAQ cruza de 8 a 22 cm; falha em 24 cm','g1_body cruza de 10 a 20 cm; falha em 8 cm e acima de 22 cm','As outras 12 políticas não cruzam nem 8 cm'],{x:6.9,y:4.2,w:5.9,h:2,fs:15});
  take(s,'Escada exige treino específico: nenhuma política de caminhada plana cruza nem 8 cm.',{y:6.5,h:0.6,fs:15});
}
{ const s=base('Step sweep: matriz completa','✓ = sobe, passa a plataforma, desce e continua de pé');
  const row=(n,a)=>[n].concat(a.map(v=>v?OK('✓'):NA('·')));
  const none=[0,0,0,0,0,0,0,0,0];
  const names=['agile_vel_height','gr00t_wbc','sonic','holosoma_fastsac','holosoma_ppo','unitree_rl_lab','mujoco_playground','unitree_rl_gym','g1_walk37_baseline','g1_walk37_robust','safe100_cbf','safe100_nominal'];
  tbl(s,['política','8 cm','10','12','14','16','18','20','22','24 cm'],
   [row('g1dwaq_stairs',[1,1,1,1,1,1,1,1,0]),row('g1_body',[0,1,1,1,1,1,1,0,0])].concat(names.map(n=>row(n,none))),
   {colW:[3.1,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],fs:12,rowH:0.33});
}
{ const s=base('Step sweep em vídeo','3 políticas, 12, 18 e 22 cm');
  gifGrid(s,[['sweep12_g1dwaq_stairs.gif','G1DWAQ 12 cm'],['sweep18_g1dwaq_stairs.gif','G1DWAQ 18 cm'],['sweep22_g1dwaq_stairs.gif','G1DWAQ 22 cm'],
    ['sweep12_g1_body.gif','g1_body 12 cm'],['sweep18_g1_body.gif','g1_body 18 cm'],['sweep22_g1_body.gif','g1_body 22 cm (cai)'],
    ['sweep12_safe100_cbf.gif','Safe100 12 cm'],['sweep18_safe100_cbf.gif','Safe100 18 cm'],['sweep22_safe100_cbf.gif','Safe100 22 cm']],
    {x:0.6,y:1.6,w:2.65,cols:3,gx:0.15,gy:0.3,fs:10});
  s.addText('Como é o teste',{x:9.3,y:1.65,w:3.4,h:0.35,fontFace:HF,fontSize:18,bold:true,color:NAVY,isTextBox:true,margin:0});
  s.addText('8 degraus subindo, plataforma, 8 descendo, com piso de 0,30 m e altura de 8 a 24 cm; comando de 0,5 m/s por 30 s. Score: a maior altura que a política cruza e ainda fica de pé.',{x:9.3,y:1.9,w:3.4,h:2.2,fontFace:BF,fontSize:14,color:INK,valign:'top',isTextBox:true,margin:0});
  s.addText('Resultado',{x:9.3,y:4.2,w:3.4,h:0.35,fontFace:HF,fontSize:18,bold:true,color:ORANGE,isTextBox:true,margin:0});
  bullets(s,['G1DWAQ: até 22 cm','g1_body: até 20 cm','Safe100 cai no CPU desde o início'],{x:9.3,y:4.6,w:3.4,h:1.8,fs:13});
}
{ const s=base('Escadas: tabela completa','14 políticas de locomoção, três protocolos');
  const r=(a,b,c,d,e,f,ok)=>[String(a),b,ok?OK(c):NO(c),d,e,f];
  tbl(s,['#','política','HB retorno (máx. 1000)','subiu no HB [m]','degrau máx. cruzado','Safe100 (13 cm)'],[
   r(1,'g1_body','622','0,87','20 cm','94%',true),r(2,'g1dwaq_stairs','594','0,88','22 cm','100%',true),
   r(3,'unitree_rl_gym','53 (caiu)','0,00','nenhum','0%'),r(4,'gr00t_wbc','52 (caiu)','0,00','nenhum','0%'),
   r(5,'holosoma_ppo','51 (caiu)','0,00','nenhum','0%'),r(6,'agile_vel_height','50 (caiu)','0,00','nenhum','0%'),
   r(7,'holosoma_fastsac','47 (caiu)','0,00','nenhum','0%'),r(8,'unitree_rl_lab','45 (caiu)','0,00','nenhum','0%'),
   r(9,'sonic','41 (caiu)','0,00','nenhum','0%'),r(10,'mujoco_playground','40 (caiu)','0,00','nenhum','0%'),
   r(11,'g1_walk37_robust','38 (caiu)','0,00','nenhum','0%'),r(12,'g1_walk37_baseline','29 (caiu)','0,00','nenhum','0%'),
   r(13,'safe100_nominal','21 (caiu)','0,00','nenhum','0%'),r(14,'safe100_cbf','16 (caiu)','0,00','nenhum','0%')],
   {colW:[0.5,2.9,2.4,2.0,2.3,2.0],fs:12,rowH:0.33,left:[1]});
}
{ const s=base('Escada padrão de 15 cm: todas as políticas (1/2)','10 degraus subindo, plataforma, 10 descendo, a 0,6 m/s');
  gifGrid(s,[['stairs_g1_body.gif','g1_body'],['stairs_g1dwaq_stairs.gif','g1dwaq_stairs'],['stairs_gr00t_wbc.gif','gr00t_wbc'],['stairs_sonic.gif','sonic'],
    ['stairs_agile_vel_height.gif','agile_vel_height'],['stairs_holosoma_fastsac.gif','holosoma_fastsac'],['stairs_holosoma_ppo.gif','holosoma_ppo'],['stairs_unitree_rl_lab.gif','unitree_rl_lab']],
    {x:0.6,y:1.5,w:2.9,cols:4,gx:0.17,gy:0.45,fs:12});
  take(s,'Só g1_body e g1dwaq_stairs cruzam a escada de 15 cm.',{y:6.3,h:0.6,fs:16});
}
{ const s=base('Escada padrão de 15 cm: todas as políticas (2/2)','Mesmo teste, demais políticas');
  gifGrid(s,[['stairs_unitree_rl_gym.gif','unitree_rl_gym'],['stairs_mujoco_playground.gif','mujoco_playground'],['stairs_safe100_cbf.gif','safe100_cbf'],
    ['stairs_safe100_nominal.gif','safe100_nominal'],['stairs_g1_walk37_baseline.gif','g1_walk37_baseline'],['stairs_g1_walk37_robust.gif','g1_walk37_robust']],
    {x:0.6,y:1.55,w:3.4,cols:3,gx:0.35,gy:0.4,fs:12});
  take(s,'Nenhuma delas cruza nem o primeiro degrau.',{y:6.3,h:0.6,fs:16});
}
// 7
{ const s=base('Safe100: 16/16 no seu simulador, cai em 1 s no CPU','Resultado de sim-to-sim, não erro de adapter');
  gif(s,'safe100_native.gif',0.6,1.7,5.6,'No simulador do paper (mjlab / MuJoCo-Warp, GPU): 16/16 no topo');
  gif(s,'hb_safe100_cbf.gif',6.9,1.7,5.6,'No arena (MuJoCo CPU): cai em 0,9 s');
  bullets(s,[
    'Observação do nosso adapter bate com a do mjlab (1,2e-5)',
    'O modelo compilado do próprio mjlab, rodado em MuJoCo CPU, também cai (0,86 s)',
    'Replay das ações do Warp em CPU: juntas divergem 0,09 rad em 10 passos'],{x:0.6,y:5.3,w:12.1,h:1.3,fs:15});
  take(s,'Política CBF-RL depende do solver de contato do Warp: não transfere. Só vale se o deploy for no mesmo simulador.',{y:6.6,h:0.6,fs:14});
}
{ const s=base('GRAIL: escadas não puderam ser medidas','Tracker da NVIDIA (fine-tune do SONIC) com height map; falha do terreno reconstruído, não das políticas');
  gifGrid(s,[['gstairs_grail_terrain.gif','GRAIL na escada reconstruída'],['gstairs_sonic_tracking.gif','SONIC na mesma escada']],{x:0.6,y:1.7,w:4.1,cols:2,gx:0.2});
  tbl(s,['clip','GMT','GRAIL','SONIC','TWIST'],[
   ['down_12steps',NO('caiu 1,5 s'),NO('caiu 1,9 s'),NO('caiu 3,2 s'),NO('caiu 1,4 s')],
   ['down_14steps',NO('caiu 1,3 s'),NO('caiu 1,7 s'),NO('caiu 3,1 s'),NO('caiu 0,9 s')],
   ['up_down_12steps',NO('caiu 1,4 s'),NO('caiu 4,1 s'),NO('caiu 2,1 s'),NO('caiu 1,7 s')]],
   {x:0.6,y:4.5,w:8.6,colW:[2.2,1.6,1.6,1.6,1.6],fs:13,rowH:0.45});
  bullets(s,['As malhas de escada do GRAIL são ativos normalizados (~1,2 × 1,25 × 2,0 m); a referência anda 2,2 m com queda de 1,25 m','Reconstruímos a escada pelos passos da referência: todos os trackers caem no primeiro degrau de descida','Falta a cena Isaac Lab do GRAIL (USD com escala em runtime)'],{x:9.5,y:1.7,w:3.3,h:4.5,fs:13});
}
// 8
{ const s=base('Potencial para estudo e expansão em escadas','Onde investir');
  const cols=[
   ['G1DWAQ_Lab',TEAL,['Melhor climber (22 cm), único que transfere para outro simulador','DreamWaQ: PPO + VAE state estimator, cego (sem terreno)','Usar como teacher principal de escadas']],
   ['g1_body (nosso)',ORANGE,['20 cm + crouch 0,52 m + 1000 N + braços livres','Piora em escadas após ~700 iterações (checkpoint 700)','Próximo run: manter o peso de imitação do DWAQ por mais tempo; teacher de rough terrain (holosoma ou AGILE)']],
   ['GRAIL (NVIDIA)',NAVY,['Fine-tune do SONIC com height map: feito para escadas, meios-fios e rampas','Stairs não medidas: malhas liberadas não casam com as referências; terreno reconstruído derruba todos os trackers','Precisa da cena Isaac Lab']]];
  cols.forEach((c,i)=>{ const x=0.6+i*4.1;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE,{x,y:1.7,w:3.9,h:4.5,fill:{color:LIGHT},rectRadius:0.1});
    s.addShape(pres.shapes.ROUNDED_RECTANGLE,{x,y:1.7,w:3.9,h:0.65,fill:{color:c[1]},rectRadius:0.1});
    s.addText(c[0],{x,y:1.7,w:3.9,h:0.65,fontFace:HF,fontSize:20,bold:true,color:WHITE,align:'center',valign:'middle',isTextBox:true,margin:0});
    bullets(s,c[2],{x:x+0.25,y:2.55,w:3.4,h:3.6,fs:15});
  });
  take(s,'Recomendação: distilar mais skills no g1_body, com G1DWAQ como teacher de escadas; GRAIL é o próximo teste de alto valor.');
}
// 9
{ const s=base('Locomoção geral: quem passa em quê','7 testes: flat, sem mãos, rough, braços balançando, pushes, crouch, stairs');
  s.addImage({path:D+'figures/loco_matrix.png',x:0.6,y:1.6,w:5.9,h:5.9*787/975});
  bullets(s,[
   'Melhor andar (erro de velocidade): WBC-AGILE, 0,016 m/s',
   'Mais robustos a pushes: G1DWAQ, SONIC, g1_body (1000 N)',
   'Rough ground (6 cm): só holosoma (2), AGILE e SONIC terminam',
   'Crouch: g1_body 0,52 m, GR00T WBC 0,50 m, SONIC 0,57 m',
   'Safe100 e g1_walk37 caem logo; unitree_rl_gym falha com braços balançando'],{x:6.9,y:1.8,w:5.9,h:4.4,fs:16});
  take(s,'Não existe uma política que ganhe tudo: SONIC é a mais completa fora de escadas; AGILE, a mais precisa.',{y:6.5,h:0.6,fs:15});
}
// 10
{ const s=base('Comparação direta: as políticas que importam','erro de velocidade em m/s (menor é melhor); rough = 6 cm de ruído');
  const rows=[
   ['WBC-AGILE','0,016',OK('ok'),'400 N','0,62 m',NO('não')],
   ['G1DWAQ_Lab','0,033',NO('cai 15 s'),'1000 N','n/a',OK('22 cm')],
   ['GR00T WBC','0,047',NO('cai 14 s'),'700 N','0,50 m',NO('não')],
   ['g1_body (nosso)','0,049',NO('cai 15 s'),'1000 N','0,52 m',OK('20 cm')],
   ['unitree_rl_lab','0,064',NO('cai 11 s'),'400 N','n/a',NO('não')],
   ['holosoma FastSAC','0,101',OK('ok'),'700 N','n/a',NO('não')],
   ['SONIC','0,104',OK('ok'),'1000 N','0,57 m',NO('não')],
   ['mujoco_playground','0,140',NO('cai 6 s'),'300 N','n/a',NO('não')]];
  const tab=[[H('política',{align:'left'}),H('flat vx'),H('rough'),H('push máx.'),H('crouch (menor)'),H('stairs')]]
    .concat(rows.map(r=>[C(r[0],{bold:true,align:'left'}),C(r[1]),typeof r[2]==='string'?C(r[2]):r[2],C(r[3]),C(r[4]),typeof r[5]==='string'?C(r[5]):r[5]]));
  s.addTable(tab,{x:0.6,y:1.7,w:12.1,colW:[3.1,1.6,1.9,1.7,2.0,1.8],rowH:0.5,border:{type:'solid',color:'D5DCE3',pt:0.75},fontSize:14});
  take(s,'Ninguém acumula tudo; o g1_body é o único que junta stairs, crouch e 1000 N, mas ainda cai em terreno irregular.',{y:6.5,h:0.6,fs:15});
}
{ const s=base('Locomoção: tabela completa','erro de velocidade em m/s (menor é melhor)');
  tbl(s,['política','flat vx / wz','sem mãos vx','rough','braços vx','push [N]','crouch (menor altura)'],[
   ['agile_vel_height','0,016 / 0,064','0,026','0,073','0,063','400','0,060 (0,62 m)'],
   ['g1_body','0,049 / 0,275','0,048',NO('caiu 15,1 s'),'0,201','1000','0,003 (0,52 m)'],
   ['g1_walk37_baseline',NO('caiu 18,2 s'),'precisa das mãos',NO('caiu 3,0 s'),'n/a','400','n/a'],
   ['g1_walk37_robust',NO('caiu 3,7 s'),'precisa das mãos',NO('caiu 3,1 s'),'n/a','300','n/a'],
   ['g1dwaq_stairs','0,033 / 0,153','0,034',NO('caiu 15,2 s'),'n/a','1000','n/a'],
   ['gr00t_wbc','0,047 / 0,137','0,052',NO('caiu 14,5 s'),'0,067','700','0,012 (0,50 m)'],
   ['holosoma_fastsac','0,101 / 0,079','0,114','0,131','n/a','700','n/a'],
   ['holosoma_ppo','0,175 / 0,056','0,174','0,181','n/a','500','n/a'],
   ['mujoco_playground','0,140 / 0,316','0,126',NO('caiu 6,2 s'),'n/a','300','n/a'],
   ['safe100_cbf',NO('caiu 0,9 s'),NO('caiu 0,9 s'),NO('caiu 0,9 s'),'n/a','0','n/a'],
   ['safe100_nominal',NO('caiu 0,9 s'),NO('caiu 0,9 s'),NO('caiu 0,9 s'),'n/a','0','n/a'],
   ['sonic','0,104 / 0,241','0,102','0,121','n/a','1000','0,032 (0,57 m)'],
   ['unitree_rl_gym','0,116 / 1,140','0,211',NO('caiu 13,3 s'),NO('caiu 4,2 s'),'500','n/a'],
   ['unitree_rl_lab','0,064 / 0,165','0,065',NO('caiu 11,1 s'),'n/a','400','n/a']],
   {colW:[2.6,1.9,1.8,1.5,1.4,1.2,1.7],fs:11,rowH:0.33});
}
{ const s=base('Locomoção plana e pushes','Gráficos dos testes flat e push');
  s.addImage({path:D+'figures/loco_flat.png',x:0.6,y:1.5,w:5.8,h:5.8*761/975});
  s.addImage({path:D+'figures/loco_push.png',x:6.9,y:1.5,w:5.8,h:5.8*761/975});
  take(s,'Empurrão máximo sobrevivido: G1DWAQ, SONIC e g1_body 1000 N; AGILE 400 N.',{y:6.55,h:0.55,fs:15});
}
{ testSlide('Teste de push (resistência a força)','Empurrões laterais crescentes enquanto o robô anda',
  [['push_g1_body.gif','g1_body'],['push_g1dwaq_stairs.gif','g1dwaq_stairs'],['push_sonic.gif','sonic'],['push_agile_vel_height.gif','agile_vel_height']],
  'Andando a 0,5 m/s, a pelve recebe um empurrão lateral de 100 a 1000 N por 0,1 s a cada 3 s. Score: o maior empurrão sobrevivido.',
  ['G1DWAQ, SONIC, g1_body: 1000 N','gr00t_wbc e holosoma_fastsac: 700 N','AGILE: 400 N; Safe100: 0']); }
{ testSlide('Teste de crouch (altura da pelve)','Comando de altura 0,62 / 0,52 / 0,70 m',
  [['crouch_g1_body.gif','g1_body'],['crouch_gr00t_wbc.gif','gr00t_wbc (teacher)'],['crouch_agile_vel_height.gif','agile_vel_height'],['crouch_sonic.gif','sonic']],
  'Só políticas com entrada de altura. A altura da pelve é comandada para baixo e para cima. Score: erro de altura e altura mais baixa alcançada.',
  ['g1_body: erro 0,003, chega a 0,52 m','gr00t_wbc: 0,012, 0,50 m (mais baixo)','sonic: 0,032, 0,57 m','AGILE: 0,060, 0,62 m']); }
{ testSlide('Teste de terreno irregular (rough)','Mesmo roteiro de 20 s em terreno aleatório de 6 cm',
  [['rough_g1_body.gif','g1_body (cai em 15 s)'],['rough_holosoma_fastsac.gif','holosoma_fastsac'],['rough_sonic.gif','sonic'],['rough_agile_vel_height.gif','agile_vel_height']],
  'Parado, 0,5 e 1,0 m/s, giro e lateral, sobre 6 cm de ruído de altura. Score: caiu ou não, e erro de velocidade.',
  ['Terminam: AGILE 0,073, SONIC 0,121, holosoma FastSAC 0,131, holosoma PPO 0,181','Caem: g1_body, G1DWAQ, GR00T WBC, unitree_rl_lab, unitree_rl_gym, mujoco_playground, Safe100']); }
{ testSlide('Teste de braços balançando','Os braços que a política não controla se movem',
  [['arms_g1_body.gif','g1_body'],['arms_gr00t_wbc.gif','gr00t_wbc'],['arms_agile_vel_height.gif','agile_vel_height'],['arms_unitree_rl_gym.gif','unitree_rl_gym (cai em 4 s)']],
  'Os braços balançam a 0,5 Hz enquanto a política controla só pernas e cintura. Mede se o equilíbrio aguenta o movimento dos braços.',
  ['AGILE 0,063 e GR00T WBC 0,067 (erro vx)','g1_body 0,201','unitree_rl_gym cai em 4,2 s']); }
// 12
{ const s=base('Motion tracking: SONIC é o melhor','SONIC, GMT, TWIST e GRAIL em 11 clips (dança, chute, agachamento, caminhada de 38 s)');
  s.addImage({path:D+'figures/track_root.png',x:0.6,y:1.6,w:4.1,h:4.1*1156/975});
  s.addImage({path:D+'figures/track_joint.png',x:4.9,y:1.6,w:4.1,h:4.1*1156/975});
  bullets(s,[
   'SONIC: menor erro de heading em 11/11 clips, menor erro de posição em 7; nunca cai',
   'GMT: menor erro de juntas em 5 clips, mas deriva (14 m em 38 s)',
   'GRAIL: ótimo em caminhada, cai em 4 de 11 clips dinâmicos',
   'TWIST: o mais fraco; cai em Gangnam Style'],{x:9.3,y:1.7,w:3.5,h:4.5,fs:14});
  take(s,'Para copiar movimentos de referência, SONIC. Para especialização em terreno, GRAIL (não medido aqui).',{y:6.55,h:0.6,fs:15});
}
{ const s=base('Tracking: erro de heading e métricas','Cada tracker começa no primeiro frame do clip e segue a referência');
  s.addImage({path:D+'figures/track_yaw.png',x:0.6,y:1.5,w:4.2,h:4.2*1156/975});
  bullets(s,['Erro de juntas: diferença média entre as 29 juntas e a referência [rad]','Erro de raiz: distância xy entre o robô e a referência, média e final [m]','Erro de heading: diferença de guinada [rad]','"Caiu": o robô perdeu o equilíbrio durante o clip','SONIC: menor heading em 11/11 clips; GMT deriva (14 m em 38 s)'],{x:5.3,y:1.7,w:7.4,h:4.2,fs:16});
  take(s,'Joint error é parecido entre trackers; a diferença está na deriva da raiz e no heading.',{y:6.5,h:0.6,fs:15});
}
{ const md=fs.readFileSync(D+'ARENA_REPORT.md','utf8').split('\n');
  const a=md.findIndex(l=>l.startsWith('## Motion tracking')), b=md.findIndex(l=>l.startsWith("unitree_rl_lab's dance policies"));
  const rows=md.slice(a,b).filter(l=>l.startsWith('| ')&&!l.startsWith('| clip |')&&!l.startsWith('|---')).map(l=>l.split('|').slice(1,-1).map(c=>c.trim())).filter(r=>r.length===6);
  const TN={sonic_tracking:'SONIC',gmt:'GMT',twist:'TWIST',grail_terrain:'GRAIL'};
  const clips=[]; rows.forEach(r=>{const n=r[0]; if(!clips.includes(n)) clips.push(n);});
  const groups=[clips.slice(0,3),clips.slice(3,6),clips.slice(6,9),clips.slice(9)];
  groups.forEach((g,gi)=>{ const s=base('Tracking: tabela completa ('+(gi+1)+'/4)','erro de juntas [rad]; erro de raiz xy [m], média (final); erro de heading [rad], média (final); ordenado por erro de juntas');
    const rs=[]; g.forEach(c=>rows.filter(r=>r[0]===c).forEach((r,i)=>{
      const own=r[1].includes('own clip'); const fellc=r[5].replace('fell','caiu');
      rs.push([i===0?c.replace(' (',' (').replace('s)','s)'):'',own?'própria política (clip dela)':(TN[r[1]]||r[1]),r[2],r[3],r[4],fellc?NO(fellc):'']); }));
    tbl(s,['clip','tracker','juntas','raiz xy','heading','caiu'],rs,{colW:[3.6,2.6,1.5,1.7,1.7,1.0],fs:12,rowH:0.36,left:[1]}); });
}
// 13
{ const s=base('Tracking em vídeo','Mesmo clip de dança e de chute, quatro trackers');
  const names=['sonic_tracking','gmt','twist','grail_terrain'], cap=['SONIC','GMT','TWIST','GRAIL'];
  names.forEach((n,i)=>{ const x=0.6+i*3.1; s.addText(cap[i],{x,y:1.6,w:2.9,h:0.35,fontFace:HF,fontSize:16,bold:true,color:NAVY,align:'center',isTextBox:true,margin:0});
    gif(s,'track_dance_'+n+'.gif',x,2.0,2.9); gif(s,'track_kick_'+n+'.gif',x,3.85,2.9); });
  s.addText('linha de cima: dança; linha de baixo: chute andando',{x:0.6,y:5.75,w:8,h:0.3,fontFace:BF,fontSize:12,color:MUTED,isTextBox:true,margin:0});
  take(s,'GRAIL cai na dança aos 20,7 s; SONIC acompanha os dois clips sem cair.',{y:6.5,h:0.6,fs:15});
}
{ const s=base('Tracking: dances da unitree_rl_lab e planner do SONIC','Políticas que seguem um clip próprio, e o planner do SONIC gerando movimento por comando de velocidade');
  gifGrid(s,[['track_unitree_dance_102.gif','unitree_rl_lab: dance_102 (21 s)'],['track_unitree_gangnam_style.gif','unitree_rl_lab: gangnam_style (20 s)'],['walk_sonic_planner.gif','SONIC com planner próprio']],
    {x:0.6,y:1.7,w:3.9,cols:3,gx:0.2,fs:12});
  bullets(s,['Dances da unitree_rl_lab: erro de juntas 0,080 e 0,081 rad, nunca caem','O planner do SONIC transforma comandos de velocidade em movimento planejado, que o tracker segue','Os mesmos clips rodados com SONIC, GMT, TWIST e GRAIL estão no slide anterior de tracking em vídeo'],{x:0.6,y:4.7,w:12.1,h:1.6,fs:15});
  take(s,'Nas dances da unitree_rl_lab, a política própria supera os trackers genéricos no mesmo clip.',{y:6.45,h:0.6,fs:15});
}
// 14
{ const s=base('Matriz de habilidades: quem faz o quê','✓ testado e funciona   ○ distribuído, não testado   ✗ falha   · não oferecido');
  const T=(a)=>a.map((v,i)=>i===0?C(v,{bold:true,align:'left'}):(v.startsWith('✓')?OK(v):v.startsWith('✗')?NO(v):v.startsWith('○')?C(v,{color:TEAL}):NA(v)));
  const tab=[[H('política',{align:'left'}),H('andar'),H('rough'),H('stairs'),H('crouch'),H('dança'),H('chute / boxe'),H('pulo, rastejar')],
   T(['SONIC (27 modos)','✓','✓','✗','✓','✓','✓ / ○','○ / ○']),
   T(['GMT, TWIST','✓','·','·','✓','✓','✓','·']),
   T(['GRAIL','✓','·','○','✗','✗','·','·']),
   T(['unitree_rl_lab','✓','✗','✗','·','✓ (2)','·','·']),
   T(['holosoma','✓','✓','✗','·','○ (2)','·','·']),
   T(['GR00T WBC','✓','✗','✗','✓','·','·','·']),
   T(['WBC-AGILE','✓','✓','✗','✓','·','·','·']),
   T(['G1DWAQ_Lab','✓','✗','✓ 22 cm','·','·','·','·']),
   T(['g1_body (nosso)','✓','✗','✓ 20 cm','✓','·','·','·']),
   T(['unitree_rl_gym, playground, g1_walk37, Safe100','✗','✗','✗','·','·','·','·'])];
  s.addTable(tab,{x:0.6,y:1.7,w:12.1,colW:[3.7,1.0,1.0,1.4,1.0,1.1,1.5,1.4],rowH:0.4,border:{type:'solid',color:'D5DCE3',pt:0.75}});
  take(s,'Só o SONIC distribui muitos modos (boxe, pulo, rastejar) numa única política; os demais são especialistas.',{y:6.6,h:0.6,fs:15});
}
{ const s=base('O que cada repositório distribui (1/2)','Amplitude anunciada vs políticas G1 realmente pré-treinadas');
  tbl(s,['repositório','amplitude anunciada','políticas G1 distribuídas','no arena'],[
   ['GR00T-WholeBodyControl','SONIC: tracker + planner com 27 modos; WBC (andar + equilíbrio); stack de teleop','SONIC (encoder, decoder, planner) e WBC (2 redes)','ambos'],
   ['holosoma','G1 e T1, PPO e FastSAC, IsaacGym / IsaacSim / MJWarp, retargeting','6 ONNX: G1 andar (2), T1 andar (2), G1 dança (2)','andar G1 (2); dances ainda não'],
   ['unitree_rl_lab','tasks para Go2, H1 e G1 no Isaac Lab','3 G1: velocidade e 2 dances','as três'],
   ['GRAIL','pipeline de dados (escadas, meios-fios, rampas, sentar, pegar); ~1000 clips por categoria','3 checkpoints de tracking','terrain em clips planos'],
   ['GMT','tracker geral de movimento','1 (23 juntas, sem mãos) + 8 clips','sim, 11 clips'],
   ['TWIST','sistema de teleoperação com dataset e treino','1 tracker geral','sim'],
   ['WBC-AGILE','velocidade, velocidade + altura, pegar e colocar (G1 e T1)','importada a política G1 de velocidade + altura','sim'],
   ['G1DWAQ_Lab','tasks Isaac Lab para G1 e H1: plano, rough, andar, correr','1 política G1 de escadas','sim; melhor em escadas']],
   {colW:[2.4,4.2,3.6,1.9],fs:12,rowH:0.62,left:[1,2,3]});
}
{ const s=base('O que cada repositório distribui (2/2)','Os que anunciam muito mas treinam pouco');
  tbl(s,['repositório','amplitude anunciada','políticas G1 distribuídas','no arena'],[
   ['mujoco_playground','~49 ambientes de locomoção e manipulação','6 ONNX demo de sim2sim, 1 é G1','a política G1 demo'],
   ['HumanoidBench','32 tarefas de corpo inteiro registradas','0 para G1 (2 .pt, tarefa de alcance); baselines são para H1','curso de escadas e reward como teste'],
   ['mujoco_menagerie','~70 modelos de robôs','não se aplica (só modelos)','modelos G1'],
   ['unitree_rl_gym','Go2, H1, H1_2, G1','1 G1 (12 juntas das pernas)','sim'],
   ['g1_walk_isaaclab_mujoco','projeto didático: treinar, fine-tune, exportar','2 (baseline e robust), para G1 antigo de 37 juntas','ambos (caem no nosso G1)'],
   ['Safe100Humanoid','uma tarefa: escadas com CBF-RL','2 (CBF e nominal)','ambos'],
   ['BFM-Zero','um modelo "promptable": reward, objetivo e tracking','1 modelo (CC-BY-NC)','baixado, ainda não adaptado']],
   {colW:[2.6,3.8,3.8,1.9],fs:12,rowH:0.55,left:[1,2,3]});
  take(s,'Quem anuncia mais ambientes (playground, HumanoidBench) entrega quase nenhuma política G1 pronta.',{y:6.5,h:0.6,fs:15});
}
// 15
{ const s=base('Nosso controlador: g1_body','Um student distilado de 2 teachers, treinado em Isaac Lab');
  const box=(x,y,w,t,c)=>{s.addShape(pres.shapes.ROUNDED_RECTANGLE,{x,y,w,h:0.9,fill:{color:c},rectRadius:0.08});
    s.addText(t,{x,y,w,h:0.9,fontFace:BF,fontSize:14,bold:true,color:WHITE,align:'center',valign:'middle',isTextBox:true,margin:0.05});};
  box(0.6,1.7,2.6,'G1DWAQ\nandar + escadas',TEAL); box(0.6,2.9,2.6,'GR00T WBC\ncrouch 0,50 a 0,72 m',TEAL);
  box(4.0,2.3,2.6,'PPO + imitação\n(gain_equivalent_target)',NAVY); box(7.4,2.3,2.4,'student g1_body\npernas + cintura',ORANGE);
  s.addShape(pres.shapes.LINE,{x:3.2,y:2.15,w:0.8,h:0.6,line:{color:MUTED,width:2,endArrowType:'triangle'}});
  s.addShape(pres.shapes.LINE,{x:3.2,y:3.35,w:0.8,h:-0.6,line:{color:MUTED,width:2,endArrowType:'triangle'},flipV:true});
  s.addShape(pres.shapes.LINE,{x:6.6,y:2.75,w:0.8,h:0,line:{color:MUTED,width:2,endArrowType:'triangle'}});
  bullets(s,['4096 G1 em 1 GPU: escadas 40%, plano 30%, rough 30%','Altura baixa: teacher GR00T WBC; senão G1DWAQ'],{x:10.2,y:1.7,w:2.6,h:2.3,fs:12});
  s.addImage({path:D+'figures/g1_body_checkpoints.png',x:0.6,y:4.15,w:5.0,h:5.0*390/975});
  bullets(s,['Checkpoints tardios perdem escadas: por isso o release é a iteração 700','Próximo run: manter o imitation weight alto por mais tempo + teacher de rough terrain'],{x:6.2,y:4.3,w:6.5,h:1.8,fs:15});
  take(s,'g1_body é o único aqui que faz stairs, crouch, 1000 N e braços livres ao mesmo tempo.',{y:6.5,h:0.6,fs:15});
}
// 16
{ const s=base('Conclusões','Quem usar para quê');
  const rows=[['Escadas','G1DWAQ_Lab (22 cm) e g1_body (20 cm, mais skills)'],['Andar com precisão','WBC-AGILE (0,016 m/s)'],['Pushes (1000 N)','G1DWAQ, SONIC, g1_body'],['Rough ground','holosoma, AGILE, SONIC'],['Crouch','GR00T WBC (0,50 m), g1_body (0,52 m)'],['Motion tracking','SONIC; GMT para erro de juntas em clips curtos'],['Mais skills num modelo','SONIC (27 modos); poucos foram testados']];
  const tab=rows.map(r=>[C(r[0],{bold:true,align:'left',fill:{color:LIGHT}}),C(r[1],{align:'left'})]);
  s.addTable(tab,{x:0.6,y:1.7,w:12.1,colW:[3.4,8.7],rowH:0.55,border:{type:'solid',color:'D5DCE3',pt:0.75},fontSize:16});
  take(s,'Escadas é um nicho: só 2 de 14 políticas resolvem, e nenhuma dança ou chuta. É aí que a distilação agrega mais.',{y:6.0,h:0.8,fs:16});
}
// 17
{ const s=base('Próximos passos');
  bullets(s,[
   'Novo run do g1_body: imitation weight do DWAQ alto por mais tempo; adicionar teacher de rough terrain (holosoma ou AGILE)',
   'GRAIL: medir escadas na cena Isaac Lab (USD stairs com scaling de runtime)',
   'Adicionar ao arena: holosoma G1 dance trackers, BFM-Zero, tarefas AGILE restantes',
   'Foundation student: distilação multi-teacher com tokenized action (SONIC / GRAIL) e supervisor hierárquico',
   'Segundo run de benchmark com mais cenários de escada (degraus irregulares, escada com corrimão)'],{x:0.6,y:1.7,w:7.4,h:4.5,fs:18});
  gif(s,'stairs_g1_body.gif',8.4,2.3,4.3,'g1_body, escada de 15 cm');
  s.addText('Relatório: docs/ARENA_REPORT.md',{x:0.6,y:6.6,w:8,h:0.4,fontFace:BF,fontSize:14,color:MUTED,isTextBox:true,margin:0});
}
pres.writeFile({fileName:'g1_arena_slides.pptx'}).then(()=>console.log('ok'));
