const $=id=>document.getElementById(id);
const messages=$("messages"),input=$("message-input"),form=$("chat-form"),send=$("send");
const canvas=$("spike-chart"),ctx=canvas.getContext("2d"),bigCanvas=$("big-chart"),bigCtx=bigCanvas.getContext("2d");
const worldCanvas=$("world-canvas"),wctx=worldCanvas.getContext("2d");
let autoWorld=null;

function set(id,v){const e=$(id);if(e)e.textContent=v??"—"}
function esc(v){return String(v).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#039;"}[c]))}
function addMessage(kind,text){const x=document.createElement("div");x.className=`message ${kind}`;x.innerHTML=`<div class="avatar">${kind==="fly"?"FLY":"YOU"}</div><div class="bubble">${esc(text)}<span class="time">NOW</span></div>`;messages.appendChild(x);messages.scrollTop=messages.scrollHeight}
function draw(values,c){
 const r=c.getBoundingClientRect(),d=devicePixelRatio||1;c.width=Math.max(1,r.width*d);c.height=Math.max(1,r.height*d);
 const x=c.getContext("2d");x.setTransform(d,0,0,d,0,0);x.clearRect(0,0,r.width,r.height);
 const w=r.width,h=r.height;
 x.strokeStyle="rgba(255,255,255,.05)";x.lineWidth=1;
 for(let i=1;i<4;i++){const gy=(h/4)*i;x.beginPath();x.moveTo(0,gy);x.lineTo(w,gy);x.stroke()}
 if(!values.length){x.fillStyle="#596476";x.font="11px system-ui";x.fillText("Waiting for neural processing...",12,28);return}
 const spikes=values.map(v=>Number(v.spikes ?? v) || 0);
 const activity=values.map(v=>Math.max(0,Math.min(1,Number(v.activity)||0)));
 const drive=values.map(v=>Math.max(0,Math.min(1,Number(v.drive)||0)));
 const max=Math.max(...spikes,1);
 const barW=Math.max(3,w/Math.max(values.length,1)-4);
 spikes.forEach((v,i)=>{const bh=Math.max(2,(v/max)*(h*.72));const bx=i*(w/values.length)+2;const by=h-2-bh;x.fillStyle=i===spikes.length-1?"#b8ff3d":"rgba(184,255,61,.24)";x.fillRect(bx,by,barW,bh)});
 const line=(arr,stroke)=>{if(arr.length<2)return;x.strokeStyle=stroke;x.lineWidth=1.8;x.beginPath();arr.forEach((v,i)=>{const px=i*(w/(arr.length-1));const py=h-8-v*(h*.78);i?x.lineTo(px,py):x.moveTo(px,py)});x.stroke()};
 line(activity,"rgba(120,190,255,.9)");line(drive,"rgba(255,170,80,.9)");
}
function updateBrain(d){
 const n=d.neural||{},i=d.internal||{},h=d.history||{},states=h.states||[];
 set("decoded",n.decoded_direction ?? i.decoded_direction);set("confidence",n.confidence==null?(i.confidence==null?"—":Number(i.confidence).toFixed(4)):Number(n.confidence).toFixed(4));
 set("spikes",n.total_spikes==null?(i.total_spikes==null?"—":Number(i.total_spikes).toLocaleString()):Number(n.total_spikes).toLocaleString());
 set("latest-spikes",n.total_spikes==null?(i.total_spikes==null?"—":Number(i.total_spikes).toLocaleString()):Number(n.total_spikes).toLocaleString());
 set("drive",n.applied_drive==null?(i.applied_drive==null?"—":Number(i.applied_drive).toFixed(3)):Number(n.applied_drive).toFixed(3));
 set("activity",i.activity==null?"—":Number(i.activity).toFixed(2));set("interactions",i.interactions);
 set("direction",i.direction);set("previous",i.previous_direction);set("behavior",d.behavior?.action);set("pattern",i.pattern);set("stability",i.stability==null?"—":Number(i.stability).toFixed(2));set("changes",i.direction_changes);
 set("brain-interactions",i.interactions);set("brain-spikes",n.total_spikes==null?(i.total_spikes==null?"—":Number(i.total_spikes).toLocaleString()):Number(n.total_spikes).toLocaleString());
 set("brain-direction",n.decoded_direction ?? i.decoded_direction);set("brain-confidence",n.confidence==null?(i.confidence==null?"—":Number(i.confidence).toFixed(3)):Number(n.confidence).toFixed(3));
 set("brain-drive",n.applied_drive==null?(i.applied_drive==null?"—":Number(i.applied_drive).toFixed(3)):Number(n.applied_drive).toFixed(3));set("brain-activity",i.activity==null?"—":Number(i.activity).toFixed(2));set("brain-concept",i.concept || states.at(-1)?.concept || "—");set("brain-behavior",d.behavior?.action || states.at(-1)?.behavior || "—");
 set("brain-state-direction",i.direction);set("brain-state-previous",i.previous_direction);set("brain-state-pattern",i.pattern);set("brain-state-stability",i.stability==null?"—":Number(i.stability).toFixed(2));set("brain-state-changes",i.direction_changes);
 const chartData=states.length?states:((h.spikes||[]).map(v=>({spikes:v,activity:0,drive:0})));
 draw(chartData,bigCanvas);draw(h.spikes||[],canvas);
 renderBrainEvents(states);
}
function renderBrainEvents(states){
 const box=$("brain-events");if(!box)return;
 if(!states.length){box.innerHTML='<div class="empty-event">Waiting for the first neural event...</div>';return}
 box.innerHTML=states.slice(-6).reverse().map((e,idx)=>`<div class="brain-event"><span class="event-dot ${idx===0?"hot":""}"></span><div><b>${esc(e.direction||"—")}</b><span>${esc(e.concept||"NEURAL")} · ${esc(e.behavior||"—")}</span></div><strong>${Number(e.spikes||0).toLocaleString()}</strong></div>`).join("");
}
async function telemetry(){
  try{
    const r=await fetch("/api/telemetry",{credentials:"same-origin"});
    const d=await r.json();

    if(r.status===503 && d.code==="SESSION_CAPACITY"){
      $("connection").textContent="BUSY";
      return;
    }

    if(d.ok){
      $("connection").textContent="ONLINE";
      updateBrain(d);
    }else{
      $("connection").textContent="OFFLINE";
    }
  }catch{
    $("connection").textContent="OFFLINE";
  }
}
async function status(){
  try{
    const r=await fetch("/api/status",{credentials:"same-origin"});
    const d=await r.json();

    if(r.status===503 && d.code==="SESSION_CAPACITY"){
      $("connection").textContent="BUSY";
      return;
    }

    $("connection").textContent=d.ok?"ONLINE":"OFFLINE";
  }catch{
    $("connection").textContent="OFFLINE";
  }
}
async function sendMessage(message){
  message=message.trim();
  if(!message)return;

  addMessage("user",message);
  input.value="";
  send.disabled=true;
  send.textContent="THINKING";

  try{
    const r=await fetch("/api/chat",{
      method:"POST",
      credentials:"same-origin",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({message})
    });

    const text=await r.text();
    let d;

    try{
      d=JSON.parse(text);
    }catch{
      throw new Error(`HTTP ${r.status}: server returned invalid JSON`);
    }

    if(r.status===503 && d.code==="SESSION_CAPACITY"){
      $("connection").textContent="BUSY";
      addMessage("fly","FLY-001 is currently in use. Please try again shortly.");
      return;
    }

    if(!d.ok){
      addMessage("fly",`System error: ${d.error}`);
      return;
    }

    $("connection").textContent="ONLINE";
    addMessage("fly",d.response);

    if(d.supported)updateBrain(d);

  }catch(e){
    $("connection").textContent="OFFLINE";
    addMessage("fly",`Connection error: ${e.message}`);
  }finally{
    send.disabled=false;
    send.textContent="SEND";
    input.focus();
  }
}

let worldTrail = [];
let lastWorld = null;

function directionVector(direction){
  const d=String(direction||"").toUpperCase();
  if(d.includes("LEFT") && !d.includes("RIGHT")) return [-1,0];
  if(d.includes("RIGHT") && !d.includes("LEFT")) return [1,0];
  return [0,0];
}

function drawWorld(w){
 const cw=worldCanvas.clientWidth,ch=worldCanvas.clientHeight||600,d=devicePixelRatio||1;
 worldCanvas.width=Math.max(1,cw*d);worldCanvas.height=Math.max(1,ch*d);wctx.setTransform(d,0,0,d,0,0);
wctx.clearRect(0,0,cw,ch);

wctx.fillStyle = "#071018";
wctx.fillRect(0,0,cw,ch);
 const scale=Math.min(cw/100,ch/100), px=v=>v==null?null:v*scale, py=v=>v==null?null:ch-v*scale;

 // Grid
 wctx.strokeStyle="rgba(184,255,61,.075)";wctx.lineWidth=1;
 for(let x=0;x<=100;x+=10){wctx.beginPath();wctx.moveTo(px(x),0);wctx.lineTo(px(x),ch);wctx.stroke()}
 for(let y=0;y<=100;y+=10){wctx.beginPath();wctx.moveTo(0,py(y));wctx.lineTo(cw,py(y));wctx.stroke()}

 // Trail of the actual observed fly positions.
 if(worldTrail.length>1){
   wctx.beginPath();worldTrail.forEach((p,i)=>{const xx=px(p.x),yy=py(p.y);i?wctx.lineTo(xx,yy):wctx.moveTo(xx,yy)});
   wctx.strokeStyle="rgba(184,255,61,.28)";wctx.lineWidth=2;wctx.stroke();
   worldTrail.slice(0,-1).forEach((p,i)=>{const a=(i+1)/worldTrail.length*.22;wctx.fillStyle=`rgba(184,255,61,${a})`;wctx.beginPath();wctx.arc(px(p.x),py(p.y),2.5,0,Math.PI*2);wctx.fill()});
 }

 const fx=w.fly?.x,fy=w.fly?.y,tx=w.target?.x,ty=w.target?.y;
 if(fx!=null&&fy!=null&&tx!=null&&ty!=null){
   // Target vector
   wctx.setLineDash([7,8]);wctx.strokeStyle="rgba(184,255,61,.18)";wctx.lineWidth=1.5;wctx.beginPath();wctx.moveTo(px(fx),py(fy));wctx.lineTo(px(tx),py(ty));wctx.stroke();wctx.setLineDash([]);

   // Target glow + marker
   const pulse=10+Math.sin(performance.now()*.004)*3;
   const grad=wctx.createRadialGradient(px(tx),py(ty),1,px(tx),py(ty),pulse*2.8);grad.addColorStop(0,"rgba(184,255,61,.28)");grad.addColorStop(1,"rgba(184,255,61,0)");wctx.fillStyle=grad;wctx.beginPath();wctx.arc(px(tx),py(ty),pulse*2.8,0,Math.PI*2);wctx.fill();
   wctx.fillStyle="#b8ff3d";wctx.beginPath();wctx.arc(px(tx),py(ty),7,0,Math.PI*2);wctx.fill();
   wctx.fillStyle="#071008";wctx.font="700 10px system-ui";wctx.textAlign="center";wctx.fillText("TARGET",px(tx),py(ty)+4);

   // Fly marker + neural direction arrow
   wctx.fillStyle="rgba(7,10,14,.78)";wctx.beginPath();wctx.arc(px(fx),py(fy),24,0,Math.PI*2);wctx.fill();
   wctx.strokeStyle="rgba(184,255,61,.32)";wctx.lineWidth=1;wctx.beginPath();wctx.arc(px(fx),py(fy),24,0,Math.PI*2);wctx.stroke();
   wctx.font="25px system-ui";wctx.fillStyle="#fff";wctx.fillText("FLY",px(fx),py(fy)+9);

   const [vx,vy]=directionVector(w.direction);if(vx||vy){
     const ax=px(fx)+vx*38, ay=py(fy)-vy*38;
     wctx.strokeStyle="#b8ff3d";wctx.lineWidth=3;wctx.beginPath();wctx.moveTo(px(fx),py(fy));wctx.lineTo(ax,ay);wctx.stroke();
     wctx.fillStyle="#b8ff3d";wctx.beginPath();wctx.moveTo(ax,ay);wctx.lineTo(ax-vx*9-vy*5,ay+vy*9-vx*5);wctx.lineTo(ax-vx*9+vy*5,ay+vy*9+vx*5);wctx.closePath();wctx.fill();
   }

   // Coordinate readout
   wctx.textAlign="left";wctx.font="11px ui-monospace, monospace";wctx.fillStyle="rgba(220,230,220,.72)";wctx.fillText(`FLY  ${fx.toFixed(1)}, ${fy.toFixed(1)}`,14,24);wctx.fillText(`TARGET  ${tx.toFixed(1)}, ${ty.toFixed(1)}`,14,42);
 }else{
   wctx.fillStyle="#596476";wctx.font="12px system-ui";wctx.fillText("Waiting for world coordinates...",20,30);
 }
}

function updateWorld(w,history){
 if(w.fly?.x!=null&&w.fly?.y!=null){
   const point={x:Number(w.fly.x),y:Number(w.fly.y)};
   const last=worldTrail.at(-1);if(!last||Math.abs(last.x-point.x)>1e-9||Math.abs(last.y-point.y)>1e-9)worldTrail.push(point);
   if(worldTrail.length>80)worldTrail.shift();
 }
 lastWorld=w;
 set("wx",w.fly?.x==null?"—":Number(w.fly.x).toFixed(2));set("wy",w.fly?.y==null?"—":Number(w.fly.y).toFixed(2));set("tx",w.target?.x==null?"—":Number(w.target.x).toFixed(2));set("ty",w.target?.y==null?"—":Number(w.target.y).toFixed(2));set("wd",w.distance==null?"—":Number(w.distance).toFixed(2));set("wdir",w.direction);set("wbeh",w.behavior);set("wstep",w.step);
 set("wdir-big",w.direction||"WAITING");set("wbeh-big",w.behavior||"No world step yet");
 const confidence=(w.direction?1:0);const bar=$("wconfidence-bar");if(bar)bar.style.width=`${confidence*100}%`;
 drawWorld(w)
}
async function getWorld(){
  try{
    const r=await fetch("/api/world",{credentials:"same-origin"});
    const d=await r.json();

    if(r.status===503 && d.code==="SESSION_CAPACITY"){
      $("connection").textContent="BUSY";
      return;
    }

    if(d.ok){
      worldTrail=[];
      (d.history||[]).forEach(x=>{
        if(x.fly?.x!=null&&x.fly?.y!=null)
          worldTrail.push({x:Number(x.fly.x),y:Number(x.fly.y)});
      });
      worldTrail=worldTrail.slice(-80);
      updateWorld(d.world,d.history||[]);
    }
  }catch{}
}
async function stepWorld(){
  set("world-status","PROCESSING");

  try{
    const r=await fetch("/api/world/step",{
      method:"POST",
      credentials:"same-origin"
    });

    const text=await r.text();
    let d;

    try{
      d=JSON.parse(text);
    }catch{
      throw new Error(`HTTP ${r.status}: invalid server response`);
    }

    if(r.status===503 && d.code==="SESSION_CAPACITY"){
      $("connection").textContent="BUSY";
      set("world-status","BUSY");
      return;
    }

    if(d.ok){
      updateWorld(d.world);
      set("world-status","STEP COMPLETE");
    }else{
      set("world-status","ERROR");
    }

  }catch{
    set("world-status","OFFLINE");
  }
}
document.querySelectorAll(".tab").forEach(t=>t.onclick=()=>{document.querySelectorAll(".tab").forEach(x=>x.classList.remove("active"));document.querySelectorAll(".tab-content").forEach(x=>x.classList.remove("active"));t.classList.add("active");$("tab-"+t.dataset.tab).classList.add("active");if(t.dataset.tab==="world")getWorld()});
form.addEventListener("submit",e=>{e.preventDefault();sendMessage(input.value)});document.querySelectorAll(".quick-actions button").forEach(b=>b.onclick=()=>sendMessage(b.dataset.message));
$("step-world").onclick=stepWorld;$("auto-world").onclick=()=>{if(autoWorld){clearInterval(autoWorld);autoWorld=null;$("auto-world").textContent="START AUTO"}else{stepWorld();autoWorld=setInterval(stepWorld,900);$("auto-world").textContent="STOP AUTO"}};
$("reset-world").onclick=getWorld;window.onresize=()=>{telemetry();getWorld()};async function initializeFly001(){
  try{
    await fetch("/api/session",{credentials:"same-origin"});
  }catch(e){
    console.error("Session initialization failed:",e);
  }

  status();
  telemetry();
  getWorld();
  setInterval(telemetry,1000);
  input.focus();
}

initializeFly001();


// --- Fixed 3D fruit-fly viewer -------------------------------------------
const flyModel = $("fly-model");

if (flyModel) {
  let cursorX = window.innerWidth * 0.5;
  let cursorY = window.innerHeight * 0.5;
  let currentYaw = 0;
  let currentPitch = 78;

  window.addEventListener("pointermove", (event) => {
    cursorX = event.clientX;
    cursorY = event.clientY;
  }, {passive:true});

  function trackCursor() {
    const rect = flyModel.getBoundingClientRect();
    const centerX = rect.left + rect.width / 2;
    const centerY = rect.top + rect.height / 2;
    const nx = Math.max(-1, Math.min(1, (cursorX - centerX) / (rect.width * 0.95)));
    const ny = Math.max(-1, Math.min(1, (cursorY - centerY) / (rect.height * 0.95)));

    // The source model's yaw is reversed, so invert X to make
    // left cursor movement produce a left-looking fly.
    const targetYaw = -nx * 42;
    const targetPitch = 78 - ny * 24;

    currentYaw += (targetYaw - currentYaw) * 0.10;
    currentPitch += (targetPitch - currentPitch) * 0.10;
    flyModel.cameraOrbit = `${currentYaw.toFixed(2)}deg ${currentPitch.toFixed(2)}deg 2.8m`;

    requestAnimationFrame(trackCursor);
  }

  requestAnimationFrame(trackCursor);
}



