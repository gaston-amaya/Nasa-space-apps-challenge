#include <Arduino.h>
#include <WiFi.h>
#include <WebServer.h>
#include "model_weights.h"

#define WIFI_SSID ""
#define WIFI_PASS ""

#define AP_SSID "NASA-EVA-TRIAGE"
#define AP_PASS "artemis2026"

#define BOOT_BUTTON_PIN 0
#define STATUS_LED_PIN  2

WebServer server(80);

enum StatIndex {
    STAT_HR = 0,
    STAT_SPO2 = 1,
    STAT_BP = 2,
    STAT_TEMP = 3,
    STAT_GLU = 4,
    STAT_RAD = 5,
    STAT_CO2 = 6,
    STAT_SLEEP = 7,
    STAT_EX = 8,
    STAT_MOOD = 9
};

struct StatMetadata {
    const char* api_key;
    const char* serial_key;
    const char* full_name;
    const char* unit;
    float nom_min;
    float nom_max;
    float min_bound;
    float max_bound;
    float jitter_amp;
    uint8_t decimals;
};

const StatMetadata STAT_META[10] = {
    { "hr",    "HR",    "Heart Rate",       "BPM",   58.0f,   92.0f,   35.0f,  180.0f, 0.8f,  0 },
    { "spo2",  "SPO2",  "Blood Oxygen",     "%",     95.0f,  100.0f,   75.0f,  100.0f, 0.15f, 1 },
    { "bp",    "BP",    "Systolic BP",      "mmHg", 105.0f,  128.0f,   70.0f,  190.0f, 0.9f,  0 },
    { "temp",  "TEMP",  "Core Body Temp",   "C",     36.4f,   37.3f,   35.0f,   40.5f, 0.03f, 1 },
    { "glu",   "GLU",   "Blood Glucose",    "mg/dL", 75.0f,  115.0f,   40.0f,  220.0f, 0.7f,  0 },
    { "rad",   "RAD",   "Radiation Dose",   "uSv/h",  5.0f,   22.0f,    0.0f,  200.0f, 0.35f, 1 },
    { "co2",   "CO2",   "Cabin CO2",        "ppm",  400.0f, 2500.0f,  300.0f, 8000.0f, 18.0f, 0 },
    { "sleep", "SLEEP", "Circadian Sleep",  "hrs",    6.5f,    8.5f,    1.0f,   11.0f, 0.02f, 1 },
    { "ex",    "EX",    "ARED Exercise",    "min",   60.0f,  120.0f,    0.0f,  150.0f, 0.2f,  0 },
    { "mood",  "MOOD",  "Behavioral Mood",  "/5",     3.0f,    5.0f,    1.0f,    5.0f, 0.0f,  0 }
};

struct EnvironmentInfo {
    const char* code;
    const char* name;
    const char* one_way_delay;
    const char* rtt_delay;
};

const EnvironmentInfo ENVIRONMENTS[4] = {
    { "ISS",  "Low Earth Orbit (ISS)",         "0.2s", "0.4s" },
    { "SHIP", "Deep Spacecraft Transit",       "8m",   "16m"  },
    { "ROVR", "Planetary Surface Rover",       "14m",  "28m"  },
    { "BASE", "Lunar / Mars Surface Habitat",  "14m",  "28m"  }
};

const char* SCENARIO_NAMES[5] = {
    "0: Nominal Baseline",
    "1: EVA Suit Hypoxia",
    "2: Solar Radiation Storm",
    "3: Circadian Exhaustion",
    "4: Acute Hypoglycemia"
};

const float SCENARIO_PRESETS[5][10] = {
    {  72.0f, 98.2f, 116.0f, 36.8f, 92.0f,  11.4f, 1180.0f, 7.6f, 85.0f, 4.0f },
    { 132.0f, 88.5f, 149.0f, 36.9f, 95.0f,  12.0f, 1350.0f, 7.2f, 78.0f, 4.0f },
    {  98.0f, 97.4f, 122.0f, 38.3f, 96.0f, 112.0f, 4450.0f, 7.0f, 75.0f, 3.0f },
    {  99.0f, 97.8f, 121.0f, 36.9f, 90.0f,  13.2f, 1420.0f, 3.8f, 68.0f, 1.0f },
    {  78.0f, 98.0f, 114.0f, 36.7f, 58.0f,  11.8f, 1250.0f, 7.4f, 16.0f, 3.0f }
};

float anchor_stats[10];
float current_stats[10];
int active_scenario = 0;
int active_env = 0;
int alert_class = 0;
int prev_alert_class = -1;
float confidence = 1.0f;
float probs[5] = { 1.0f, 0.0f, 0.0f, 0.0f, 0.0f };
uint32_t infer_us = 0;
bool alert_acknowledged = false;
bool jitter_enabled = true;

int crew_xp = 100;
int streak_days = 5;
bool habit_done[4] = { false, false, false, false };

uint32_t last_tick_ms = 0;
uint32_t last_led_ms = 0;
bool led_state = false;
bool btn_pressed = false;
bool btn_long_fired = false;
uint32_t btn_down_ms = 0;
String serial_buffer = "";

// Offline UI stored in Flash (PROGMEM) to preserve ESP32 RAM
const char INDEX_HTML[] PROGMEM = R"rawliteral(<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Health Monitor</title>
<style>
:root{
  --bg:#0b1120;--card:#131d33;--card2:#1a2744;--border:#283b66;
  --text:#f8fafc;--muted:#94a3b8;--green:#10b981;--amber:#f59e0b;--red:#ef4444;--blue:#3b82f6;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:system-ui,-apple-system,sans-serif;padding:18px;max-width:820px;margin:0 auto}

.status-banner{background:rgba(16,185,129,0.14);border:2px solid var(--green);border-radius:14px;padding:16px 20px;margin-bottom:18px;display:flex;justify-content:space-between;align-items:center;gap:14px}
.status-banner.warn{background:rgba(245,158,11,0.16);border-color:var(--amber)}
.status-banner.crit{background:rgba(239,68,68,0.22);border-color:var(--red);animation:pulse 0.6s infinite}
@keyframes pulse{0%,100%{border-color:var(--red)}50%{border-color:#fff}}
.status-title{font-size:20px;font-weight:800}
.status-sub{font-size:14px;color:#e2e8f0;margin-top:4px;display:none}
.ack-btn{background:#fff;color:#0f172a;border:none;padding:10px 16px;border-radius:10px;font-weight:800;font-size:13px;cursor:pointer;display:none;white-space:nowrap}

.main{display:grid;grid-template-columns:230px 1fr;gap:16px;margin-bottom:18px}
@media(max-width:650px){.main{grid-template-columns:1fr}}

.suit-box{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:14px;display:flex;flex-direction:column;align-items:center;justify-content:center}
.suit-svg{width:170px;height:220px}
.body-zone{cursor:pointer;transition:0.2s;stroke-width:2.2}
.zone-ok{fill:rgba(16,185,129,0.25);stroke:var(--green)}
.zone-warn{fill:rgba(245,158,11,0.4);stroke:var(--amber)}
.zone-crit{fill:rgba(239,68,68,0.5);stroke:var(--red)}

.cards-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
@media(max-width:500px){.cards-grid{grid-template-columns:1fr}}
.tile{background:var(--card);border:2px solid var(--border);border-radius:14px;padding:16px;display:flex;flex-direction:column;justify-content:space-between}
.tile.warn{border-color:var(--amber);background:rgba(245,158,11,0.1)}
.tile.crit{border-color:var(--red);background:rgba(239,68,68,0.15)}
.tile-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}
.tile-title{font-size:14px;font-weight:700;color:var(--muted)}
.tile-badge{font-size:11px;font-weight:800;padding:3px 8px;border-radius:6px;background:rgba(16,185,129,0.2);color:#6ee7b7}
.tile.warn .tile-badge{background:rgba(245,158,11,0.25);color:#fcd34d}
.tile.crit .tile-badge{background:rgba(239,68,68,0.3);color:#fca5a5}
.tile-main{font-size:24px;font-weight:800;margin-bottom:4px}
.tile-sub{font-size:13px;color:var(--muted)}

details{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:10px 14px}
summary{cursor:pointer;font-size:13px;font-weight:700;color:var(--muted);user-select:none}
.sim-area{margin-top:12px;padding-top:12px;border-top:1px solid var(--border)}
.scen-row{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:12px}
.btn{background:var(--card2);color:var(--text);border:1px solid var(--border);padding:8px 13px;border-radius:8px;font-size:12px;font-weight:700;cursor:pointer}
.btn.active{background:var(--blue);border-color:#93c5fd}
.sliders{display:grid;grid-template-columns:repeat(2,1fr);gap:8px;margin-top:10px}
.sl{background:var(--card2);padding:8px 10px;border-radius:8px;font-size:12px}
.sl label{display:flex;justify-content:space-between;color:var(--muted);font-weight:600}
.sl label span{color:var(--text)}
.sl input{width:100%;margin-top:4px;accent-color:var(--blue)}
.send-btn{background:var(--blue);color:#fff;border:none;padding:9px 16px;border-radius:8px;font-weight:700;font-size:12px;cursor:pointer;margin-top:10px}
</style>
</head>
<body>

<section class="status-banner" id="banner">
  <div>
    <div class="status-title" id="st-title">Astronaut is Healthy</div>
    <div class="status-sub" id="st-sub"></div>
  </div>
  <button class="ack-btn" id="ack-btn" onclick="ackAlert()">Dismiss Alert</button>
</section>

<div class="main">
  <div class="suit-box">
    <svg class="suit-svg" viewBox="0 0 200 250">
      <rect id="z-suit" class="body-zone zone-ok" x="22" y="8" width="156" height="234" rx="28"/>
      <circle id="z-head" class="body-zone zone-ok" cx="100" cy="48" r="26"/>
      <rect id="z-chest" class="body-zone zone-ok" x="64" y="82" width="72" height="54" rx="10"/>
      <rect x="36" y="84" width="22" height="72" rx="10" fill="rgba(148,163,184,0.15)" stroke="#334155"/>
      <rect x="142" y="84" width="22" height="72" rx="10" fill="rgba(148,163,184,0.15)" stroke="#334155"/>
      <path id="z-core" class="body-zone zone-ok" d="M66 142 H134 V180 H112 V220 H104 V180 H96 V220 H88 V180 H66 Z"/>
    </svg>
  </div>

  <div class="cards-grid">
    <div class="tile" id="tile-cardio">
      <div class="tile-head"><span class="tile-title">HEART &amp; OXYGEN</span><span class="tile-badge" id="b-cardio">OK</span></div>
      <div class="tile-main" id="v-cardio">72 BPM</div>
      <div class="tile-sub" id="s-cardio">Oxygen: 98% &bull; BP: 116</div>
    </div>

    <div class="tile" id="tile-env">
      <div class="tile-head"><span class="tile-title">RADIATION &amp; AIR</span><span class="tile-badge" id="b-env">OK</span></div>
      <div class="tile-main" id="v-env">11 uSv/h</div>
      <div class="tile-sub" id="s-env">Temp: 36.8&deg;C &bull; CO2: 1180</div>
    </div>

    <div class="tile" id="tile-neuro">
      <div class="tile-head"><span class="tile-title">SLEEP &amp; MOOD</span><span class="tile-badge" id="b-neuro">OK</span></div>
      <div class="tile-main" id="v-neuro">7.6 hrs</div>
      <div class="tile-sub" id="s-neuro">Mood Score: 4 / 5</div>
    </div>

    <div class="tile" id="tile-meta">
      <div class="tile-head"><span class="tile-title">BLOOD SUGAR</span><span class="tile-badge" id="b-meta">OK</span></div>
      <div class="tile-main" id="v-meta">92 mg/dL</div>
      <div class="tile-sub" id="s-meta">Exercise: 85 min</div>
    </div>
  </div>
</div>

<details>
  <summary>&#9881; Test &amp; Simulate Scenarios</summary>
  <div class="sim-area">
    <div class="scen-row">
      <button class="btn active" id="scen-0" onclick="setScenario(0)">Normal</button>
      <button class="btn" id="scen-1" onclick="setScenario(1)">Simulate Low Oxygen</button>
      <button class="btn" id="scen-2" onclick="setScenario(2)">Simulate Radiation Storm</button>
      <button class="btn" id="scen-3" onclick="setScenario(3)">Simulate Fatigue</button>
      <button class="btn" id="scen-4" onclick="setScenario(4)">Simulate Low Blood Sugar</button>
    </div>

    <details>
      <summary>Fine-Tune Individual Sensors</summary>
      <div class="sliders" id="sliders-grid"></div>
      <button class="send-btn" onclick="sendCustomStats()">Apply Values</button>
    </details>
  </div>
</details>

<script>
const META = [
  {k:'hr',   name:'Heart Rate (BPM)', min:35,  max:180, step:1,   dec:0},
  {k:'spo2', name:'Oxygen (%)',       min:75,  max:100, step:0.5, dec:1},
  {k:'bp',   name:'Blood Pressure',   min:70,  max:190, step:1,   dec:0},
  {k:'temp', name:'Body Temp (C)',    min:35.0,max:40.5,step:0.1, dec:1},
  {k:'glu',  name:'Blood Sugar',      min:40,  max:220, step:1,   dec:0},
  {k:'rad',  name:'Radiation',        min:0.0, max:200, step:0.5, dec:1},
  {k:'co2',  name:'Cabin CO2',        min:300, max:8000,step:50,  dec:0},
  {k:'sleep',name:'Sleep (hrs)',      min:1.0, max:11.0,step:0.2, dec:1},
  {k:'ex',   name:'Exercise (min)',   min:0,   max:150, step:5,   dec:0},
  {k:'mood', name:'Mood (1-5)',       min:1,   max:5,   step:1,   dec:0}
];

const ALERTS = [
  {t:'Astronaut is Healthy',          s:''},
  {t:'Warning: Heart / Low Oxygen',   s:'Check spacesuit oxygen flow, sit down, and drink water.'},
  {t:'Warning: High Radiation / CO2', s:'Move to the radiation shelter and check air fan.'},
  {t:'Warning: High Fatigue',         s:'Pause work, rest for 20 minutes, and use bright light.'},
  {t:'Warning: Low Blood Sugar',      s:'Eat a glucose energy gel and log daily exercise.'}
];

let slidersInit = false;

function initUI() {
  document.getElementById('sliders-grid').innerHTML = META.map(m => `
    <div class="sl">
      <label>${m.name} <span id="lbl-${m.k}">--</span></label>
      <input type="range" id="sl-${m.k}" min="${m.min}" max="${m.max}" step="${m.step}"
             oninput="document.getElementById('lbl-${m.k}').textContent=this.value">
    </div>
  `).join('');
}

function syncSliders(d) {
  META.forEach(m => {
    document.getElementById('sl-'+m.k).value = d[m.k].toFixed(m.dec);
    document.getElementById('lbl-'+m.k).textContent = d[m.k].toFixed(m.dec);
  });
}

function setTileState(tileId, badgeId, zoneId, level) {
  const t = document.getElementById(tileId);
  const b = document.getElementById(badgeId);
  const z = document.getElementById(zoneId);
  t.className = 'tile' + (level===2 ? ' crit' : (level===1 ? ' warn' : ''));
  b.textContent = level===2 ? 'DANGER' : (level===1 ? 'CHECK' : 'OK');
  z.setAttribute('class', 'body-zone ' + (level===2 ? 'zone-crit' : (level===1 ? 'zone-warn' : 'zone-ok')));
}

function updateUI(d) {
  if(!slidersInit){ syncSliders(d); slidersInit = true; }

  const ban = document.getElementById('banner');
  ban.className = 'status-banner' + (d.alert>0 && !d.ack ? (d.severity===2 ? ' crit' : ' warn') : '');
  document.getElementById('st-title').textContent = ALERTS[d.alert].t;
  const sub = document.getElementById('st-sub');
  sub.style.display = d.alert>0 ? 'block' : 'none';
  sub.textContent = ALERTS[d.alert].s;
  document.getElementById('ack-btn').style.display = (d.alert>0 && !d.ack) ? 'block' : 'none';

  document.getElementById('v-cardio').textContent = `${d.hr.toFixed(0)} BPM`;
  document.getElementById('s-cardio').innerHTML = `Oxygen: ${d.spo2.toFixed(0)}% &bull; BP: ${d.bp.toFixed(0)}`;
  const lvCardio = (d.spo2<88||d.hr>135||d.hr<42||d.bp>165) ? 2 : ((d.spo2<92||d.hr>110||d.hr<48||d.bp>142||d.bp<90) ? 1 : 0);
  setTileState('tile-cardio', 'b-cardio', 'z-chest', lvCardio);

  document.getElementById('v-env').textContent = `${d.rad.toFixed(0)} uSv/h`;
  document.getElementById('s-env').innerHTML = `Temp: ${d.temp.toFixed(1)}&deg;C &bull; CO2: ${d.co2.toFixed(0)}`;
  const lvEnv = (d.rad>80||d.co2>6000||d.temp>=39) ? 2 : ((d.rad>50||d.co2>4000||d.temp>37.8) ? 1 : 0);
  setTileState('tile-env', 'b-env', 'z-suit', lvEnv);

  document.getElementById('v-neuro').textContent = `${d.sleep.toFixed(1)} hrs`;
  document.getElementById('s-neuro').textContent = `Mood: ${d.mood.toFixed(0)} / 5`;
  const lvNeuro = (d.sleep<2.5 && d.mood<=1) ? 2 : ((d.sleep<5.0||d.mood<2.5) ? 1 : 0);
  setTileState('tile-neuro', 'b-neuro', 'z-head', lvNeuro);

  document.getElementById('v-meta').textContent = `${d.glu.toFixed(0)} mg/dL`;
  document.getElementById('s-meta').textContent = `Exercise: ${d.ex.toFixed(0)} min`;
  const lvMeta = (d.glu<60||d.glu>190) ? 2 : ((d.glu<70||d.glu>145||d.ex<30) ? 1 : 0);
  setTileState('tile-meta', 'b-meta', 'z-core', lvMeta);

  for(let s=0;s<5;s++) document.getElementById('scen-'+s).classList.toggle('active', d.scenario===s);
}

async function poll() {
  try { const r = await fetch('/api/telemetry'); if(r.ok) updateUI(await r.json()); } catch(e){}
}
async function ctrl(q, sync=false) {
  try {
    const r = await fetch('/api/control?'+q, {method:'POST'});
    if(r.ok){ const d = await r.json(); if(sync) syncSliders(d); updateUI(d); }
  } catch(e){}
}
function setScenario(s){ ctrl('scenario='+s, true); }
function ackAlert(){ ctrl('ack=1'); }
function sendCustomStats(){
  ctrl(META.map(m => m.k+'='+encodeURIComponent(document.getElementById('sl-'+m.k).value)).join('&'));
}

initUI();
poll();
setInterval(poll, 1000);
</script>
</body>
</html>)rawliteral";

// Returns 0 = Nominal, 1 = Crew Self-Action, 2 = Critical Flight Surgeon Escalation
int compute_severity(int cls, const float s[10]) {
    if (cls == 0) return 0;
    if (cls == 1) return (s[STAT_SPO2] < 88.0f || s[STAT_HR] > 135.0f || s[STAT_HR] < 42.0f || s[STAT_BP] > 165.0f) ? 2 : 1;
    if (cls == 2) return (s[STAT_RAD] > 80.0f || s[STAT_CO2] > 6000.0f || s[STAT_TEMP] >= 39.0f) ? 2 : 1;
    if (cls == 3) return (s[STAT_SLEEP] < 2.5f && s[STAT_MOOD] <= 1.0f) ? 2 : 1;
    if (cls == 4) return (s[STAT_GLU] < 60.0f || s[STAT_GLU] > 190.0f) ? 2 : 1;
    return 1;
}

void run_esp32_inference(bool log_to_serial, const char* reason) {
    uint32_t t0 = micros();
    float conf = 0.0f;
    int pred = run_astronaut_triage(current_stats, &conf, probs);
    infer_us = micros() - t0;

    alert_class = pred;
    confidence = conf;

    bool changed = (alert_class != prev_alert_class);
    if (changed) {
        if (alert_class > 0) alert_acknowledged = false;
        prev_alert_class = alert_class;
    }

    if (log_to_serial || changed) {
        int sev = compute_severity(alert_class, current_stats);
        Serial.println(F("----------------------------------------------------------------------------"));
        Serial.printf(" [AI TRIAGE] Trigger: %s | Env: %s (%s Earth Delay)\n",
                      reason, ENVIRONMENTS[active_env].code, ENVIRONMENTS[active_env].one_way_delay);
        Serial.printf("   * Predicted Class : %d (Confidence: %.1f%%) | ESP32 Inference: %lu us (<1 ms)\n",
                      alert_class, confidence * 100.0f, (unsigned long)infer_us);
        Serial.printf("   * Escalation Tier : %s\n",
                      sev == 0 ? "[TIER 1: NOMINAL]" :
                      sev == 2 ? "[TIER 3: FLIGHT SURGEON PRIORITY TX]" : "[TIER 2: CREW SELF-ACTION]");
        Serial.printf("   * Vitals Snapshot : HR=%.0f SpO2=%.1f%% BP=%.0f Temp=%.1fC Glu=%.0f Rad=%.1f CO2=%.0f Slp=%.1fh Ex=%.0fm Mood=%.0f\n",
                      current_stats[0], current_stats[1], current_stats[2], current_stats[3], current_stats[4],
                      current_stats[5], current_stats[6], current_stats[7], current_stats[8], current_stats[9]);
        Serial.println(F("----------------------------------------------------------------------------"));
    }
}

void load_scenario(int scen, bool log_serial) {
    if (scen < 0 || scen > 4) return;
    active_scenario = scen;
    for (int i = 0; i < 10; ++i) {
        anchor_stats[i] = SCENARIO_PRESETS[scen][i];
        current_stats[i] = SCENARIO_PRESETS[scen][i];
    }
    alert_acknowledged = false;
    run_esp32_inference(log_serial, SCENARIO_NAMES[scen]);
}

void apply_sensor_jitter() {
    if (!jitter_enabled) {
        run_esp32_inference(false, "Periodic Tick");
        return;
    }
    for (int i = 0; i < 10; ++i) {
        float amp = STAT_META[i].jitter_amp;
        if (amp <= 0.0f) continue;
        float delta = ((float)random(-100, 101) / 100.0f) * amp;
        float next_val = anchor_stats[i] + delta;
        if (active_scenario == 0) {
            next_val = constrain(next_val, STAT_META[i].nom_min + 0.1f, STAT_META[i].nom_max - 0.1f);
        } else {
            next_val = constrain(next_val, STAT_META[i].min_bound, STAT_META[i].max_bound);
        }
        current_stats[i] = next_val;
    }
    run_esp32_inference(false, "1Hz Sensor Jitter");
}

void send_telemetry_json() {
    int sev = compute_severity(alert_class, current_stats);
    uint32_t up_s = millis() / 1000;

    char json[640];
    snprintf(json, sizeof(json),
        "{\"hr\":%.1f,\"spo2\":%.1f,\"bp\":%.1f,\"temp\":%.2f,\"glu\":%.1f,"
        "\"rad\":%.2f,\"co2\":%.0f,\"sleep\":%.1f,\"ex\":%.0f,\"mood\":%.0f,"
        "\"alert\":%d,\"conf\":%.4f,\"probs\":[%.4f,%.4f,%.4f,%.4f,%.4f],"
        "\"infer_us\":%lu,\"severity\":%d,\"ack\":%s,"
        "\"scenario\":%d,\"env\":%d,\"xp\":%d,\"streak\":%d,\"uptime_s\":%lu,"
        "\"jitter\":%s,\"tasks\":[%d,%d,%d,%d]}",
        current_stats[0], current_stats[1], current_stats[2], current_stats[3], current_stats[4],
        current_stats[5], current_stats[6], current_stats[7], current_stats[8], current_stats[9],
        alert_class, confidence, probs[0], probs[1], probs[2], probs[3], probs[4],
        (unsigned long)infer_us, sev, alert_acknowledged ? "true" : "false",
        active_scenario, active_env, crew_xp, streak_days, (unsigned long)up_s,
        jitter_enabled ? "true" : "false",
        habit_done[0] ? 1 : 0, habit_done[1] ? 1 : 0, habit_done[2] ? 1 : 0, habit_done[3] ? 1 : 0
    );

    server.sendHeader("Cache-Control", "no-cache, no-store, must-revalidate");
    server.send(200, "application/json", json);
}

void handle_root() {
    server.send_P(200, "text/html", INDEX_HTML);
}

void handle_control() {
    bool custom_stats_updated = false;

    if (server.hasArg("scenario")) {
        load_scenario(server.arg("scenario").toInt(), true);
    }
    if (server.hasArg("env")) {
        int e = server.arg("env").toInt();
        if (e >= 0 && e <= 3) active_env = e;
    }
    if (server.hasArg("ack")) {
        if (alert_class > 0 && !alert_acknowledged) {
            alert_acknowledged = true;
            crew_xp += 10;
        }
    }
    if (server.hasArg("jitter")) {
        jitter_enabled = (server.arg("jitter").toInt() != 0);
    }
    if (server.hasArg("task")) {
        int t = server.arg("task").toInt() - 1;
        if (t >= 0 && t < 4) {
            habit_done[t] = !habit_done[t];
            if (habit_done[t]) crew_xp += 25;
        }
    }

    for (int i = 0; i < 10; ++i) {
        if (server.hasArg(STAT_META[i].api_key)) {
            float v = constrain(server.arg(STAT_META[i].api_key).toFloat(), STAT_META[i].min_bound, STAT_META[i].max_bound);
            anchor_stats[i] = v;
            current_stats[i] = v;
            custom_stats_updated = true;
        }
    }

    if (custom_stats_updated) {
        alert_acknowledged = false;
        run_esp32_inference(true, "Web UI Manual Sliders");
    }

    send_telemetry_json();
}

void update_hardware_io() {
    uint32_t now = millis();

    // GPIO 0 button: short press (<1.2s) = ACK or next scenario; long press (>=1.2s) = cycle scenario
    bool is_low = (digitalRead(BOOT_BUTTON_PIN) == LOW);
    if (is_low && !btn_pressed) {
        btn_pressed = true;
        btn_long_fired = false;
        btn_down_ms = now;
    } else if (is_low && btn_pressed) {
        if (!btn_long_fired && (now - btn_down_ms >= 1200)) {
            btn_long_fired = true;
            load_scenario((active_scenario + 1) % 5, true);
        }
    } else if (!is_low && btn_pressed) {
        uint32_t held = now - btn_down_ms;
        btn_pressed = false;
        if (!btn_long_fired && held >= 30) {
            if (alert_class > 0 && !alert_acknowledged) {
                alert_acknowledged = true;
                crew_xp += 10;
            } else {
                load_scenario((active_scenario + 1) % 5, true);
            }
        }
    }

    // GPIO 2 LED: OFF when nominal/ACK, 1Hz blink for caution, 5Hz strobe for critical
    if (alert_class == 0 || alert_acknowledged) {
        digitalWrite(STATUS_LED_PIN, LOW);
        led_state = false;
    } else {
        int sev = compute_severity(alert_class, current_stats);
        uint32_t half_period = (sev == 2) ? 100 : 500;
        if (now - last_led_ms >= half_period) {
            last_led_ms = now;
            led_state = !led_state;
            digitalWrite(STATUS_LED_PIN, led_state ? HIGH : LOW);
        }
    }
}

void print_serial_banner() {
    Serial.println();
    Serial.println(F("============================================================================"));
    Serial.println(F("  2026 NASA SPACE APPS — AUTONOMOUS EDGE HEALTH TRIAGE WEB SERVER (ESP32)"));
    Serial.println(F("============================================================================"));
    Serial.printf("  [WIFI HOTSPOT] SSID: %s  |  Password: %s\n", AP_SSID, AP_PASS);
    Serial.printf("  [WEB CONSOLE]  Open in browser: http://%s\n", WiFi.softAPIP().toString().c_str());
    if (WiFi.status() == WL_CONNECTED) {
        Serial.printf("  [LOCAL LAN IP] Also reachable at: http://%s\n", WiFi.localIP().toString().c_str());
    }
    Serial.println(F("  COMMANDS: 0..4 | HR:128 SPO2:89 GLU:62 RAD:95 CO2:4500 | ACK | STATUS | HELP"));
    Serial.println(F("============================================================================"));
}

void handle_serial_command(String raw) {
    raw.trim();
    if (raw.length() == 0) return;
    String cmd = raw;
    cmd.toUpperCase();

    if (cmd.length() == 1 && cmd[0] >= '0' && cmd[0] <= '4') {
        load_scenario(cmd[0] - '0', true);
        return;
    }
    if (cmd == "HELP" || cmd == "?" || cmd == "IP") {
        print_serial_banner();
        return;
    }
    if (cmd == "STATUS") {
        run_esp32_inference(true, "Manual Serial STATUS");
        return;
    }
    if (cmd == "ACK") {
        alert_acknowledged = true;
        crew_xp += 10;
        return;
    }

    int colon = cmd.indexOf(':');
    if (colon > 0) {
        String k = cmd.substring(0, colon);
        String v_str = cmd.substring(colon + 1);
        k.trim();
        v_str.trim();

        if (k == "ENV") {
            if (v_str == "ISS" || v_str == "0") active_env = 0;
            else if (v_str == "SHIP" || v_str == "1") active_env = 1;
            else if (v_str == "ROVR" || v_str == "2") active_env = 2;
            else if (v_str == "BASE" || v_str == "MARS" || v_str == "3") active_env = 3;
            return;
        }

        for (int i = 0; i < 10; ++i) {
            if (k == STAT_META[i].serial_key) {
                float val = constrain(v_str.toFloat(), STAT_META[i].min_bound, STAT_META[i].max_bound);
                anchor_stats[i] = val;
                current_stats[i] = val;
                alert_acknowledged = false;
                char reason[48];
                snprintf(reason, sizeof(reason), "Serial %s:%.1f", STAT_META[i].serial_key, val);
                run_esp32_inference(true, reason);
                return;
            }
        }
    }
}

void setup() {
    Serial.begin(115200);
    delay(250);

    pinMode(BOOT_BUTTON_PIN, INPUT_PULLUP);
    pinMode(STATUS_LED_PIN, OUTPUT);
    digitalWrite(STATUS_LED_PIN, LOW);
    randomSeed(analogRead(0) ^ micros());

    if (strlen(WIFI_SSID) > 0) {
        WiFi.mode(WIFI_AP_STA);
        WiFi.softAP(AP_SSID, AP_PASS);
        WiFi.begin(WIFI_SSID, WIFI_PASS);
        uint32_t t_start = millis();
        while (WiFi.status() != WL_CONNECTED && (millis() - t_start < 6000)) {
            delay(250);
        }
    } else {
        WiFi.mode(WIFI_AP);
        WiFi.softAP(AP_SSID, AP_PASS);
    }

    server.on("/", HTTP_GET, handle_root);
    server.on("/api/telemetry", HTTP_GET, send_telemetry_json);
    server.on("/api/control", HTTP_GET, handle_control);
    server.on("/api/control", HTTP_POST, handle_control);
    server.begin();

    load_scenario(0, false);
    print_serial_banner();
    run_esp32_inference(true, "System Boot Nominal");
}

void loop() {
    server.handleClient();
    update_hardware_io();

    while (Serial.available() > 0) {
        char c = (char)Serial.read();
        if (c == '\n' || c == '\r') {
            if (serial_buffer.length() > 0) {
                handle_serial_command(serial_buffer);
                serial_buffer = "";
            }
        } else {
            serial_buffer += c;
        }
    }

    uint32_t now = millis();
    if (now - last_tick_ms >= 1000) {
        last_tick_ms = now;
        apply_sensor_jitter();
    }
}
