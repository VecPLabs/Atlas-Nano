/* global React, ReactDOM, MonitoringScreen, ModelsScreen, CalibrationScreen, CategoriesScreen, GauntletScreen, AuditScreen */

const { useState, useEffect } = React;

const SCREENS = {
  monitoring:  { title: "Live Monitoring", section: "Operations", C: MonitoringScreen },
  models:      { title: "Models",          section: "Operations", C: ModelsScreen },
  calibration: { title: "Calibration",     section: "Operations", C: CalibrationScreen },
  categories:  { title: "Categories",      section: "Reference",  C: CategoriesScreen },
  gauntlet:    { title: "Gauntlets",       section: "Reference",  C: GauntletScreen },
  audit:       { title: "Audit Log",       section: "Reference",  C: AuditScreen }
};

function App() {
  const [screen, setScreen] = useState(() => {
    return localStorage.getItem("atlas-nano:screen") || "monitoring";
  });
  const [streamOn, setStreamOn] = useState(true);

  useEffect(() => { localStorage.setItem("atlas-nano:screen", screen); }, [screen]);

  // wire sidebar navigation (DOM in index.html)
  useEffect(() => {
    const items = document.querySelectorAll(".nav-item");
    function onClick(ev) {
      const el = ev.currentTarget;
      const s = el.getAttribute("data-screen");
      if (s) setScreen(s);
    }
    items.forEach(el => {
      el.addEventListener("click", onClick);
    });
    return () => items.forEach(el => el.removeEventListener("click", onClick));
  }, []);

  // update active state + crumb
  useEffect(() => {
    document.querySelectorAll(".nav-item").forEach(el => {
      el.classList.toggle("active", el.getAttribute("data-screen") === screen);
    });
    const crumb = document.getElementById("crumb");
    if (crumb) {
      const info = SCREENS[screen];
      crumb.innerHTML = `${info.section} <span class="crumb-sep">/</span> <b>${info.title}</b>`;
    }
  }, [screen]);

  // listen for stream tweak
  useEffect(() => {
    function onMsg(e) {
      if (e && e.data && e.data.type === "atlas-stream") setStreamOn(e.data.on);
    }
    window.addEventListener("message", onMsg);
    return () => window.removeEventListener("message", onMsg);
  }, []);

  const Screen = SCREENS[screen].C;
  return <Screen streamOn={streamOn}/>;
}

ReactDOM.createRoot(document.getElementById("screen-host")).render(<App/>);
