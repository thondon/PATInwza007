/* ---------- Theme helper ---------- */
function applyTheme(mode = 'light') {
    document.documentElement.classList.toggle('dark', mode === 'dark');
    localStorage.setItem('theme', mode);
    const sel = document.getElementById('theme-select');
    if (sel) sel.value = mode;            // sync dropdownถ้ามี
  }
  
  function initTheme() {
    const saved = localStorage.getItem('theme') || 'light';
    applyTheme(saved);
  
    // ถ้าหน้านี้มี <select> ให้ผูก event
    const select = document.getElementById('theme-select');
    if (select && !select._themeBound) {   // กัน bind ซ้ำ
      select.addEventListener('change', e => applyTheme(e.target.value));
      select._themeBound = true;
    }
  }
  
  document.addEventListener('DOMContentLoaded', initTheme);
  