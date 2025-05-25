
window.addEventListener("DOMContentLoaded", () => {
    const currentpath = window.location.pathname;

    if (currentpath.includes("test")) {
        document.getElementById("nav-test")?.classList.add("active-link")
    }
    initVideoPlayer();
})
function initVideoPlayer() {
  const video = document.getElementById('myVideo');
  const playBtn = document.getElementById('playButton');
  const container = document.getElementById('videoContainer');
  const progress = document.getElementById('progress');
  const currentTimeLabel = document.getElementById('currentTime');
  const durationLabel = document.getElementById('duration');

  const playSVG = `
    <svg width="42" height="46" viewBox="0 0 42 46" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path id="Polygon 1" d="M37.5117 15.7959C42.9153 19.1651 42.828 27.1408 37.25 30.3613L13.25 44.2178L12.9834 44.3652C7.36371 47.3607 0.500027 43.2976 0.5 36.8564L0.5 9.14355C0.500023 2.6003 7.58336 -1.48935 13.25 1.78223L37.25 15.6387L37.5117 15.7959Z" fill="white" stroke="url(#paint0_linear_64_78)"/>
      <defs>
        <linearGradient id="paint0_linear_64_78" x1="17" y1="-9" x2="17" y2="55" gradientUnits="userSpaceOnUse">
          <stop stop-color="#7D84FF"/>
          <stop offset="1" stop-color="#D04AFF"/>
        </linearGradient>
      </defs>
    </svg>`;

  const stopSVG = `
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 384 512">
      <path d="M0 128C0 92.7 28.7 64 64 64H320c35.3 0 64 28.7 64 64V384c0 35.3-28.7 64-64 64H64c-35.3 0-64-28.7-64-64V128z" fill="white"/>
    </svg>`;

  function formatTime(seconds) {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
  }

  playBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    video.paused ? video.play() : video.pause();
  });

  video.addEventListener('play', () => {
    playBtn.classList.add('hidden');
    playBtn.innerHTML = stopSVG;
  });

  video.addEventListener('pause', () => {
    playBtn.classList.remove('hidden');
    playBtn.innerHTML = playSVG;
  });

  timeoutId = null;

  container.addEventListener('click', () => {
    if (!video.paused) {
      playBtn.innerHTML = stopSVG;
      playBtn.classList.remove('hidden');
      if(timeoutId !== null){
        clearTimeout(timeoutId);
        timeoutId = null;
        playBtn.classList.add("hidden")
      }else{
        timeoutId = setTimeout(() => {
            playBtn.classList.add("hidden")
            timeoutId = null;
        },3000)
      }
    }
  });

  function updateDurationLabel() {
    if (!isNaN(video.duration)) {
      durationLabel.textContent = formatTime(video.duration);
    } else {
      console.warn("⚠️ ไม่สามารถโหลด duration ได้");
    }
  }

  video.addEventListener('loadedmetadata', updateDurationLabel);
  video.addEventListener('canplay', updateDurationLabel);
  video.addEventListener('durationchange', updateDurationLabel);

  video.addEventListener('timeupdate', () => {
    const percent = (video.currentTime / video.duration) * 100;
    progress.value = percent;
    currentTimeLabel.textContent = formatTime(video.currentTime);
  });

  progress.addEventListener('input', () => {
    const newTime = (progress.value / 100) * video.duration;
    video.currentTime = newTime;
    currentTimeLabel.textContent = formatTime(newTime);
  });
}


function showsidebar() {
    const sidebar = document.querySelector(".sidebar");
    sidebar.classList.add("active");
}

function hidesidebar() {
    const sidebar = document.querySelector(".sidebar");
    sidebar.classList.remove("active");
}

document.getElementById("sun").addEventListener("click",function(){
  document.body.classList.toggle("sun")
})

//    document.addEventListener("click",function(){
//     backchange.forEach(btn => btn.classList.remove("change"))
//    })

