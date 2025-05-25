function showsidebar(){
  const sidebar = document.querySelector(".sidebar");
  sidebar.classList.add("active")
}
function hidesidebar(){
  const sidebar = document.querySelector(".sidebar");
  sidebar.classList.remove("active");
}

   localStorage.setItem("userpopup","popup")
   const userpopup = localStorage.getItem("userpopup");
    const password = localStorage.getItem("password");

window.addEventListener("DOMContentLoaded", () => {
    /* ─────────────────── highlight menu ─────────────────── */
    if (location.pathname.includes("page.html")) {
      document.getElementById("nav-page")?.classList.add("active-link");
    }
  
    
  
    // --- Recording setup ---
    let mediaRecorder;
    let audioChunks = [];

  
    const startBtn = document.getElementById("startBtn");
    const stopBtn = document.getElementById("stopBtn");
  
    const containerwrapper = document.querySelector(".container-wrapper");

    if(localStorage.getItem("containervisible") === "true"){
        containerwrapper.style.display = "flex"
    }else{
        containerwrapper.style.display = "none"
    }

    const userid = localStorage.getItem("userid");
    console.log(userid)
    const username = localStorage.getItem("username"); 
    // --- Upload helper ---
    async function uploadPitch(pitch) {
      try {
        const res = await fetch("http://127.0.0.1:8000/api/save_pitch", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({pitch:pitch , username:username,userid:userid,password:password}),
        });
        if (!res.ok) throw new Error(`status ${res.status}`);


        containerwrapper.style.display = "flex";
        localStorage.setItem("containervisible","true");
        document.getElementById("log").innerText = "✅ อัปโหลดสำเร็จ!";

      } catch (err) {
        alert("❌ อัปโหลดล้มเหลว:", err.message);
      }
    }
  
    startBtn.addEventListener("click", async () => {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder = new MediaRecorder(stream);
      audioChunks = [];
  
      mediaRecorder.addEventListener("dataavailable", (e) =>
        audioChunks.push(e.data)
      );
  
      mediaRecorder.addEventListener("stop", async () => {
        const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
        const arrayBuffer = await audioBlob.arrayBuffer();
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
        const pitch = extractPitch(audioBuffer);

        
        await uploadPitch(pitch); // ← อัปโหลดทันที
  

      });
  
      mediaRecorder.start();
      document.getElementById("log").innerText = "🔴 กำลังบันทึก...";
      startBtn.style.display = "none";
      stopBtn.style.display = "block";
    });
  
    stopBtn.addEventListener("click", (e) => {
    e.preventDefault();
        mediaRecorder.stop();
      document.getElementById("log").innerText = "⏹️ หยุดบันทึกแล้ว";
    });
  
    // --- Simple pitch extraction (zero-crossing rate) ---
    function extractPitch(buffer) {
      const samples = buffer.getChannelData(0);
      const sr = buffer.sampleRate;
      const frame = 1024;
      const hop = 512;
      const freqArr = [];

      console.log("first-frame samples (20 ตัวแรก) =", samples.slice(0, 20));
  
      for (let i = 0; i < samples.length - frame; i += hop) {
        let crossings = 0;
        for (let j = 1; j < frame; j++) {
          if (samples[i + j - 1] > 0 && samples[i + j] <= 0) crossings++;
        }
        if (i === 0) {                     // เฟรมแรก
            console.log("crossings in frame #0 =", crossings);
          } else if (i < 10 * hop) {         // แค่ 10 เฟรมแรกพอ
            console.log(`frame #${i / hop}: crossings=${crossings}`);
          }
        freqArr.push((crossings * sr) / (2 * frame));
      }
      console.log("pitch[0..10] =", freqArr.slice(0, 10));
      return freqArr;
    }


    fetch("http://127.0.0.1:8000/popup", {
    method: "POST",
    headers: {
        "Content-Type": "application/json"
    },
    body: JSON.stringify({ username: username,userpopup:userpopup})
})
.then(res => res.json())
.then(data => {
    document.getElementById("point").innerText = data.score;       // ใช้คะแนนแทน
    document.getElementById("messager").innerText = data.username; // ชื่อผู้ใช้
    document.getElementById("check").innerText = data.detail1;     // จุดที่ควรปรับปรุง
})
.catch(err => {
    console.error(err);
});
   
  

    const closebtn = document.querySelector(".btn-close");
    if(closebtn){
        closebtn.addEventListener("click",function(){
            containerwrapper.style.display = "none";
            localStorage.removeItem("containervisible")
        })
    }




















  



    document
      .getElementById("sun")
      .addEventListener("click", () => document.body.classList.toggle("sun"));
  });

 
  