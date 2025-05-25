from fastapi import FastAPI, UploadFile, File, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import HTTPException
import bcrypt
import re
from typing import List
from datetime import datetime
import librosa
import numpy as np
import tempfile
import json




app = FastAPI()





app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ปลอดภัยมากขึ้นถ้าใส่ URL เฉพาะ เช่น ["http://localhost:5500"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

user_data = {}

class User(BaseModel):
    name:str
    username:str
    email:str
    password:str
    confirmpassword:str

class Userlogin(BaseModel):
    name:str
    username:str
    email:str
    password:str

def is_Valid_email(email: str) -> bool:
    pattern = r"^[a-zA-Z0-9._%+-]+@(gmail|hotmail|outlook|yahoo)\.(com|net|org|co\.th|ac\.th|edu|gov)$"
    return re.match(pattern, email) is not None

def is_Valid_name(name:str) -> bool:
    return re.match(r"^[\u0E00-\u0E7F\s]+$", name) is not None

# ดึง request จาหน้า register
@app.post("/register")
def register(user:User):
    if not is_Valid_email(user.email):
        raise HTTPException(status_code=400,detail="email format not valid or not allowed")
    
    if not is_Valid_name(user.name):
        raise HTTPException(status_code=400,detail="กรุณาใส่ชื่อภาษาไทย")
    
    if user.confirmpassword != user.password:
        raise HTTPException(status_code=400,detail="Password not matched")
    
    if not re.match("^[a-zA-Z0-9_]+$", user.username):
        raise HTTPException(status_code=400,detail="กรุณาใส่ usernmae เป็นภาษาอังกฤษหรือตัวเลข")
    
    if user.username in user_data:
        raise HTTPException(status_code=400,detail="username has already exists")
    
    for u in user_data.values():
        if u["email"] == user.email:
            raise HTTPException(status_code=400,detail="email has already exists")
    #hash_password
    hash_password = bcrypt.hashpw(user.password.encode(),bcrypt.gensalt())

    user_data[user.username] = {
        "name":user.name,
        "email":user.email, 
        "password":hash_password
    }
    return{"message":"register successfully"}

#ข้อมูลดึงจากหน้า login
@app.post("/login")
def login_user(user:Userlogin):
    user_record = user_data.get(user.username)

    if not user_record:
        raise HTTPException(status_code=400, detail="Username not found")
    
    if not bcrypt.checkpw(user.password.encode(), user_record["password"]):
        raise HTTPException(status_code=400, detail="Incorrect password")
    
    if user.email != user_record["email"]:
        raise HTTPException(status_code=400,detail="Email not correct")
    
    return{"message":"login successfully"}
    
@app.get("/user-all")
def user_all():
    return {"detail":user_data}


class PitchData(BaseModel):
    username: str         # <─ ชื่อผู้ใช้ที่บันทึกไว้ใน localStorage
    password:str
    userid:str #userid คือค่า id ที่รับเข้ามาจาก test ต่างๆ
    pitch: List[float]

#ข้อมูล request มาจากหน้า page
@app.post("/api/save_pitch")

async def save_pitch(data: PitchData):
        
    if data.username not in user_data:
        raise HTTPException(status_code=400, detail="Unknown username")
    timestamp = datetime.utcnow().isoformat()

    save_data = {
        "timestamp": timestamp,
        "password":data.password,
        "userid":data.userid,
        "username": data.username,   # <─ ผูกกับชื่อผู้ใช้
        "pitch": data.pitch,
    }

    try:
        with open("pitch_data.json", "a", encoding="utf-8") as f:
            f.write(json.dumps(save_data) + "\n")
        return {"message": "Saved successfully", "length": len(data.pitch)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

    
class UserRequest(BaseModel):
    username: str

class popup(BaseModel):
    userpopup:str
#mock_data ข้อมูลด้านในควรอยู่ใน user_data ที่มี username เป็น key
#ไม่ได้ setItem ไว้ไม่ต้องเอาออก
popup_data = {
    "popup": {"score": "80%","message":"เก่งมาก", "detail1": "1 จุด"},
    "jane": {"score": "92%","message":"เก่งมาก","detail1": "ฝึกการจัดการข้อมูล array ให้คล่องขึ้น"}
}

# ของจริงต้องเช็คจาก username จาก user_data
@app.post("/popup")
def popup(user_request: popup):
    username = user_request.userpopup
    if username in popup_data:
        score = popup_data[username]["score"]
        message = popup_data[username]["message"]
        detail1 = popup_data[username]["detail1"]
        return {
            "username": message,
            "score": score,
            "detail1": detail1
        }
    else:
        raise HTTPException(status_code=404, detail="User not found")


#ต้องไปเอา localstorage ปลอมออกในหน้า deepresult
# ข้อมูล request จากไฟล์ deepresult
deepresult_data = {
    "demo": {
        "result": [
            {"correctspace":"5/8(51%)", "correctsound": "80%", "overall": "ดีเยี่ยม"},
        ]
    }
}

@app.post("/deepresult")
def result(user_request: UserRequest):
    username = user_request.username
    if username in deepresult_data:
        #เปลี่ยนจาก email เป็น point ที่ได้จากที่ AI ประมวลผล และเวลา + คะแนนเฉลี่ยที่ได้จาก AI ประมวลผล
        correct = deepresult_data[username]["result"]
        # นำข้อมูลทีไ่ด้จาก AI ประมวลผลมา return ตรงนี้
        return {"username": username, "result":correct}
    else:
        raise HTTPException(status_code=404, detail="User not found")



#ดึง request จากหน้า deepresult
# mock_database ข้อมูลนี้ควรอยู่ใน user_data ที่มี username เป็น key
record_data = {
    "demo": {
        "records": [
            {"time": "2024-05-01 10:00", "score": 80, "detail": "ผลลัพธ์ที่ 1"},
            # {"time": "2024-05-02 14:30", "score": 75, "detail": "ผลลัพธ์ที่ 2"},
            # {"time": "2024-05-02 14:30", "score": 90, "detail": "ผลลัพธ์ที่ 3"},
        ]
        
    }
}
    

#end-point นี้รับ request จากหน้า result ต้องใช้ username เป็น key แทน
#ต้องเอา localstorage ปลอมออในหน้า result
@app.post("/result")
def result(user_request: UserRequest):
    username = user_request.username
    #จากบรรทัดนี้ควรเปลี่ยนเป็น user_data เมื่อใช้จริง ที่มีการเก็บ username เป็น key
    #เอาข้อมูลจาก data_base มาใช้
    if username in record_data:
        records = record_data[username]["records"]
        return {"username": username, "records": records}
    else:
        raise HTTPException(status_code=404, detail="User not found")
    

#เหมือนกันต้องไปปรับใน localstorage ให้เอา username ปลอมออก
# mock ของจริงต้องดึงจากผู้ใช้
arr_data = {
    "demo":{
        "arr":[25, 45, 19]  
    }
}

class HighlightIndexResponse(BaseModel):
    detail: str   
    wrong_indexes: List[int]



highlight_data = {
    "demo": {
        "detail": "บัด เดี้๋ยว ดัง หง่าง เหง่ง",
        "wrong_indexes": [1, 2]
    }
}
#รับ request จากหน้า deepresult
#เหมือนกันต้องไปปรับใน localstorage ให้เอา username ปลอมออก
# mock ของจริงต้องดึงจากผู้ใช้
arr_data = {
    "demo":{
        "arr":[25, 45, 19]  
    }
}
#รับ request จากหน้า deepresult
@app.post("/highlight")
def result(user_request: UserRequest):
    username = user_request.username
    #จากบรรทัดนี้ควรเปลี่ยนเป็น user_data เมื่อใช้จริง ที่มีการเก็บ username เป็น key
    if username in arr_data:
        arr = arr_data[username]["arr"]
        return {"username": username, "arr": arr}
    else:
        raise HTTPException(status_code=404, detail="User not found")
    




@app.get("/wrong-words")
def get_wrong_words():
    return [
        {"word": "บัดเดี๋ยว", "char_index": 0},
        {"word": "สะดุ้ง", "char_index": 34},
        {"word": "มนุษย์", "char_index": 132},
        {"word": "เห็นโยคีขี้รุ้งพุ่งออกมา", "char_index": 63},
        {"word": "แม้นใครรักรักมั่ง", "char_index": 327},
        {"word": "ถึงเถาววัลย์", "char_index": 165},
        {"word": "มนุษย์นี้ที่รักอยู่สองสถาน", "char_index": 225},
        {"word": "ที่พึ่งหนึ่งพึ่งได้เเต่กายตน", "char_index": 274}
    ]