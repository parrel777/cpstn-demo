# 한국어로 이미지생성 (한복ver)

![alt text](image.png)

============

cd frontend  
npm install

============

cd backend  
python -m venv venv

Windows  
venv\Scripts\activate

macOS/Linux  
source venv/bin/activate

pip install -r requirements.txt  
pip install fastapi uvicorn

============

/frontend  
npm start

/backend  
uvicorn main:app --reload --host 127.0.0.1 --port 8000
