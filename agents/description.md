Input: - 
Output: -
To do: - 

*refactor notebook*: thinking and optimize notebook **.ipynb* 
struct of notebook: 
- cell md: Overview all step in note book and description. draw a table
- cell md: Step 0 import libraries - code from *.py and setup
- cell code: import libraries - code from *.py and setup
- cell md: STEP 1 read data : raw data + light process data
- cell code: use pandas read data. df.head(), df.info(), df.stats()
- cell md and code: STEP 2 EDA with raw - có sẵn trong notebook này. use code in /Users/quangmanh/Project/lab01/notebooks/eda.py
- cell md: STEP 3 Check Missing in data of all feature. - raw
- cell code: check missing - raw
- cell md: check duplicate - raw
- cell code: check duplicate - raw
- cell md and code: STEP 4 process raw. use code in /Users/quangmanh/Project/lab01/notebooks/preprocess.py
- check missing and duplicate in data after process.
- STEP 5 EDA with data after process.
- STEP 6 Balance and Unbalance data - have EDA
- STEP 7 Feature Engineering. TF IDF
- STEP 8 Split data in Balance and Unbalance
- STEP 9 train with 3 model. có check lại với thư viện và: dùng code trong 
/Users/quangmanh/Project/lab01/notebooks/model_from_scratch.py
- STEP 10 tìm threshold tốt với tập val.
- STEP 10 test với metric: dùng cái ma trận có sẵn và in ra các giá trị của ma trận. tính FPR, TPR. so sánh ngưỡng mặc định với ngưỡng tìm ra. so sánh data balance và unbalance. nhớ kĩ là cần tập Val để tìm ngưỡng tốt. 
- STEP 11 save model để deploy.

*NOTICE: BẠN NÊN ĐỌC LẠI CÁI NOTEBOOK HIỆN CÓ. VÀ TỐI ƯU CẤU TRÚC TRÊN ĐÓ, NHẤT LÀ GIỮ LẠI CÁC BIỂU ĐỒ ĐÃ CÓ, KÉT QUẢ MODEL. VÌ ĐÂY LÀ CODE LÀM LẠI CẤU TRÚC CHỨ KHÔNG PHẢI THAY ĐÔI KẾT QUẢ HIỆN CÓ CỦA LAB01.IPYNB* 

Bỏ V2 data đi khong dùng nữa. dùng balance và unbalance thôi.
Dùng model scratch chứ không dùng cái từ thư viện. 
chia sẵn 3 tập từ đầu.
