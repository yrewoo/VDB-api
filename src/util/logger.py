import os
import logging
from datetime import datetime

# 로그 디렉터리 생성
LOG_DIR = "log"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# 로그 설정
exp_time = datetime.now().strftime("%Y%m%d%H")
LOG_FILE = os.path.join(LOG_DIR, exp_time + ".log")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(funcName)s - %(message)s"

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOG_FILE),  # 로그 파일 저장
        logging.StreamHandler()  # 콘솔 출력
    ]
)


logger = logging.getLogger("app_logger")  # 앱 전용 로거
logger.setLevel(logging.INFO)

logging.getLogger("httpx").setLevel(logging.WARNING)

__all__ = ["logger"]
