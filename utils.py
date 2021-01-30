from datetime import datetime

def get_current_time():
    now = datetime.now()
    return now.strftime("%m-%d-%Y-%H:%M:%S")