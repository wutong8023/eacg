API_KEY_BASE = '/path/to/your/API_KEYSET/'
try:
    with open(API_KEY_BASE + 'deepseek.txt','r') as f:
        DEEPSEEK_APIKEY = f.read().strip()
    with open(API_KEY_BASE + 'closeai.txt','r') as f:
        CLOSEAI_APIKEY = f.read().strip()
    with open(API_KEY_BASE + 'qdd.txt','r') as f:
        QDD_APIKEY = f.read().strip()
except Exception as e:
    print(f"Error reading API keys: {e}, please check the API_KEY_BASE path")