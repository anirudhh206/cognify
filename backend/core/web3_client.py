import os
from web3 import Web3
from dotenv import load_dotenv
load_dotenv()

RPC = os.getenv("RPC_URL")
w3 = Web3(Web3.HTTPProvider(RPC))

def load_account():
    pk = os.getenv("PRIVATE_KEY")
    if not pk:
        raise RuntimeError("PRIVATE_KEY not set")
    return w3.eth.account.from_key(pk)