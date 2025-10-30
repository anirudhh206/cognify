from pydantic import BaseModel

class WalletQuery(BaseModel):
    type: str
    address: str

class WalletReport(BaseModel):
    type: str
    source: str
    payload: dict

class ShipmentUpdate(BaseModel):
    type: str
    shipment_id: str
    status: str
    eta: str
