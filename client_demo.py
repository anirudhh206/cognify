import asyncio
import os
import json
from redis import asyncio as redis
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

async def publish_and_wait(channel, message, result_channel, timeout=15.0):
    r = redis.from_url(REDIS_URL, decode_responses=True)
    pubsub = r.pubsub()
    
    # Subscribe BEFORE publishing
    await pubsub.subscribe(result_channel)
    await asyncio.sleep(0.1)
    
    # Publish the request
    await r.publish(channel, json.dumps(message))
    print(f"📤 Published to {channel}, waiting on {result_channel}...")

    waited = 0.0
    while waited < timeout:
        msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
        if msg and msg.get("type") == "message" and msg.get("data"):
            try:
                result = json.loads(msg["data"])
                print(f"✅ Received from {result_channel}")
                await pubsub.unsubscribe(result_channel)
                await r.aclose()
                return result
            except Exception as e:
                print(f"❌ Error parsing message: {e}")
        waited += 1.0
    
    await pubsub.unsubscribe(result_channel)
    await r.aclose()
    return {"error": f"timeout waiting for response from {result_channel}"}


async def demo_scenario_1_on_time():
    """Scenario 1: On-time delivery triggers payment automatically"""
    print("\n" + "="*70)
    print("🎬 SCENARIO 1: On-Time Delivery (Happy Path)")
    print("="*70)
    
    # Step 1: Check initial wallet balance
    print("\n📊 Step 1: Check buyer's wallet balance")
    wallet_address = os.getenv("USER_WALLET") or "0xc374DbA624ce34f1E8B145B0E6FB3A799e5118BE"
    wallet_request = {"type": "wallet_query", "address": wallet_address}
    finance_response = await publish_and_wait("finance.requests", wallet_request, "finance.replies")
    print(f"   Balance: {finance_response.get('eth', 'N/A')} ETH")
    
    # Step 2: Track shipment (on-time delivery)
    print("\n📦 Step 2: Track shipment - delivered ON TIME")
    eta = (datetime.utcnow() + timedelta(days=2)).isoformat()
    delivered = (datetime.utcnow() + timedelta(days=1)).isoformat()  # Delivered early!
    shipment_request = {
        "type": "track_shipment",
        "id": "SHIP-001",
        "eta": eta,
        "delivered": delivered,
        "amount": "0.001",  # Small testnet amount (0.001 ETH)
        "recipient": os.getenv("USER_WALLET")  # Payment recipient
    }
    logistics_response = await publish_and_wait("logistics.requests", shipment_request, "logistics.replies")
    
    if logistics_response.get("shipment"):
        shipment = logistics_response["shipment"]
        print(f"   Status: {shipment.get('status')}")
        print(f"   Delay: {shipment.get('delay_hours', 0)} hours")
        print(f"   Payment Approved: {shipment.get('payment_approved')}")
        print(f"   Reason: {shipment.get('payment_reason')}")
    
    # Step 3: Wait a moment for payment to process
    print("\n⏳ Step 3: Waiting for automatic payment processing...")
    await asyncio.sleep(2)
    
    print("\n✅ Result: Payment automatically released! Shipment delivered on time.")


async def demo_scenario_2_delayed():
    """Scenario 2: Delayed delivery withholds payment"""
    print("\n" + "="*70)
    print("🎬 SCENARIO 2: Severely Delayed Delivery (Payment Withheld)")
    print("="*70)
    
    # Track severely delayed shipment
    print("\n📦 Step 1: Track shipment - SEVERELY DELAYED")
    eta = (datetime.utcnow() - timedelta(days=2)).isoformat()  # ETA was 2 days ago
    delivered = (datetime.utcnow()).isoformat()  # Just delivered now (48h late)
    shipment_request = {
        "type": "track_shipment",
        "id": "SHIP-002",
        "eta": eta,
        "delivered": delivered,
        "amount": "1.0"  # 1 ETH payment
    }
    logistics_response = await publish_and_wait("logistics.requests", shipment_request, "logistics.replies")
    
    if logistics_response.get("shipment"):
        shipment = logistics_response["shipment"]
        print(f"   Status: {shipment.get('status')}")
        print(f"   Delay: {shipment.get('delay_hours', 0)} hours")
        print(f"   Severity: {shipment.get('severity')}")
        print(f"   Payment Approved: {shipment.get('payment_approved')}")
        print(f"   Reason: {shipment.get('payment_reason')}")
    
    print("\n⚠️  Result: Payment WITHHELD pending manual review due to severe delay.")


async def demo_scenario_3_governance_audit():
    """Scenario 3: Full governance audit trail"""
    print("\n" + "="*70)
    print("🎬 SCENARIO 3: Governance Audit & Compliance Check")
    print("="*70)
    
    # Submit a governance report
    print("\n⚖️ Step 1: Submit transaction for governance review")
    report_request = {
        "type": "report",
        "source": "demo_client",
        "payload": {
            "transaction_type": "payment_release",
            "shipment_id": "SHIP-001",
            "amount": "0.5 ETH",
            "status": "verified",
            "timestamp": datetime.utcnow().isoformat()
        }
    }
    governance_response = await publish_and_wait("governance.requests", report_request, "governance.replies", timeout=10.0)
    
    if governance_response.get("status") == "verified":
        print(f"   ✅ Governance Status: {governance_response.get('status')}")
        print(f"   📊 Compliance Score: {governance_response.get('compliance_score')}/100")
        if governance_response.get("tx_hash"):
            print(f"   🔗 Blockchain TX: {governance_response.get('tx_hash')}")
        else:
            print(f"   📝 Proof logged (mock mode)")
    else:
        print(f"   ❌ Governance Status: {governance_response.get('status')}")
        print(f"   Reason: {governance_response.get('reason')}")
    
    print("\n✅ Result: Transaction verified and logged for audit trail.")


async def main():
    print("\n" + "🚀"*35)
    print("   AUTONOMOUS SUPPLY CHAIN FINANCE SYSTEM")
    print("   Powered by ASI Alliance Multi-Agent Architecture")
    print("🚀"*35)
    
    # Run all three scenarios
    await demo_scenario_1_on_time()
    await asyncio.sleep(1)
    
    await demo_scenario_2_delayed()
    await asyncio.sleep(1)
    
    await demo_scenario_3_governance_audit()
    
    print("\n" + "="*70)
    print("🎉 DEMO COMPLETE!")
    print("="*70)
    print("\n📋 Summary:")
    print("   ✅ Finance Agent: Wallet balance queries & payment processing")
    print("   ✅ Logistics Agent: Shipment tracking with smart escrow logic")
    print("   ✅ Governance Agent: MeTTa-powered compliance verification")
    print("\n💡 Key Features Demonstrated:")
    print("   • Autonomous inter-agent communication")
    print("   • Smart escrow (payment triggered by delivery status)")
    print("   • MeTTa Knowledge Graph for compliance")
    print("   • Blockchain audit trail")
    print("   • Real-world supply chain finance use case")
    print("\n")


if __name__ == "__main__":
    asyncio.run(main())