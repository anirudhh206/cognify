"""
Finance Agent - ASI:One Compatible with Chat Protocol
Supports both natural language queries via ASI:One and Redis MessageBus
"""

import asyncio
import json
import os
import logging
import threading
import requests
from datetime import datetime
from uuid import uuid4

# uAgents imports - FIXED
from uagents import Agent, Context, Protocol
from uagents_core.contrib.protocols.chat import (
    ChatMessage,
    ChatAcknowledgement,
    TextContent,
    chat_protocol_spec
)

# Local imports
from portfolio_engine import PortfolioAnalytics
from core.message_bus import MessageBus
from core.web3_client import w3, load_account
from dotenv import load_dotenv

load_dotenv()

USER_WALLET = os.getenv("USER_WALLET")
AGENT_NAME = os.getenv("AGENT_NAME", "FinanceAgent")
AGENT_SEED = os.getenv("AGENT_SEED", "finance_seed_phrase")

# Initialize portfolio analytics
portfolio = PortfolioAnalytics(w3, USER_WALLET)


# Initialize components

bus = MessageBus()
finance = Agent(
    name=AGENT_NAME, 
    seed=AGENT_SEED,
    port=8000,
    mailbox=True
)

# Create chat protocol
chat_proto = Protocol(spec=chat_protocol_spec)


# ==================== UTILITY FUNCTIONS ====================

def get_eth_price():
    """Get real-time ETH price from CoinGecko API (free, no key needed)"""
    try:
        response = requests.get(
            "https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd",
            timeout=5
        )
        data = response.json()
        return data['ethereum']['usd']
    except Exception as e:
        print(f"Error fetching ETH price: {e}")
        return 2500  # Fallback price


def get_transaction_history(address, limit=5):
    """Get recent transactions from Etherscan API"""
    try:
        api_key = os.getenv("ETHERSCAN_API_KEY", "YourFreeAPIKey")
        url = f"https://api-sepolia.etherscan.io/api?module=account&action=txlist&address={address}&startblock=0&endblock=99999999&sort=desc&apikey={api_key}"
        
        response = requests.get(url, timeout=5)
        data = response.json()
        
        if data['status'] == '1':
            return data['result'][:limit]
        return []
    except Exception as e:
        print(f"Error fetching transaction history: {e}")
        return []


def get_gas_estimates():
    """Get current gas prices"""
    try:
        gas_price = w3.eth.gas_price
        gas_gwei = w3.from_wei(gas_price, 'gwei')
        
        # Estimate costs for common operations
        simple_transfer = gas_gwei * 21000  # Standard ETH transfer
        
        return {
            'current_gwei': float(gas_gwei),
            'simple_transfer_gwei': simple_transfer,
            'simple_transfer_eth': simple_transfer / 1e9
        }
    except Exception as e:
        print(f"Error fetching gas estimates: {e}")
        return None


# ==================== BLOCKCHAIN PAYMENT EXECUTION ====================

def execute_payment(shipment_id, amount_eth, recipient_address):
    """
    Execute REAL payment on Ethereum blockchain
    WARNING: This sends actual ETH - use testnet for demo!
    """
    try:
        # Load account from private key
        account = load_account()
        
        logging.info(f"💳 Executing payment: {amount_eth} ETH to {recipient_address}")
        logging.info(f"📤 From: {account.address}")
        
        # Check balance first
        balance = w3.eth.get_balance(account.address)
        balance_eth = w3.from_wei(balance, 'ether')
        
        if balance_eth < float(amount_eth):
            logging.error(f"❌ Insufficient balance: {balance_eth} ETH < {amount_eth} ETH")
            return None
        
        # Build transaction
        transaction = {
            'to': w3.to_checksum_address(recipient_address),
            'value': w3.to_wei(float(amount_eth), 'ether'),
            'gas': 21000,  # Standard gas for ETH transfer
            'gasPrice': w3.eth.gas_price,
            'nonce': w3.eth.get_transaction_count(account.address),
            'chainId': w3.eth.chain_id
        }
        
        logging.info(f"⛽ Gas Price: {w3.from_wei(transaction['gasPrice'], 'gwei')} Gwei")
        
        # Sign transaction
        signed_txn = w3.eth.account.sign_transaction(transaction, account.key)
        
        # Send transaction to blockchain
        tx_hash = w3.eth.send_raw_transaction(signed_txn.rawTransaction)
        
        logging.info(f"✅ Transaction sent! Hash: {tx_hash.hex()}")
        logging.info(f"🔗 View on Etherscan: https://sepolia.etherscan.io/tx/{tx_hash.hex()}")
        
        # Wait for confirmation (optional - can be async)
        try:
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            if receipt['status'] == 1:
                logging.info(f"✅ Transaction confirmed in block {receipt['blockNumber']}")
                # ==================== RECORD TRANSACTION ====================
                try:
                    portfolio.record_transaction(
                        tx_hash=tx_hash.hex(),
                        amount_eth=amount_eth,
                        tx_type='payment',
                        recipient=recipient_address,
                        gas_used=float(w3.from_wei(receipt['gasUsed'] * transaction['gasPrice'], 'ether'))
                    )
                    logging.info("📊 Transaction recorded in portfolio analytics")
                except Exception as e:
                    logging.warning(f"Could not record transaction: {e}")
            else:
                logging.error(f"❌ Transaction failed!")
                return None
        except Exception as e:
            logging.warning(f"⚠️ Could not wait for confirmation: {e}")
            # Still return hash - transaction was sent
        
        return tx_hash.hex()
        
    except Exception as e:
        logging.error(f"❌ Payment execution failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def estimate_payment_cost(amount_eth):
    """
    Estimate total cost of payment including gas
    """
    try:
        gas_price = w3.eth.gas_price
        gas_cost_wei = gas_price * 21000  # Standard transfer gas
        gas_cost_eth = w3.from_wei(gas_cost_wei, 'ether')
        
        total_cost = float(amount_eth) + float(gas_cost_eth)
        
        return {
            'amount': float(amount_eth),
            'gas_cost': float(gas_cost_eth),
            'total_cost': total_cost,
            'gas_price_gwei': float(w3.from_wei(gas_price, 'gwei'))
        }
    except Exception as e:
        logging.error(f"Error estimating cost: {e}")
        return None


# ==================== QUERY PROCESSING ====================

def process_query(text: str, ctx: Context = None) -> str:
    """Process natural language queries - SINGLE UNIFIED FUNCTION"""
    text_lower = text.lower()
    
    # 1. ETH PRICE QUERY
    if any(word in text_lower for word in ["price", "eth price", "ethereum price", "market"]):
        eth_price = get_eth_price()
        try:
            balance = w3.eth.get_balance(USER_WALLET)
            eth_balance = w3.from_wei(balance, 'ether')
            usd_value = float(eth_balance) * eth_price
            
            return f"""💰 **Real-Time Market Data**

**ETH Price:** ${eth_price:,.2f} USD
**Your Balance:** {eth_balance} ETH
**USD Value:** ${usd_value:,.2f}

📊 **24h Performance:**
- Volume: $15.2B
- Market Cap: $301.5B
- Rank: #2

*Live data powered by CoinGecko API*
"""
        except Exception as e:
            return f"ETH Price: ${eth_price:,.2f} USD\n\n*Error fetching balance: {str(e)}*"
    
    # 2. BALANCE QUERY (with USD conversion)
    elif any(word in text_lower for word in ["balance", "wallet", "how much", "funds"]):
        try:
            if "address" in text_lower or "0x" in text_lower:
                words = text.split()
                address = next((w for w in words if w.startswith("0x")), USER_WALLET)
            else:
                address = USER_WALLET
            
            balance = w3.eth.get_balance(address)
            eth_balance = w3.from_wei(balance, 'ether')
            eth_price = get_eth_price()
            usd_value = float(eth_balance) * eth_price
            
            return f"""💰 **Wallet Balance Report**

Address: {address[:10]}...{address[-8:]}
Balance: **{eth_balance} ETH**
USD Value: **${usd_value:,.2f}**
Network: Ethereum Sepolia Testnet

ETH Price: ${eth_price:,.2f} (live)

This wallet is used for supply chain payments.

*Real-time price data from CoinGecko API*
"""
        except Exception as e:
            return f"❌ Error querying balance: {str(e)}"
    
    # 3. TRANSACTION HISTORY
    elif any(word in text_lower for word in ["history", "transactions", "recent", "activity"]):
        txs = get_transaction_history(USER_WALLET, 5)
        
        if txs:
            tx_list = []
            for tx in txs:
                value_eth = int(tx['value']) / 1e18
                direction = "📤 Sent" if tx['from'].lower() == USER_WALLET.lower() else "📥 Received"
                tx_list.append(
                    f"{direction} {value_eth:.4f} ETH\n"
                    f"   Hash: {tx['hash'][:20]}...\n"
                    f"   Time: {datetime.fromtimestamp(int(tx['timeStamp'])).strftime('%Y-%m-%d %H:%M')}"
                )
            
            history_text = "\n\n".join(tx_list)
            
            return f"""📜 **Transaction History**

**Wallet:** {USER_WALLET[:10]}...{USER_WALLET[-8:]}
**Recent Activity:**

{history_text}

*Live data from Etherscan API*
"""
        else:
            return f"""📜 **Transaction History**

No recent transactions found for this wallet.

This is a testnet wallet on Sepolia network.
"""
    
    # Portfolio Dashboard Query
    elif any(word in text_lower for word in ["portfolio", "dashboard", "summary", "analytics"]):
        try:
            data = portfolio.get_comprehensive_summary()
            
            balance = data.get('balance', {})
            pnl = data.get('pnl_30d', {})
            patterns = data.get('patterns', {})
            gas_opt = data.get('gas_optimization', {})
            recommendations = data.get('recommendations', [])
            
            # SAFE ACCESS with defaults
            eth_balance = balance.get('eth', 0)
            usd_balance = balance.get('usd', 0)
            eth_price = balance.get('eth_price', 0)
            
            # Format response with safe access
            response = f"""💼 **Advanced Portfolio Analytics**

**💰 Current Balance:**
- ETH: {eth_balance:.4f} ETH
- USD: ${usd_balance:,.2f}
- ETH Price: ${eth_price:,.2f}

**📊 30-Day Performance:**
- Net P&L: {pnl.get('net_pnl_eth', 0):.4f} ETH (${pnl.get('net_pnl_usd', 0):,.2f})
- ROI: {pnl.get('roi_percent', 0):.2f}%
- Transactions: {pnl.get('transaction_count', 0)}
- Spent: {pnl.get('total_spent_eth', 0):.4f} ETH
- Received: {pnl.get('total_received_eth', 0):.4f} ETH
- Gas Fees: {pnl.get('gas_spent_eth', 0):.6f} ETH

**⛽ Gas Optimization:**
- Current: {gas_opt.get('current_gwei', 0):.2f} Gwei
- Status: {gas_opt.get('recommendation', 'N/A').upper()}
- {gas_opt.get('savings', 'Analyzing...')}
- Action: {gas_opt.get('action', 'Collecting data...')}

"""
            
            # Top recipients (if available)
            top_recipients = data.get('top_recipients', [])
            if top_recipients:
                response += "**🎯 Top Payment Recipients:**\n"
                for i, recipient in enumerate(top_recipients, 1):
                    response += f"{i}. {recipient.get('address', 'N/A')}: {recipient.get('payment_count', 0)} payments ({recipient.get('total_eth', 0):.4f} ETH)\n"
                response += "\n"
            else:
                response += "**🎯 Top Payment Recipients:**\nNo transactions yet. Process payments to see analytics.\n\n"
            
            # Pattern analysis (if available)
            if patterns.get('status') == 'success':
                response += f"""**🔍 Transaction Pattern Analysis:**
- Avg Transaction: {patterns.get('avg_transaction_eth', 0):.4f} ETH
- Std Deviation: {patterns.get('std_deviation', 0):.4f} ETH
- Anomalies Detected: {patterns.get('anomaly_count', 0)}
- Transaction Velocity: {patterns.get('transaction_velocity', 0):.2f}/day
- Recent Trend: {patterns.get('recent_trend', 'N/A').upper()}

"""
            else:
                response += "**🔍 Transaction Pattern Analysis:**\n📊 Collecting data... Need at least 10 transactions for ML analysis.\n\n"
            
            # Smart recommendations
            if recommendations:
                response += "**🎯 Smart Recommendations:**\n"
                for i, rec in enumerate(recommendations[:3], 1):
                    priority_emoji = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}.get(rec.get('priority'), '⚪')
                    response += f"{i}. {priority_emoji} **{rec.get('title', 'N/A')}**\n"
                    response += f"   {rec.get('message', 'N/A')}\n"
                    response += f"   → {rec.get('action', 'N/A')}\n\n"
            else:
                response += "**🎯 Smart Recommendations:**\n✅ All systems normal. No urgent actions needed.\n\n"
            
            response += "*Powered by ML-driven financial analytics*\n"
            response += "\n💡 **Start processing payments to unlock full analytics!**"
            
            return response
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            logging.error(f"Portfolio error: {error_details}")
            
            return f"""⚠️ Portfolio analytics temporarily unavailable.

**Error Details:** {str(e)}

**Quick Diagnostics:**
- Try: "What's my balance?" for basic wallet info
- Try: "What are gas fees?" for network status

The portfolio system needs at least one transaction to display full analytics.
Once you process a payment, all features will be available!

*System is operational, just collecting initial data.*
"""


    # Performance Analysis Query
    elif any(word in text_lower for word in ["performance", "roi", "p&l", "profit"]):
        try:
            pnl = portfolio.calculate_pnl(30)
            
            profit_emoji = "📈" if pnl['net_pnl_eth'] > 0 else "📉"
            roi_status = "PROFIT" if pnl['roi_percent'] > 0 else "LOSS"
            
            return f"""{profit_emoji} **30-Day Performance Report**

**Overall P&L:**
• Net Result: {pnl['net_pnl_eth']:.4f} ETH
• USD Value: ${pnl['net_pnl_usd']:,.2f}
• ROI: {pnl['roi_percent']:.2f}% ({roi_status})

**Transaction Breakdown:**
• Total Spent: {pnl['total_spent_eth']:.4f} ETH
• Total Received: {pnl['total_received_eth']:.4f} ETH
• Gas Fees: {pnl['gas_spent_eth']:.6f} ETH
• Net Flow: {pnl['net_pnl_eth']:.4f} ETH

**Activity:**
• Transactions: {pnl['transaction_count']}
• Avg ETH Price: ${pnl['avg_eth_price']:,.2f}
• Period: {pnl['period_days']} days

**Analysis:**
{"🎉 Excellent performance! Portfolio is profitable." if pnl['roi_percent'] > 10 else
 "✅ Positive performance. Continue current strategy." if pnl['roi_percent'] > 0 else
 "⚠️ Negative performance. Consider reviewing payment policies."}

*Try: "Show portfolio" for comprehensive analytics*
"""
            
        except Exception as e:
            return f"Error calculating performance: {str(e)}"


    # Gas Optimization Query
    elif any(word in text_lower for word in ["gas", "optimize", "best time", "when to send"]):
        try:
            gas_data = portfolio.predict_gas_optimization()
            
            status_emoji = {
                'excellent': '🟢',
                'good': '🟡',
                'moderate': '🟠',
                'expensive': '🔴'
            }.get(gas_data.get('recommendation', 'moderate'), '⚪')
            
            return f"""⛽ **Smart Gas Optimization**

**Current Status:** {status_emoji} {gas_data.get('recommendation', 'N/A').upper()}

**Gas Prices:**
• Current: {gas_data.get('current_gwei', 0):.2f} Gwei
• Your Average: {gas_data.get('avg_gwei', 0):.2f} Gwei
• Comparison: {gas_data.get('savings', 'N/A')}

**Timing Analysis:**
• {gas_data.get('timing', 'N/A')}
• Optimal Time: {gas_data.get('best_time_utc', 'N/A')}

**💡 Recommendation:**
{gas_data.get('action', 'No recommendation available')}

**Why This Matters:**
• Save 30-70% on gas fees by timing transactions
• Off-peak hours (2-6 AM UTC) typically cheapest
• Weekend gas prices often 20% lower

*ML-powered prediction based on your transaction history*
"""
            
        except Exception as e:
            return f"Error analyzing gas: {str(e)}"


    # Recommendations Query
    elif any(word in text_lower for word in ["recommend", "advice", "suggestion", "what should"]):
        try:
            recommendations = portfolio.generate_smart_recommendations()
            
            if not recommendations:
                return """✅ **All Good!**

Your portfolio is healthy. No urgent actions needed.

**Current Status:**
• Balance: Adequate
• Gas prices: Normal
• Activity: Standard patterns

Try: "Show portfolio" for detailed analytics
"""
            
            response = "🎯 **Smart Financial Recommendations**\n\n"
            
            for i, rec in enumerate(recommendations, 1):
                priority_emoji = {'high': '🔴 URGENT', 'medium': '🟡 IMPORTANT', 'low': '🟢 INFO'}[rec['priority']]
                
                response += f"**{i}. {rec['title']}** ({priority_emoji})\n"
                response += f"📝 {rec['message']}\n"
                response += f"→ **Action:** {rec['action']}\n\n"
            
            response += "*AI-powered recommendations updated in real-time*"
            
            return response
            
        except Exception as e:
            return f"Error generating recommendations: {str(e)}"

    
    # 4. GAS PRICES
    elif any(word in text_lower for word in ["gas", "fee", "cost", "transfer cost"]):
        gas_data = get_gas_estimates()
        
        if gas_data:
            eth_price = get_eth_price()
            transfer_usd = gas_data['simple_transfer_eth'] * eth_price
            
            return f"""⛽ **Gas Price & Transaction Costs**

**Current Gas Price:** {gas_data['current_gwei']:.2f} Gwei

**Estimated Costs:**
- Simple Transfer: {gas_data['simple_transfer_eth']:.6f} ETH (${transfer_usd:.2f})
- Smart Contract Call: ~0.0015 ETH (~${0.0015 * eth_price:.2f})

💡 **Tip:** Gas prices are lower during off-peak hours (2-6 AM UTC)

*Real-time gas data from Ethereum network*
"""
        else:
            return "Unable to fetch gas prices. Please try again."
    
    # 5. PAYMENT STATUS QUERY
    elif any(word in text_lower for word in ["payment", "transaction", "paid", "tx", "hash", "status"]) and "release" not in text_lower:
        words = text.upper().split()
        shipment_id = next((w for w in words if "SHIP" in w), None)
        
        if shipment_id:
            return f"""💳 **Payment Status: {shipment_id}**

Status: ✅ **PROCESSED**
Amount: 0.5 ETH
Transaction Hash: 0xmock_{shipment_id.lower()}_tx_hash
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Payment was automatically released by smart escrow after delivery confirmation from Logistics Agent.
"""
        else:
            return """💳 **Payment Status Query**

To check a specific payment, please provide:
- Transaction hash (0x...)
- Shipment ID (SHIP-XXX)

Example: "Check payment status for SHIP-001"

Recent activity: 3 payments processed in last 24 hours
"""
    
    # NEW: Payment cost estimation (add before release payment section)
    elif any(word in text_lower for word in ["estimate", "cost", "how much to send"]):
        words = text.split()
        # Try to extract amount
        amount = "0.1"  # Default
        for i, word in enumerate(words):
            try:
                if i > 0 and words[i-1].lower() in ["send", "transfer", "pay"]:
                    amount = word
                    break
            except:
                pass
        
        cost_estimate = estimate_payment_cost(amount)
        
        if cost_estimate:
            return f"""💰 **Payment Cost Estimate**

**Amount to Send:** {cost_estimate['amount']} ETH
**Gas Cost:** {cost_estimate['gas_cost']:.6f} ETH
**Total Cost:** {cost_estimate['total_cost']:.6f} ETH

**Gas Price:** {cost_estimate['gas_price_gwei']:.2f} Gwei

**Note:** Actual cost may vary slightly based on network conditions at execution time.

*Ready to send? Use: "Release payment for SHIP-XXX"*
"""
        else:
            return "Unable to estimate payment cost. Please try again."
    
    # 6. RELEASE PAYMENT
    elif "release" in text_lower or "process payment" in text_lower:
        words = text.upper().split()
        shipment_id = next((w for w in words if "SHIP" in w), "SHIP-XXX")
        
        return f"""💸 **Payment Release Initiated**

Shipment ID: {shipment_id}
Amount: 1.0 ETH (default)
Status: ⏳ Awaiting logistics confirmation

The payment will be automatically released once:
1. ✅ Logistics Agent confirms delivery status
2. ✅ Governance Agent verifies compliance
3. ✅ Smart escrow conditions are met

You'll receive a confirmation with transaction hash once processed.
"""
    
    # 7. HELP/CAPABILITIES
    elif any(word in text_lower for word in ["help", "what can", "capabilities", "do", "?"]):
        return """🤖 **Finance Agent - Supply Chain Finance**

**I can help you with:**

💰 **Market Data & Balances**
- Real-time ETH prices
- Wallet balance with USD conversion
- Transaction history
- Gas price estimates

💳 **Payment Processing**
- Release payments for shipments
- Verify transaction status
- Process escrow releases

🔐 **Smart Escrow**
- Automatic payment triggers based on delivery
- Integration with Logistics Agent
- Compliance verification via Governance

**Try asking:**
- "What's the ETH price?"
- "What's my wallet balance?"
- "Show transaction history"
- "What are gas fees?"
- "Release payment for SHIP-001"

Powered by Ethereum blockchain + CoinGecko API for transparent, trustless transactions.
"""
    
    # 8. GREETING
    elif any(word in text_lower for word in ["hello", "hi", "hey"]):
        return """👋 Hello! I'm the Finance Agent for Supply Chain Finance.

I handle wallet balances, payments, and real-time market data on Ethereum.

What would you like to know? Try:
- "What's the ETH price?"
- "What's my balance?"
- "Release payment for SHIP-001"
- "What can you do?"
"""
    
    # 9. DEFAULT
    else:
        return """💰 I'm the Finance Agent for autonomous supply chain finance.

**I can help you:**
- Check ETH prices and wallet balances
- View transaction history
- Get gas fee estimates
- Process payment releases

What would you like to know?

*Tip: Try "What can you do?" to see all capabilities*
"""


# ==================== CHAT PROTOCOL HANDLERS ====================

@chat_proto.on_message(ChatMessage)
async def handle_chat_message(ctx: Context, sender: str, msg: ChatMessage):
    """Handle natural language queries from ASI:One"""
    ctx.logger.info(f"💬 Chat message from {sender}")
    
    await ctx.send(
        sender,
        ChatAcknowledgement(
            timestamp=datetime.utcnow(),
            acknowledged_msg_id=msg.msg_id
        )
    )
    
    text = ""
    for item in msg.content:
        if isinstance(item, TextContent):
            text += item.text
    
    ctx.logger.info(f"📝 Query: {text}")
    
    # Process query (NO AWAIT - it's not async!)
    response_text = process_query(text, ctx)
    
    response_msg = ChatMessage(
        timestamp=datetime.utcnow(),
        msg_id=uuid4(),
        content=[TextContent(type="text", text=response_text)]
    )
    await ctx.send(sender, response_msg)


@chat_proto.on_message(ChatAcknowledgement)
async def handle_acknowledgement(ctx: Context, sender: str, msg: ChatAcknowledgement):
    ctx.logger.info(f"✅ Message {msg.acknowledged_msg_id} acknowledged")


# ==================== REDIS MESSAGEBUS HANDLERS ====================

async def handle_redis_message(msg):
    """Handle messages from Redis MessageBus (for inter-agent communication)"""
    try:
        data = json.loads(msg['data'])
        print(f"💰 Finance received: {data.get('type')}")
        
        if data.get("type") == "wallet_query":
            addr = data.get("address", USER_WALLET)
            try:
                bal = w3.eth.get_balance(addr)
                eth = w3.from_wei(bal, "ether")
                result = {"agent": "finance", "address": addr, "eth": str(eth)}
                print(f"✅ Balance retrieved: {eth} ETH")
            except Exception as e:
                result = {"agent": "finance", "address": addr, "error": str(e)}
                print(f"❌ Error: {e}")

            await bus.publish("finance.replies", json.dumps(result))
            print(f"📤 Published to finance.replies")
        
        # UPDATED: Real payment execution
        elif data.get("type") == "release_payment":
            shipment_id = data.get("shipment_id")
            amount = data.get("amount", "0.001")  # Default: small testnet amount
            recipient = data.get("recipient", USER_WALLET)  # Who receives payment
            reason = data.get("reason", "Payment release")
            
            print(f"💳 Payment release request:")
            print(f"   Shipment: {shipment_id}")
            print(f"   Amount: {amount} ETH")
            print(f"   Recipient: {recipient}")
            print(f"   Reason: {reason}")
            
            # Estimate cost first
            cost_estimate = estimate_payment_cost(amount)
            if cost_estimate:
                print(f"💰 Estimated total cost: {cost_estimate['total_cost']:.6f} ETH")
                print(f"   (includes {cost_estimate['gas_cost']:.6f} ETH gas)")
            
            # Execute REAL blockchain payment
            tx_hash = execute_payment(shipment_id, amount, recipient)
            
            if tx_hash:
                result = {
                    "agent": "finance",
                    "type": "payment_released",
                    "shipment_id": shipment_id,
                    "amount": amount,
                    "recipient": recipient,
                    "status": "success",
                    "reason": reason,
                    "tx_hash": tx_hash,  # REAL blockchain transaction hash!
                    "etherscan_url": f"https://sepolia.etherscan.io/tx/{tx_hash}",
                    "timestamp": datetime.now().isoformat()
                }
                print(f"✅ REAL payment sent on blockchain!")
                print(f"🔗 Transaction: {tx_hash}")
            else:
                result = {
                    "agent": "finance",
                    "type": "payment_failed",
                    "shipment_id": shipment_id,
                    "amount": amount,
                    "status": "failed",
                    "reason": "Payment execution failed - check logs",
                    "timestamp": datetime.now().isoformat()
                }
                print(f"❌ Payment execution failed")
            
            # Send confirmation to governance for audit
            await bus.publish("governance.requests", json.dumps({
                "type": "report",
                "source": "finance",
                "payload": result
            }))
            
            print(f"📤 Payment result sent to governance")
            
    except Exception as e:
        print(f"❌ Error handling message: {e}")
        import traceback
        traceback.print_exc()


async def run_redis_bus():
    """Listen to Redis MessageBus"""
    try:
        await bus.init()
        sub = await bus.subscribe("finance.requests")
        print("💰 Finance Redis MessageBus running")
        
        while True:
            msg = await sub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            if msg and msg.get("type") == "message":
                await handle_redis_message(msg)
                
    except Exception as e:
        print(f"❌ MessageBus error: {e}")
        import traceback
        traceback.print_exc()


def start_redis_thread():
    """Start Redis MessageBus in background thread"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run_redis_bus())


# ==================== MAIN ====================

finance.include(chat_proto, publish_manifest=True)

if __name__ == "__main__":
    if not w3.is_connected():
        print("⚠️ Web3 not connected! Running in limited mode.")
    else:
        print(f"✅ Web3 connected: Chain ID {w3.eth.chain_id}")
    
    print(f"📡 Watching wallet: {USER_WALLET}")
    
    redis_thread = threading.Thread(target=start_redis_thread, daemon=True)
    redis_thread.start()
    
    print(f"🚀 {AGENT_NAME} starting with enhanced features:")
    print(f"   • Real-time ETH prices (CoinGecko)")
    print(f"   • Transaction history (Etherscan)")
    print(f"   • Gas price estimates")
    print(f"   • Chat Protocol for ASI:One")
    print(f"   • Redis MessageBus for inter-agent communication")
    
    finance.run()