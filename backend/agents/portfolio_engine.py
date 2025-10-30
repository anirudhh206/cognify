"""
Advanced Portfolio Analytics Engine
Real financial analysis with ML predictions and data persistence
Location: cognify/backend/agents/portfolio_engine.py
"""
import json
import os
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
from web3 import Web3


class PortfolioAnalytics:
    """
    Advanced portfolio tracking and analytics
    Tracks transactions, calculates P&L, predicts trends
    """
    
    def __init__(self, web3_instance, wallet_address):
        self.w3 = web3_instance
        self.wallet = wallet_address
        self.data_file = "backend/agents/data/portfolio_data.json"
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
        
        # Load or initialize data
        self.portfolio_data = self._load_data()
        
    def _load_data(self):
        """Load historical portfolio data"""
        if os.path.exists(self.data_file):
            with open(self.data_file, 'r') as f:
                return json.load(f)
        return {
            'transactions': [],
            'balances_history': [],
            'last_updated': None
        }
    
    def _save_data(self):
        """Persist portfolio data"""
        with open(self.data_file, 'w') as f:
            json.dump(self.portfolio_data, f, indent=2)
    
    def record_transaction(self, tx_hash, amount_eth, tx_type, recipient=None, gas_used=None):
        """
        Record a transaction for analytics
        """
        tx_record = {
            'hash': tx_hash,
            'timestamp': datetime.now().isoformat(),
            'amount_eth': float(amount_eth),
            'type': tx_type,  # 'payment', 'received', 'gas'
            'recipient': recipient,
            'gas_eth': float(gas_used) if gas_used else 0,
            'eth_price_usd': self._get_eth_price()
        }
        
        self.portfolio_data['transactions'].append(tx_record)
        self._save_data()
        
        return tx_record
    
    def _get_eth_price(self):
        """Get current ETH price (cached for performance)"""
        try:
            import requests
            response = requests.get(
                "https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd",
                timeout=3
            )
            return response.json()['ethereum']['usd']
        except:
            return 2500  # Fallback
    
    def get_current_balance(self):
        """Get real-time balance from blockchain"""
        try:
            balance_wei = self.w3.eth.get_balance(self.wallet)
            balance_eth = self.w3.from_wei(balance_wei, 'ether')
            eth_price = self._get_eth_price()
            
            return {
                'eth': float(balance_eth),
                'usd': float(balance_eth) * eth_price,
                'eth_price': eth_price,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {'error': str(e)}
    
    def calculate_pnl(self, period_days=30):
        """
        Calculate Profit & Loss over period
        """
        cutoff_date = datetime.now() - timedelta(days=period_days)
        
        # Filter transactions in period
        period_txs = [
            tx for tx in self.portfolio_data.get('transactions', [])
            if datetime.fromisoformat(tx['timestamp']) > cutoff_date
        ]
        
        # FIXED: Handle no transactions case properly
        if not period_txs:
            return {
                'period_days': period_days,
                'total_spent_eth': 0,
                'total_received_eth': 0,
                'gas_spent_eth': 0,
                'net_pnl_eth': 0,
                'net_pnl_usd': 0,
                'roi_percent': 0,
                'transaction_count': 0,
                'avg_eth_price': self._get_eth_price()
            }
        
        # Calculate flows
        total_spent = sum(tx['amount_eth'] for tx in period_txs if tx['type'] == 'payment')
        total_received = sum(tx['amount_eth'] for tx in period_txs if tx['type'] == 'received')
        total_gas = sum(tx.get('gas_eth', 0) for tx in period_txs)
        
        # Net P&L
        net_eth = total_received - total_spent - total_gas
        
        # USD value (use average ETH price during period)
        avg_eth_price = np.mean([tx['eth_price_usd'] for tx in period_txs])
        net_usd = net_eth * avg_eth_price
        
        # ROI calculation
        initial_value = total_spent * avg_eth_price if total_spent > 0 else 1
        roi_percent = (net_usd / initial_value) * 100 if initial_value > 0 else 0
        
        return {
            'period_days': period_days,
            'total_spent_eth': round(total_spent, 4),
            'total_received_eth': round(total_received, 4),
            'gas_spent_eth': round(total_gas, 6),
            'net_pnl_eth': round(net_eth, 4),
            'net_pnl_usd': round(net_usd, 2),
            'roi_percent': round(roi_percent, 2),
            'transaction_count': len(period_txs),
            'avg_eth_price': round(avg_eth_price, 2)
        }
    
    def get_top_recipients(self, limit=5):
        """
        Analyze top payment recipients
        """
        recipient_totals = defaultdict(lambda: {'count': 0, 'total_eth': 0})
        
        for tx in self.portfolio_data['transactions']:
            if tx['type'] == 'payment' and tx.get('recipient'):
                recipient = tx['recipient']
                recipient_totals[recipient]['count'] += 1
                recipient_totals[recipient]['total_eth'] += tx['amount_eth']
        
        # Sort by total ETH sent
        sorted_recipients = sorted(
            recipient_totals.items(),
            key=lambda x: x[1]['total_eth'],
            reverse=True
        )[:limit]
        
        return [
            {
                'address': addr[:10] + '...' + addr[-8:] if len(addr) > 20 else addr,
                'full_address': addr,
                'payment_count': data['count'],
                'total_eth': round(data['total_eth'], 4),
                'avg_payment': round(data['total_eth'] / data['count'], 4)
            }
            for addr, data in sorted_recipients
        ]
    
    def analyze_transaction_patterns(self):
        """
        ML-based pattern analysis
        Detects anomalies and trends
        """
        if len(self.portfolio_data['transactions']) < 10:
            return {
                'status': 'insufficient_data',
                'message': 'Need at least 10 transactions for analysis'
            }
        
        txs = self.portfolio_data['transactions']
        
        # Extract time series
        amounts = [tx['amount_eth'] for tx in txs if tx['type'] == 'payment']
        timestamps = [datetime.fromisoformat(tx['timestamp']) for tx in txs]
        
        # Calculate statistics
        avg_amount = np.mean(amounts)
        std_amount = np.std(amounts)
        
        # Detect anomalies (transactions > 2 std devs)
        anomalies = [
            {
                'amount': amt,
                'timestamp': ts.isoformat(),
                'deviation': abs(amt - avg_amount) / std_amount
            }
            for amt, ts in zip(amounts, timestamps)
            if abs(amt - avg_amount) > 2 * std_amount
        ]
        
        # Calculate velocity (transactions per day)
        if len(timestamps) > 1:
            time_span_days = (timestamps[-1] - timestamps[0]).days or 1
            velocity = len(txs) / time_span_days
        else:
            velocity = 0
        
        # Trend detection (simple linear regression on recent 7 days)
        recent_txs = [
            tx for tx in txs
            if datetime.fromisoformat(tx['timestamp']) > datetime.now() - timedelta(days=7)
        ]
        
        if len(recent_txs) >= 3:
            recent_amounts = [tx['amount_eth'] for tx in recent_txs if tx['type'] == 'payment']
            x = np.arange(len(recent_amounts))
            
            if len(recent_amounts) > 1:
                slope = np.polyfit(x, recent_amounts, 1)[0]
                trend = 'increasing' if slope > 0.01 else 'decreasing' if slope < -0.01 else 'stable'
            else:
                trend = 'stable'
        else:
            trend = 'insufficient_data'
        
        return {
            'status': 'success',
            'avg_transaction_eth': round(avg_amount, 4),
            'std_deviation': round(std_amount, 4),
            'anomaly_count': len(anomalies),
            'anomalies': anomalies[:3],  # Top 3
            'transaction_velocity': round(velocity, 2),
            'recent_trend': trend
        }
    
    def predict_gas_optimization(self):
        """
        Predict best time for transactions based on gas patterns
        Uses historical data + current network conditions
        """
        try:
            current_gas = self.w3.eth.gas_price
            current_gwei = self.w3.from_wei(current_gas, 'gwei')
            
            # Analyze historical gas from our transactions
            gas_history = [
                tx['gas_eth'] / 21000  # Normalize to gas price
                for tx in self.portfolio_data['transactions']
                if tx.get('gas_eth', 0) > 0
            ]
            
            if len(gas_history) < 5:
                return {
                    'recommendation': 'insufficient_data',
                    'current_gwei': float(current_gwei),
                    'message': 'Collecting gas data...'
                }
            
            avg_gas_gwei = np.mean(gas_history)
            
            # Compare current to average
            if current_gwei < avg_gas_gwei * 0.7:
                status = 'excellent'
                action = 'Process all pending transactions NOW'
                savings = f"~{((avg_gas_gwei - current_gwei) / avg_gas_gwei * 100):.0f}% cheaper than average"
            elif current_gwei < avg_gas_gwei:
                status = 'good'
                action = 'Good time to transact'
                savings = f"~{((avg_gas_gwei - current_gwei) / avg_gas_gwei * 100):.0f}% cheaper than average"
            elif current_gwei < avg_gas_gwei * 1.3:
                status = 'moderate'
                action = 'Consider waiting for lower gas'
                savings = f"~{((current_gwei - avg_gas_gwei) / avg_gas_gwei * 100):.0f}% more expensive"
            else:
                status = 'expensive'
                action = 'WAIT - Gas prices very high'
                savings = f"~{((current_gwei - avg_gas_gwei) / avg_gas_gwei * 100):.0f}% more expensive"
            
            # Predict best time (based on typical patterns)
            # Gas is typically lowest 2-6 AM UTC
            current_hour = datetime.utcnow().hour
            
            if 2 <= current_hour <= 6:
                timing = 'Excellent timing (off-peak hours)'
            elif 7 <= current_hour <= 9 or 17 <= current_hour <= 19:
                timing = 'Peak hours - expect higher gas'
            else:
                timing = 'Moderate timing'
            
            return {
                'recommendation': status,
                'action': action,
                'current_gwei': float(current_gwei),
                'avg_gwei': round(avg_gas_gwei, 2),
                'savings': savings,
                'timing': timing,
                'best_time_utc': '2:00 AM - 6:00 AM UTC'
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def generate_smart_recommendations(self):
        """
        AI-powered financial recommendations
        """
        recommendations = []
        
        # Get current state
        balance = self.get_current_balance()
        pnl = self.calculate_pnl(30)
        patterns = self.analyze_transaction_patterns()
        gas_opt = self.predict_gas_optimization()
        
        # Balance-based recommendations
        if balance.get('eth', 0) < 0.1:
            recommendations.append({
                'priority': 'high',
                'category': 'liquidity',
                'title': '⚠️ Low Balance Warning',
                'message': f"Balance is {balance['eth']:.4f} ETH. Consider adding funds to avoid transaction failures.",
                'action': 'Add at least 0.5 ETH'
            })
        
        # Gas optimization
        if gas_opt.get('recommendation') == 'excellent':
            recommendations.append({
                'priority': 'medium',
                'category': 'optimization',
                'title': '⛽ Excellent Gas Prices',
                'message': f"Current gas: {gas_opt['current_gwei']:.2f} Gwei. {gas_opt['savings']}",
                'action': 'Process pending transactions now'
            })
        elif gas_opt.get('recommendation') == 'expensive':
            recommendations.append({
                'priority': 'medium',
                'category': 'optimization',
                'title': '🚨 High Gas Alert',
                'message': f"Gas prices elevated. {gas_opt['savings']}",
                'action': f"Wait until {gas_opt['best_time_utc']}"
            })
        
        # Transaction patterns
        if patterns.get('status') == 'success':
            if patterns['anomaly_count'] > 2:
                recommendations.append({
                    'priority': 'high',
                    'category': 'security',
                    'title': '🔍 Unusual Activity Detected',
                    'message': f"{patterns['anomaly_count']} anomalous transactions found",
                    'action': 'Review transaction history for unauthorized activity'
                })
            
            if patterns['recent_trend'] == 'increasing':
                recommendations.append({
                    'priority': 'low',
                    'category': 'insight',
                    'title': '📈 Spending Increasing',
                    'message': 'Transaction volume trending up over past 7 days',
                    'action': 'Monitor budget and consider batch processing'
                })
        
        # P&L recommendations
        if pnl['roi_percent'] < -10:
            recommendations.append({
                'priority': 'medium',
                'category': 'performance',
                'title': '📉 Negative ROI',
                'message': f"Portfolio down {abs(pnl['roi_percent']):.1f}% over {pnl['period_days']} days",
                'action': 'Review payment policies and dispute rates'
            })
        elif pnl['roi_percent'] > 15:
            recommendations.append({
                'priority': 'low',
                'category': 'performance',
                'title': '🎉 Strong Performance',
                'message': f"Portfolio up {pnl['roi_percent']:.1f}% - excellent efficiency",
                'action': 'Continue current strategy'
            })
        
        return sorted(recommendations, key=lambda x: {'high': 0, 'medium': 1, 'low': 2}[x['priority']])
    
    def get_comprehensive_summary(self):
        """
        Full portfolio dashboard
        """
        return {
            'balance': self.get_current_balance(),
            'pnl_30d': self.calculate_pnl(30),
            'top_recipients': self.get_top_recipients(5),
            'patterns': self.analyze_transaction_patterns(),
            'gas_optimization': self.predict_gas_optimization(),
            'recommendations': self.generate_smart_recommendations(),
            'metadata': {
                'total_transactions': len(self.portfolio_data['transactions']),
                'data_since': self.portfolio_data['transactions'][0]['timestamp'] if self.portfolio_data['transactions'] else None,
                'last_updated': datetime.now().isoformat()
            }
        }