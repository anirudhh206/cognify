"""
Advanced Governance Engine
ML-powered fraud detection, compliance scoring, and dispute resolution
Location: cognify/backend/agents/governance_engine.py
"""
import json
import os
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
from sklearn.ensemble import IsolationForest
import logging


class FraudDetectionEngine:
    """
    ML-powered fraud detection using Isolation Forest
    Detects anomalous transactions and suspicious patterns
    """
    
    def __init__(self):
        self.model = IsolationForest(
            contamination=0.1,  # Expect 10% anomalies
            random_state=42,
            n_estimators=100
        )
        self.transaction_history = []
        self.trained = False
        
        # Fraud indicators
        self.fraud_patterns = {
            'rapid_succession': {'threshold': 5, 'window_minutes': 10},
            'unusual_amount': {'multiplier': 3.0},
            'off_hours': {'start': 1, 'end': 5},  # 1 AM - 5 AM
            'new_recipient': {'days': 7},
            'geographical_anomaly': {'enabled': True}
        }
        
        logging.info("🛡️ Fraud Detection Engine initialized")
    
    def analyze_transaction(self, transaction_data):
        """
        Comprehensive fraud analysis using ML + rule-based detection
        """
        fraud_score = 0.0
        risk_indicators = []
        
        # Extract features
        amount = float(transaction_data.get('amount', 0))
        timestamp = transaction_data.get('timestamp', datetime.now().isoformat())
        recipient = transaction_data.get('recipient', 'unknown')
        source = transaction_data.get('source', 'unknown')
        
        tx_time = datetime.fromisoformat(timestamp) if isinstance(timestamp, str) else timestamp
        
        # 1. AMOUNT-BASED DETECTION
        if self.transaction_history:
            amounts = [tx.get('amount', 0) for tx in self.transaction_history]
            avg_amount = np.mean(amounts)
            std_amount = np.std(amounts)
            
            # Unusual amount (> 3 std deviations)
            if amount > avg_amount + (self.fraud_patterns['unusual_amount']['multiplier'] * std_amount):
                fraud_score += 30
                risk_indicators.append({
                    'type': 'unusual_amount',
                    'severity': 'HIGH',
                    'details': f"Amount {amount:.4f} ETH is {((amount - avg_amount) / avg_amount * 100):.0f}% above average",
                    'score_impact': 30
                })
        
        # 2. RAPID SUCCESSION DETECTION
        recent_txs = self._get_recent_transactions(minutes=self.fraud_patterns['rapid_succession']['window_minutes'])
        if len(recent_txs) >= self.fraud_patterns['rapid_succession']['threshold']:
            fraud_score += 25
            risk_indicators.append({
                'type': 'rapid_succession',
                'severity': 'HIGH',
                'details': f"{len(recent_txs)} transactions in {self.fraud_patterns['rapid_succession']['window_minutes']} minutes",
                'score_impact': 25
            })
        
        # 3. OFF-HOURS ACTIVITY
        hour = tx_time.hour
        if self.fraud_patterns['off_hours']['start'] <= hour <= self.fraud_patterns['off_hours']['end']:
            fraud_score += 15
            risk_indicators.append({
                'type': 'off_hours',
                'severity': 'MEDIUM',
                'details': f"Transaction at {hour}:00 (suspicious hours)",
                'score_impact': 15
            })
        
        # 4. NEW RECIPIENT CHECK
        recipient_history = [tx.get('recipient') for tx in self.transaction_history]
        if recipient not in recipient_history:
            fraud_score += 20
            risk_indicators.append({
                'type': 'new_recipient',
                'severity': 'MEDIUM',
                'details': f"First transaction to {recipient[:10]}...",
                'score_impact': 20
            })
        
        # 5. VELOCITY CHECK (transactions per hour)
        recent_hour = self._get_recent_transactions(minutes=60)
        if len(recent_hour) > 10:
            fraud_score += 20
            risk_indicators.append({
                'type': 'high_velocity',
                'severity': 'HIGH',
                'details': f"{len(recent_hour)} transactions in last hour",
                'score_impact': 20
            })
        
        # 6. ML-BASED ANOMALY DETECTION
        if self.trained and len(self.transaction_history) > 50:
            features = self._extract_ml_features(transaction_data)
            ml_score = self.model.decision_function([features])[0]
            
            # Anomaly detected (negative score = anomaly)
            if ml_score < -0.5:
                fraud_score += 25
                risk_indicators.append({
                    'type': 'ml_anomaly',
                    'severity': 'HIGH',
                    'details': f"ML model detected unusual pattern (score: {ml_score:.2f})",
                    'score_impact': 25
                })
        
        # Calculate final fraud probability
        fraud_probability = min(100, fraud_score)
        
        # Determine risk level
        risk_level = self._categorize_fraud_risk(fraud_probability)
        
        # Store transaction
        self._record_transaction(transaction_data)
        
        return {
            'fraud_probability': round(fraud_probability, 1),
            'risk_level': risk_level,
            'risk_indicators': risk_indicators,
            'recommendation': self._get_fraud_recommendation(fraud_probability),
            'should_block': fraud_probability >= 70,
            'requires_review': fraud_probability >= 50,
            'timestamp': datetime.now().isoformat()
        }
    
    def _extract_ml_features(self, transaction_data):
        """Extract features for ML model"""
        amount = float(transaction_data.get('amount', 0))
        timestamp = transaction_data.get('timestamp', datetime.now().isoformat())
        
        if isinstance(timestamp, str):
            tx_time = datetime.fromisoformat(timestamp)
        else:
            tx_time = timestamp
        
        # Feature vector
        features = [
            amount,  # Transaction amount
            tx_time.hour,  # Hour of day
            tx_time.weekday(),  # Day of week
            len(self._get_recent_transactions(60)),  # Transactions in last hour
            len(self._get_recent_transactions(1440)),  # Transactions in last 24h
        ]
        
        return features
    
    def _get_recent_transactions(self, minutes):
        """Get transactions within time window"""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        return [
            tx for tx in self.transaction_history
            if datetime.fromisoformat(tx['timestamp']) > cutoff
        ]
    
    def _record_transaction(self, transaction_data):
        """Record transaction for pattern analysis"""
        tx_record = {
            'amount': float(transaction_data.get('amount', 0)),
            'timestamp': transaction_data.get('timestamp', datetime.now().isoformat()),
            'recipient': transaction_data.get('recipient', 'unknown'),
            'source': transaction_data.get('source', 'unknown')
        }
        
        self.transaction_history.append(tx_record)
        
        # Train ML model periodically
        if len(self.transaction_history) >= 50 and len(self.transaction_history) % 10 == 0:
            self._train_ml_model()
    
    def _train_ml_model(self):
        """Train Isolation Forest on transaction history"""
        try:
            features = [self._extract_ml_features(tx) for tx in self.transaction_history]
            self.model.fit(features)
            self.trained = True
            logging.info(f"🤖 Fraud detection model trained on {len(features)} transactions")
        except Exception as e:
            logging.error(f"ML training error: {e}")
    
    def _categorize_fraud_risk(self, probability):
        """Categorize fraud risk level"""
        if probability < 20:
            return {'level': 'MINIMAL', 'emoji': '🟢', 'color': 'green'}
        elif probability < 40:
            return {'level': 'LOW', 'emoji': '🟡', 'color': 'yellow'}
        elif probability < 60:
            return {'level': 'MODERATE', 'emoji': '🟠', 'color': 'orange'}
        elif probability < 80:
            return {'level': 'HIGH', 'emoji': '🔴', 'color': 'red'}
        else:
            return {'level': 'CRITICAL', 'emoji': '🚨', 'color': 'darkred'}
    
    def _get_fraud_recommendation(self, probability):
        """Get action recommendation based on fraud score"""
        if probability < 20:
            return "APPROVE - Low fraud risk, proceed with transaction"
        elif probability < 40:
            return "APPROVE with MONITORING - Watch for patterns"
        elif probability < 60:
            return "REVIEW REQUIRED - Additional verification needed"
        elif probability < 80:
            return "HOLD - High fraud risk, manual review mandatory"
        else:
            return "BLOCK - Critical fraud indicators, reject transaction"


class ComplianceScorecardSystem:
    """
    Track and score compliance for all parties
    Maintains reputation scores and violation history
    """
    
    def __init__(self):
        self.data_file = "backend/agents/data/compliance_scores.json"
        os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
        
        self.scorecard_data = self._load_data()
        logging.info("📋 Compliance Scorecard System initialized")
    
    def _load_data(self):
        """Load compliance data"""
        if os.path.exists(self.data_file):
            with open(self.data_file, 'r') as f:
                return json.load(f)
        return {'parties': {}, 'violations': []}
    
    def _save_data(self):
        """Save compliance data"""
        with open(self.data_file, 'w') as f:
            json.dump(self.scorecard_data, f, indent=2)
    
    def get_party_score(self, party_id, party_type='carrier'):
        """Get compliance score for a party"""
        if party_id not in self.scorecard_data['parties']:
            # Initialize new party
            self.scorecard_data['parties'][party_id] = {
                'type': party_type,
                'score': 100,  # Start at perfect score
                'transactions': 0,
                'violations': 0,
                'on_time_deliveries': 0,
                'late_deliveries': 0,
                'disputes': 0,
                'created_at': datetime.now().isoformat()
            }
            self._save_data()
        
        party = self.scorecard_data['parties'][party_id]
        
        # Calculate dynamic metrics
        on_time_rate = (party['on_time_deliveries'] / party['transactions'] * 100) if party['transactions'] > 0 else 100
        dispute_rate = (party['disputes'] / party['transactions'] * 100) if party['transactions'] > 0 else 0
        
        return {
            'party_id': party_id,
            'party_type': party['type'],
            'score': party['score'],
            'rating': self._score_to_rating(party['score']),
            'transactions': party['transactions'],
            'violations': party['violations'],
            'on_time_rate': round(on_time_rate, 1),
            'dispute_rate': round(dispute_rate, 1),
            'status': self._get_status(party['score']),
            'created_at': party['created_at']
        }
    
    def record_transaction(self, party_id, on_time=True, compliant=True):
        """Record transaction outcome"""
        if party_id not in self.scorecard_data['parties']:
            self.get_party_score(party_id)  # Initialize
        
        party = self.scorecard_data['parties'][party_id]
        party['transactions'] += 1
        
        if on_time:
            party['on_time_deliveries'] += 1
            party['score'] = min(100, party['score'] + 0.5)  # Small bonus
        else:
            party['late_deliveries'] += 1
            party['score'] = max(0, party['score'] - 2)  # Penalty
        
        if not compliant:
            party['violations'] += 1
            party['score'] = max(0, party['score'] - 5)  # Larger penalty
            
            self.scorecard_data['violations'].append({
                'party_id': party_id,
                'timestamp': datetime.now().isoformat(),
                'type': 'late_delivery' if not on_time else 'compliance_violation'
            })
        
        self._save_data()
    
    def _score_to_rating(self, score):
        """Convert numeric score to letter rating"""
        if score >= 95:
            return 'A+'
        elif score >= 90:
            return 'A'
        elif score >= 85:
            return 'B+'
        elif score >= 80:
            return 'B'
        elif score >= 75:
            return 'C+'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'
    
    def _get_status(self, score):
        """Get status based on score"""
        if score >= 90:
            return {'label': 'EXCELLENT', 'emoji': '🌟', 'color': 'green'}
        elif score >= 80:
            return {'label': 'GOOD', 'emoji': '✅', 'color': 'lightgreen'}
        elif score >= 70:
            return {'label': 'ACCEPTABLE', 'emoji': '⚠️', 'color': 'yellow'}
        elif score >= 60:
            return {'label': 'POOR', 'emoji': '❌', 'color': 'orange'}
        else:
            return {'label': 'CRITICAL', 'emoji': '🚨', 'color': 'red'}
    
    def get_system_overview(self):
        """Get overview of all parties"""
        parties = self.scorecard_data['parties']
        
        if not parties:
            return {
                'total_parties': 0,
                'avg_score': 0,
                'top_performers': [],
                'at_risk': []
            }
        
        scores = [p['score'] for p in parties.values()]
        
        # Sort parties by score
        sorted_parties = sorted(
            parties.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )
        
        top_performers = [
            {
                'id': pid[:10] + '...' if len(pid) > 10 else pid,
                'score': data['score'],
                'rating': self._score_to_rating(data['score']),
                'transactions': data['transactions']
            }
            for pid, data in sorted_parties[:5]
        ]
        
        at_risk = [
            {
                'id': pid[:10] + '...' if len(pid) > 10 else pid,
                'score': data['score'],
                'rating': self._score_to_rating(data['score']),
                'violations': data['violations']
            }
            for pid, data in sorted_parties
            if data['score'] < 70
        ]
        
        return {
            'total_parties': len(parties),
            'avg_score': round(np.mean(scores), 1),
            'top_performers': top_performers,
            'at_risk': at_risk,
            'total_violations': len(self.scorecard_data['violations'])
        }


class DisputeResolutionEngine:
    """
    AI-powered automated dispute resolution
    Analyzes evidence and proposes fair settlements
    """
    
    def __init__(self):
        self.disputes = []
        logging.info("⚖️ Dispute Resolution Engine initialized")
    
    def analyze_dispute(self, dispute_data):
        """
        Analyze dispute and propose resolution
        """
        shipment_id = dispute_data.get('shipment_id', 'UNKNOWN')
        complaint_type = dispute_data.get('type', 'delivery_delay')
        evidence = dispute_data.get('evidence', {})
        
        # Analyze evidence
        analysis = {
            'dispute_id': f"DISP-{len(self.disputes)+1:04d}",
            'shipment_id': shipment_id,
            'type': complaint_type,
            'severity': self._assess_severity(evidence),
            'liability': self._determine_liability(evidence),
            'proposed_resolution': self._propose_resolution(complaint_type, evidence),
            'confidence': self._calculate_confidence(evidence),
            'timestamp': datetime.now().isoformat()
        }
        
        self.disputes.append(analysis)
        
        return analysis
    
    def _assess_severity(self, evidence):
        """Assess dispute severity"""
        delay_hours = evidence.get('delay_hours', 0)
        customer_impact = evidence.get('customer_impact', 'low')
        
        if delay_hours > 48 or customer_impact == 'high':
            return {'level': 'HIGH', 'emoji': '🔴'}
        elif delay_hours > 24 or customer_impact == 'medium':
            return {'level': 'MEDIUM', 'emoji': '🟡'}
        else:
            return {'level': 'LOW', 'emoji': '🟢'}
    
    def _determine_liability(self, evidence):
        """Determine who's at fault"""
        delay_hours = evidence.get('delay_hours', 0)
        weather = evidence.get('weather', 'clear')
        carrier_fault = evidence.get('carrier_fault', False)
        
        if carrier_fault:
            return {'party': 'carrier', 'percentage': 100}
        elif weather in ['heavy_rain', 'snow'] and delay_hours < 36:
            return {'party': 'force_majeure', 'percentage': 0}
        elif delay_hours > 48:
            return {'party': 'carrier', 'percentage': 75}
        else:
            return {'party': 'shared', 'percentage': 50}
    
    def _propose_resolution(self, complaint_type, evidence):
        """AI proposes fair resolution"""
        delay_hours = evidence.get('delay_hours', 0)
        payment_amount = evidence.get('payment_amount', 1.0)
        
        if complaint_type == 'delivery_delay':
            if delay_hours > 48:
                return {
                    'action': 'full_refund',
                    'amount': payment_amount,
                    'reason': 'Severe delay (>48h) - full refund warranted'
                }
            elif delay_hours > 24:
                return {
                    'action': 'partial_refund',
                    'amount': payment_amount * 0.5,
                    'reason': 'Moderate delay (24-48h) - 50% refund'
                }
            else:
                return {
                    'action': 'discount_next',
                    'amount': payment_amount * 0.2,
                    'reason': 'Minor delay - 20% credit on next shipment'
                }
        
        return {
            'action': 'manual_review',
            'amount': 0,
            'reason': 'Complex case requires human judgment'
        }
    
    def _calculate_confidence(self, evidence):
        """Calculate AI confidence in resolution"""
        evidence_count = len(evidence)
        
        if evidence_count >= 5:
            return round(0.85 + (evidence_count - 5) * 0.02, 2)
        else:
            return round(0.60 + evidence_count * 0.05, 2)