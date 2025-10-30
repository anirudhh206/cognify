"""
MeTTa Knowledge Base - Real Symbolic AI Reasoning
Location: cognify/backend/agents/metta/knowledge_base.py
"""
import json
import os
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class MeTTaKnowledgeBase:
    """
    Production-grade MeTTa-inspired knowledge base
    Implements symbolic reasoning for compliance and governance
    """
    
    def __init__(self):
        self.kb_file = "backend/agents/metta/kb_rules.json"
        os.makedirs(os.path.dirname(self.kb_file), exist_ok=True)
        
        # Initialize or load knowledge base
        self.knowledge_graph = self._load_or_initialize_kb()
        
        # Inference engine
        self.inference_cache = {}
        
        logging.info("🧠 MeTTa Knowledge Base initialized")
        logging.info(f"   • Rules loaded: {self._count_rules()}")
        logging.info(f"   • Jurisdictions: {len(self.knowledge_graph.get('jurisdictions', {}))}")
    
    def _load_or_initialize_kb(self) -> Dict:
        """Load existing KB or create comprehensive initial knowledge base"""
        if os.path.exists(self.kb_file):
            try:
                with open(self.kb_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        
        # Create comprehensive initial knowledge base
        kb = {
            "meta": {
                "version": "1.0",
                "created": datetime.now().isoformat(),
                "type": "supply_chain_compliance"
            },
            
            # Core ontology - defines concepts
            "ontology": {
                "entities": ["payment", "shipment", "carrier", "buyer", "seller"],
                "attributes": ["amount", "delay", "jurisdiction", "status", "compliance_score"],
                "relations": ["requires", "triggers", "validates", "conflicts_with"]
            },
            
            # Jurisdictional rules
            "jurisdictions": {
                "US": {
                    "name": "United States",
                    "currency_rules": {
                        "kyc_threshold": 10000,
                        "reporting_threshold": 10000,
                        "sanctions_check": True
                    },
                    "delivery_rules": {
                        "max_acceptable_delay_hours": 48,
                        "requires_proof_of_delivery": True,
                        "dispute_window_days": 30
                    },
                    "data_privacy": {
                        "requires_consent": True,
                        "data_retention_days": 2555  # 7 years
                    }
                },
                "EU": {
                    "name": "European Union",
                    "currency_rules": {
                        "kyc_threshold": 10000,
                        "reporting_threshold": 10000,
                        "gdpr_compliance": True
                    },
                    "delivery_rules": {
                        "max_acceptable_delay_hours": 72,
                        "requires_proof_of_delivery": True,
                        "dispute_window_days": 14
                    },
                    "data_privacy": {
                        "requires_explicit_consent": True,
                        "right_to_deletion": True,
                        "data_retention_days": 2555
                    }
                },
                "UK": {
                    "name": "United Kingdom",
                    "currency_rules": {
                        "kyc_threshold": 8500,  # ~10k USD in GBP
                        "reporting_threshold": 10000,
                        "sanctions_check": True
                    },
                    "delivery_rules": {
                        "max_acceptable_delay_hours": 48,
                        "requires_proof_of_delivery": True,
                        "consumer_rights_act": True
                    }
                },
                "ASIA_PACIFIC": {
                    "name": "Asia Pacific",
                    "currency_rules": {
                        "kyc_threshold": 10000,
                        "cross_border_restrictions": True
                    },
                    "delivery_rules": {
                        "max_acceptable_delay_hours": 96,
                        "customs_clearance_required": True
                    }
                }
            },
            
            # Compliance rules in logical format
            "rules": {
                "payment_approval": {
                    "rule_id": "PAY_001",
                    "description": "Payment approval requires delivery confirmation and compliance check",
                    "logic": {
                        "conditions": [
                            {"type": "delivery_confirmed", "required": True},
                            {"type": "delay_within_threshold", "required": True},
                            {"type": "compliance_verified", "required": True}
                        ],
                        "action": "approve_payment",
                        "priority": "high"
                    }
                },
                
                "kyc_requirement": {
                    "rule_id": "KYC_001",
                    "description": "KYC required for high-value transactions",
                    "logic": {
                        "conditions": [
                            {"type": "amount_exceeds", "threshold": "jurisdiction.kyc_threshold"}
                        ],
                        "action": "require_kyc_verification",
                        "priority": "critical"
                    }
                },
                
                "sanctions_screening": {
                    "rule_id": "SANC_001",
                    "description": "All parties must be screened against sanctions lists",
                    "logic": {
                        "conditions": [
                            {"type": "jurisdiction_requires", "attribute": "sanctions_check"}
                        ],
                        "action": "screen_all_parties",
                        "priority": "critical"
                    }
                },
                
                "delivery_delay_tolerance": {
                    "rule_id": "DEL_001",
                    "description": "Payment withholding based on delivery delay",
                    "logic": {
                        "conditions": [
                            {"type": "delay_exceeds", "threshold": "jurisdiction.max_acceptable_delay_hours"}
                        ],
                        "action": "withhold_payment",
                        "priority": "high"
                    }
                },
                
                "weather_exception": {
                    "rule_id": "EXCEP_001",
                    "description": "Weather-related delays may be excused",
                    "logic": {
                        "conditions": [
                            {"type": "severe_weather", "required": True},
                            {"type": "delay_justifiable", "required": True}
                        ],
                        "action": "excuse_delay",
                        "modifier": "contextual",
                        "priority": "medium"
                    }
                },
                
                "dispute_resolution": {
                    "rule_id": "DISP_001",
                    "description": "Disputes must be filed within jurisdiction window",
                    "logic": {
                        "conditions": [
                            {"type": "within_dispute_window", "threshold": "jurisdiction.dispute_window_days"}
                        ],
                        "action": "accept_dispute",
                        "priority": "medium"
                    }
                },
                
                "multi_sig_requirement": {
                    "rule_id": "AUTH_001",
                    "description": "High-value transactions require multi-signature",
                    "logic": {
                        "conditions": [
                            {"type": "amount_exceeds", "threshold": 50000}
                        ],
                        "action": "require_multi_signature",
                        "priority": "high"
                    }
                },
                
                "data_retention": {
                    "rule_id": "DATA_001",
                    "description": "Transaction data must be retained per jurisdiction",
                    "logic": {
                        "conditions": [
                            {"type": "transaction_completed", "required": True}
                        ],
                        "action": "retain_data",
                        "duration": "jurisdiction.data_retention_days",
                        "priority": "high"
                    }
                }
            },
            
            # Inference patterns - how to combine rules
            "inference_patterns": {
                "payment_decision": {
                    "pattern": "IF (delivery_confirmed AND delay_acceptable AND compliance_passed) THEN approve",
                    "required_rules": ["PAY_001", "DEL_001"],
                    "optional_rules": ["EXCEP_001"],
                    "confidence_threshold": 0.8
                },
                
                "risk_assessment": {
                    "pattern": "COMBINE(kyc_check, sanctions_check, delay_analysis) => risk_score",
                    "required_rules": ["KYC_001", "SANC_001", "DEL_001"],
                    "aggregation": "weighted_sum"
                }
            },
            
            # Contextual knowledge - situational awareness
            "context": {
                "force_majeure_events": ["hurricane", "earthquake", "war", "pandemic", "severe_weather"],
                "acceptable_excuses": ["weather", "natural_disaster", "customs_delay", "port_strike"],
                "red_flags": ["sanctioned_country", "high_risk_wallet", "unusual_pattern", "rapid_succession"]
            }
        }
        
        # Save initial KB
        self._save_kb(kb)
        return kb
    
    def _save_kb(self, kb: Dict):
        """Persist knowledge base"""
        try:
            with open(self.kb_file, 'w') as f:
                json.dump(kb, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save KB: {e}")
    
    def _count_rules(self) -> int:
        """Count total rules in KB"""
        return len(self.knowledge_graph.get('rules', {}))
    
    def query_rule(self, rule_id: str) -> Optional[Dict]:
        """Retrieve specific rule by ID"""
        return self.knowledge_graph.get('rules', {}).get(rule_id)
    
    def query_jurisdiction(self, jurisdiction: str) -> Optional[Dict]:
        """Get rules for specific jurisdiction"""
        return self.knowledge_graph.get('jurisdictions', {}).get(jurisdiction.upper())
    
    def infer_compliance(self, transaction_data: Dict) -> Dict:
        """
        Main inference engine - determines compliance using symbolic reasoning
        This is where MeTTa magic happens!
        """
        # Extract transaction context
        amount = transaction_data.get('amount', 0)
        delay_hours = transaction_data.get('delay_hours', 0)
        jurisdiction = transaction_data.get('jurisdiction', 'US')
        weather = transaction_data.get('weather', 'clear')
        delivery_confirmed = transaction_data.get('delivery_confirmed', False)
        
        # Get jurisdiction rules
        juris_rules = self.query_jurisdiction(jurisdiction) or self.query_jurisdiction('US')
        
        # Apply inference patterns
        applicable_rules = []
        violations = []
        exceptions = []
        
        # Rule 1: KYC Check
        kyc_threshold = juris_rules.get('currency_rules', {}).get('kyc_threshold', 10000)
        if amount > kyc_threshold:
            applicable_rules.append({
                'rule_id': 'KYC_001',
                'triggered': True,
                'reason': f'Amount ${amount} exceeds KYC threshold ${kyc_threshold}'
            })
        
        # Rule 2: Delivery Delay Check
        max_delay = juris_rules.get('delivery_rules', {}).get('max_acceptable_delay_hours', 48)
        if delay_hours > max_delay:
            # Check for weather exception
            if weather in self.knowledge_graph['context']['force_majeure_events']:
                exceptions.append({
                    'rule_id': 'EXCEP_001',
                    'type': 'weather_exception',
                    'reason': f'Delay ({delay_hours}h) excusable due to {weather}'
                })
            else:
                violations.append({
                    'rule_id': 'DEL_001',
                    'violated': True,
                    'reason': f'Delay {delay_hours}h exceeds maximum {max_delay}h'
                })
        
        # Rule 3: Delivery Confirmation
        if not delivery_confirmed:
            violations.append({
                'rule_id': 'PAY_001',
                'violated': True,
                'reason': 'Payment requires delivery confirmation'
            })
        
        # Rule 4: Multi-signature requirement
        if amount > 50000:
            applicable_rules.append({
                'rule_id': 'AUTH_001',
                'triggered': True,
                'reason': 'High-value transaction requires multi-signature'
            })
        
        # Inference: Combine rules to make decision
        is_compliant = len(violations) == 0 or len(exceptions) >= len(violations)
        
        # Calculate confidence based on rule matching
        confidence = self._calculate_inference_confidence(
            applicable_rules, violations, exceptions
        )
        
        # Generate explanation (symbolic reasoning chain)
        explanation = self._generate_reasoning_chain(
            applicable_rules, violations, exceptions, is_compliant
        )
        
        return {
            'is_compliant': is_compliant,
            'confidence': confidence,
            'jurisdiction': jurisdiction,
            'applicable_rules': applicable_rules,
            'violations': violations,
            'exceptions': exceptions,
            'reasoning_chain': explanation,
            'recommendation': 'APPROVE' if is_compliant else 'REVIEW_REQUIRED',
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_inference_confidence(self, rules: List, violations: List, 
                                         exceptions: List) -> float:
        """Calculate confidence in inference"""
        if len(violations) == 0:
            return 0.95  # High confidence if no violations
        elif len(exceptions) >= len(violations):
            return 0.75  # Medium-high if exceptions cover violations
        elif len(rules) > 0:
            return 0.60  # Medium if rules apply but violations exist
        else:
            return 0.40  # Low confidence
    
    def _generate_reasoning_chain(self, rules: List, violations: List,
                                   exceptions: List, is_compliant: bool) -> str:
        """
        Generate human-readable reasoning chain
        This shows HOW the symbolic AI reached its conclusion
        """
        chain = []
        
        chain.append("SYMBOLIC REASONING CHAIN:")
        chain.append("")
        
        # Step 1: Rules evaluated
        if rules:
            chain.append("1. RULES EVALUATED:")
            for rule in rules:
                chain.append(f"   → {rule['rule_id']}: {rule['reason']}")
            chain.append("")
        
        # Step 2: Violations found
        if violations:
            chain.append("2. VIOLATIONS DETECTED:")
            for v in violations:
                chain.append(f"   ✗ {v['rule_id']}: {v['reason']}")
            chain.append("")
        
        # Step 3: Exceptions applied
        if exceptions:
            chain.append("3. EXCEPTIONS APPLIED:")
            for e in exceptions:
                chain.append(f"   ✓ {e['rule_id']}: {e['reason']}")
            chain.append("")
        
        # Step 4: Inference
        chain.append("4. INFERENCE:")
        if is_compliant:
            if len(violations) == 0:
                chain.append("   → All rules satisfied")
            else:
                chain.append("   → Violations excused by valid exceptions")
            chain.append("   → CONCLUSION: COMPLIANT")
        else:
            chain.append("   → Unexcused violations present")
            chain.append("   → CONCLUSION: NON-COMPLIANT")
        
        return "\n".join(chain)
    
    def semantic_query(self, natural_language_query: str) -> Dict:
        """
        Convert natural language to knowledge graph query
        Examples:
        - "What are payment rules for US?"
        - "Is delay of 50 hours acceptable?"
        - "What exceptions exist for weather delays?"
        """
        query_lower = natural_language_query.lower()
        
        # Pattern matching for semantic understanding
        if "payment" in query_lower and any(j in query_lower for j in ["us", "eu", "uk"]):
            # Query about payment rules
            jurisdiction = next((j.upper() for j in ["us", "eu", "uk"] if j in query_lower), "US")
            juris_data = self.query_jurisdiction(jurisdiction)
            
            return {
                'query_type': 'jurisdiction_rules',
                'jurisdiction': jurisdiction,
                'rules': juris_data,
                'relevant_rule_ids': ['PAY_001', 'KYC_001']
            }
        
        elif "delay" in query_lower or "acceptable" in query_lower:
            # Query about delay thresholds
            hours_mentioned = None
            words = query_lower.split()
            for i, word in enumerate(words):
                if word.isdigit():
                    hours_mentioned = int(word)
                    break
            
            return {
                'query_type': 'delay_rules',
                'hours_mentioned': hours_mentioned,
                'relevant_rule': self.query_rule('DEL_001'),
                'exceptions': self.query_rule('EXCEP_001')
            }
        
        elif "exception" in query_lower or "weather" in query_lower:
            # Query about exceptions
            return {
                'query_type': 'exceptions',
                'force_majeure': self.knowledge_graph['context']['force_majeure_events'],
                'acceptable_excuses': self.knowledge_graph['context']['acceptable_excuses'],
                'relevant_rule': self.query_rule('EXCEP_001')
            }
        
        else:
            # General query - return ontology
            return {
                'query_type': 'general',
                'ontology': self.knowledge_graph['ontology'],
                'available_queries': [
                    'jurisdiction rules',
                    'payment requirements',
                    'delay thresholds',
                    'exceptions and force majeure'
                ]
            }
    
    def add_rule(self, rule_id: str, rule_data: Dict):
        """Dynamically add new rule to knowledge base"""
        self.knowledge_graph['rules'][rule_id] = rule_data
        self._save_kb(self.knowledge_graph)
        logging.info(f"✅ Rule {rule_id} added to knowledge base")
    
    def update_jurisdiction(self, jurisdiction: str, updates: Dict):
        """Update jurisdiction rules"""
        if jurisdiction.upper() in self.knowledge_graph['jurisdictions']:
            self.knowledge_graph['jurisdictions'][jurisdiction.upper()].update(updates)
            self._save_kb(self.knowledge_graph)
            logging.info(f"✅ Jurisdiction {jurisdiction} updated")


# ==================== HELPER FUNCTIONS ====================

def format_metta_response(inference_result: Dict) -> str:
    """Format MeTTa inference result for beautiful display"""
    
    is_compliant = inference_result['is_compliant']
    confidence = inference_result['confidence']
    jurisdiction = inference_result['jurisdiction']
    
    # Header
    response = f"""🧠 **MeTTa Symbolic AI Inference**

**Jurisdiction:** {jurisdiction}
**Compliance Status:** {"✅ COMPLIANT" if is_compliant else "❌ NON-COMPLIANT"}
**Confidence:** {confidence*100:.1f}%
**Recommendation:** {inference_result['recommendation']}

---

"""
    
    # Applicable Rules
    if inference_result['applicable_rules']:
        response += "**📋 Rules Applied:**\n"
        for rule in inference_result['applicable_rules']:
            response += f"• **{rule['rule_id']}**: {rule['reason']}\n"
        response += "\n"
    
    # Violations
    if inference_result['violations']:
        response += "**⚠️ Violations Detected:**\n"
        for v in inference_result['violations']:
            response += f"• **{v['rule_id']}**: {v['reason']}\n"
        response += "\n"
    
    # Exceptions
    if inference_result['exceptions']:
        response += "**✓ Exceptions Applied:**\n"
        for e in inference_result['exceptions']:
            response += f"• **{e['rule_id']}**: {e['reason']}\n"
        response += "\n"
    
    # Reasoning Chain
    response += "---\n\n"
    response += "**🔍 Symbolic Reasoning Chain:**\n\n"
    response += f"```\n{inference_result['reasoning_chain']}\n```\n\n"
    
    response += "*Powered by MeTTa Symbolic AI Knowledge Graph*"
    
    return response