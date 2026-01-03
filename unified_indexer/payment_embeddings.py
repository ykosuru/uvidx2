"""
Payment Domain Embeddings

Structured semantic dimensions specifically designed for payment systems.
Instead of one dimension per keyword, this creates meaningful categorical
dimensions that capture the semantic structure of payment processing.

Dimension Categories:
1. Message Types (MT, ISO 20022, etc.)
2. Payment Networks/Rails
3. Compliance & Regulatory
4. Processing Stages
5. Transaction Parties
6. Data Elements
7. Error/Exception Types
8. Business Capabilities
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
from enum import Enum


class PaymentDimension(Enum):
    """Semantic dimensions for payment systems"""
    
    # Message Format Dimensions (0-9)
    MSG_MT_CUSTOMER = 0          # MT 1xx - Customer payments
    MSG_MT_INSTITUTION = 1       # MT 2xx - Institution transfers
    MSG_MT_TREASURY = 2          # MT 3xx - Treasury markets
    MSG_MT_COLLECTION = 3        # MT 4xx - Collections
    MSG_MT_SECURITIES = 4        # MT 5xx - Securities
    MSG_MT_PRECIOUS = 5          # MT 6xx - Precious metals
    MSG_MT_DOCUMENTARY = 6       # MT 7xx - Documentary credits
    MSG_MT_TRAVELLERS = 7        # MT 8xx - Travellers cheques
    MSG_MT_STATUS = 8            # MT 9xx - Cash management/status
    MSG_ISO20022 = 9             # ISO 20022 messages
    
    # Payment Network Dimensions (10-19)
    NET_SWIFT = 10               # SWIFT network
    NET_FEDWIRE = 11             # Federal Reserve Wire
    NET_CHIPS = 12               # CHIPS
    NET_ACH = 13                 # ACH/NACHA
    NET_SEPA = 14                # SEPA (Europe)
    NET_TARGET2 = 15             # TARGET2
    NET_RTGS = 16                # Real-Time Gross Settlement (generic)
    NET_CORRESPONDENT = 17       # Correspondent banking
    NET_DOMESTIC = 18            # Domestic wires
    NET_CROSS_BORDER = 19        # Cross-border/International
    
    # Compliance Dimensions (20-29)
    COMP_OFAC = 20               # OFAC sanctions
    COMP_AML = 21                # Anti-Money Laundering
    COMP_KYC = 22                # Know Your Customer
    COMP_BSA = 23                # Bank Secrecy Act
    COMP_FATF = 24               # FATF guidelines
    COMP_PEP = 25                # Politically Exposed Persons
    COMP_CTR = 26                # Currency Transaction Reports
    COMP_SAR = 27                # Suspicious Activity Reports
    COMP_SANCTIONS = 28          # General sanctions
    COMP_WATCHLIST = 29          # Watchlist screening
    
    # Processing Stage Dimensions (30-39)
    STAGE_INITIATION = 30        # Payment initiation
    STAGE_VALIDATION = 31        # Validation/verification
    STAGE_SCREENING = 32         # Compliance screening
    STAGE_ENRICHMENT = 33        # Data enrichment
    STAGE_ROUTING = 34           # Payment routing
    STAGE_EXECUTION = 35         # Execution/transmission
    STAGE_SETTLEMENT = 36        # Settlement
    STAGE_RECONCILIATION = 37    # Reconciliation
    STAGE_REPORTING = 38         # Reporting
    STAGE_ARCHIVAL = 39          # Archival/retention
    
    # Transaction Party Dimensions (40-49)
    PARTY_ORIGINATOR = 40        # Ordering party/originator
    PARTY_BENEFICIARY = 41       # Beneficiary
    PARTY_ORDERING_INST = 42     # Ordering institution
    PARTY_BENEFICIARY_INST = 43  # Beneficiary institution
    PARTY_INTERMEDIARY = 44      # Intermediary/correspondent
    PARTY_ACCOUNT_WITH = 45      # Account with institution
    PARTY_SENDER = 46            # Sender
    PARTY_RECEIVER = 47          # Receiver
    PARTY_AGENT = 48             # Agent
    PARTY_DEBTOR_CREDITOR = 49   # Debtor/Creditor
    
    # Data Element Dimensions (50-59)
    DATA_AMOUNT = 50             # Amount/currency
    DATA_DATE = 51               # Value date, execution date
    DATA_REFERENCE = 52          # References (UETR, TRN, etc.)
    DATA_ACCOUNT = 53            # Account numbers
    DATA_BIC = 54                # BIC/SWIFT codes
    DATA_IBAN = 55               # IBAN
    DATA_ABA = 56                # ABA/routing numbers
    DATA_REMITTANCE = 57         # Remittance info
    DATA_CHARGES = 58            # Charges/fees
    DATA_EXCHANGE = 59           # FX/exchange rate
    
    # Error/Exception Dimensions (60-69)
    ERR_VALIDATION = 60          # Validation errors
    ERR_COMPLIANCE = 61          # Compliance failures
    ERR_ROUTING = 62             # Routing errors
    ERR_INSUFFICIENT = 63        # Insufficient funds
    ERR_DUPLICATE = 64           # Duplicate detection
    ERR_TIMEOUT = 65             # Timeout/SLA breach
    ERR_FORMAT = 66              # Format/parsing errors
    ERR_REJECT = 67              # Rejections
    ERR_RETURN = 68              # Returns
    ERR_REPAIR = 69              # Repair queue
    
    # Business Capability Dimensions (70-79)
    CAP_HIGH_VALUE = 70          # High-value payments
    CAP_LOW_VALUE = 71           # Low-value/batch
    CAP_URGENT = 72              # Urgent/priority
    CAP_SCHEDULED = 73           # Scheduled/future-dated
    CAP_RECURRING = 74           # Recurring payments
    CAP_INQUIRY = 75             # Inquiry/investigation
    CAP_AMENDMENT = 76           # Amendments/modifications
    CAP_CANCELLATION = 77        # Cancellations
    CAP_CONFIRMATION = 78        # Confirmations/acknowledgments
    CAP_NOTIFICATION = 79        # Notifications/alerts


# Total dimensions
N_PAYMENT_DIMENSIONS = 80


@dataclass
class PaymentTermMapping:
    """Mapping from terms to payment dimensions with weights"""
    term: str
    dimensions: List[Tuple[PaymentDimension, float]]  # (dimension, weight)


# Comprehensive term-to-dimension mapping
PAYMENT_TERM_MAPPINGS: List[PaymentTermMapping] = [
    # MT Message Types
    PaymentTermMapping("MT-103", [
        (PaymentDimension.MSG_MT_CUSTOMER, 1.0),
        (PaymentDimension.NET_SWIFT, 0.8),
        (PaymentDimension.CAP_HIGH_VALUE, 0.6),
        (PaymentDimension.PARTY_ORIGINATOR, 0.5),
        (PaymentDimension.PARTY_BENEFICIARY, 0.5),
    ]),
    PaymentTermMapping("MT103", [
        (PaymentDimension.MSG_MT_CUSTOMER, 1.0),
        (PaymentDimension.NET_SWIFT, 0.8),
    ]),
    PaymentTermMapping("MT-202", [
        (PaymentDimension.MSG_MT_INSTITUTION, 1.0),
        (PaymentDimension.NET_SWIFT, 0.8),
        (PaymentDimension.NET_CORRESPONDENT, 0.7),
        (PaymentDimension.PARTY_INTERMEDIARY, 0.6),
    ]),
    PaymentTermMapping("MT202", [
        (PaymentDimension.MSG_MT_INSTITUTION, 1.0),
        (PaymentDimension.NET_SWIFT, 0.8),
    ]),
    PaymentTermMapping("MT-202COV", [
        (PaymentDimension.MSG_MT_INSTITUTION, 1.0),
        (PaymentDimension.MSG_MT_CUSTOMER, 0.5),
        (PaymentDimension.NET_SWIFT, 0.8),
        (PaymentDimension.NET_CORRESPONDENT, 0.7),
    ]),
    PaymentTermMapping("MT-199", [
        (PaymentDimension.MSG_MT_CUSTOMER, 0.7),
        (PaymentDimension.CAP_INQUIRY, 0.9),
        (PaymentDimension.NET_SWIFT, 0.8),
    ]),
    PaymentTermMapping("MT-299", [
        (PaymentDimension.MSG_MT_INSTITUTION, 0.7),
        (PaymentDimension.CAP_INQUIRY, 0.9),
        (PaymentDimension.NET_SWIFT, 0.8),
    ]),
    PaymentTermMapping("MT-900", [
        (PaymentDimension.MSG_MT_STATUS, 1.0),
        (PaymentDimension.CAP_CONFIRMATION, 0.8),
        (PaymentDimension.STAGE_RECONCILIATION, 0.6),
    ]),
    PaymentTermMapping("MT-910", [
        (PaymentDimension.MSG_MT_STATUS, 1.0),
        (PaymentDimension.CAP_CONFIRMATION, 0.8),
        (PaymentDimension.STAGE_RECONCILIATION, 0.6),
    ]),
    PaymentTermMapping("MT-940", [
        (PaymentDimension.MSG_MT_STATUS, 1.0),
        (PaymentDimension.STAGE_RECONCILIATION, 0.9),
        (PaymentDimension.STAGE_REPORTING, 0.7),
    ]),
    PaymentTermMapping("MT-942", [
        (PaymentDimension.MSG_MT_STATUS, 1.0),
        (PaymentDimension.STAGE_RECONCILIATION, 0.9),
        (PaymentDimension.CAP_NOTIFICATION, 0.7),
    ]),
    
    # ISO 20022 Messages
    PaymentTermMapping("pacs.008", [
        (PaymentDimension.MSG_ISO20022, 1.0),
        (PaymentDimension.MSG_MT_CUSTOMER, 0.6),
        (PaymentDimension.CAP_HIGH_VALUE, 0.5),
    ]),
    PaymentTermMapping("pacs.009", [
        (PaymentDimension.MSG_ISO20022, 1.0),
        (PaymentDimension.MSG_MT_INSTITUTION, 0.6),
        (PaymentDimension.NET_CORRESPONDENT, 0.5),
    ]),
    PaymentTermMapping("pacs.004", [
        (PaymentDimension.MSG_ISO20022, 1.0),
        (PaymentDimension.ERR_RETURN, 1.0),
        (PaymentDimension.ERR_REJECT, 0.7),
    ]),
    PaymentTermMapping("pacs.002", [
        (PaymentDimension.MSG_ISO20022, 1.0),
        (PaymentDimension.CAP_CONFIRMATION, 0.9),
        (PaymentDimension.MSG_MT_STATUS, 0.5),
    ]),
    PaymentTermMapping("pain.001", [
        (PaymentDimension.MSG_ISO20022, 1.0),
        (PaymentDimension.STAGE_INITIATION, 0.9),
        (PaymentDimension.PARTY_ORIGINATOR, 0.6),
    ]),
    PaymentTermMapping("pain.002", [
        (PaymentDimension.MSG_ISO20022, 1.0),
        (PaymentDimension.CAP_CONFIRMATION, 0.8),
        (PaymentDimension.ERR_REJECT, 0.5),
    ]),
    PaymentTermMapping("camt.053", [
        (PaymentDimension.MSG_ISO20022, 1.0),
        (PaymentDimension.STAGE_RECONCILIATION, 0.9),
        (PaymentDimension.STAGE_REPORTING, 0.8),
    ]),
    PaymentTermMapping("camt.054", [
        (PaymentDimension.MSG_ISO20022, 1.0),
        (PaymentDimension.CAP_NOTIFICATION, 0.9),
        (PaymentDimension.STAGE_RECONCILIATION, 0.6),
    ]),
    PaymentTermMapping("ISO 20022", [
        (PaymentDimension.MSG_ISO20022, 1.0),
    ]),
    PaymentTermMapping("ISO20022", [
        (PaymentDimension.MSG_ISO20022, 1.0),
    ]),
    
    # Payment Networks
    PaymentTermMapping("SWIFT", [
        (PaymentDimension.NET_SWIFT, 1.0),
        (PaymentDimension.NET_CROSS_BORDER, 0.6),
    ]),
    PaymentTermMapping("Fedwire", [
        (PaymentDimension.NET_FEDWIRE, 1.0),
        (PaymentDimension.NET_RTGS, 0.8),
        (PaymentDimension.NET_DOMESTIC, 0.7),
        (PaymentDimension.CAP_HIGH_VALUE, 0.6),
    ]),
    PaymentTermMapping("CHIPS", [
        (PaymentDimension.NET_CHIPS, 1.0),
        (PaymentDimension.NET_DOMESTIC, 0.6),
        (PaymentDimension.CAP_HIGH_VALUE, 0.7),
    ]),
    PaymentTermMapping("ACH", [
        (PaymentDimension.NET_ACH, 1.0),
        (PaymentDimension.NET_DOMESTIC, 0.8),
        (PaymentDimension.CAP_LOW_VALUE, 0.7),
    ]),
    PaymentTermMapping("NACHA", [
        (PaymentDimension.NET_ACH, 1.0),
        (PaymentDimension.NET_DOMESTIC, 0.7),
    ]),
    PaymentTermMapping("SEPA", [
        (PaymentDimension.NET_SEPA, 1.0),
        (PaymentDimension.NET_CROSS_BORDER, 0.6),
    ]),
    PaymentTermMapping("TARGET2", [
        (PaymentDimension.NET_TARGET2, 1.0),
        (PaymentDimension.NET_RTGS, 0.9),
        (PaymentDimension.CAP_HIGH_VALUE, 0.7),
    ]),
    PaymentTermMapping("RTGS", [
        (PaymentDimension.NET_RTGS, 1.0),
        (PaymentDimension.CAP_HIGH_VALUE, 0.8),
        (PaymentDimension.STAGE_SETTLEMENT, 0.7),
    ]),
    PaymentTermMapping("correspondent", [
        (PaymentDimension.NET_CORRESPONDENT, 1.0),
        (PaymentDimension.PARTY_INTERMEDIARY, 0.8),
    ]),
    PaymentTermMapping("nostro", [
        (PaymentDimension.NET_CORRESPONDENT, 0.9),
        (PaymentDimension.DATA_ACCOUNT, 0.8),
        (PaymentDimension.STAGE_RECONCILIATION, 0.6),
    ]),
    PaymentTermMapping("vostro", [
        (PaymentDimension.NET_CORRESPONDENT, 0.9),
        (PaymentDimension.DATA_ACCOUNT, 0.8),
        (PaymentDimension.STAGE_RECONCILIATION, 0.6),
    ]),
    
    # Compliance Terms
    PaymentTermMapping("OFAC", [
        (PaymentDimension.COMP_OFAC, 1.0),
        (PaymentDimension.COMP_SANCTIONS, 0.9),
        (PaymentDimension.STAGE_SCREENING, 0.8),
        (PaymentDimension.COMP_WATCHLIST, 0.7),
    ]),
    PaymentTermMapping("sanctions", [
        (PaymentDimension.COMP_SANCTIONS, 1.0),
        (PaymentDimension.COMP_OFAC, 0.7),
        (PaymentDimension.STAGE_SCREENING, 0.8),
    ]),
    PaymentTermMapping("AML", [
        (PaymentDimension.COMP_AML, 1.0),
        (PaymentDimension.STAGE_SCREENING, 0.7),
        (PaymentDimension.COMP_BSA, 0.6),
    ]),
    PaymentTermMapping("anti-money laundering", [
        (PaymentDimension.COMP_AML, 1.0),
        (PaymentDimension.STAGE_SCREENING, 0.7),
    ]),
    PaymentTermMapping("KYC", [
        (PaymentDimension.COMP_KYC, 1.0),
        (PaymentDimension.STAGE_VALIDATION, 0.6),
        (PaymentDimension.PARTY_ORIGINATOR, 0.5),
    ]),
    PaymentTermMapping("know your customer", [
        (PaymentDimension.COMP_KYC, 1.0),
    ]),
    PaymentTermMapping("BSA", [
        (PaymentDimension.COMP_BSA, 1.0),
        (PaymentDimension.COMP_AML, 0.8),
        (PaymentDimension.COMP_CTR, 0.6),
    ]),
    PaymentTermMapping("Bank Secrecy Act", [
        (PaymentDimension.COMP_BSA, 1.0),
        (PaymentDimension.COMP_AML, 0.7),
    ]),
    PaymentTermMapping("FATF", [
        (PaymentDimension.COMP_FATF, 1.0),
        (PaymentDimension.COMP_AML, 0.7),
    ]),
    PaymentTermMapping("PEP", [
        (PaymentDimension.COMP_PEP, 1.0),
        (PaymentDimension.COMP_WATCHLIST, 0.8),
        (PaymentDimension.STAGE_SCREENING, 0.7),
    ]),
    PaymentTermMapping("politically exposed", [
        (PaymentDimension.COMP_PEP, 1.0),
        (PaymentDimension.COMP_WATCHLIST, 0.7),
    ]),
    PaymentTermMapping("CTR", [
        (PaymentDimension.COMP_CTR, 1.0),
        (PaymentDimension.COMP_BSA, 0.8),
        (PaymentDimension.STAGE_REPORTING, 0.7),
    ]),
    PaymentTermMapping("currency transaction report", [
        (PaymentDimension.COMP_CTR, 1.0),
        (PaymentDimension.STAGE_REPORTING, 0.8),
    ]),
    PaymentTermMapping("SAR", [
        (PaymentDimension.COMP_SAR, 1.0),
        (PaymentDimension.COMP_AML, 0.8),
        (PaymentDimension.STAGE_REPORTING, 0.7),
    ]),
    PaymentTermMapping("suspicious activity", [
        (PaymentDimension.COMP_SAR, 1.0),
        (PaymentDimension.COMP_AML, 0.8),
    ]),
    PaymentTermMapping("watchlist", [
        (PaymentDimension.COMP_WATCHLIST, 1.0),
        (PaymentDimension.STAGE_SCREENING, 0.8),
    ]),
    PaymentTermMapping("screening", [
        (PaymentDimension.STAGE_SCREENING, 1.0),
        (PaymentDimension.COMP_WATCHLIST, 0.6),
    ]),
    PaymentTermMapping("SDN", [
        (PaymentDimension.COMP_OFAC, 0.9),
        (PaymentDimension.COMP_SANCTIONS, 1.0),
        (PaymentDimension.COMP_WATCHLIST, 0.8),
    ]),
    PaymentTermMapping("blocked", [
        (PaymentDimension.COMP_SANCTIONS, 0.8),
        (PaymentDimension.ERR_COMPLIANCE, 0.7),
    ]),
    
    # Processing Stages
    PaymentTermMapping("initiation", [
        (PaymentDimension.STAGE_INITIATION, 1.0),
    ]),
    PaymentTermMapping("validation", [
        (PaymentDimension.STAGE_VALIDATION, 1.0),
        (PaymentDimension.ERR_VALIDATION, 0.4),
    ]),
    PaymentTermMapping("enrichment", [
        (PaymentDimension.STAGE_ENRICHMENT, 1.0),
    ]),
    PaymentTermMapping("routing", [
        (PaymentDimension.STAGE_ROUTING, 1.0),
        (PaymentDimension.ERR_ROUTING, 0.4),
    ]),
    PaymentTermMapping("execution", [
        (PaymentDimension.STAGE_EXECUTION, 1.0),
    ]),
    PaymentTermMapping("settlement", [
        (PaymentDimension.STAGE_SETTLEMENT, 1.0),
        (PaymentDimension.NET_RTGS, 0.4),
    ]),
    PaymentTermMapping("reconciliation", [
        (PaymentDimension.STAGE_RECONCILIATION, 1.0),
    ]),
    PaymentTermMapping("reporting", [
        (PaymentDimension.STAGE_REPORTING, 1.0),
    ]),
    
    # Transaction Parties
    PaymentTermMapping("originator", [
        (PaymentDimension.PARTY_ORIGINATOR, 1.0),
        (PaymentDimension.PARTY_DEBTOR_CREDITOR, 0.5),
    ]),
    PaymentTermMapping("ordering party", [
        (PaymentDimension.PARTY_ORIGINATOR, 1.0),
    ]),
    PaymentTermMapping("beneficiary", [
        (PaymentDimension.PARTY_BENEFICIARY, 1.0),
        (PaymentDimension.PARTY_DEBTOR_CREDITOR, 0.5),
    ]),
    PaymentTermMapping("ordering institution", [
        (PaymentDimension.PARTY_ORDERING_INST, 1.0),
    ]),
    PaymentTermMapping("beneficiary institution", [
        (PaymentDimension.PARTY_BENEFICIARY_INST, 1.0),
    ]),
    PaymentTermMapping("intermediary", [
        (PaymentDimension.PARTY_INTERMEDIARY, 1.0),
        (PaymentDimension.NET_CORRESPONDENT, 0.6),
    ]),
    PaymentTermMapping("account with", [
        (PaymentDimension.PARTY_ACCOUNT_WITH, 1.0),
    ]),
    PaymentTermMapping("sender", [
        (PaymentDimension.PARTY_SENDER, 1.0),
    ]),
    PaymentTermMapping("receiver", [
        (PaymentDimension.PARTY_RECEIVER, 1.0),
    ]),
    PaymentTermMapping("debtor", [
        (PaymentDimension.PARTY_DEBTOR_CREDITOR, 1.0),
        (PaymentDimension.PARTY_ORIGINATOR, 0.7),
    ]),
    PaymentTermMapping("creditor", [
        (PaymentDimension.PARTY_DEBTOR_CREDITOR, 1.0),
        (PaymentDimension.PARTY_BENEFICIARY, 0.7),
    ]),
    
    # Data Elements
    PaymentTermMapping("amount", [
        (PaymentDimension.DATA_AMOUNT, 1.0),
    ]),
    PaymentTermMapping("currency", [
        (PaymentDimension.DATA_AMOUNT, 0.8),
        (PaymentDimension.DATA_EXCHANGE, 0.5),
    ]),
    PaymentTermMapping("value date", [
        (PaymentDimension.DATA_DATE, 1.0),
        (PaymentDimension.STAGE_SETTLEMENT, 0.5),
    ]),
    PaymentTermMapping("execution date", [
        (PaymentDimension.DATA_DATE, 1.0),
        (PaymentDimension.STAGE_EXECUTION, 0.5),
    ]),
    PaymentTermMapping("UETR", [
        (PaymentDimension.DATA_REFERENCE, 1.0),
        (PaymentDimension.NET_SWIFT, 0.6),
    ]),
    PaymentTermMapping("TRN", [
        (PaymentDimension.DATA_REFERENCE, 1.0),
    ]),
    PaymentTermMapping("reference", [
        (PaymentDimension.DATA_REFERENCE, 1.0),
    ]),
    PaymentTermMapping("account number", [
        (PaymentDimension.DATA_ACCOUNT, 1.0),
    ]),
    PaymentTermMapping("BIC", [
        (PaymentDimension.DATA_BIC, 1.0),
        (PaymentDimension.NET_SWIFT, 0.6),
    ]),
    PaymentTermMapping("SWIFT code", [
        (PaymentDimension.DATA_BIC, 1.0),
        (PaymentDimension.NET_SWIFT, 0.7),
    ]),
    PaymentTermMapping("IBAN", [
        (PaymentDimension.DATA_IBAN, 1.0),
        (PaymentDimension.DATA_ACCOUNT, 0.7),
    ]),
    PaymentTermMapping("ABA", [
        (PaymentDimension.DATA_ABA, 1.0),
        (PaymentDimension.NET_DOMESTIC, 0.6),
    ]),
    PaymentTermMapping("routing number", [
        (PaymentDimension.DATA_ABA, 1.0),
        (PaymentDimension.STAGE_ROUTING, 0.5),
    ]),
    PaymentTermMapping("remittance", [
        (PaymentDimension.DATA_REMITTANCE, 1.0),
    ]),
    PaymentTermMapping("charges", [
        (PaymentDimension.DATA_CHARGES, 1.0),
    ]),
    PaymentTermMapping("fees", [
        (PaymentDimension.DATA_CHARGES, 1.0),
    ]),
    PaymentTermMapping("OUR", [
        (PaymentDimension.DATA_CHARGES, 1.0),
        (PaymentDimension.PARTY_ORIGINATOR, 0.5),
    ]),
    PaymentTermMapping("BEN", [
        (PaymentDimension.DATA_CHARGES, 1.0),
        (PaymentDimension.PARTY_BENEFICIARY, 0.5),
    ]),
    PaymentTermMapping("SHA", [
        (PaymentDimension.DATA_CHARGES, 1.0),
    ]),
    PaymentTermMapping("exchange rate", [
        (PaymentDimension.DATA_EXCHANGE, 1.0),
    ]),
    PaymentTermMapping("FX", [
        (PaymentDimension.DATA_EXCHANGE, 1.0),
    ]),
    
    # Errors and Exceptions
    PaymentTermMapping("validation error", [
        (PaymentDimension.ERR_VALIDATION, 1.0),
        (PaymentDimension.STAGE_VALIDATION, 0.6),
    ]),
    PaymentTermMapping("compliance failure", [
        (PaymentDimension.ERR_COMPLIANCE, 1.0),
        (PaymentDimension.STAGE_SCREENING, 0.7),
    ]),
    PaymentTermMapping("routing error", [
        (PaymentDimension.ERR_ROUTING, 1.0),
        (PaymentDimension.STAGE_ROUTING, 0.6),
    ]),
    PaymentTermMapping("insufficient funds", [
        (PaymentDimension.ERR_INSUFFICIENT, 1.0),
    ]),
    PaymentTermMapping("duplicate", [
        (PaymentDimension.ERR_DUPLICATE, 1.0),
    ]),
    PaymentTermMapping("timeout", [
        (PaymentDimension.ERR_TIMEOUT, 1.0),
    ]),
    PaymentTermMapping("SLA", [
        (PaymentDimension.ERR_TIMEOUT, 0.7),
    ]),
    PaymentTermMapping("format error", [
        (PaymentDimension.ERR_FORMAT, 1.0),
    ]),
    PaymentTermMapping("parse error", [
        (PaymentDimension.ERR_FORMAT, 1.0),
    ]),
    PaymentTermMapping("reject", [
        (PaymentDimension.ERR_REJECT, 1.0),
    ]),
    PaymentTermMapping("rejection", [
        (PaymentDimension.ERR_REJECT, 1.0),
    ]),
    PaymentTermMapping("return", [
        (PaymentDimension.ERR_RETURN, 1.0),
    ]),
    PaymentTermMapping("repair", [
        (PaymentDimension.ERR_REPAIR, 1.0),
    ]),
    PaymentTermMapping("exception", [
        (PaymentDimension.ERR_REPAIR, 0.8),
    ]),
    PaymentTermMapping("STP", [
        (PaymentDimension.ERR_REPAIR, 0.3),  # Inverse - STP means no repair
        (PaymentDimension.STAGE_EXECUTION, 0.7),
    ]),
    PaymentTermMapping("straight through", [
        (PaymentDimension.STAGE_EXECUTION, 0.8),
    ]),
    
    # Business Capabilities
    PaymentTermMapping("wire transfer", [
        (PaymentDimension.CAP_HIGH_VALUE, 0.9),
        (PaymentDimension.NET_FEDWIRE, 0.5),
        (PaymentDimension.NET_SWIFT, 0.5),
    ]),
    PaymentTermMapping("high value", [
        (PaymentDimension.CAP_HIGH_VALUE, 1.0),
    ]),
    PaymentTermMapping("low value", [
        (PaymentDimension.CAP_LOW_VALUE, 1.0),
    ]),
    PaymentTermMapping("batch", [
        (PaymentDimension.CAP_LOW_VALUE, 0.8),
    ]),
    PaymentTermMapping("urgent", [
        (PaymentDimension.CAP_URGENT, 1.0),
    ]),
    PaymentTermMapping("priority", [
        (PaymentDimension.CAP_URGENT, 0.9),
    ]),
    PaymentTermMapping("scheduled", [
        (PaymentDimension.CAP_SCHEDULED, 1.0),
    ]),
    PaymentTermMapping("future dated", [
        (PaymentDimension.CAP_SCHEDULED, 1.0),
    ]),
    PaymentTermMapping("recurring", [
        (PaymentDimension.CAP_RECURRING, 1.0),
    ]),
    PaymentTermMapping("standing order", [
        (PaymentDimension.CAP_RECURRING, 1.0),
    ]),
    PaymentTermMapping("inquiry", [
        (PaymentDimension.CAP_INQUIRY, 1.0),
    ]),
    PaymentTermMapping("investigation", [
        (PaymentDimension.CAP_INQUIRY, 1.0),
    ]),
    PaymentTermMapping("amendment", [
        (PaymentDimension.CAP_AMENDMENT, 1.0),
    ]),
    PaymentTermMapping("modification", [
        (PaymentDimension.CAP_AMENDMENT, 1.0),
    ]),
    PaymentTermMapping("cancellation", [
        (PaymentDimension.CAP_CANCELLATION, 1.0),
    ]),
    PaymentTermMapping("recall", [
        (PaymentDimension.CAP_CANCELLATION, 0.9),
    ]),
    PaymentTermMapping("confirmation", [
        (PaymentDimension.CAP_CONFIRMATION, 1.0),
    ]),
    PaymentTermMapping("acknowledgment", [
        (PaymentDimension.CAP_CONFIRMATION, 0.9),
    ]),
    PaymentTermMapping("notification", [
        (PaymentDimension.CAP_NOTIFICATION, 1.0),
    ]),
    PaymentTermMapping("alert", [
        (PaymentDimension.CAP_NOTIFICATION, 0.9),
    ]),
]


class PaymentDomainEmbedder:
    """
    Creates structured payment domain embeddings with semantic dimensions.
    
    Instead of one dimension per keyword (sparse, unstructured), this creates
    80 meaningful dimensions that capture the semantic structure of payments.
    """
    
    def __init__(self):
        self.n_dimensions = N_PAYMENT_DIMENSIONS
        
        # Build lookup from term to dimensions
        self.term_to_dims: Dict[str, List[Tuple[int, float]]] = {}
        for mapping in PAYMENT_TERM_MAPPINGS:
            term_lower = mapping.term.lower()
            dims = [(d.value, w) for d, w in mapping.dimensions]
            self.term_to_dims[term_lower] = dims
            
            # Also add without dashes/spaces
            term_clean = term_lower.replace("-", "").replace(" ", "")
            if term_clean != term_lower:
                self.term_to_dims[term_clean] = dims
        
        # Build dimension names for interpretability
        self.dimension_names = [d.name for d in PaymentDimension]
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Generate payment domain embedding for text.
        
        Returns:
            80-dimensional vector with semantic payment dimensions
        """
        vector = np.zeros(self.n_dimensions)
        text_lower = text.lower()
        
        # Match all known terms
        matched_terms = set()
        for term, dims in self.term_to_dims.items():
            if term in text_lower:
                matched_terms.add(term)
                for dim_idx, weight in dims:
                    vector[dim_idx] = max(vector[dim_idx], weight)
        
        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector.tolist()
    
    def transform(self, text: str) -> np.ndarray:
        """Alias for get_embedding returning numpy array"""
        return np.array(self.get_embedding(text))
    
    def explain_embedding(self, text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Explain which dimensions are activated for a text.
        
        Returns:
            List of (dimension_name, weight) for top activated dimensions
        """
        embedding = self.transform(text)
        
        # Get indices sorted by value
        top_indices = np.argsort(embedding)[::-1][:top_k]
        
        explanations = []
        for idx in top_indices:
            if embedding[idx] > 0:
                explanations.append((self.dimension_names[idx], float(embedding[idx])))
        
        return explanations
    
    def get_dimension_groups(self) -> Dict[str, List[str]]:
        """Return dimension names grouped by category"""
        return {
            "Message Types": [d.name for d in PaymentDimension if d.name.startswith("MSG_")],
            "Networks": [d.name for d in PaymentDimension if d.name.startswith("NET_")],
            "Compliance": [d.name for d in PaymentDimension if d.name.startswith("COMP_")],
            "Processing Stages": [d.name for d in PaymentDimension if d.name.startswith("STAGE_")],
            "Parties": [d.name for d in PaymentDimension if d.name.startswith("PARTY_")],
            "Data Elements": [d.name for d in PaymentDimension if d.name.startswith("DATA_")],
            "Errors": [d.name for d in PaymentDimension if d.name.startswith("ERR_")],
            "Capabilities": [d.name for d in PaymentDimension if d.name.startswith("CAP_")],
        }


class HybridPaymentEmbedder:
    """
    Combines payment domain embeddings with text embeddings.
    
    Total dimensions: 80 (payment) + text_dim
    """
    
    def __init__(self, 
                 text_dim: int = 512,
                 payment_weight: float = 0.7,
                 text_weight: float = 0.3):
        """
        Initialize hybrid payment embedder.
        
        Args:
            text_dim: Dimension for text embeddings (hash-based)
            payment_weight: Weight for payment domain component
            text_weight: Weight for general text component
        """
        self.payment_embedder = PaymentDomainEmbedder()
        self.text_embedder = None  # Lazy load to avoid circular import
        self._text_dim = text_dim
        
        self.payment_weight = payment_weight
        self.text_weight = text_weight
        
        self.n_dimensions = self.payment_embedder.n_dimensions + text_dim
    
    def _get_text_embedder(self):
        """Lazy load text embedder to avoid circular import"""
        if self.text_embedder is None:
            from .embeddings import HashEmbedder
            self.text_embedder = HashEmbedder(n_features=self._text_dim)
        return self.text_embedder
    
    @property
    def n_features(self) -> int:
        """Alias for n_dimensions"""
        return self.n_dimensions
    
    def get_embedding(self, text: str) -> List[float]:
        """Generate hybrid embedding"""
        payment_vec = self.payment_embedder.transform(text) * self.payment_weight
        text_vec = self._get_text_embedder().transform(text) * self.text_weight
        
        combined = np.concatenate([payment_vec, text_vec])
        return combined.tolist()
    
    def transform(self, text: str) -> np.ndarray:
        """Alias for get_embedding returning numpy array"""
        return np.array(self.get_embedding(text))
    
    def explain_embedding(self, text: str, top_k: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        """
        Explain the embedding with payment dimensions.
        """
        return {
            "payment_dimensions": self.payment_embedder.explain_embedding(text, top_k)
        }


# Convenience function
def create_payment_embedder(
    embedder_type: str = "hybrid",
    text_dim: int = 512,
    payment_weight: float = 0.7,
    **kwargs
) -> PaymentDomainEmbedder:
    """
    Factory function to create payment domain embedders.
    
    Args:
        embedder_type: "domain" for payment-only, "hybrid" for payment+text
        text_dim: Dimension for text component (hybrid only)
        payment_weight: Weight for payment component (hybrid only)
        
    Returns:
        Configured embedder
    """
    if embedder_type == "domain":
        return PaymentDomainEmbedder()
    elif embedder_type == "hybrid":
        return HybridPaymentEmbedder(
            text_dim=text_dim,
            payment_weight=payment_weight,
            text_weight=1.0 - payment_weight
        )
    else:
        raise ValueError(f"Unknown embedder type: {embedder_type}")


# ============================================================
# Dimension Analysis Utilities
# ============================================================

def analyze_dimension_coverage(texts: List[str]) -> Dict[str, any]:
    """
    Analyze which dimensions are activated across a corpus.
    
    Returns:
        Statistics about dimension usage
    """
    embedder = PaymentDomainEmbedder()
    
    # Accumulate dimension activations
    activation_counts = np.zeros(N_PAYMENT_DIMENSIONS)
    activation_sums = np.zeros(N_PAYMENT_DIMENSIONS)
    
    for text in texts:
        embedding = embedder.transform(text)
        activation_counts += (embedding > 0).astype(int)
        activation_sums += embedding
    
    # Compute statistics
    dimension_stats = []
    for i, name in enumerate(embedder.dimension_names):
        dimension_stats.append({
            "dimension": name,
            "activation_count": int(activation_counts[i]),
            "activation_pct": float(activation_counts[i] / len(texts) * 100),
            "avg_weight": float(activation_sums[i] / max(activation_counts[i], 1))
        })
    
    # Sort by activation count
    dimension_stats.sort(key=lambda x: x["activation_count"], reverse=True)
    
    # Group by category
    category_stats = {}
    for group_name, dims in embedder.get_dimension_groups().items():
        group_activations = sum(
            activation_counts[PaymentDimension[d].value] 
            for d in dims
        )
        category_stats[group_name] = {
            "total_activations": int(group_activations),
            "dimensions": len(dims)
        }
    
    return {
        "total_texts": len(texts),
        "total_dimensions": N_PAYMENT_DIMENSIONS,
        "dimensions_used": int(np.sum(activation_counts > 0)),
        "dimension_details": dimension_stats,
        "category_summary": category_stats
    }
