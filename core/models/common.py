"""Common enums and types shared across the Dhan API."""

from enum import Enum


class ExchangeSegment(str, Enum):
    """Exchange segment identifiers."""

    NSE_EQ = "NSE_EQ"
    NSE_FNO = "NSE_FNO"
    NSE_CURRENCY = "NSE_CURRENCY"
    BSE_EQ = "BSE_EQ"
    BSE_FNO = "BSE_FNO"
    BSE_CURRENCY = "BSE_CURRENCY"
    MCX_COMM = "MCX_COMM"
    IDX_I = "IDX_I"


class TransactionType(str, Enum):
    """Order transaction type."""

    BUY = "BUY"
    SELL = "SELL"


class ProductType(str, Enum):
    """Product type for orders."""

    CNC = "CNC"  # Cash and Carry (delivery)
    INTRADAY = "INTRADAY"
    MARGIN = "MARGIN"
    MTF = "MTF"  # Margin Trading Facility
    CO = "CO"  # Cover Order
    BO = "BO"  # Bracket Order


class OrderType(str, Enum):
    """Order type."""

    LIMIT = "LIMIT"
    MARKET = "MARKET"
    STOP_LOSS = "STOP_LOSS"
    STOP_LOSS_MARKET = "STOP_LOSS_MARKET"


class Validity(str, Enum):
    """Order validity."""

    DAY = "DAY"
    IOC = "IOC"  # Immediate or Cancel


class OrderStatus(str, Enum):
    """Order status."""

    TRANSIT = "TRANSIT"
    PENDING = "PENDING"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"
    TRADED = "TRADED"
    EXPIRED = "EXPIRED"


class Instrument(str, Enum):
    """Instrument type."""

    EQUITY = "EQUITY"
    FUTIDX = "FUTIDX"
    FUTSTK = "FUTSTK"
    OPTIDX = "OPTIDX"
    OPTSTK = "OPTSTK"
    FUTCUR = "FUTCUR"
    OPTCUR = "OPTCUR"
    FUTCOM = "FUTCOM"
    OPTCOM = "OPTCOM"
    INDEX = "INDEX"


class ExpiryCode(int, Enum):
    """Expiry code for derivatives."""

    NEAR_EXPIRY = 0
    NEXT_EXPIRY = 1
    FAR_EXPIRY = 2
    NOT_APPLICABLE = -2147483648


class IntradayInterval(int, Enum):
    """Intraday candle interval in minutes."""

    ONE_MINUTE = 1
    FIVE_MINUTES = 5
    FIFTEEN_MINUTES = 15
    TWENTY_FIVE_MINUTES = 25
    ONE_HOUR = 60


class PositionType(str, Enum):
    """Position type."""

    LONG = "LONG"
    SHORT = "SHORT"
